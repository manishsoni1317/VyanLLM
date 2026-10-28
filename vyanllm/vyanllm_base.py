
from typing import List, Optional, Tuple, Union
from tqdm import tqdm
from pathlib import Path
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer, GenerationConfig
from transformers.generation import GenerationMixin
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device
from transformers.quantizers import AutoHfQuantizer

from .profiler import LayeredProfiler

from .utils import clean_memory, trim_host_memory, uncompress_layer_state_dict, \
    is_compressed_layer_state_dict, load_layer, layer_tensor_names, load_layer_subset, \
    find_or_create_local_splitted_path, load_merged_ngram_embedding, \
    open_ngram_mmap_table, MmapEmbedding, _force_meta_embeddings, default_device
from .persist import ModelPersister

try:
    import bitsandbytes as bnb

    bitsandbytes_installed = True
    print('>>>> bitsandbytes installed')
except ImportError:
    bitsandbytes_installed = False


# Helpers that transformers 5.0 moved out of transformers.utils.generic. Remote model code is
# routinely written against an older transformers and still imports them from the old location,
# which makes such models fail to import at all. Re-exporting is enough to load them.
_RELOCATED_TRANSFORMERS_SYMBOLS = {
    'OutputRecorder': 'transformers.utils.output_capturing',
    'check_model_inputs': 'transformers.utils.output_capturing',
}


def restore_relocated_transformers_symbols():
    """Re-export moved transformers helpers under their old names, where they are missing."""
    import importlib
    import transformers.utils.generic as generic

    for name, new_home in _RELOCATED_TRANSFORMERS_SYMBOLS.items():
        if hasattr(generic, name):
            continue
        try:
            module = importlib.import_module(new_home)
        except ImportError:
            continue
        symbol = getattr(module, name, None)
        if symbol is not None:
            setattr(generic, name, symbol)


class VyanLLMBaseModel:
    """
    Memory-frugal wrapper around a Hugging Face ``*ForCausalLM`` model.

    The checkpoint is split into per-layer shards on disk. The real transformers model is
    instantiated on the ``meta`` device (no memory used) and owns the full forward / generation
    logic. VyanLLM only attaches forward hooks to each big module (embeddings, every decoder
    layer, the final norm and the lm_head) to stream that module's weights disk -> GPU right
    before it runs and free them right after, prefetching the next module on a worker thread.

    Because transformers drives the forward pass, VyanLLM no longer needs to track per-architecture
    attention/rotary/cache details: new model architectures work as soon as transformers supports
    them.
    """

    # Upper bound on how much pinned (page-locked) host memory a single prefetched layer may use.
    # Layers larger than this are loaded into ordinary pageable memory instead.
    max_pinned_layer_bytes = 2 * 1024 ** 3

    # Subclasses override this to point at non-standard module names.
    def set_layer_names_dict(self):
        self.layer_names_dict = {'embed': 'model.embed_tokens',
                                 'layer_prefix': 'model.layers',
                                 'norm': 'model.norm',
                                 'lm_head': 'lm_head'}

    def __init__(self, model_local_path_or_repo_id, device=None, dtype=None, max_seq_len=512,
                 layer_shards_saving_path=None, profiling_mode=False, compression=None,
                 hf_token=None, prefetching=True, delete_original=False,
                 install_hooks=True, load_resident=True, async_copy=None,
                 memory_cleanup="forward", precast_shards=True, cache_gb=None,
                 gpu_cache_gb=None):
        """
        Parameters
        ----------
        model_local_path_or_repo_id : str or Path
            path to the local model checkpoint or huggingface repo id
        device : str, optional
            runtime device. By default the best available is picked: CUDA, else Apple MPS,
            else CPU. Pass an explicit value (e.g. "cuda:1", "cpu") to override.
        dtype : torch.dtype, optional
            runtime dtype; defaults to the model's own config.torch_dtype (usually bfloat16 for
            modern models). float16 has too narrow a range for very deep models and overflows to
            inf/NaN, which silently corrupts the output, so we don't force it.
        max_seq_len : int, optional
            max seq length, by default 512
        layer_shards_saving_path : str, optional
            optional path to save the splitted shards, by default next to the model cache
        profiling_mode : bool, optional
            whether to profile the model loading time, default False
        compression: str, optional
            '4bit' or '8bit' to enable block-wise quantization of the on-disk shards
        hf_token: str, optional
            huggingface api token
        prefetching: bool, optional
            overlap the next layer's disk load with the current layer's compute
        delete_original: bool, optional
            delete the original downloaded checkpoint after splitting to save disk space
        install_hooks: bool, optional
            attach inference streaming hooks. Training drives load/evict itself, so LoRA
            training passes False — the hooks call ``module.to('meta')`` after forward and
            that breaks backward.
        load_resident: bool, optional
            load ``resident`` modules (e.g. a vision tower) onto the GPU. Text-only LoRA
            training leaves them on meta.
        async_copy: bool, optional
            with prefetching on CUDA, also copy the next layer to the GPU (and dequantize/cast it
            there) on a side CUDA stream, so the host->device transfer overlaps the current
            layer's compute. Costs one extra layer of VRAM. Default: on when CUDA is used.
        memory_cleanup: str, optional
            how often to run the expensive gc / malloc_trim / CUDA cache release:
            "layer" (old behaviour: after every module), "forward" (default: trim host heap once
            per forward pass, full cleanup after generate) or "end" (only after generate).
        cache_gb: float or "auto", optional
            memory budget for keeping layers between tokens instead of re-reading them from disk
            every token. On CUDA this is a pinned host-RAM tier; on Apple MPS / CPU (where the
            device shares system RAM) layers simply stay resident on the device. "auto" uses the
            available RAM minus a safety margin. Default None (off).
        gpu_cache_gb: float or "auto", optional
            CUDA only: VRAM budget for keeping layers resident on the GPU (no disk read and no
            PCIe copy). "auto" uses free VRAM minus headroom for activations and the KV cache.
            Filled before the cache_gb tier. Default None (off).
        precast_shards: bool, optional
            when the stored shards use a wider float dtype than the runtime dtype (e.g. fp32
            checkpoint run in bf16), write a one-time converted copy so every later load reads
            fewer bytes from disk and skips the cast. Default True; ignored with compression.
        """

        self.profiling_mode = profiling_mode
        self.profiler = LayeredProfiler()

        self.total_disk_loading_time = None
        self.total_gpu_loading_time = None
        self.total_compression_overhead_time = None
        self.hf_quantizer = None

        if compression is not None and not bitsandbytes_installed:
            raise ImportError('WARNING: bitsandbytes not found. Compression needs bitsandbytes. '
                              'To use compression, please install bitsandbytes: `pip install bitsandbytes`')

        self.compression = compression
        self.hf_token = hf_token

        restore_relocated_transformers_symbols()

        # This path streams safetensors shards, including on macOS (Apple GPU via MPS), where the
        # process-wide default would otherwise be the MLX backend's format.
        ModelPersister.use('safetensors')

        self.set_layer_names_dict()

        self.model_local_path, self.checkpoint_path = find_or_create_local_splitted_path(
            model_local_path_or_repo_id,
            layer_shards_saving_path,
            compression=compression,
            layer_names=self.layer_names_dict,
            hf_token=hf_token,
            delete_original=delete_original)

        self.running_device = device if device is not None else default_device()
        self.device = torch.device(self.running_device)

        # Prefer transformers' native implementation; only trust the model's bundled remote code when
        # transformers doesn't recognize the architecture. Vendored remote code is frequently pinned
        # to an old transformers and breaks against the current cache/generation APIs (e.g.
        # DeepSeek-V2's modeling_deepseek.py calls the long-removed DynamicCache.seen_tokens).
        token_kwargs = {'token': hf_token} if hf_token is not None else {}
        try:
            self.config = AutoConfig.from_pretrained(
                self.model_local_path, trust_remote_code=False, **token_kwargs)
            self.trust_remote_code = False
        except Exception:
            self.config = AutoConfig.from_pretrained(
                self.model_local_path, trust_remote_code=True, **token_kwargs)
            self.trust_remote_code = True

        # Default to the model's native dtype (bf16 for most modern models). Forcing fp16 overflows
        # on deep models (e.g. Qwen3-235B's 94 layers) and produces garbage; bf16's wider range
        # avoids it. Users can still override via dtype=.
        if dtype is None:
            cfg_dtype = getattr(self.config, "torch_dtype", None)
            if cfg_dtype is None:
                cfg_dtype = getattr(self.config, "dtype", None)
            if cfg_dtype is None:
                text_cfg = getattr(self.config, "text_config", None)
                cfg_dtype = getattr(text_cfg, "torch_dtype", None) or getattr(text_cfg, "dtype", None)
            if isinstance(cfg_dtype, str):
                cfg_dtype = getattr(torch, cfg_dtype, None)
            dtype = cfg_dtype if isinstance(cfg_dtype, torch.dtype) else torch.float16
        self.running_dtype = dtype
        self.dtype = self.running_dtype

        self.generation_config = self.get_generation_config()
        self.tokenizer = self.get_tokenizer(hf_token=hf_token)

        if memory_cleanup not in ("layer", "forward", "end"):
            raise ValueError(f"memory_cleanup must be 'layer', 'forward' or 'end', got {memory_cleanup!r}")
        self.memory_cleanup = memory_cleanup

        if precast_shards and self.compression is None:
            self.checkpoint_path = self._precast_shards_if_smaller(self.checkpoint_path)

        # prefetch executor / state. Compression no longer disables prefetching: the worker reads
        # the compressed bytes and dequantization happens on the device (see _prepare_layer).
        self.prefetching = prefetching
        self._executor = ThreadPoolExecutor(max_workers=1) if self.prefetching else None
        self._prefetch_future = None
        self._prefetched_idx = None

        on_cuda = self.device.type == 'cuda' and torch.cuda.is_available()
        if async_copy is None:
            async_copy = on_cuda and self.prefetching
        if async_copy and not (on_cuda and self.prefetching):
            print("async_copy needs CUDA and prefetching; disabling it.")
            async_copy = False
        self.async_copy = bool(async_copy)
        self._copy_stream = torch.cuda.Stream(device=self.device) if self.async_copy else None

        # Layer cache (see _plan_cache). Budgets are resolved once hooks are installed, when
        # shard sizes are known.
        self._cache_request = (cache_gb, gpu_cache_gb)
        self._host_cache = {}           # idx -> host-ready state dict (CUDA RAM tier)
        self._host_cache_bytes = 0
        self._host_cache_budget = 0
        self._device_resident = set()   # idx of modules kept materialised on the device
        self._device_cache_bytes = 0
        self._device_cache_budget = 0
        self._cache_lock = threading.Lock()
        self._cache_reported = False
        self.cache_stats = {'disk_bytes': 0, 'disk_reads': 0, 'ram_hits': 0, 'device_hits': 0}

        self.init_model()

        # compute layer count from the instantiated model
        model_attr = self.model
        for attr_name in self.layer_names_dict["layer_prefix"].split("."):
            model_attr = getattr(model_attr, attr_name)
        layers_count = len(model_attr)

        self.layer_names = [self.layer_names_dict['embed']] + \
                           [f'{self.layer_names_dict["layer_prefix"]}.{i}' for i in range(layers_count)] + \
                           [self.layer_names_dict['norm'], self.layer_names_dict['lm_head']]

        self.max_seq_len = max_seq_len

        self.set_layers_from_layer_names()
        if load_resident:
            self._load_resident_modules()
        self._load_cpu_resident_modules()
        if install_hooks:
            self._install_streaming_hooks()

    # ---- customization hooks for subclasses -------------------------------------------------

    def get_generation_config(self):
        try:
            return GenerationConfig.from_pretrained(self.model_local_path)
        except Exception:
            return GenerationConfig()

    def get_tokenizer(self, hf_token=None):
        if hf_token is not None:
            return AutoTokenizer.from_pretrained(self.model_local_path, token=hf_token, trust_remote_code=True)
        else:
            return AutoTokenizer.from_pretrained(self.model_local_path, trust_remote_code=True)

    # ---- model construction -----------------------------------------------------------------

    def _propagate_attn_implementation(self, impl):
        """Push the attention choice down into nested sub-configs.

        Multimodal wrappers keep the real decoder under a sub-config -- Kimi K3 uses ``text_config``
        -- and transformers records the request only on the config it was handed. The sub-model then
        reads an unset value and falls through to a flash-attention path, which fails outright on a
        machine without flash-attn installed.
        """
        from transformers import PretrainedConfig

        def walk(cfg, depth=0):
            if depth > 2:
                return
            for sub in vars(cfg).values():
                if isinstance(sub, PretrainedConfig):
                    sub._attn_implementation = impl
                    walk(sub, depth + 1)

        walk(self.config)

    def _model_matches_layer_names(self, model):
        """True when the instantiated class actually has the modules we plan to stream.

        Auto factories key off ``model_type``, so a VL checkpoint can land on a text-only or
        backbone class whose tree doesn't match ``layer_names_dict``. Reject those and keep trying.
        """
        try:
            for key in ('embed', 'layer_prefix', 'norm', 'lm_head'):
                mod = model
                for attr in self.layer_names_dict[key].split('.'):
                    mod = getattr(mod, attr)
            return True
        except AttributeError:
            return False

    def _auto_model_classes(self):
        """Auto* factories to try, in order, when building the empty model.

        ``AutoModelForCausalLM`` maps some VL model_types (notably ``qwen3_5``) onto a text-only
        ``*ForCausalLM`` class. That class expects a text config and either crashes or builds a
        model whose module names don't match the checkpoint. Conditional-generation architectures
        therefore try the image-text factories first.
        """
        archs = getattr(self.config, "architectures", None) or []
        arch = archs[0] if archs else ""
        names = []
        if any(tag in arch for tag in ("ConditionalGeneration", "ImageTextToText", "Multimodal")):
            names.extend(["AutoModelForImageTextToText", "AutoModelForMultimodalLM"])
        names.extend([
            "AutoModelForCausalLM",
            "AutoModelForImageTextToText",
            "AutoModelForMultimodalLM",
            "AutoModel",
        ])
        seen = set()
        classes = []
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            cls = getattr(transformers, name, None)
            if cls is not None:
                classes.append((name, cls))
        return classes

    def _instantiate_on_meta(self, attn_implementation):
        """Build the transformers model on the meta device for one attention implementation."""
        self._propagate_attn_implementation(attn_implementation)
        kwargs = {
            "attn_implementation": attn_implementation,
            "trust_remote_code": self.trust_remote_code,
        }
        errors = []
        with _force_meta_embeddings():
            for name, cls in self._auto_model_classes():
                try:
                    with init_empty_weights(include_buffers=False):
                        model = cls.from_config(self.config, **kwargs)
                except TypeError:
                    # Older Auto factories don't take attn_implementation.
                    try:
                        with init_empty_weights(include_buffers=False):
                            model = cls.from_config(self.config, trust_remote_code=self.trust_remote_code)
                    except Exception as e:  # noqa: BLE001 - try the next factory
                        errors.append(f"{name}: {type(e).__name__}: {e}")
                        continue
                except Exception as e:  # noqa: BLE001 - try the next factory
                    errors.append(f"{name}: {type(e).__name__}: {e}")
                    continue
                if not self._model_matches_layer_names(model):
                    errors.append(
                        f"{name}: built {type(model).__name__} but it is missing "
                        f"{self.layer_names_dict['layer_prefix']}"
                    )
                    continue
                print(f"built empty {type(model).__name__} via {name} (attn={attn_implementation})")
                return model
        raise RuntimeError(
            "Could not instantiate the model on meta from config. "
            f"architecture={getattr(self.config, 'architectures', None)} "
            f"model_type={getattr(self.config, 'model_type', None)}. "
            f"Tried: {'; '.join(errors)}"
        )

    def init_model(self):
        # Build the real model on meta (no memory). include_buffers=False so non-persistent
        # buffers such as rotary inv_freq are actually computed (they aren't in the checkpoint).
        try:
            self.model = self._instantiate_on_meta("sdpa")
        except Exception as e:
            print(f"attn_implementation='sdpa' not available ({e}), falling back to eager attention")
            # Some (often remote-code) architectures don't support sdpa and also default to it, so we
            # must request eager explicitly; otherwise transformers re-selects sdpa and errors again.
            self.model = self._instantiate_on_meta("eager")

        quantization_config = getattr(self.config, "quantization_config", None)
        if quantization_config is None:
            # Nested multimodal configs (Kimi K3) keep it under text_config.
            quantization_config = getattr(getattr(self.config, "text_config", None),
                                          "quantization_config", None)
        if quantization_config is not None:
            self.hf_quantizer = AutoHfQuantizer.from_config(quantization_config, pre_quantized=True)
            device_map = self.hf_quantizer.update_device_map(None)
            self.hf_quantizer.preprocess_model(model=self.model, device_map=device_map)
            # compressed-tensors registers a hook that expands every packed module on the first
            # forward. That undoes per-expert streaming (a K3 layer becomes ~56GB) and is also
            # unnecessary: we decompress each expert ourselves as it loads. Remove the hook.
            hook = getattr(self.model, "ct_decompress_hook", None)
            if hook is not None:
                hook.remove()
                delattr(self.model, "ct_decompress_hook")

        self.model.eval()
        self.model.tie_weights()
        self.model.generation_config = self.generation_config

        # Move all (already-materialized) buffers to the running device, preserving their dtype.
        # This includes rotary inv_freq, which transformers computes once at the model level and
        # passes down to every decoder layer.
        for buffer_name, buffer in self.model.named_buffers():
            if buffer is not None and buffer.device.type != 'meta':
                set_module_tensor_to_device(self.model, buffer_name, self.running_device, value=buffer)

        # Force the model to report the running (cuda) device even though its parameters live on
        # meta between layer executions, so transformers' generation utilities place inputs/cache
        # tensors on the right device.
        self._patch_device_property()

    def _patch_device_property(self):
        running_device = torch.device(self.running_device)
        running_dtype = self.running_dtype
        base_cls = type(self.model)

        # transformers >= 4.50 removed GenerationMixin from PreTrainedModel, so model classes that
        # predate that change (or ship as remote code, like Kimi K3's multimodal wrapper) no longer
        # have .generate(). They still define prepare_inputs_for_generation, so mixing the class
        # back in restores generation. It must come after the model class, per transformers.
        extra_bases = () if isinstance(self.model, GenerationMixin) else (GenerationMixin,)
        if extra_bases:
            print(f"{base_cls.__name__} does not inherit GenerationMixin; mixing it in so "
                  f"generate() works.")

        class _VyanLLMRuntimeModel(base_cls, *extra_bases):
            @property
            def device(self):
                return running_device

            @property
            def dtype(self):
                return running_dtype

        self.model.__class__ = _VyanLLMRuntimeModel

    def set_layers_from_layer_names(self):
        self.layers = []

        model_attr = self.model
        for attr_name in self.layer_names_dict["embed"].split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.append(model_attr)

        model_attr = self.model
        for attr_name in self.layer_names_dict["layer_prefix"].split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.extend(list(model_attr))

        model_attr = self.model
        for attr_name in self.layer_names_dict["norm"].split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.append(model_attr)

        model_attr = self.model
        for attr_name in self.layer_names_dict["lm_head"].split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.append(model_attr)

    # ---- weight streaming -------------------------------------------------------------------

    def load_layer_to_cpu(self, layer_name):
        t = time.time()
        load_layer_output = load_layer(self.checkpoint_path, layer_name, self.profiling_mode)
        elapsed_time = time.time() - t

        if self.profiling_mode:
            state_dict, compression_time = load_layer_output
            disk_loading_time = elapsed_time - compression_time
            self.profiler.add_profiling_time('load_safe_tensor', disk_loading_time)
            self.profiler.add_profiling_time('compression_time', compression_time)
        else:
            state_dict = load_layer_output

        if self.prefetching and torch.cuda.is_available():
            # pin_memory() returns a pinned copy rather than pinning in place, so the result has to
            # be kept for the faster host->device copy to actually happen. Pinned memory can't be
            # paged out, so we only spend it on layers small enough to be safe: a frontier MoE
            # checkpoint has ~17GB layers, and with prefetching two are in flight at once, which
            # would lock up ~34GB of RAM for a copy that is dwarfed by the disk read anyway.
            total_bytes = sum(v.numel() * v.element_size() for v in state_dict.values())
            if total_bytes <= self.max_pinned_layer_bytes:
                try:
                    for k in state_dict.keys():
                        state_dict[k] = state_dict[k].pin_memory()
                except RuntimeError:
                    # Out of pinned memory: fall back to pageable, which is slower but always works.
                    pass

        return state_dict

    def _restore_plain_weight_modules(self, state_dict):
        """Undo CT's packed-parameter layout when the checkpoint still ships a plain ``weight``.

        Some checkpoints (Kimi K3) list residual/router Linears under the MXFP4 target list but
        store them as bf16. After ``preprocess_model`` those modules expose ``weight_packed`` and
        reject the real ``weight`` tensor. Restore a meta ``weight`` Parameter so the shard can load.
        """
        plain = {k[:-len('.weight')] for k in state_dict if k.endswith('.weight')}
        packed = {k[:-len('.weight_packed')] for k in state_dict if k.endswith('.weight_packed')}
        for prefix in plain - packed:
            try:
                module = self.model.get_submodule(prefix)
            except AttributeError:
                continue
            names = list(module._parameters.keys())
            if 'weight' in names or 'weight_packed' not in names:
                continue
            weight = state_dict[f'{prefix}.weight']
            for name in names:
                if name == 'weight' or name.startswith('weight_'):
                    module._parameters.pop(name, None)
            module.register_parameter(
                'weight',
                torch.nn.Parameter(torch.empty(weight.shape, device='meta', dtype=weight.dtype),
                                   requires_grad=False),
            )
            for attr in ('quantization_scheme', 'quantization_status', 'quantization_format'):
                if hasattr(module, attr):
                    delattr(module, attr)

    def _decompress_state_dict(self, state_dict):
        """Expand packed payloads into the plain weights the modules expect.

        compressed-tensors has no quantized compute kernels: it registers a hook that decompresses
        the whole model before the first forward, after which every quantized module wants a plain
        ``weight``. Our shards still hold ``weight_packed``/``weight_scale``, so we expand them here.

        The expansion happens on the GPU, after transferring the *packed* bytes. For MXFP4 that
        moves 4x less data across PCIe than transferring an already-expanded weight would.
        """
        if self.hf_quantizer is None:
            return state_dict

        packed_prefixes = {k[: -len('.weight_packed')]
                           for k in state_dict if k.endswith('.weight_packed')}
        if not packed_prefixes:
            return state_dict

        from compressed_tensors.compressors.base import BaseCompressor
        try:
            from compressed_tensors.linear.compressed_linear import CompressedLinear
        except ImportError:
            CompressedLinear = None

        out = dict(state_dict)
        for prefix in packed_prefixes:
            try:
                module = self.model.get_submodule(prefix)
            except AttributeError:
                continue
            # CompressedLinear's own forward consumes the packed payload, so leave it packed.
            if CompressedLinear is not None and isinstance(module, CompressedLinear):
                continue
            scheme = getattr(module, 'quantization_scheme', None)
            if scheme is None or getattr(scheme, 'format', None) is None:
                continue

            local = {k[len(prefix) + 1:]: v.to(self.running_device)
                     for k, v in state_dict.items() if k.startswith(prefix + '.')}
            # scheme.format is an enum on some compressed-tensors versions and a plain str on others.
            fmt = getattr(scheme.format, 'value', scheme.format)
            compressor = BaseCompressor.get_value_from_registry(fmt)
            decompressed = compressor.decompress(local, scheme)

            for k in local:
                out.pop(f'{prefix}.{k}', None)
            if 'weight' in decompressed:
                # Once expanded, the module reads only `weight`; the scales that came back merely
                # describe how the checkpoint stored it, and keeping them would waste VRAM.
                out[f'{prefix}.weight'] = decompressed['weight']
                self._expose_plain_weight(module, decompressed['weight'])
            else:
                for k, v in decompressed.items():
                    out[f'{prefix}.{k}'] = v
        return out

    def _expose_plain_weight(self, module, weight):
        """Swap a module's packed parameters for the plain ``weight`` its forward reads.

        compressed-tensors patches a ``quantized_forward`` onto quantized Linears that reads
        ``self.weight``; the packed parameters only describe how the checkpoint stores the value.
        Marking the module COMPRESSED tells that forward the weight is already on the quantization
        grid, so it skips a fake-quantize that would only reproduce what we just decompressed.
        """
        existing = module._parameters.get('weight')
        if existing is None or existing.shape != weight.shape:
            for name in [n for n in list(module._parameters) if n.startswith('weight')]:
                module._parameters.pop(name, None)
            module.register_parameter(
                'weight',
                torch.nn.Parameter(torch.empty(weight.shape, device='meta', dtype=weight.dtype),
                                   requires_grad=False),
            )
        try:
            from compressed_tensors.quantization import QuantizationStatus
        except ImportError:
            return
        module.quantization_status = QuantizationStatus.COMPRESSED

    def _adopt_checkpoint_shape(self, param_name, value):
        """Resize a meta placeholder whose shape disagrees with the checkpoint.

        A model class builds its parameters from the config, and that construction can disagree
        with the weights actually shipped. Kimi K3 sizes ``A_log`` from ``num_heads`` while its
        checkpoint stores one entry per head channel, which would abort the load. For a parameter
        still on meta -- i.e. one we have never materialised -- the checkpoint is the source of
        truth, so adopt its shape.
        """
        module_path, _, attr = param_name.rpartition('.')
        try:
            module = self.model.get_submodule(module_path) if module_path else self.model
        except AttributeError:
            return
        current = module._parameters.get(attr)
        if current is None or current.device.type != 'meta' or current.shape == value.shape:
            return

        if not hasattr(self, '_shape_adoption_warned'):
            self._shape_adoption_warned = set()
        if attr not in self._shape_adoption_warned:
            self._shape_adoption_warned.add(attr)
            print(f"{attr}: checkpoint ships {tuple(value.shape)} but the model class builds "
                  f"{tuple(current.shape)}; using the checkpoint shape.")

        module.register_parameter(
            attr,
            torch.nn.Parameter(torch.empty(value.shape, device='meta', dtype=current.dtype),
                               requires_grad=False),
        )

    def move_layer_to_device(self, state_dict):
        if self.compression is not None and is_compressed_layer_state_dict(state_dict):
            # Shards from the prefetch worker arrive still compressed unless async_copy already
            # dequantized them on the side stream. Expand them on the device, in the runtime dtype.
            state_dict = uncompress_layer_state_dict(state_dict, device=self.running_device,
                                                     dtype=self.running_dtype)
        self._restore_plain_weight_modules(state_dict)
        state_dict = self._decompress_state_dict(state_dict)
        marker = self.layer_names_dict.get('cpu_resident_marker')
        moved = []
        for param_name in self._param_names_from_state_dict(state_dict):
            # A nested CPU table (Flash-Next PLE) must never be placed on the GPU, even if a
            # decoder-layer shard still contains it.
            if param_name in getattr(self, '_cpu_resident_params', ()):
                continue
            if marker and marker in param_name:
                continue
            if self.hf_quantizer is not None and self._needs_quantization(param_name):
                # On-the-fly-quantizing schemes (e.g. bitsandbytes) reconstruct the param from the
                # weight plus companion quant-state tensors carried in state_dict.
                self.hf_quantizer.create_quantized_param(self.model, state_dict[param_name], param_name,
                                                         self.running_device, state_dict)
            else:
                # Normal load. Only ordinary high-precision tensors get cast to the runtime dtype;
                # pre-quantized payloads must be placed verbatim (see _should_load_verbatim).
                value = state_dict[param_name]
                self._adopt_checkpoint_shape(param_name, value)
                if self._should_load_verbatim(param_name, value):
                    set_module_tensor_to_device(self.model, param_name, self.running_device, value=value)
                else:
                    set_module_tensor_to_device(self.model, param_name, self.running_device,
                                                value=value, dtype=self.running_dtype)
            moved.append(param_name)
        return moved

    # Suffixes of the companion tensors that pre-quantized checkpoints ship alongside a weight
    # (fp8 block scales, compressed-tensors/MXFP4 packed payloads and their scales, GPTQ indices).
    _QUANT_COMPANION_SUFFIXES = ("_scale", "_scale_inv", "_packed", "_zero_point", "_g_idx", "_shape")

    def _should_load_verbatim(self, param_name, value):
        """Whether a checkpoint tensor must be placed on the device without a dtype cast.

        Casting a pre-quantized payload to the runtime dtype destroys it: an fp8 weight loses its
        quantization, and a 4-bit MXFP4 ``weight_packed`` tensor (stored as packed integers) becomes
        meaningless floats. Equally important for very large models, decompressing on load would
        multiply a layer's footprint by ~4x, which is what keeps Kimi-K3-class checkpoints from
        fitting on a single GPU. So we keep anything that isn't a plain high-precision float as-is.
        """
        if not value.is_floating_point():
            # Packed 4-bit payloads, zero points, g_idx, shape metadata.
            return True
        if value.element_size() == 1:
            # Any 8-bit float: fp8 e4m3/e5m2 weights, and the e8m0 scales MXFP4 uses.
            return True
        return param_name.endswith(self._QUANT_COMPANION_SUFFIXES)

    def _needs_quantization(self, param_name):
        q = self.hf_quantizer
        # transformers renamed check_quantized_param -> param_needs_quantization.
        if hasattr(q, "param_needs_quantization"):
            return q.param_needs_quantization(self.model, param_name)
        return q.check_quantized_param(self.model, param_value=None, param_name=param_name, state_dict={})

    def _param_names_from_state_dict(self, state_dict):
        names = []
        for param_name in state_dict.keys():
            # bitsandbytes stores a weight plus companion quant-state tensors named
            # "<weight>.4bit.*" / "<weight>.8bit.*"; those are reconstructed together via
            # create_quantized_param, so collapse them down to the base weight name. Everything
            # else (including fp8 weight + weight_scale_inv pairs) is kept as distinct params.
            if '.4bit.' in param_name or '.8bit.' in param_name:
                base = param_name.split('.4bit.')[0].split('.8bit.')[0]
                if base not in names:
                    names.append(base)
            elif param_name not in names:
                names.append(param_name)
        return names

    def _load_resident_modules(self):
        """Load modules that sit outside the streamed embed -> layers -> norm -> lm_head sequence.

        Multimodal checkpoints carry a vision tower and projector, and some architectures add
        extra top-level norms. They never get a streaming hook, so without this they would stay on
        the meta device and fail the moment they run. They are small (well under a GB), so we load
        them once and leave them resident.
        """
        for name in self.layer_names_dict.get('resident', []):
            try:
                state_dict = self.load_layer_to_cpu(name)
            except FileNotFoundError:
                # Not every checkpoint of a given architecture ships every optional module.
                continue
            self.move_layer_to_device(state_dict)

    def _cpu_resident_names(self):
        """Module prefixes whose weights stay on CPU for the lifetime of the process.

        Flash-Next's n-gram table is the motivating case: transformers already gathers rows onto
        the embedding's own device, so keeping ~102GB of bf16 on the host is the designed path.
        Names come from ``cpu_resident`` and from split files matching ``cpu_resident_marker``.
        """
        names = list(self.layer_names_dict.get('cpu_resident', []))
        marker = self.layer_names_dict.get('cpu_resident_marker')
        if marker:
            for path in Path(self.checkpoint_path).glob('*.mmap.json'):
                stem = path.name[:-len('.mmap.json')]
                if stem.endswith(marker) and stem not in names:
                    names.append(stem)
            for path in Path(self.checkpoint_path).glob('*.safetensors'):
                stem = path.name[:-len('.safetensors')]
                if stem.endswith(marker) and stem not in names:
                    names.append(stem)
        return names

    def _install_mmap_embedding(self, module_name, table):
        parent_name, _, attr = module_name.rpartition('.')
        parent = self.model.get_submodule(parent_name)
        setattr(parent, attr, MmapEmbedding(table))

    def _load_cpu_resident_modules(self):
        """Load oversized lookup tables onto CPU and never stream them to the GPU.

        Flash-Next's n-gram table is ~102GB bf16. We mmap it from disk so a 64GB host can still
        run; transformers already gathers rows on ``ngram_embedding.weight.device``.
        ``module.to('meta')`` after a decoder layer would otherwise evict a child embedding that
        lives under that layer, so we also record the parameter names and evict selectively.
        """
        self._cpu_resident_params = set()
        for name in self._cpu_resident_names():
            mmap_meta = Path(self.checkpoint_path) / f"{name}.mmap.json"
            if mmap_meta.exists():
                table = open_ngram_mmap_table(self.checkpoint_path, name)
                self._install_mmap_embedding(name, table)
                self._cpu_resident_params.add(f"{name}.weight")
                print(f"cpu-resident mmap embedding: {name} "
                      f"{tuple(table.shape)} {table.dtype} (gathered from disk, not copied to RAM)")
                continue
            try:
                state_dict = load_merged_ngram_embedding(self.checkpoint_path, name)
            except FileNotFoundError:
                continue
            for param_name, value in state_dict.items():
                self._adopt_checkpoint_shape(param_name, value)
                if self._should_load_verbatim(param_name, value):
                    set_module_tensor_to_device(self.model, param_name, 'cpu', value=value)
                else:
                    set_module_tensor_to_device(
                        self.model, param_name, 'cpu', value=value, dtype=self.running_dtype)
                self._cpu_resident_params.add(param_name)
        if self._cpu_resident_params:
            n = len(self._cpu_resident_params)
            print(f"cpu-resident modules: left {n} tensors on host RAM "
                  f"(n-gram / PLE table is gathered on CPU, not streamed to the GPU).")

    def _install_streaming_hooks(self):
        # Modules execute in this order during a forward: embed -> layers -> norm -> lm_head.
        n = len(self.layer_names)

        # Detect tied input/output embeddings. When tied, lm_head shares the embedding weight, so
        # there is no separate lm_head shard. We keep the embedding resident on the GPU (it is the
        # only copy and such models are small) and re-tie lm_head to it, then stream only the
        # decoder layers and the final norm.
        self.tie_word_embeddings = bool(getattr(self.config, "tie_word_embeddings", False))

        if self.tie_word_embeddings:
            embed_state = self.load_layer_to_cpu(self.layer_names[0])
            self.move_layer_to_device(embed_state)
            self.model.tie_weights()
            self._streamed_indices = list(range(1, n - 1))  # decoder layers + final norm
        else:
            self._streamed_indices = list(range(n))

        self._streamed_set = set(self._streamed_indices)

        self._setup_expert_streaming()

        for idx in self._streamed_indices:
            module = self.layers[idx]
            module._vyanllm_idx = idx
            module.register_forward_pre_hook(self._pre_hook)
            module.register_forward_hook(self._post_hook)

        self._plan_cache(*self._cache_request)

    # ---- per-expert streaming ---------------------------------------------------------------

    def _setup_expert_streaming(self):
        """Stream individual MoE experts instead of whole decoder layers, where that is possible.

        A sparse MoE layer holds hundreds of experts but routes each token to a handful of them.
        Materialising the whole layer is therefore enormously wasteful: for Kimi K3 a layer's
        experts are ~55GB expanded, of which a token touches ~1GB. Because the model calls each
        selected expert as its own module (and skips unselected ones), a forward hook per expert
        loads exactly the experts that actually run.

        This needs `expert_prefix` in layer_names_dict and safetensors shards, since it relies on
        reading individual tensors out of a shard.
        """
        self._expert_streaming = False
        self._expert_keys = {}
        self._non_expert_keys = {}

        expert_prefix = self.layer_names_dict.get('expert_prefix')
        if not expert_prefix:
            return
        if type(ModelPersister.get_model_persister()).__name__ != 'SafetensorModelPersister':
            return
        if self.compression is not None:
            # Per-expert loading reads tensors straight out of the shard with load_layer_subset(),
            # which -- unlike load_layer() -- does not run uncompress_layer_state_dict(). With
            # compression on, the shard holds a quantized payload plus companion .4bit./.8bit.
            # quant-state tensors, so streaming subsets of it would hand still-quantized weights to
            # move_layer_to_device(). Fall back to whole-layer streaming, which does decompress.
            print("per-expert streaming is not supported together with compression for now; "
                  "streaming whole layers instead.")
            return

        layer_prefix = self.layer_names_dict['layer_prefix']
        hooked = 0

        for idx in self._streamed_indices:
            layer_name = self.layer_names[idx]
            if not layer_name.startswith(layer_prefix + '.'):
                continue
            try:
                names = layer_tensor_names(self.checkpoint_path, layer_name)
            except Exception:
                continue

            marker = f'.{expert_prefix}.'
            per_expert = {}
            others = []
            for key in names:
                pos = key.find(marker)
                if pos == -1:
                    others.append(key)
                    continue
                rest = key[pos + len(marker):]
                head = rest.split('.', 1)[0]
                if not head.isdigit():
                    others.append(key)
                    continue
                per_expert.setdefault(int(head), []).append(key)

            if not per_expert:
                continue

            layer_module = self.layers[idx]
            experts_container = layer_module
            try:
                for attr in expert_prefix.split('.'):
                    experts_container = getattr(experts_container, attr)
            except AttributeError:
                continue

            self._non_expert_keys[idx] = others
            self._expert_keys[idx] = per_expert

            for expert_idx, keys in per_expert.items():
                if expert_idx >= len(experts_container):
                    continue
                expert_module = experts_container[expert_idx]
                expert_module._vyanllm_expert = (idx, expert_idx)
                expert_module.register_forward_pre_hook(self._expert_pre_hook)
                expert_module.register_forward_hook(self._expert_post_hook)
                hooked += 1

        if hooked:
            self._expert_streaming = True
            n_layers = len(self._expert_keys)
            print(f"per-expert streaming enabled: {hooked} experts across {n_layers} layers "
                  f"load on demand, so only the experts a token routes to are materialised.")

    def _expert_pre_hook(self, module, args):
        layer_idx, expert_idx = module._vyanllm_expert
        keys = self._expert_keys[layer_idx][expert_idx]
        state_dict = load_layer_subset(self.checkpoint_path, self.layer_names[layer_idx], keys)
        module._vyanllm_moved = self.move_layer_to_device(state_dict)

    def _expert_post_hook(self, module, args, output):
        for param_name in getattr(module, '_vyanllm_moved', []):
            set_module_tensor_to_device(self.model, param_name, 'meta')
        module._vyanllm_moved = []
        return output

    def _next_streamed_idx(self, idx):
        """The next module (cyclically) that will actually need loading.

        Device-resident modules are skipped, so while they run the worker is already reading the
        next module that is not cached. After the last module the next forward pass (the next
        generated token) starts again at the beginning, so the search wraps around.
        """
        order = self._streamed_indices
        if not order:
            return None
        pos = order.index(idx) if idx in self._streamed_set else -1
        for step in range(1, len(order) + 1):
            cand = order[(pos + step) % len(order)]
            if cand not in self._device_resident:
                return cand
        return None

    def _read_streamed_layer(self, idx):
        """Read one streamed module's shard from disk. Experts are excluded when they stream themselves.

        Compressed shards are returned still compressed: moving the compressed bytes and
        dequantizing on the device is both less PCIe traffic and safe to do off the main thread.
        """
        keys = self._non_expert_keys.get(idx) if getattr(self, '_expert_streaming', False) else None
        if keys is not None:
            return load_layer_subset(self.checkpoint_path, self.layer_names[idx], keys)

        t = time.time()
        state_dict = load_layer(self.checkpoint_path, self.layer_names[idx], decompress=False)
        if self.profiling_mode:
            self.profiler.add_profiling_time('load_safe_tensor', time.time() - t)
        self.cache_stats['disk_reads'] += 1
        self.cache_stats['disk_bytes'] += self._layer_bytes(state_dict)
        return state_dict

    def _layer_bytes(self, state_dict):
        return sum(v.numel() * v.element_size() for v in state_dict.values())

    def _castable(self, key, value):
        """Whether a tensor would be cast to the runtime dtype when placed (mirrors move_layer_to_device)."""
        return (self.hf_quantizer is None
                and value.is_floating_point()
                and value.dtype != self.running_dtype
                and not self._should_load_verbatim(key, value))

    def _stays_on_host(self, key):
        marker = self.layer_names_dict.get('cpu_resident_marker')
        return key in getattr(self, '_cpu_resident_params', ()) or bool(marker and marker in key)

    # ---- layer cache ------------------------------------------------------------------------
    #
    # Every token sweeps the modules in the same fixed order. For that cyclic pattern an LRU cache
    # smaller than the model never hits (it always evicts exactly the layer needed next), so the
    # cache is static: the first modules that fit are kept for good and only the rest stream.
    # That is the optimal policy for cyclic access, and admission happens during the first pass
    # using each module's real size.

    @staticmethod
    def _fmt_bytes(n):
        return f"{n / 1024 ** 3:.2f} GB" if n >= 1024 ** 3 else f"{n / 1024 ** 2:.1f} MB"

    def _shard_bytes(self, idx):
        try:
            return os.path.getsize(Path(self.checkpoint_path) / (self.layer_names[idx] + ".safetensors"))
        except OSError:
            return 0

    def _plan_cache(self, cache_gb, gpu_cache_gb):
        on_cuda = self.device.type == 'cuda'
        largest = max((self._shard_bytes(i) for i in self._streamed_indices), default=0)
        if self.compression == '4bit':
            largest_expanded = int(largest / 0.28)
        elif self.compression == '8bit':
            largest_expanded = largest * 2
        else:
            largest_expanded = largest
        GB = 1024 ** 3

        def host_auto():
            import psutil
            vm = psutil.virtual_memory()
            # Leave room for the OS and other apps, plus the layers in flight while streaming.
            margin = max(4 * GB, int(0.15 * vm.total)) + 3 * largest_expanded
            return max(0, vm.available - margin)

        def resolve(req, auto_fn):
            if req is None or req == 0:
                return 0
            if isinstance(req, str):
                if req.strip().lower() == 'auto':
                    return auto_fn()
                try:
                    req = float(req)  # e.g. "6" from a command line
                except ValueError:
                    raise ValueError(f"cache size must be a number of GB or 'auto', got {req!r}")
            if float(req) < 0:
                raise ValueError(f"cache size must not be negative, got {req!r}")
            return int(float(req) * GB)

        if on_cuda:
            def device_auto():
                free, _ = torch.cuda.mem_get_info(self.device)
                # Activations, the KV cache and up to two streamed layers in flight.
                return max(0, free - 2 * GB - 3 * largest_expanded)
            self._device_cache_budget = resolve(gpu_cache_gb, device_auto)
            self._host_cache_budget = resolve(cache_gb, host_auto)
        else:
            # MPS / CPU: the device *is* system RAM, so keeping modules resident on it is strictly
            # better than a separate host copy that would be copied again every token.
            if gpu_cache_gb not in (None, 0):
                print("gpu_cache_gb applies to CUDA only; on this device use cache_gb.")
            self._device_cache_budget = resolve(cache_gb, host_auto)
            self._host_cache_budget = 0

        if isinstance(cache_gb, str) and cache_gb.strip().lower() == 'auto' and not (self._device_cache_budget or self._host_cache_budget):
            import psutil
            vm = psutil.virtual_memory()
            print(f"layer cache: 'auto' found no spare memory ({self._fmt_bytes(vm.available)} available "
                  f"of {self._fmt_bytes(vm.total)}, and the safety margin keeps at least "
                  f"{self._fmt_bytes(max(4 * GB, int(0.15 * vm.total)))} free), so the cache is off. "
                  f"Pass cache_gb=<GB> to set a budget yourself.")

        if self._device_cache_budget or self._host_cache_budget:
            total = sum(self._shard_bytes(i) for i in self._streamed_indices)
            parts = []
            if self._device_cache_budget:
                parts.append(f"{self._fmt_bytes(self._device_cache_budget)} resident on {self.device.type}")
            if self._host_cache_budget:
                parts.append(f"{self._fmt_bytes(self._host_cache_budget)} pinned RAM")
            print(f"layer cache: up to {' + '.join(parts)} for a {self._fmt_bytes(total)} model "
                  f"(filled during the first forward pass)")

    def _tensor_bytes(self, module):
        n = 0
        for t in list(module.parameters(recurse=True)) + list(module.buffers(recurse=True)):
            if t.device.type not in ('meta', 'cpu') or self.device.type == 'cpu' and t.device.type == 'cpu':
                n += t.numel() * t.element_size()
        return n

    def _try_admit_device(self, idx, module):
        """Keep a module that just ran materialised on the device if it fits the budget."""
        if idx in self._device_resident or not self._device_cache_budget:
            return idx in self._device_resident
        if getattr(self, '_expert_streaming', False) and idx in self._expert_keys:
            return False  # experts manage their own lifetime; keep MoE layers streaming
        size = self._tensor_bytes(module)
        if size == 0 or self._device_cache_bytes + size > self._device_cache_budget:
            return False
        self._device_resident.add(idx)
        self._device_cache_bytes += size
        with self._cache_lock:
            # Already held on the device: a host copy would only waste the RAM budget.
            dropped = self._host_cache.pop(idx, None)
            if dropped is not None:
                self._host_cache_bytes -= self._layer_bytes(dropped)
        return True

    def _try_admit_host(self, idx, host):
        if not self._host_cache_budget or idx in self._device_resident:
            return
        size = self._layer_bytes(host)
        with self._cache_lock:
            if idx in self._host_cache or self._host_cache_bytes + size > self._host_cache_budget:
                return
            self._host_cache[idx] = dict(host)
            self._host_cache_bytes += size

    def clear_cache(self):
        """Release every cached layer (RAM and device tiers). Budgets stay as configured."""
        self.drop_prefetch()
        with self._cache_lock:
            self._host_cache.clear()
            self._host_cache_bytes = 0
        for idx in list(self._device_resident):
            module = self.layers[idx]
            if (self.hf_quantizer is not None or getattr(self, '_expert_streaming', False)
                    or getattr(self, '_cpu_resident_params', None)):
                for name in getattr(module, '_vyanllm_moved', []):
                    set_module_tensor_to_device(self.model, name, 'meta')
            else:
                module.to('meta')
            module._vyanllm_moved = []
        self._device_resident.clear()
        self._device_cache_bytes = 0
        self._cache_reported = False
        clean_memory()

    def _report_cache(self):
        if self._cache_reported or not (self._device_cache_budget or self._host_cache_budget):
            return
        self._cache_reported = True
        GB = 1024 ** 3
        n = len(self._streamed_indices)
        streamed = n - len(self._device_resident) - len(self._host_cache)
        disk = sum(self._shard_bytes(i) for i in self._streamed_indices
                   if i not in self._device_resident and i not in self._host_cache)
        msg = [f"layer cache filled: {len(self._device_resident)}/{n} modules resident on "
               f"{self.device.type} ({self._fmt_bytes(self._device_cache_bytes)})"]
        if self._host_cache_budget:
            msg.append(f"{len(self._host_cache)} in pinned RAM ({self._fmt_bytes(self._host_cache_bytes)})")
        msg.append(f"{streamed} still stream from disk (~{self._fmt_bytes(disk)} per token)")
        print("; ".join(msg))

    # ---- preparing a module's weights -------------------------------------------------------

    def _host_ready(self, idx):
        """The module's weights in host memory, cast and pinned as the transfer path wants them.

        Served from the RAM cache when possible; otherwise read from disk, prepared, and offered
        to the cache. Returns (state_dict, compressed, use_device_path).
        """
        with self._cache_lock:
            cached = self._host_cache.get(idx)
        if cached is not None:
            self.cache_stats['ram_hits'] += 1
            state_dict = dict(cached)
        else:
            state_dict = self._read_streamed_layer(idx)
        compressed = self.compression is not None and is_compressed_layer_state_dict(state_dict)

        # hf_quantizer checkpoints are reconstructed by the quantizer from host tensors, and
        # oversized layers would cost too much VRAM to double-buffer; both keep the plain path.
        use_device_path = (self.async_copy
                           and self.hf_quantizer is None
                           and self._layer_bytes(state_dict) <= self.max_pinned_layer_bytes)
        if cached is not None:
            return state_dict, compressed, use_device_path

        if not use_device_path and not compressed:
            # Cast here (worker thread) rather than inside move_layer_to_device on the main
            # thread; the later placement then finds the tensor already in the right dtype.
            # (On the device path the cast happens on the GPU instead, which is cheaper still.)
            for k, v in state_dict.items():
                if self._castable(k, v):
                    state_dict[k] = v.to(self.running_dtype)
        if torch.cuda.is_available() and (self.prefetching or self._host_cache_budget) and \
                self._layer_bytes(state_dict) <= self.max_pinned_layer_bytes:
            try:
                for k in state_dict:
                    if not self._stays_on_host(k):
                        state_dict[k] = state_dict[k].pin_memory()
            except RuntimeError:
                pass  # out of pinned memory: pageable copies are slower but always work
        self._try_admit_host(idx, state_dict)
        return state_dict, compressed, use_device_path

    def _prepare_layer(self, idx):
        """Everything needed before a module can run, except binding tensors to it.

        Runs on the prefetch worker thread when prefetching, so all of it overlaps the previous
        module's compute: disk read (or RAM-cache hit), dtype cast and -- with async_copy -- the
        host->device copy on a side CUDA stream and on-device dequantization.

        Returns (state_dict, on_device). When on_device is True the tensors already live on the
        runtime device, were produced on the copy stream, and the copy has completed.
        """
        host, compressed, use_device_path = self._host_ready(idx)
        if not use_device_path:
            return host, False

        t = time.time()
        with torch.cuda.stream(self._copy_stream):
            if compressed:
                out = uncompress_layer_state_dict(
                    {k: v for k, v in host.items() if not self._stays_on_host(k)},
                    device=self.device, dtype=self.running_dtype, non_blocking=True)
            else:
                out = {}
                for k, v in host.items():
                    if self._stays_on_host(k):
                        continue
                    dv = v.to(self.device, non_blocking=True)
                    # Cast on the GPU: far cheaper than on the CPU, and still off the compute stream.
                    out[k] = dv.to(self.running_dtype) if self._castable(k, dv) else dv
            for k, v in host.items():
                if self._stays_on_host(k):
                    out[k] = v
            done = torch.cuda.Event()
            done.record(self._copy_stream)
        # Blocks only this worker thread. Once the copy is done the host buffers may be released
        # (or stay in the RAM cache), and the main thread can use the tensors without waiting.
        done.synchronize()
        if self.profiling_mode:
            self.profiler.add_profiling_time('async_h2d', time.time() - t)
        return out, True

    def _load_streamed_layer(self, idx):
        """Back-compat entry point: the module's state dict, ready for move_layer_to_device."""
        return self._prepare_layer(idx)[0]

    def _pre_hook(self, module, args):
        idx = module._vyanllm_idx

        if idx in self._device_resident:
            # Cached on the device: nothing to load. Make sure the worker is busy with the next
            # module that does need loading.
            self.cache_stats['device_hits'] += 1
            if self.prefetching and self._prefetch_future is None:
                nxt = self._next_streamed_idx(idx)
                if nxt is not None:
                    self._prefetch_future = self._executor.submit(self._prepare_layer, nxt)
                    self._prefetched_idx = nxt
            return

        if self.prefetching and self._prefetch_future is not None and self._prefetched_idx == idx:
            state_dict, on_device = self._prefetch_future.result()
            self._prefetch_future = None
        else:
            if self._prefetch_future is not None:
                # A stale prefetch (e.g. an out-of-order forward). Let it finish and drop it so
                # its buffers are freed before we allocate this module's.
                self._prefetch_future.result()
                self._prefetch_future = None
            state_dict, on_device = self._prepare_layer(idx)

        if on_device:
            # These tensors were allocated on the copy stream but will be read on the compute
            # stream. record_stream stops the caching allocator from handing their memory to the
            # next prefetch while this module's kernels may still be reading it.
            compute = torch.cuda.current_stream(self.device)
            for v in state_dict.values():
                if v.device.type == 'cuda':
                    v.record_stream(compute)

        module._vyanllm_moved = self.move_layer_to_device(state_dict)
        del state_dict

        if self.prefetching:
            nxt = self._next_streamed_idx(idx)
            if nxt is not None:
                self._prefetch_future = self._executor.submit(self._prepare_layer, nxt)
                self._prefetched_idx = nxt

    def _post_hook(self, module, args, output):
        idx = module._vyanllm_idx
        is_last = idx == self._streamed_indices[-1]

        if not self._try_admit_device(idx, module):
            # module.to('meta') would also evict experts (which manage their own lifetime) and any
            # cpu-resident child such as Flash-Next's n-gram table, which lives under a decoder
            # layer but must stay materialised on the host. When any of those are in play, only
            # release the tensors this hook actually placed.
            if (self.hf_quantizer is not None
                    or getattr(self, '_expert_streaming', False)
                    or getattr(self, '_cpu_resident_params', None)):
                for param_name in getattr(module, '_vyanllm_moved', []):
                    set_module_tensor_to_device(self.model, param_name, 'meta')
            else:
                module.to('meta')
            module._vyanllm_moved = []

        if is_last:
            self._report_cache()

        if self.memory_cleanup == "layer":
            clean_memory()
        elif self.memory_cleanup == "forward" and is_last:
            # Once per forward pass: hand freed host pages back to the OS. gc.collect and
            # torch.cuda.empty_cache are deliberately not run here -- the evicted layer's tensors
            # are freed by refcount already, and emptying the CUDA cache would only force the
            # next layer to cudaMalloc the same blocks again.
            trim_host_memory()
        return output

    def drop_prefetch(self):
        """Discard any in-flight prefetch (it holds one layer in RAM or VRAM)."""
        if self._prefetch_future is not None:
            try:
                self._prefetch_future.result()
            finally:
                self._prefetch_future = None
                self._prefetched_idx = None

    def release_memory(self):
        """Drop the speculative prefetch and run a full gc / malloc_trim / CUDA cache release."""
        self.drop_prefetch()
        clean_memory()

    # ---- one-time dtype conversion of the shards (precast_shards) ------------------------------

    _ST_DTYPES = {'F64': torch.float64, 'F32': torch.float32, 'F16': torch.float16,
                  'BF16': torch.bfloat16}

    def _precast_shards_if_smaller(self, checkpoint_path):
        """Return a shard directory whose plain float tensors are already in the runtime dtype.

        Only done when it shrinks the shards (e.g. fp32 -> bf16), because then every token reads
        proportionally fewer bytes from disk -- the dominant cost of layer streaming. When the
        dtypes already match, or the conversion would not save bytes, the original path is used.
        """
        from safetensors import safe_open
        from safetensors.torch import save_file
        import os, shutil

        if getattr(self.config, 'quantization_config', None) is not None:
            return checkpoint_path  # leave pre-quantized checkpoints exactly as shipped

        checkpoint_path = Path(checkpoint_path)
        target = self.running_dtype
        target_size = torch.tensor([], dtype=target).element_size()
        shards = sorted(checkpoint_path.glob('*.safetensors'))

        def wider(path):
            with safe_open(str(path), framework='pt') as f:
                for k in f.keys():
                    dt = self._ST_DTYPES.get(f.get_slice(k).get_dtype())
                    if dt is not None and dt.itemsize > target_size and \
                            not k.endswith(self._QUANT_COMPANION_SUFFIXES):
                        return True
            return False

        if not shards or not any(wider(p) for p in shards):
            return checkpoint_path

        tag = str(target).replace('torch.', '')
        out_dir = checkpoint_path.parent / f"{checkpoint_path.name}.{tag}"
        done_marker = out_dir / '.precast.done'
        if done_marker.exists():
            print(f"using pre-cast {tag} shards: {out_dir}")
            return out_dir

        needed = sum(p.stat().st_size for p in shards)  # upper bound
        out_dir.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(out_dir).free < needed:
            print(f"precast_shards: not enough free disk for a {tag} copy; keeping original shards.")
            return checkpoint_path

        print(f"precast_shards: converting shards to {tag} once (fewer bytes to read per token) -> {out_dir}")
        for entry in checkpoint_path.iterdir():
            dst = out_dir / entry.name
            if dst.exists():
                continue
            if entry.suffix == '.safetensors' and wider(entry):
                tensors = {}
                with safe_open(str(entry), framework='pt') as f:
                    meta = f.metadata()
                    for k in f.keys():
                        v = f.get_tensor(k)
                        if v.is_floating_point() and v.element_size() > target_size and \
                                not k.endswith(self._QUANT_COMPANION_SUFFIXES):
                            v = v.to(target)
                        tensors[k] = v.contiguous()
                tmp = dst.with_name(dst.name + '.tmp')
                save_file(tensors, str(tmp), metadata=meta)
                os.replace(tmp, dst)
                del tensors
            else:
                # Everything else (narrow shards, done markers, n-gram mmaps) is reused as-is.
                try:
                    os.symlink(entry.resolve(), dst)
                except OSError:
                    if entry.is_file():
                        shutil.copy2(entry, dst)
        done_marker.touch()
        return out_dir

    # ---- delegation to the underlying transformers model ------------------------------------

    def generate(self, *args, **kwargs):
        try:
            return self.model.generate(*args, **kwargs)
        finally:
            if self.memory_cleanup in ("forward", "end"):
                # The wrap-around prefetch of the first module is speculative; release it and do
                # the full cleanup once here instead of after every layer.
                self.release_memory()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
