from sys import platform

is_on_mac_os = False

if platform == "darwin":
    is_on_mac_os = True

if is_on_mac_os:
    from .vyanllm_llama_mlx import VyanLLMLlamaMlx
    from .auto_model import AutoModel
else:
    # Core entry points. These have no model-specific optional dependencies, so a plain
    # `import vyanllm` always works as long as torch/transformers are installed.
    from .vyanllm_base import VyanLLMBaseModel
    from .auto_model import AutoModel
    from .utils import split_and_save_layers
    from .utils import NotEnoughSpaceException
    from .utils import compress_layer_state_dict, uncompress_layer_state_dict

    # Dedicated subclasses for a handful of custom-architecture models. Some of them pull in
    # optional extras (e.g. the Baichuan tokenizer needs `sentencepiece`). Import them defensively
    # so a missing optional dependency for one niche family never breaks the whole package; the
    # generic VyanLLMBaseModel path keeps working regardless.
    import warnings as _warnings

    for _name, _module in (
        ("VyanLLMLlama2", ".vyanllm"),
        ("VyanLLMChatGLM", ".vyanllm_chatglm"),
        ("VyanLLMQWen", ".vyanllm_qwen"),
        ("VyanLLMQWen2", ".vyanllm_qwen2"),
        ("VyanLLMBaichuan", ".vyanllm_baichuan"),
        ("VyanLLMInternLM", ".vyanllm_internlm"),
        ("VyanLLMMistral", ".vyanllm_mistral"),
        ("VyanLLMMixtral", ".vyanllm_mixtral"),
        ("VyanLLMKimiK3", ".vyanllm_kimi_k3"),
        ("VyanLLMQwen3_5", ".vyanllm_qwen3_5"),
        ("VyanLLMQwen4Exp", ".vyanllm_qwen4_exp"),
        ("VyanLLMLoRA", ".vyanllm_lora"),
        ("VyanLLMLoRAQwen4Exp", ".vyanllm_lora"),
    ):
        try:
            _mod = __import__(__name__ + _module, fromlist=[_name])
            globals()[_name] = getattr(_mod, _name)
        except Exception as _e:  # noqa: BLE001 - optional family, keep package importable
            _warnings.warn(
                f"vyanllm: optional model class {_name} is unavailable ({_e}). "
                f"This only affects that specific model family; the generic streaming path still works."
            )

