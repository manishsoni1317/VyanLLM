import importlib
from transformers import AutoConfig
from sys import platform

is_on_mac_os = False

if platform == "darwin":
    is_on_mac_os = True

if is_on_mac_os:
    from vyanllm import VyanLLMLlamaMlx

# Architectures that need a dedicated VyanLLM subclass because of a non-standard module layout
# (custom remote-code models). Everything else uses the generic VyanLLMBaseModel, which streams any
# standard *ForCausalLM (model.model.layers + lm_head / norm) and lets transformers own the
# forward pass, so newly released architectures work without code changes.
ARCH_OVERRIDES = {
    "ChatGLMModel": "VyanLLMChatGLM",
    "ChatGLMForConditionalGeneration": "VyanLLMChatGLM",
    "QWenLMHeadModel": "VyanLLMQWen",
    "BaichuanForCausalLM": "VyanLLMBaichuan",
    "BaiChuanForCausalLM": "VyanLLMBaichuan",
    "InternLMForCausalLM": "VyanLLMInternLM",
    "KimiK3ForConditionalGeneration": "VyanLLMKimiK3",
    "Qwen3_5ForConditionalGeneration": "VyanLLMQwen3_5",
    "Qwen4ExpForConditionalGeneration": "VyanLLMQwen4Exp",
}

# On macOS every model goes through VyanLLMLlamaMlx, an MLX port of the Llama decoder block, so only
# checkpoints with that layout can run there. Mistral uses the same block and weight names.
MLX_ARCHITECTURES = {"LlamaForCausalLM", "MistralForCausalLM"}


class AutoModel:
    def __init__(self):
        raise EnvironmentError(
            "AutoModel is designed to be instantiated "
            "using the `AutoModel.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def get_architecture(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        if 'hf_token' in kwargs:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True,
                                                token=kwargs['hf_token'])
        else:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)

        architectures = getattr(config, "architectures", None) or []
        return architectures[0] if architectures else ""

    @classmethod
    def get_module_class(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        arch = AutoModel.get_architecture(pretrained_model_name_or_path, *inputs, **kwargs)

        cls_name = ARCH_OVERRIDES.get(arch)
        if cls_name is None:
            print(f"using generic VyanLLM streaming model for architecture: {arch or 'unknown'}")
            cls_name = "VyanLLMBaseModel"
        return "vyanllm", cls_name

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):

        if is_on_mac_os:
            # Check this before VyanLLMLlamaMlx downloads and splits the whole checkpoint, otherwise an
            # unsupported model only fails (or gives garbage) after that is done.
            arch = AutoModel.get_architecture(pretrained_model_name_or_path, *inputs, **kwargs)
            if arch and arch not in MLX_ARCHITECTURES:
                raise ValueError(
                    f"{arch} is not supported on macOS yet. The macOS backend (MLX) only runs "
                    f"Llama-style models: {', '.join(sorted(MLX_ARCHITECTURES))}."
                )
            return VyanLLMLlamaMlx(pretrained_model_name_or_path, *inputs, **kwargs)

        module, class_name = AutoModel.get_module_class(pretrained_model_name_or_path, *inputs, **kwargs)
        module = importlib.import_module(module)
        class_ = getattr(module, class_name)
        return class_(pretrained_model_name_or_path, *inputs, **kwargs)
