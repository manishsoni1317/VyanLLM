
from transformers import GenerationConfig

from .vyanllm_base import VyanLLMBaseModel



class VyanLLMInternLM(VyanLLMBaseModel):


    def __init__(self, *args, **kwargs):


        super(VyanLLMInternLM, self).__init__(*args, **kwargs)

    def get_use_better_transformer(self):
        return False
    def get_generation_config(self):
        return GenerationConfig()


