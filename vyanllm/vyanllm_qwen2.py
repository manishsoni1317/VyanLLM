
from transformers import GenerationConfig


from .vyanllm_base import VyanLLMBaseModel



class VyanLLMQWen2(VyanLLMBaseModel):


    def __init__(self, *args, **kwargs):


        super(VyanLLMQWen2, self).__init__(*args, **kwargs)

    def get_use_better_transformer(self):
        return False


