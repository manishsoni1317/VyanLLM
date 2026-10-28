import sys
import unittest

#sys.path.insert(0, '../vyanllm')

from ..vyanllm.auto_model import AutoModel



class TestAutoModel(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass

    def test_auto_model_should_return_correct_model(self):
        mapping_dict = {
            'garage-bAInd/Platypus2-7B': 'VyanLLMLlama2',
            'Qwen/Qwen-7B': 'VyanLLMQWen',
            'internlm/internlm-chat-7b': 'VyanLLMInternLM',
            'THUDM/chatglm3-6b-base': 'VyanLLMChatGLM',
            'baichuan-inc/Baichuan2-7B-Base': 'VyanLLMBaichuan',
            'mistralai/Mistral-7B-Instruct-v0.1': 'VyanLLMMistral',
            'mistralai/Mixtral-8x7B-v0.1': 'VyanLLMMixtral'
        }


        for k,v in mapping_dict.items():
            module, cls = AutoModel.get_module_class(k)
            self.assertEqual(cls, v, f"expecting {v}")

