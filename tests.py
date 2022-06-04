import unittest

import numpy as np


from typing import Mapping, List
from transformers.onnx import OnnxConfig
from collections import OrderedDict


class CustomOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "sequence"}),
                ("attention_mask", {0: "batch", 1: "sequence"}),
                ("token_type_ids", {0: "batch", 1: "sequence"}),
            ]
        )
    
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("last_hidden_state", {0: "batch", 1: "sequence"}),
                # ("pooler_output", {0: "batch", 1: "sequence"}),
            ]
        )


class ExportONNX(unittest.TestCase):
    def test_export_onnx(self):
        from pathlib import Path
        from transformers.onnx import export
        from transformers import AutoTokenizer, AutoModel

        pretrained: str = 'resource/model/simcse-chinese-roberta-wwm-ext'

        
        model = AutoModel.from_pretrained(pretrained)
        tokenizer = AutoTokenizer.from_pretrained(pretrained)

        onnx_path = Path("model.onnx")
        onnx_config = CustomOnnxConfig(model.config)

        onnx_inputs, onnx_outputs = export(tokenizer, model, onnx_config, onnx_config.default_onnx_opset, onnx_path)


class TestORM(unittest.TestCase):
    def test_init(self):
        from about.orm import Trace
        Trace.create(ip="127.0.0.1")


class TestRawModel(unittest.TestCase):
    def test_raw_model(self):
        from transformers import AutoModel, AutoTokenizer
        from about.model import ModelV2, ModelV3
        pretrained: str = 'resource/model/simcse-chinese-roberta-wwm-ext'
        model: AutoModel = AutoModel.from_pretrained(pretrained)
        tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(pretrained)
        text: str = 'Hello World!'
        token = tokenizer([text], return_tensors='pt')
        prediction = model(**token)
        embedding = prediction.last_hidden_state.detach().numpy()[:, 1:].mean(axis=1)[0]
        # print(embedding[:10])

        model_v2 = ModelV2(pretrained)
        embedding_v2 = model_v2.embedding(text)
        # print(embedding_v2[:10])

        model_v3 = ModelV3(pretrained, skip_cls=True)
        embedding_v3_1 = model_v3.embedding(text)
        # print(embedding_v3_1[:10])

        self.assertTrue(np.isclose(embedding, embedding_v3_1).all())

        model_v3.skip_cls = False
        embedding_v3_2 = model_v3.embedding(text)
        # print(embedding_v3_2[:10])
        self.assertTrue(np.isclose(embedding_v2, embedding_v3_2).all())

        self.assertTrue(True)


class TestModel(unittest.TestCase):
    def setUp(self) -> None:
        from about.model import Model
        self.model = Model('resource/model/albert_chinese_tiny', embedding_type='MAX')

    def test_model(self):
        result = self.model.predict('现在几点了')
        last_hidden_state, pooler_output = result['last_hidden_state'], result['pooler_output']
        print(last_hidden_state.shape)
        print(pooler_output.shape)
        self.assertTrue(True)

    def test_recall(self):
        from about.tools import cosine_similarity

        candidate_text: List[str] = [
            '几点了',
            '啥时候了',
            '我饿了',
            '我想看电视',
            '今天天气怎么样',
            '你是男生还是女生'
        ]

        candidate_vector: List[np.ndarray] = self.model.embedding_batch(candidate_text)

        result = self.model.embedding('现在几点了')
        for text, vector in zip(candidate_text, candidate_vector):
            print(text, cosine_similarity(result, vector))
        self.assertTrue(True)


class TestEmbedding(unittest.TestCase):
    def setUp(self) -> None:
        from about.model import ModelV2
        self.model_v2 = ModelV2('resource/model/simcse-chinese-roberta-wwm-ext')

    def test_embedding_v2(self):
        batch_embedding_result = self.model_v2.embedding_batch(['现在几点了'])
        print(batch_embedding_result.shape)
        embedding_result = self.model_v2.embedding('现在几点了')
        print(embedding_result.shape)


class TestChat(unittest.TestCase):
    def setUp(self) -> None:
        from about.chat import Chat
        self.chat = Chat('resource/model/simcse-chinese-roberta-wwm-ext')

    def test_chat(self):
        response = self.chat.reply('你叫什么')
        print(response.json())

    def test_reply_input(self):
        while True:
            text = input()
            if text == 'exit':
                break
            response = self.chat.reply(text)
            print(response.json())


if __name__ == '__main__':
    unittest.main()
