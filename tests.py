import os
import unittest

import numpy as np


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

        candidate_text: [str] = [
            '几点了',
            '啥时候了',
            '我饿了',
            '我想看电视',
            '今天天气怎么样',
            '你是男生还是女生'
        ]

        candidate_vector: [np.ndarray] = self.model.embedding_batch(candidate_text)

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
        with open(os.path.join(self.chat.corpus_base_dir, 'test.tsv'), 'w') as f:
            f.writelines([
                '几点了\t',
                '啥时候了\t',
                '我饿了\t',
                '我想看电视\t',
                '今天天气怎么样\t',
                '你是男生还是女生\t',
                '你的名字是什么\t'
            ])

    def test_reply(self):
        while True:
            text = input()
            if text == 'exit':
                break
            response, response_list = self.chat.reply(text)
            if response is None:
                print('没有找到答案，猜你想问：', [each.question for each in response_list])
            else:
                print(response)


if __name__ == '__main__':
    unittest.main()
