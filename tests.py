import unittest
import os
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


class TestChat(unittest.TestCase):
    def setUp(self) -> None:
        from about.chat import Chat
        self.chat = Chat('resource/model/albert_chinese_tiny', embedding_type='MEAN')

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

    def test_conversation(self):
        while True:
            text = input()
            if text == 'exit':
                break
            print(self.chat.response(text).json())


if __name__ == '__main__':
    unittest.main()
