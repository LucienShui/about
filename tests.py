import unittest
import numpy as np


class TestModel(unittest.TestCase):
    def setUp(self) -> None:
        from about.model import Model
        self.model = Model('resource/model/albert_chinese_tiny')

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
            '今天天气怎么样'
        ]

        _type = 'MAX'  # 在这个超迷你测试集上，MAX 效果最好
        candidate_vector: [np.ndarray] = self.model.embedding_batch(candidate_text, _type)

        result = self.model.embedding('现在几点了', _type)
        for text, vector in zip(candidate_text, candidate_vector):
            print(text, cosine_similarity(result, vector))
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
