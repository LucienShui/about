from .model import Model


class Chat:
    def __init__(self):
        self.model = Model('resource/model/albert_chinese_tiny')
        self.batch_size = 32

        self.corpus = self.load_corpus('resource/corpus/')

    def load_corpus(self, path: str) -> [(str, str, list)]:
        result: [(str, str, list)] = []
        with open(path) as f:
            buffer_q: [str] = []
            buffer_a: [str] = []
            for line in f:
                q, a = line.strip().split('\t')
                buffer_q.append(q)
                buffer_a.append(a)

            buffer_v = self.model.embedding_batch(buffer_q)
            for q, a, v in zip(buffer_q, buffer_a, buffer_v):
                result.append((q, a, v))
