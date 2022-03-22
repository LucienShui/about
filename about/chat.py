import numpy as np
import pickle

from .model import Model
from .tools import cosine_similarity


class ChatResponse:
    def __init__(self, text, question, answer, score):
        self.text = text
        self.question = question
        self.answer = answer
        self.score = score

    def __str__(self):
        return self.answer

    def json(self):
        return {
            'text': self.text,
            'question': self.question,
            'answer': self.answer,
            'score': self.score
        }


class Chat:
    def __init__(self, pretrained: str = 'resource/model/albert_chinese_tiny',
                 embedding_type: str = 'CLS', skip_pickle: bool = False):
        self.model = Model(pretrained, embedding_type)
        self.batch_size = 32

        try:
            if skip_pickle:
                raise FileNotFoundError
            with open('resource/corpus/corpus.pkl', 'rb') as f:
                self.corpus = pickle.load(f)
        except FileNotFoundError:
            self.corpus: [(str, str, list)] = np.concatenate([
                self.load_corpus('resource/corpus/test.tsv'),
                # self.load_corpus('resource/corpus/ptt.tsv'),
                # self.load_corpus('resource/corpus/xiaohuangji.tsv'),
            ])

        with open('resource/corpus/corpus.pkl', 'wb') as f:
            pickle.dump(self.corpus, f)

    def load_corpus(self, path: str) -> [(str, str, list)]:
        result: [(str, str, list)] = []
        with open(path) as f:
            buffer_q: [str] = []
            buffer_a: [str] = []
            for line in f:
                q, a = line.strip().split('\t')
                buffer_q.append(q)
                buffer_a.append(a)

            buffer_v = self.model.embedding_batch(buffer_q, show_progress_bar=True)
            for q, a, v in zip(buffer_q, buffer_a, buffer_v):
                result.append((q, a, v))
            print('load corpus from %s' % path)
            return result

    def response(self, text: str) -> ChatResponse:
        vector = self.model.embedding(text)
        cosine_similarity_list = [cosine_similarity(vector, v) for q, a, v in self.corpus]
        response_index = np.argmax(cosine_similarity_list)
        q, a, _ = self.corpus[response_index]
        return ChatResponse(text, q, a, cosine_similarity_list[response_index])
