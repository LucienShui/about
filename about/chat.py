import os
import pickle
from typing import Callable

from .entity import Knowledge
from .func_set import now_date
from .model import Model
from .tools import cosine_similarity, concatenate_lists


class ChatResponse:
    def __init__(self, question: str, answer: str, score: float):
        self.question = question
        self.__answer = answer
        self.score = score

        self.func_map: {str, Callable} = {
            func.__name__: func
            for func in [now_date]
        }

    def __str__(self):
        return self.answer

    @property
    def answer(self):
        if self.__answer.startswith('func:'):
            func_name = self.__answer[len('func:'):]
            return self.func_map[func_name]()
        return self.__answer

    def json(self):
        return {
            'question': self.question,
            'answer': self.answer,
            'score': self.score
        }


class Chat:
    def __init__(self, pretrained: str = 'resource/model/simcse-chinese-roberta-wwm-ext',
                 embedding_type: str = 'CLS', skip_pickle: bool = False):
        self.model = Model(pretrained, embedding_type)
        self.batch_size = 32

        self.text_to_vector: {str, list} = {}

        self.skip_pickle: bool = skip_pickle
        self.corpus_base_dir = 'resource/corpus'  # standard question directory
        self.none_response = ChatResponse('', '嘤嘤嘤，这个问题我还不会', 0.0)

        self.corpus: [(str, str, list)] = concatenate_lists([
            self.load_corpus(os.path.join(self.corpus_base_dir, each))
            for each in os.listdir(self.corpus_base_dir) if each.endswith('.tsv')
        ])

        self.knowledge_list: [Knowledge] = ...

    def load_knowledge(self, path: str) -> [Knowledge]:
        pass

    def load_corpus(self, path: str) -> [(str, str, list)]:
        path_without_ext, ext = os.path.splitext(path)
        if ext != '.tsv':
            raise ValueError('File extension must be .tsv')
        pickle_path: str = path_without_ext + '.pkl'
        try:
            if self.skip_pickle:
                raise FileNotFoundError
            with open(pickle_path, 'rb') as f:
                print('load corpus from %s' % pickle_path)
                return pickle.load(f)
        except FileNotFoundError:
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
            print('load corpus from %s' % path)
            with open(pickle_path, 'wb') as f:
                pickle.dump(result, f)
            return result

    def cosine_similarity(self, text_vector: [float], knowledge: Knowledge) -> (float, str):
        best_score = 0.0
        best_question = ''
        for question in knowledge.similar_question_list:
            score = cosine_similarity(self.text_to_vector[question], text_vector)
            if score > best_score:
                best_question = question
                best_score = score

        return best_score, best_question

    def reply(self, text: str, top_k: int = 3, threshold: float = 0.7) -> (ChatResponse, [ChatResponse]):
        vector = self.model.embedding(text)
        cosine_similarity_list: [(float, str)] = [
            self.cosine_similarity(vector, knowledge) for knowledge in self.knowledge_list]
        sorted_index = sorted(range(len(self.knowledge_list)), key=lambda x: cosine_similarity_list[x][0], reverse=True)
        response_list = [ChatResponse(question=cosine_similarity_list[i][1], answer=self.knowledge_list[i].answer,
                                      score=cosine_similarity_list[i][0]) for i in sorted_index[:top_k]]
        has_result = response_list[0].score >= threshold
        return response_list[0] if has_result else self.none_response, response_list[has_result:]
