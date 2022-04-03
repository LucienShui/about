import os
import pickle
from typing import Callable

import numpy as np

from .entity import Knowledge, Question, MatchResult
from .func_set import now_date
from .model import Model
from .tools import concatenate_lists, argmax


class ChatResponse:
    def __init__(self, knowledge: Knowledge, question: str, score: float):
        self.knowledge = knowledge
        self.question = question
        self.score = score

        self.func_map: {str, Callable} = {
            func.__name__: func
            for func in [now_date]
        }

    def __str__(self):
        return self.answer

    @property
    def answer(self):
        if self.knowledge.answer.startswith('func:'):
            func_name = self.knowledge.answer[len('func:'):]
            return self.func_map[func_name]()
        return self.knowledge.answer

    def json(self):
        return {
            'knowledge': self.knowledge.name,
            'question': self.question,
            'answer': self.answer,
            'score': self.score
        }


class Chat:
    def __init__(self, pretrained: str = 'resource/model/simcse-chinese-roberta-wwm-ext',
                 embedding_type: str = 'CLS', skip_pickle: bool = False):
        self.model: Model = Model(pretrained, embedding_type)
        self.batch_size: int = 32

        self.skip_pickle: bool = skip_pickle
        self.corpus_base_dir: str = 'resource/corpus'  # standard question directory

        self.none_knowledge: Knowledge = Knowledge('未命中', '嘤嘤嘤，这个问题我还不会')
        self.none_response: ChatResponse = ChatResponse(self.none_knowledge, '', -1)

        self.knowledge_list: [Knowledge] = concatenate_lists([
            self.load_knowledge(os.path.join(self.corpus_base_dir, each))
            for each in os.listdir(self.corpus_base_dir) if each.endswith('.tsv')
        ])

    def load_knowledge(self, path: str) -> ([Knowledge], {str, np.ndarray}):
        path_without_ext, ext = os.path.splitext(path)
        if ext != '.tsv':
            raise ValueError('File extension must be .tsv')
        pickle_path: str = path_without_ext + '.pkl'
        try:
            if self.skip_pickle:
                raise FileNotFoundError
            with open(pickle_path, 'rb') as f:
                print('load corpus from %s' % pickle_path)
                text_to_vector: {str, np.ndarray} = pickle.load(f)
        except FileNotFoundError:
            text_to_vector: {str, np.ndarray} = {}

        knowledge_list: [Knowledge] = []
        text_list: [str] = []
        with open(path, encoding='gbk') as f:  # Excel 导出的文件是 GBK 编码
            for line_numer, line in enumerate(f):
                if line_numer == 0:  # 跳过列名
                    continue
                key, value, is_knowledge = line.strip().split('\t')
                if int(is_knowledge):
                    name, answer = key, value
                    knowledge_list.append(Knowledge(name, answer))
                else:
                    question, threshold = key, float(value) if value != '' else 0.7
                    knowledge_list[-1].question_list.append(Question(question, threshold))
                    if question not in text_to_vector:
                        text_list.append(question)
        vector_list: [str] = self.model.embedding_batch(text_list)
        for text, vector in zip(text_list, vector_list):
            text_to_vector[text] = vector
        print('load corpus from %s' % path)
        with open(pickle_path, 'wb') as f:
            pickle.dump(text_to_vector, f)

        for knowledge in knowledge_list:
            for question in knowledge.question_list:
                question.vector = text_to_vector[question.text]
        return knowledge_list

    def reply(self, text: str) -> ChatResponse:
        vector = self.model.embedding(text)
        match_result_list: [MatchResult] = [knowledge.match(vector) for knowledge in self.knowledge_list]
        best_index: int = argmax(match_result_list, lambda x: x.score)
        if match_result_list[best_index]:
            return ChatResponse(self.knowledge_list[best_index],
                                match_result_list[best_index].question, match_result_list[best_index].score)
        return self.none_response
