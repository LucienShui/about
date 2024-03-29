import json
import os
import pickle
import random

import numpy as np

from typing import Dict, List

from .entity import Knowledge, Question, MatchResult
from .func_set import func_map
from .model import OnnxModel as Model
from .tools import concatenate_lists, argmax


class ChatResponse:
    def __init__(self, knowledge: Knowledge, question: str, score: float):
        self.knowledge = knowledge
        self.question = question
        self.score = score

    def __str__(self):
        return self.answer

    @property
    def answer(self):
        random_answer: str = random.choice(self.knowledge.answer_list)
        if random_answer.startswith('func:'):
            func_name = random_answer[len('func:'):]
            return func_map[func_name]()
        return random_answer

    def json(self):
        return {
            'knowledge': self.knowledge.name,
            'question': self.question,
            'answer': self.answer,
            'score': self.score
        }


class Chat:
    def __init__(self, pretrained: str = 'resource/model/simcse-chinese-roberta-wwm-ext',
                 embedding_type: str = 'MEAN', skip_pickle: bool = False, device: str = 'cpu'):
        self.supported_ext: List[str] = ['.tsv', '.json']

        self.model: Model = Model(pretrained, embedding_type, device=device)
        self.knowledge_base_dir: str = 'resource/knowledge'  # standard question directory
        self.none_response: ChatResponse = ChatResponse(Knowledge('未命中', ['嘤嘤嘤，这个问题我还不会']), '', -1)
        self.knowledge_list: List[Knowledge] = self.load_knowledge(self.knowledge_base_dir, skip_pickle)

    def is_ext_supported(self, path: str) -> bool:
        for ext in self.supported_ext:
            if path.endswith(ext):
                return True
        return False

    def load_knowledge(self, base_dir: str, skip_pickle: bool = False) -> List[Knowledge]:
        return concatenate_lists([
            self.__load_knowledge(os.path.join(base_dir, each), skip_pickle)
            for each in os.listdir(base_dir) if self.is_ext_supported(each)
        ])

    def __load_knowledge(self, path: str, skip_pickle: bool = False) -> List[Knowledge]:
        path_without_ext, ext = os.path.splitext(path)
        if ext not in self.supported_ext:
            raise ValueError('File extension must be .tsv or .json')
        pickle_path: str = path_without_ext + '.pkl'
        try:
            if skip_pickle:
                raise FileNotFoundError
            with open(pickle_path, 'rb') as f:
                print('load knowledge from %s' % pickle_path)
                text_to_vector: Dict[str, np.ndarray] = pickle.load(f)
        except FileNotFoundError:
            text_to_vector: Dict[str, np.ndarray] = {}

        knowledge_list: List[Knowledge] = []
        text_list: List[str] = []
        if ext == '.tsv':
            with open(path, encoding='gbk') as f:  # Excel 导出的文件是 GBK 编码
                for line_numer, line in enumerate(f):
                    if line_numer == 0:  # 跳过列名
                        continue
                    key, value, is_knowledge = line.strip().split('\t')
                    if int(is_knowledge):
                        name, answer = key, value
                        knowledge_list.append(Knowledge(name, [answer]))
                    else:
                        question, threshold = key, float(value) if value != '' else 0.7
                        knowledge_list[-1].question_list.append(Question(question, threshold))
        elif ext == '.json':
            with open(path, encoding='utf-8') as f:
                data: dict = json.load(f)
                for name, value in data.items():
                    question_list: list = value['question_list']
                    answer_list: list = value['answer_list']

                    knowledge = Knowledge(name, answer_list)

                    for each in question_list:
                        if isinstance(each, str):
                            knowledge.question_list.append(Question(each, 0.7))
                        elif (isinstance(each, tuple) or isinstance(each, list)) and len(each) == 2:
                            question, threshold = each
                            knowledge.question_list.append(Question(question, threshold))
                        else:
                            raise ValueError('question_list must be list of str or list of tuple')

                    knowledge_list.append(knowledge)

        for knowledge in knowledge_list:
            for question in knowledge.question_list:
                text = question.text
                if text not in text_to_vector:
                    text_list.append(text)

        if text_list:
            vector_list: np.ndarray = self.model.embedding_batch(text_list)
            for text, vector in zip(text_list, vector_list):
                text_to_vector[text] = vector
            print('load knowledge from %s' % path)
            with open(pickle_path, 'wb') as f:
                pickle.dump(text_to_vector, f)

        for knowledge in knowledge_list:
            for question in knowledge.question_list:
                question.vector = text_to_vector[question.text]
        return knowledge_list

    def reply(self, text: str) -> ChatResponse:
        vector = self.model.embedding(text)
        match_result_list: List[MatchResult] = [knowledge.match(vector) for knowledge in self.knowledge_list]
        best_index: int = argmax(match_result_list, lambda x: x.score)
        if match_result_list[best_index]:
            return ChatResponse(self.knowledge_list[best_index],
                                match_result_list[best_index].question, match_result_list[best_index].score)
        return self.none_response
