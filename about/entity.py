import numpy as np

from .tools import cosine_similarity


class MatchResult(object):
    def __init__(self, question: str, score: float):
        self.question = question
        self.score = score

    def __bool__(self):
        return self.score > 0


class Question(object):
    def __init__(self, text: str, threshold: float, vector: np.ndarray = None):
        self.text = text
        self.threshold = threshold
        self.vector = vector


class Knowledge(object):
    def __init__(self, name: str, answer: str, question_list: [Question] = None):
        self.name: str = name
        self.question_list: [Question] = question_list or []
        self.answer: str = answer

    def match(self, vector: np.ndarray) -> MatchResult:
        best_question = ''
        best_score = -1
        for question in self.question_list:
            score = cosine_similarity(question.vector, vector)
            if score >= question.threshold and score > best_score:
                best_question = question.text
                best_score = score

        return MatchResult(best_question, best_score)
