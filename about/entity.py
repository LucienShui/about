class Knowledge(object):
    def __init__(self, name: str, similar_question_list: [str], answer: str):
        self.name: str = name
        self.similar_question_list: [str] = similar_question_list
        self.answer: str = answer
