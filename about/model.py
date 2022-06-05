import numpy as np
from functools import partial  # tqdm 同行进度条
from tqdm import tqdm
from typing import List, Dict, Callable
from .tools import concatenate_lists

try:
    import torch
except ImportError:
    class SomeClass:
        pass


    torch = SomeClass
    torch.cuda = None
    torch.device = None
    torch.no_grad = object

tqdm = partial(tqdm, position=0, leave=True)  # tqdm 同行进度条


class SentenceEmbedding(object):
    def embedding(self, text: str) -> np.ndarray:
        raise NotImplementedError

    def embedding_batch(self, text_list: List[str]) -> np.ndarray:
        raise NotImplementedError


class ModelV1(SentenceEmbedding):
    def __init__(self, pretrained: str, embedding_type: str = 'CLS',
                 skip_cls: bool = False, max_length: int = 128, skip_load_model: bool = False):
        self.embedding_type = embedding_type
        self.skip_cls = skip_cls  # skip_cls or not when calc with last_hidden_state
        self.max_length = max_length
        if self.embedding_type not in ['CLS', 'MEAN', 'MAX', 'MIN', 'AVG']:
            raise ValueError('self.embedding_type must be one of "CLS", "MEAN", "MAX", "MIN", "AVG"')

        self.pretrained = pretrained

        if skip_load_model:
            return

        from transformers import AlbertModel, BertTokenizerFast

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(self.pretrained)
        self.model = AlbertModel.from_pretrained(self.pretrained)
        self.model.to(device=self.device)

        self.model.eval()

        print('load finished')

    def _predict_batch(self, text_list: List[str]) -> Dict[str, np.ndarray]:
        if self.max_length > 0:
            encoded_ids = self.tokenizer(text_list, padding='max_length', truncation=True,
                                         max_length=self.max_length, return_tensors="pt")
        else:
            encoded_ids = self.tokenizer(text_list, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in encoded_ids.items()}

        with torch.no_grad():
            huggingface_prediction = self.model(**inputs)

        return {
            "last_hidden_state": huggingface_prediction.last_hidden_state.cpu().numpy(),
            "pooler_output": huggingface_prediction.pooler_output.cpu().numpy()
        }

    def predict_batch(self, text_list: List[str], batch_size: int = 32, show_progress_bar: bool = False) -> dict:
        last_hidden_state_list: List[np.ndarray] = []
        pooler_output_list: List[np.ndarray] = []

        rg = range(0, len(text_list), batch_size)
        if show_progress_bar:
            rg = tqdm(rg)
        for i in rg:
            text_batch = text_list[i:i + batch_size]
            prediction = self._predict_batch(text_batch)

            last_hidden_state_list.append(prediction['last_hidden_state'])
            pooler_output_list.append(prediction['pooler_output'])

        last_hidden_state: np.ndarray = np.concatenate(last_hidden_state_list, axis=0)
        pooler_output: np.ndarray = np.concatenate(pooler_output_list, axis=0)

        return {"last_hidden_state": last_hidden_state, "pooler_output": pooler_output}

    def predict(self, text: str) -> dict:
        result = self.predict_batch([text])
        for key in result.keys():
            result[key] = result[key][0]
        return result

    def pooling(self, outputs: Dict[str, np.ndarray]) -> np.ndarray:
        if self.embedding_type == 'CLS':
            return outputs['pooler_output']
        if self.embedding_type == 'MEAN':
            return outputs['last_hidden_state'][:, self.skip_cls:].mean(axis=1)
        if self.embedding_type == 'MAX':
            return outputs['last_hidden_state'][:, self.skip_cls:].max(axis=1)
        if self.embedding_type == 'MIN':
            return outputs['last_hidden_state'][:, self.skip_cls:].min(axis=1)

    def embedding_batch(self, text_list: List[str], batch_size: int = 32,
                        show_progress_bar: bool = False) -> np.ndarray:
        predict_result = self.predict_batch(text_list, batch_size, show_progress_bar)
        return self.pooling(predict_result)

    def embedding(self, text: str) -> np.ndarray:
        return self.embedding_batch([text])[0]


class ModelV2(SentenceEmbedding):
    def __init__(self, pretrained: str, _: str = 'MEAN'):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise e
        self.pretrained = pretrained
        self.encoder = SentenceTransformer(self.pretrained)

    def embedding(self, text: str) -> np.ndarray:
        return self.embedding_batch([text])[0]

    def embedding_batch(self, text_list: List[str]) -> np.ndarray:
        sentence_embeddings = self.encoder.encode(text_list)
        return sentence_embeddings


class ModelV3(ModelV1):
    def __init__(self, pretrained: str, embedding_type: str = 'MEAN', skip_cls: bool = False, max_length: int = 0):
        from transformers import BertTokenizerFast, AutoModel

        super(ModelV3, self).__init__(pretrained, embedding_type, skip_cls, max_length, skip_load_model=True)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(self.pretrained)
        self.model: AutoModel = AutoModel.from_pretrained(self.pretrained)
        self.model.to(device=self.device)

        self.model.eval()

        print('load finished')


class OnnxModel(ModelV1):
    def __init__(self, pretrained: str, embedding_type: str = 'MEAN', skip_cls: bool = False, max_length: int = 0):
        from onnxruntime import InferenceSession
        from transformers import BertTokenizerFast
        import os

        super(OnnxModel, self).__init__(pretrained, embedding_type, skip_cls, max_length, skip_load_model=True)

        self.pretrained = pretrained
        self.tokenizer: Callable = BertTokenizerFast.from_pretrained(self.pretrained)
        self.model: InferenceSession = InferenceSession(os.path.join(self.pretrained, 'model.onnx'))

        self.output_names = ['last_hidden_state', 'pooler_output']

    def embedding(self, text: str) -> np.ndarray:
        return self.pooling(self._predict(text))[0]

    def embedding_batch(self, text_list: List[str], batch_size: int = 32,
                        show_progress_bar: bool = False) -> np.ndarray:
        if self.max_length > 0:
            pass
        else:
            return np.array([self.embedding(text) for text in text_list])

    def _predict(self, text: str) -> Dict[str, np.ndarray]:
        encoded_ids: np.ndarray = self.tokenizer(text, return_tensors="np")
        outputs = self.model.run(output_names=self.output_names, input_feed=dict(encoded_ids))
        return {k: v for k, v in zip(self.output_names, outputs)}

    """
    def _predict_batch(self, text_list: List[str]) -> Dict[str, np.ndarray]:
        if self.max_length > 0:
            encoded_ids: np.ndarray = self.tokenizer(text_list, padding='max_length', truncation=True,
                                                     max_length=self.max_length, return_tensors="np")
            outputs = self.model.run(output_names=['last_hidden_state', 'pooler_output'], input_feed=dict(encoded_ids))
            return {"last_hidden_state": outputs[0], "pooler_output": outputs[1]}
        else:
            last_hidden_state_list: List[List] = []
            pooler_output_list: List[List] = []
            for text in text_list:
                outputs: List = self._predict(text)
                last_hidden_state_list.append(outputs[0])
                pooler_output_list.append(outputs[1])

            last_hidden_state: np.ndarray = np.concatenate(last_hidden_state_list, axis=0)
            pooler_output: np.ndarray = np.concatenate(pooler_output_list, axis=0)

            return {"last_hidden_state": last_hidden_state, "pooler_output": pooler_output}
    """


Model = OnnxModel
