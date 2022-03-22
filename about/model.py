import re
from functools import partial  # tqdm 同行进度条

import numpy as np
import torch
from tqdm import tqdm
# from transformers import AlbertForSequenceClassification, BertTokenizerFast
from transformers import AlbertModel, BertTokenizerFast
from transformers.modeling_outputs import BaseModelOutputWithPooling

tqdm = partial(tqdm, position=0, leave=True)  # tqdm 同行进度条


class Module(AlbertModel, torch.nn.Module):
    pass


class Model:
    def __init__(self, pretrained: str):
        self.max_length = 128

        self.pretrained = pretrained

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(self.pretrained)
        self.model: Module = AlbertModel.from_pretrained(self.pretrained)
        self.model.to(device=self.device)

        self.model.eval()

        print('load finished')

    def predict_batch(self, text_list: [str], batch_size: int = 32, show_progress_bar: bool = False) -> dict:
        last_hidden_state_list: [np.ndarray] = []
        pooler_output_list: [np.ndarray] = []

        with torch.no_grad():
            rg = range(0, len(text_list), batch_size)
            if show_progress_bar:
                rg = tqdm(rg)
            for i in rg:
                text_batch = text_list[i:i + batch_size]
                encoded_ids = self.tokenizer(text_batch, padding='max_length', truncation=True,
                                             max_length=128, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in encoded_ids.items()}
                huggingface_prediction: BaseModelOutputWithPooling = self.model(**inputs)

                pooler_output_list.append(huggingface_prediction.pooler_output.cpu().numpy())
                last_hidden_state_list.append(huggingface_prediction.last_hidden_state.cpu().numpy())

        last_hidden_state: np.ndarray = np.concatenate(last_hidden_state_list, axis=0)
        pooler_output: np.ndarray = np.concatenate(pooler_output_list, axis=0)

        return {"last_hidden_state": last_hidden_state, "pooler_output": pooler_output}

    def predict(self, text: str) -> dict:
        result = self.predict_batch([text])
        for key in result.keys():
            result[key] = result[key][0]
        return result

    def embedding_batch(self, text_list: [str], _type: str = 'CLS') -> np.ndarray:
        if _type not in ['CLS', 'MEAN', 'MAX', 'MIN', 'AVG']:
            raise ValueError('_type must be one of "CLS", "MEAN", "MAX", "MIN", "AVG"')
        predict_result = self.predict_batch(text_list)
        if _type == 'CLS':
            return predict_result['pooler_output']
        if _type == 'MEAN':
            return predict_result['last_hidden_state'][:, 1:].mean(axis=1)
        if _type == 'AVG':
            # 除以每个句子的长度，计算 cosine 时没有区别
            total = predict_result['last_hidden_state'][:, 1:].sum(axis=1)
            length = np.array([len(each) - 1 for each in self.tokenizer(text_list)['input_ids']])
            return total / length.reshape(len(text_list), 1)
        if _type == 'MAX':
            return predict_result['last_hidden_state'][:, 1:].max(axis=1)
        if _type == 'MIN':
            return predict_result['last_hidden_state'][:, 1:].min(axis=1)

    def embedding(self, text: str, _type: str = 'CLS') -> np.ndarray:
        return self.embedding_batch([text], _type)[0]


