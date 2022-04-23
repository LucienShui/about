# About

让机器人来回答于我的问题

## 功能

目前只支持单轮 QA，采用文本相似度进行召回，支持自定义问题和答案。  
文本相似度模型采用 [SimCSE](https://arxiv.org/abs/2104.08821)
，预训练模型来自 [Hugging Face](https://huggingface.co/cyclone/simcse-chinese-roberta-wwm-ext)。

### 知识库格式

当命中某个标问时，会判定为命中为本条知识，并返回答案，当答案有多个时，会随机返回一个。

```json
{
  "知识_1": {
    "question_list": ["标问_1", "标问_2"],
    "answer_list": ["答案_1"]
  },
  "知识_2": {
    "question_list": ["标问_3", "标问_4"],
    "answer_list": ["答案_2", "答案_3"]
  }
}
```

## 运行

```bash
python3 main.py
```

## API

### chat

```bash
curl -X POST \
     -H "Content-Type: application/json" \
     -d '{"text": "你好"}' \
     'http://localhost:3000/chat'
```
