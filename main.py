import json

from flask import Flask, request, jsonify, Response, render_template

from about.chat import Chat, ChatResponse
from about.orm import Trace

bot = Chat()
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/reload', methods=['POST'])
def reload() -> Response:
    bot.knowledge_list = bot.load_knowledge(bot.knowledge_base_dir)
    return jsonify({'code': 200})


@app.route('/api/chat', methods=['POST'])
def chat() -> Response:
    text = request.json['text']
    response: ChatResponse = bot.reply(text)
    json_response: dict = response.json()
    ip = request.headers.get('X-Real-IP', request.remote_addr)
    Trace.create(module='chat', input={'text': text}, output=json_response, ip=ip)
    return jsonify(json_response)


def main():
    with open('config.json') as file:
        config: dict = json.load(file)
    host: str = config.get('host', '0.0.0.0')
    port: int = config.get('port', 3000)
    app.run(host=host, port=port)


if __name__ == '__main__':
    main()
