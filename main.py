from flask import Flask, request, jsonify, Response, render_template
from about.chat import Chat

bot = Chat(embedding_type='MEAN')
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat() -> Response:
    text = request.json['text']
    return jsonify(bot.response(text).json())


def main():
    app.run(host='0.0.0.0', port=3000)


if __name__ == '__main__':
    main()
