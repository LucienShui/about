from flask import Flask, request, jsonify, Response, render_template

from about.chat import Chat

bot = Chat()
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat() -> Response:
    text = request.json['text']
    response, response_list = bot.reply(text)
    return jsonify(response.json())


def main():
    app.run(host='0.0.0.0', port=5000)


if __name__ == '__main__':
    main()
