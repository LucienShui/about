from flask import Flask, request, jsonify, Response
from about.chat import Chat

app = Flask(__name__)
chat = Chat()


@app.route('/', methods=['GET', 'POST'])
def response() -> Response:
    text = request.json()['text']
    return jsonify(chat.response(text).json())


def main():
    app.run(host='0.0.0.0', port=3000)


if __name__ == '__main__':
    main()
