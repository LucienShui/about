FROM python:3.9
COPY requirements.txt /requirements.txt
RUN pip install -i https://mirrors.aliyun.com/pypi/simple/ -r /requirements.txt
WORKDIR /app

CMD ["python", "main.py"]
