FROM python:3.9
COPY requirements.txt /requirements.txt
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
	pip install -r /requirements.txt
