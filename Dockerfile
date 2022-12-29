FROM python:3.8-alpine
COPY requirements-runtime.txt /requirements-runtime.txt
RUN pip install --no-cache -i https://pypi.tuna.tsinghua.edu.cn/simple -r /requirements-runtime.txt
WORKDIR /app

CMD ["python", "main.py"]
