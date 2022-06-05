FROM python:3.9
COPY requirements-runtime.txt /requirements-runtime.txt
RUN pip install -r /requirements-runtime.txt
WORKDIR /app

CMD ["python", "main.py"]
