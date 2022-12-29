FROM nvidia/cuda:11.6.2-base-ubuntu20.04
COPY requirements-gpu-runtime.txt /requirements-gpu-runtime.txt
RUN  echo 'deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal main restricted universe multiverse' > /etc/apt/sources.list && \
     echo 'deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-updates main restricted universe multiverse' >> /etc/apt/sources.list && \
     echo 'deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-backports main restricted universe multiverse' >> /etc/apt/sources.list && \
     echo 'deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-security main restricted universe multiverse' >> /etc/apt/sources.list && \
     apt-get update && apt-get install python3 python3-pip -y
RUN pip3 install --no-cache -i https://pypi.tuna.tsinghua.edu.cn/simple -r /requirements-gpu-runtime.txt
WORKDIR /app

CMD ["python", "main.py"]
