# ubuntu=22.04  pytorch=2.4.1  cuda=12.4  cudnn=90.1  python=3.8
FROM ubuntu:latest
LABEL authors="caochong"

# Install openSSH and openSSL
RUN apt-get update && apt-get openssh-server openssl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /var/run/sshd
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    cat /etc/ssh/sshd_config | grep -v PermitRootLogin > /etc/ssh/sshd_config.new && \
    echo "PermitRootLogin yes" >> /etc/ssh/sshd_config.new && \
    mv -f /etc/ssh/ssh_config.new /etc/ssh/ssh_config && \
    mv -f /etc/ssh/sshd_config.new /etc/ssh/sshd_config > /dev/null 2>&1;

# Install mmcv and mmegine
RUN pip install mmengine==0.10.5 -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN git clone -b v2.1.0  https://github.com/open-mmlab/mmcv.git /mmcv
WORKDIR /mmcv
RUN pip install -r requirements/optional.txt
RUN pip install -e . -v

# Install mmdetection
RUN git clone -b v3.3.0 https://github.com/open-mmlab/mmdetection.git
WORKDIR mmdetection
RUN pip install -v -e .

ENTRYPOINT ["top", "-b"]