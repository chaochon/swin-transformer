FROM nvcr.io/nvidia/pytorch:24.08-py3

ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1"

# Avoid Public GPG key error
# https://github.com/NVIDIA/nvidia-docker/issues/1631
RUN rm -f /etc/apt/sources.list.d/cuda.list \
    && rm -f /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# (Optional, use Mirror to speed up downloads)
RUN sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//http:\/\/mirrors.aliyun.com\/ubuntu\//g' /etc/apt/sources.list && \
  pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install the required packages
RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 openssh-server openssl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /var/run/sshd
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    cat /etc/ssh/sshd_config | grep -v PermitRootLogin > /etc/ssh/sshd_config.new && \
    echo "PermitRootLogin yes" >> /etc/ssh/sshd_config.new && \
    mv -f /etc/ssh/ssh_config.new /etc/ssh/ssh_config && \
    mv -f /etc/ssh/sshd_config.new /etc/ssh/sshd_config > /dev/null 2>&1;

# uninstall all versions of opencv
RUN pip uninstall $(pip list --format=freeze | grep opencv) -y \
    && rm -rf /usr/local/lib/python3.10/dist-packages/cv2/ \
    && pip install opencv-python-headless

# Install MMEngine MMCV and Segmentation
RUN pip install mmengine==0.10.1 mmcv==2.1.0
RUN pip install mmsegmentation==1.2.2 mmpretrain==1.2.0 ftfy regex

# Install MMDetection
#RUN git clone -b v3.3.0 https://github.com/open-mmlab/mmdetection.git /mmdetection \
#    && cd /mmdetection \
#    && pip install --no-cache-dir -e . -v

# Install apex
#RUN git clone https://github.com/NVIDIA/apex /apex \
#    && cd /apex \
#    && pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# Install miniconda

# Install vmamba in new conda env
#RUN git clone https://github.com/MzeroMiko/VMamba.git /VMamba \
#    && cd /VMamba \
#    && pip install -r requirements.txt \
#    && cd kernels/selective_scan && pip install .

WORKDIR /home
