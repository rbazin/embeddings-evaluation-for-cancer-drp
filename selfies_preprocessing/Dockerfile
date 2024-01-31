FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    unzip && \
    rm -rf /var/lib/apt/lists/*

ARG GROUP_ID
ARG USER_ID
RUN groupadd -g ${GROUP_ID} python && \
    useradd --uid ${USER_ID} \
    --gid ${GROUP_ID} \
    --create-home \
    --shell /bin/bash \
    python

USER python
WORKDIR /home/python
ENV PATH="$PATH:/home/python/.local/bin"

# Copying the repository contents into the container
RUN git clone https://github.com/HUBioDataLab/SELFormer.git

RUN pip install --upgrade --no-cache-dir pip && \
    pip install --no-cache-dir gdown transformers pandas ipykernel joblib

RUN mkdir -p /home/python/SELFormer/data/pretrained_models && \
    cd /home/python/SELFormer/data/pretrained_models && \
    gdown 1zuVAKXCMc-HZHQo9y3Hu5zmQy51FGduI -O pretrained_models.zip && \
    unzip -q pretrained_models.zip && \
    rm pretrained_models.zip

ENTRYPOINT ["/bin/bash"]
