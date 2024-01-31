FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

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

RUN pip install --upgrade --no-cache-dir pip && \
    pip install --no-cache-dir pandas transformers joblib ipykernel

ENTRYPOINT ["/bin/bash"]
