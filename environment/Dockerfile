# hash:sha256:bf984cf63c7df4f73667744c219d6caea3e25f36068becc3d9b0449e835cc4b8
FROM registry.codeocean.allenneuraldynamics.org/codeocean/tensorflow:2.11-python3.10-cuda11.7.0-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg=7:4.4.2-0ubuntu0.22.04.1 \
        libcublas-12-0=12.0.2.224-1 \
        libcudnn8=8.9.5.29-1+cuda12.2 \
        libsm6=2:1.2.3-1build2 \
        libxext6=2:1.3.4-1build1 \
    && rm -rf /var/lib/apt/lists/*

RUN conda install -y \
        ruamel.yaml==0.16.12 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    deeplabcut[tf]==2.3.7 \
    moviepy==1.0.3

COPY postInstall /
RUN /postInstall
