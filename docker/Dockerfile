FROM nvidia/cuda:11.7.1-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive


# get python modules
RUN apt-get update \
    && apt-get install -yq --no-install-recommends \
    python3-dev \
    python3-pip \
    python3-opencv \
    python3-tk \
    gcc

RUN apt-get install -y git

RUN pip install --upgrade pip \
    && pip install setuptools


COPY requirements.txt /tmp
WORKDIR /tmp
RUN pip install -r requirements.txt


RUN  pip install 'git+https://github.com/facebookresearch/detectron2.git'