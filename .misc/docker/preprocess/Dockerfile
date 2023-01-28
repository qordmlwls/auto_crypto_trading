FROM 763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-training:1.12.1-gpu-py38-cu116-ubuntu20.04-ec2

# Update pip
RUN pip install -U pip

# Project setting
RUN mkdir /opt/ml && \
    mkdir /opt/ml/code && \
    mkdir /opt/ml/logs && \
    mkdir /opt/ml/tmp
WORKDIR /opt/ml

COPY . /opt/ml/code

# ROOT PATH
ENV PYTHONPATH="/opt/ml/code"

# Install deps
ADD ./.misc/requirements/base.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt