FROM 763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-training:1.12.1-gpu-py38-cu113-ubuntu20.04-sagemaker


ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code
ENV PATH="/opt/ml/code:${PATH}"

COPY ./.misc/requirements/base.txt /home
RUN pip3 install -r /home/base.txt


# env

ENV SAGEMAKER_PROGRAM train.py