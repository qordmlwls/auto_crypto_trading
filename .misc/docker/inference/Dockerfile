FROM 763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-inference:1.12.1-gpu-py38-cu113-ubuntu20.04-sagemaker

COPY .misc/requirements/base.txt /home

RUN pip3 install -r /home/base.txt

ENV SAGEMAKER_PROGRAM inference.py

ENV SAGEMAKER_REGION ap-northeast-2

ENV MMS_DEFAILT_RESPONSE_TIMEOUT 500

RUN pip3 install multi-model-server==1.1.2
RUN pip3 install model-archiver==1.0.3
RUN pip3 install sagemaker-inference==1.5.5
RUN pip3 install sagemaker-pytorch-inference==1.5.1.post1
ENTRYPOINT ["python", "/usr/local/bin/dockerd-entrypoint.py"]
CMD ["multi-model-server", "--start", "--mms-config", "/home/model-server/config.properties"]