FROM 080366477338.dkr.ecr.ap-northeast-2.amazonaws.com/autotrading-sagemaker:base

COPY . /opt/ml/code

# ROOT PATH
ENV PYTHONPATH="/opt/ml/code"

# Install deps
ADD ./.misc/requirements/base.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt