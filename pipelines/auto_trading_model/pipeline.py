import os
from typing import Dict
from sagemaker.workflow.entities import PipelineVariable

import boto3
import sagemaker
import sagemaker.session
from sagemaker.huggingface import HuggingFace
from sagemaker.inputs import (
    TrainingInput, CreateModelInput
)

from sagemaker.huggingface.model import HuggingFaceModel
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import (
    ProcessingInput, ProcessingOutput, Processor
)

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import (
    ProcessingStep, TrainingStep, CreateModelStep
)
from src.component.binance.constraint import (MOVING_AVERAGE_WINDOW, LOSS_TYPE, TIME_WINDOW, 
                                              PREDICTION_OUTPUT_SIZE, TIME_MINUTE_LIMIT)


BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def get_sagemaker_client(region):
    """Gets the sagemaker client.
       Args:
           region: the aws region to start the session
       Returns:
           `sagemaker.session.Session instance
       """
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.
    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts
    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )
    

def get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region.
    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts
    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )


def get_preprocessing_step(role, 
                           image_uri,
                           instance_type,
                           pipeline_session,
                           moving_average_window,
                           time_minute_limit):
    processor = Processor(
        entrypoint=['python3', 'code/pipelines/auto_trading_model/preprocess.py'],
        image_uri=image_uri,
        role=role,
        instance_type=instance_type,
        instance_count=1,
        sagemaker_session=pipeline_session,
        volume_size_in_gb=50
    )
    return ProcessingStep(
        name='Preprocessing',
        processor=processor,
        job_arguments=[
            "--moving_average_window", moving_average_window,
            "--time_minute_limit", time_minute_limit,
        ],
        inputs=[
            ProcessingInput(
                source='s3://sagemaker-autocryptotrading/data',
                destination='/opt/ml/processing/data'
            ),
        ],
        outputs=[
            ProcessingOutput(output_name='preprocessed', source='/opt/ml/processing/output'),
            ProcessingOutput(output_name='code', source='/opt/ml/processing/code', 
                             destination='s3://sagemaker-autocryptotrading/code'),
            ProcessingOutput(output_name='deploy_code', source='/opt/ml/processing/deploy_code',
                             destination='s3://sagemaker-autocryptotrading/deploy_code')
        ]
    )
    

def get_train_step(role,
                   image_uri,
                   instance_type,
                   pipeline_session,
                   inputs: Dict,
                   epochs,
                   column_limit,
                   activation_function,
                   moving_average_window,
                   frame_size,
                   output_size,
                   scaler_x,
                   scaler_y,
                   loss_type,
                   addtional_layer,
                   learning_rate,
                   patience,
                   hidden_size):
    estimator = HuggingFace(
        py_version='py38',
        image_uri=image_uri,
        role=role,
        instance_type=instance_type,
        instance_count=1,
        volume_size=50,
        input_mode='File',
        source_dir='s3://sagemaker-autocryptotrading/code/code.tar.gz',
        entry_point='train.py',
        hyperparameters={
            'epochs': epochs,
            'column_limit': column_limit,
            'activation_function': activation_function,
            'moving_average_window': moving_average_window,
            'scaler_x': scaler_x,
            'scaler_y': scaler_y,
            'loss_type': loss_type,
            'addtional_layer': addtional_layer,
            'learning_rate': learning_rate,
            'patience': patience,
            'frame_size': frame_size,
            'output_size': output_size,
            'hidden_size': hidden_size,
        },
        sagemaker_session=pipeline_session
    )
    return TrainingStep(
        name='Training',
        estimator=estimator,
        inputs=inputs
    )
    
    
def get_create_model_step(role,
                           image_uri,
                           endpoint_instance_type,
                           pipeline_session,
                           model_data: PipelineVariable):
    model = HuggingFaceModel(
        image_uri=image_uri,
        role=role,
        model_data=model_data,
        pytorch_version="1.12.1",
        sagemaker_session=pipeline_session
    )
    inputs = CreateModelInput(
        instance_type=endpoint_instance_type
    )
    return CreateModelStep(
        name='CreatingModel',
        model=model,
        inputs=inputs
    )   
    

def get_deploy_processing_step(role,
                               instance_type,
                               pipeline_session,
                               model_name: PipelineVariable,
                               endpoint_instance_type,
                               endpoint_instance_count):
    processor = SKLearnProcessor(
        framework_version='0.23-1',
        role=role,
        instance_type=instance_type,
        instance_count=1,
        sagemaker_session=pipeline_session
    )
    return ProcessingStep(
        name='DeployingModel',
        processor=processor,
        job_arguments=[
            "--model_name", model_name,
            "--endpoint_instance_type", endpoint_instance_type,
            "--endpoint_instance_count", endpoint_instance_count,
            "--endpoint_name", "Autotrading-Endpoint"
            # "--endpoint_name", "Test"
        ],
        code="s3://sagemaker-autocryptotrading/deploy_code/deploy_model.py"
    )


def get_pipeline(
        region,
        role=None,
        default_bucket=None,
        pipeline_name="AutotradingTrainPipeline",
        train_instance_type='ml.g4dn.8xlarge',
        epochs=1000,
        column_limit=50,
        scaler_x='robust',
        scaler_y='robust',
        loss_type=LOSS_TYPE,
        endpoint_instance_type="ml.t2.medium",
        endpoint_instance_count=1,
        activation_function="gelu",
        moving_average_window=MOVING_AVERAGE_WINDOW,
        frame_size=TIME_WINDOW,
        output_size=PREDICTION_OUTPUT_SIZE,
        addtional_layer=True,
        learning_rate=0.0001,
        patience=100,
        time_minute_limit=TIME_MINUTE_LIMIT,
        hidden_size=200):
    """_summary_

    Args:
        region (_type_): _description_
        role (_type_, optional): _description_. Defaults to None.
        default_bucket (_type_, optional): _description_. Defaults to None.
        pipeline_name (str, optional): _description_. Defaults to "AutotradingTrainPipeline".
        train_instance_type (str, optional): _description_. Defaults to 'ml.g4dn.8xlarge'.
        epochs (int, optional): _description_. Defaults to 2000.
        column_limit (int, optional): _description_. Defaults to 50.
        endpoint_instance_type (str, optional): _description_. Defaults to "ml.t2.medium".
        endpoint_instance_count (int, optional): _description_. Defaults to 1.
        activation_function (str, optional): _description_. Defaults to "none". or "tanh" or "relu"

    Returns:
        _type_: _description_
    """
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)
    
    processing_image_uri = '080366477338.dkr.ecr.ap-northeast-2.amazonaws.com/autotrading-sagemaker:preprocess'
    training_image_uri = '080366477338.dkr.ecr.ap-northeast-2.amazonaws.com/autotrading-sagemaker:train'
    inference_image_uri = '080366477338.dkr.ecr.ap-northeast-2.amazonaws.com/autotrading-sagemaker:inference'
    
    pipeline_session = get_pipeline_session(region, default_bucket)
    
    step_preprocess = get_preprocessing_step(role,
                                             processing_image_uri,
                                             pipeline_session=pipeline_session,
                                            #  instance_type="ml.m4.16xlarge",
                                             instance_type="ml.r5.4xlarge",
                                             moving_average_window=str(moving_average_window),
                                             time_minute_limit=str(time_minute_limit)
                                             )
    training_inputs = {
        "crypto_data": TrainingInput(
            s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs[
                'preprocessed'
            ].S3Output.S3Uri
        )
    }
    step_train = get_train_step(role,
                                training_image_uri,
                                instance_type=train_instance_type,
                                pipeline_session=pipeline_session,
                                inputs=training_inputs,
                                epochs=str(epochs),
                                column_limit=str(column_limit),
                                activation_function=activation_function,
                                moving_average_window=str(moving_average_window),
                                frame_size=str(frame_size),
                                output_size=str(output_size),
                                scaler_x=scaler_x,
                                scaler_y=scaler_y,
                                loss_type=loss_type,
                                addtional_layer=str(addtional_layer),
                                learning_rate=str(learning_rate),
                                patience=str(patience),
                                hidden_size=str(hidden_size)
                                )
    model_data = step_train.properties.ModelArtifacts.S3ModelArtifacts 
    step_create_model = get_create_model_step(role,
                                              inference_image_uri,
                                              endpoint_instance_type=endpoint_instance_type,
                                              pipeline_session=pipeline_session,
                                              model_data=model_data
                                              )
    model_name = step_create_model.properties.ModelName
    step_deploy = get_deploy_processing_step(role,
                                             instance_type="ml.m5.xlarge",
                                             pipeline_session=pipeline_session,
                                             model_name=model_name,
                                             endpoint_instance_type=endpoint_instance_type,
                                             endpoint_instance_count=str(endpoint_instance_count)
                                             )
    
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[],
        steps=[step_preprocess, step_train, step_create_model, step_deploy],
        sagemaker_session=pipeline_session,
    )
    return pipeline
