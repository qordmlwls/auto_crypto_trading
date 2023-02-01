import time
import boto3
import argparse
import sys
import json
import logging.handlers

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Parse argument variables passed via the DeployModel processing step
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str)
parser.add_argument('--region', type=str, default="ap-northeast-2")
parser.add_argument('--endpoint_instance_type', type=str, default='ml.t3.medium')
parser.add_argument('--endpoint_name', type=str)
parser.add_argument('--endpoint_instance_count', type=int, default=1)
args = parser.parse_args()

logger.info("#############################################")
logger.info(f"args.model_name: {args.model_name}")
logger.info(f"args.region: {args.region}")
logger.info(f"args.endpoint_instance_type: {args.endpoint_instance_type}")
logger.info(f"args.endpoint_name: {args.endpoint_name}")

region = args.region
instance_type = args.endpoint_instance_type
instance_count = args.endpoint_instance_count
model_name = args.model_name

boto3.setup_default_session(region_name=region)
sagemaker_boto_client = boto3.client('sagemaker')

# name truncated per sagameker length requirememnts (63 char max)
endpoint_config_name = f'{args.model_name[:56]}-config'
existing_configs = sagemaker_boto_client.list_endpoint_configs(NameContains=endpoint_config_name)['EndpointConfigs']

logger.info("Creating Endpoint Config")

if not existing_configs:
    create_ep_config_response = sagemaker_boto_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[{
            'InstanceType': instance_type,
            'InitialVariantWeight': 1,
            'InitialInstanceCount': instance_count,
            'ModelName': model_name,
            'VariantName': 'AllTraffic'
        }]
    )

existing_endpoints = sagemaker_boto_client.list_endpoints(NameContains=args.endpoint_name)['Endpoints']

logger.info("Creating Endpoint")

if not existing_endpoints:
    logger.info(f"Creating endpoint")
    create_endpoint_response = sagemaker_boto_client.create_endpoint(
        EndpointName=args.endpoint_name,
        EndpointConfigName=endpoint_config_name)
else:
    logger.info(f"Endpoint exists. Updating endpoint")
    undate_endpoint_response = sagemaker_boto_client.update_endpoint(
        EndpointName=args.endpoint_name,
        EndpointConfigName=endpoint_config_name)

endpoint_info = sagemaker_boto_client.describe_endpoint(EndpointName=args.endpoint_name)
endpoint_status = endpoint_info['EndpointStatus']

logger.info(f'Endpoint status is creating or updating')
while endpoint_status == 'Creating':
    endpoint_info = sagemaker_boto_client.describe_endpoint(EndpointName=args.endpoint_name)
    endpoint_status = endpoint_info['EndpointStatus']
    logger.info(f'Endpoint status: {endpoint_status}')
    if endpoint_status == 'Creating':
        time.sleep(30)

while endpoint_status == 'Updating':
    endpoint_info = sagemaker_boto_client.describe_endpoint(EndpointName=args.endpoint_name)
    endpoint_status = endpoint_info['EndpointStatus']
    logger.info(f'Endpoint status: {endpoint_status}')
    if endpoint_status == 'Updating':
        time.sleep(30)


client = boto3.client('sagemaker-runtime')

# @TODO: auto-scaling polucy 추가 - 배포될 엔드포인트에서 auto-scaling설정
asg = boto3.client('application-autoscaling')

# Resource type is variant and the unique identifier is the resource ID.
resource_id=f"endpoint/{args.endpoint_name}/variant/AllTraffic"

# scaling configuration
# response = asg.register_scalable_target(
#     ServiceNamespace='sagemaker', 
#     ResourceId=resource_id,
#     ScalableDimension='sagemaker:variant:DesiredInstanceCount', 
#     MinCapacity=2,
#     MaxCapacity=5
# )


# #Target Scaling
# response = asg.put_scaling_policy(
#     PolicyName=f'Request-ScalingPolicy-{endpoint_name}',
#     ServiceNamespace='sagemaker',
#     ResourceId=resource_id,
#     ScalableDimension='sagemaker:variant:DesiredInstanceCount',
#     PolicyType='TargetTrackingScaling',
#     TargetTrackingScalingPolicyConfiguration={
#         'TargetValue': 10.0, # Threshold
#         'PredefinedMetricSpecification': {
#             'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance',
#         },
#         'ScaleInCooldown': 300, # duration until scale in
#         'ScaleOutCooldown': 60 # duration between scale out
#     }
# )
