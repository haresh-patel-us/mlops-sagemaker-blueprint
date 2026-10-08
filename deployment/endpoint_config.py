"""Real-time endpoint with target-tracking autoscaling.

Deploy the approved model package to a real-time endpoint and scale on
invocations-per-instance — the standard pattern for online inference.
"""
import boto3

client = boto3.client("sagemaker")
autoscaling = boto3.client("application-autoscaling")


def deploy_endpoint(model_name: str, endpoint_config_name: str, endpoint_name: str,
                    instance_type: str = "ml.m5.large") -> None:
    client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[{
            "VariantName": "primary",
            "ModelName": model_name,
            "InitialInstanceCount": 1,
            "InstanceType": instance_type,
            "InitialVariantWeight": 1,
        }],
    )
    client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name,
    )

    resource_id = f"endpoint/{endpoint_name}/variant/primary"
    autoscaling.register_scalable_target(
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        MinCapacity=1,
        MaxCapacity=4,
    )
    autoscaling.put_scaling_policy(
        PolicyName="invocations-per-instance",
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        PolicyType="TargetTrackingScaling",
        TargetTrackingScalingPolicyConfiguration={
            "TargetValue": 100.0,  # invocations per instance per minute
            "PredefinedMetricSpecification": {
                "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance",
            },
            "ScaleInCooldown": 300,
            "ScaleOutCooldown": 60,
        },
    )
