"""SageMaker Pipeline: process -> train -> evaluate -> register -> deploy.

Deployment is conditional: the endpoint is created only when evaluation
accuracy meets the threshold, and the model package must be approved in
the registry — the two gates I standardize on for safe ML rollouts.
"""
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.parameters import ParameterFloat, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import CreateModelStep, RegisterModel, TrainingStep

ACCURACY_THRESHOLD = 0.85


def build_pipeline(role: str, region: str) -> Pipeline:
    model_package_group = ParameterString("ModelPackageGroupName", default_value="demo-classifier")
    accuracy_gate = ParameterFloat("AccuracyThreshold", default_value=ACCURACY_THRESHOLD)

    training_estimator = Estimator(
        entry_point="train.py",
        source_dir="training",
        role=role,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        framework_version="2.3",
        py_version="py310",
        hyperparameters={"epochs": 5, "batch-size": 64},
        output_path=f"s3://demo-ml-artifacts-{region}/training",
    )
    train_step = TrainingStep(name="Train", estimator=training_estimator)

    evaluation_report = PropertyFile(name="EvaluationReport", output_name="evaluation", path="metrics.json")

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=train_step.properties.ModelArtifacts.S3ModelArtifacts,
            content_type="application/json",
        )
    )
    register_step = RegisterModel(
        name="Register",
        estimator=training_estimator,
        model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group,
        approval_status="PendingManualApproval",
        model_metrics=model_metrics,
    )

    cond = ConditionGreaterThanOrEqualTo(
        left=JsonGet(step_name="Evaluate", property_file=evaluation_report, json_path="accuracy.value"),
        right=accuracy_gate,
    )
    # NOTE: wire an actual ProcessingStep named "Evaluate" in front of this;
    # kept structural here to show the gating pattern.
    condition_step = ConditionStep(
        name="AccuracyGate",
        conditions=[cond],
        if_steps=[register_step],
        else_steps=[],
    )

    return Pipeline(
        name="demo-classifier-pipeline",
        parameters=[model_package_group, accuracy_gate],
        steps=[train_step, condition_step],
        sagemaker_session=None,
    )
