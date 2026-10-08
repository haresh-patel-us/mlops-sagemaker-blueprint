# MLOps on Amazon SageMaker — starter blueprint

Portfolio sample of an end-to-end SageMaker ML workflow: a training script
(PyTorch), a SageMaker Pipeline definition, and a deployment sketch for a
real-time endpoint behind autoscaling. Patterns I use on ML platforms:
reproducible training jobs, model registry promotion, and infrastructure for
both batch and online inference.

## Layout

- `training/train.py` — PyTorch training script (SageMaker Script Mode)
- `pipeline/pipeline.py` — SageMaker Pipelines: processing, training,
  evaluation, register, deploy (conditional on accuracy)
- `deployment/endpoint_config.py` — real-time endpoint + autoscaling policy

## Run (AWS)

```bash
# Train
python training/train.py --epochs 5

# Build + run the pipeline
python pipeline/pipeline.py
```

## Patterns demonstrated

- Script-mode training jobs with explicit hyperparameters and metrics
- Pipeline steps with caching, so re-runs skip unchanged stages
- Model registry with approval workflow before endpoint deployment
- Endpoint autoscaling on `SageMakerVariantInvocationsPerInstance`
