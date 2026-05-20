# Historical Backfill with Argo Workflows

This WorkflowTemplate splits a historical range into calendar-month backfill tasks, runs one month at a time, then runs Silver and Gold.

## Prerequisites

- Argo Workflows CRDs and controller are installed in the cluster.
- Optional: Argo CLI is installed locally. If not, use the `kubectl create` example below.
- MinIO manifests and `secret/openaq-secret.yaml` are applied.
- Images are built inside minikube:

```bash
eval "$(minikube docker-env)"
docker build -f batch/backfill/Dockerfile -t air-quality-backfill:0.1 .
docker build -f batch/spark/Dockerfile -t air-quality-spark:0.1 .
```

- Historical ConfigMaps are applied:

```bash
kubectl apply -f batch/backfill/historical-backfill-configmap.yaml
kubectl apply -f batch/spark/historical-spark-configmap.yaml
kubectl apply -f batch/argo/historical-workflow-rbac.yaml
```

The Spark ConfigMap is tuned for an 8-core, 8 GB minikube node:

```text
SPARK_LOCAL_CORES=4
SPARK_DRIVER_MEMORY=3g
SPARK_EXECUTOR_MEMORY=3g
SPARK_SQL_SHUFFLE_PARTITIONS=8
```

The backfill image defaults to `--fetch-mode range --request-chunk-days 7`, which fetches each sensor in 7-day windows for the monthly range and then writes daily Bronze partitions. Rebuild `air-quality-backfill:0.1` after changing `backfill_to_bronze.py`.

## Install the WorkflowTemplate

```bash
kubectl apply -f batch/argo/historical-batch-workflow-template.yaml
```

## Run 2000 Days

```bash
argo submit --from workflowtemplate/historical-batch-pipeline -p days=2000 --watch
```

Without the Argo CLI:

```bash
kubectl create -f batch/argo/historical-batch-workflow.yaml
```

By default, the workflow ends at today's UTC date. To pin the range:

```bash
argo submit --from workflowtemplate/historical-batch-pipeline \
  -p date-from=2023-01-01 \
  -p date-to=2025-09-26 \
  --watch
```

To run Bronze only:

```bash
argo submit --from workflowtemplate/historical-batch-pipeline \
  -p days=2000 \
  -p request-chunk-days=7 \
  -p run-silver-gold=false \
  --watch
```

The template sets `parallelism: 1`, so generated month tasks run sequentially. Silver and Gold are later workflow steps, so they still wait for every Bronze month to finish and then run one by one.

## Recover a Stuck Backfill Pod

If a backfill pod stays in `Init:0/1` for a long time, the backfill container has not started yet. Terminate that workflow and submit a new one from the first unfinished month:

```bash
argo terminate historical-batch-pipeline-nwztn -n bigdata
argo submit -n bigdata --from workflowtemplate/historical-batch-pipeline \
  -p date-from=2024-09-01 \
  -p date-to=2026-05-20 \
  -p run-silver-gold=true \
  --watch
```

The updated template gives backfill pods a 4-hour deadline and retries them twice, so a future Argo init/runtime hang should fail and retry instead of blocking the workflow indefinitely.
