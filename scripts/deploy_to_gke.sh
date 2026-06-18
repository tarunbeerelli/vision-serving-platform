#!/bin/bash
set -e

PROJECT="vision-serving-platform"
REGION="europe-west4"
CLUSTER="vision-serving"

echo "Getting GKE credentials..."
gcloud container clusters get-credentials $CLUSTER \
    --region $REGION \
    --project $PROJECT

echo "Applying k8s manifests..."
kubectl apply -k k8s/overlays/prod

echo "Waiting for Triton to be ready..."
kubectl rollout status deployment/triton-server -n triton --timeout=600s

echo "Waiting for gateway to be ready..."
kubectl rollout status deployment/gateway -n triton --timeout=300s

echo "Getting external IP..."
kubectl get svc gateway-service -n triton

echo "✓ Deployment complete"
