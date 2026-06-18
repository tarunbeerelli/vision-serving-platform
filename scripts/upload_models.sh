#!/bin/bash
set -e

BUCKET="vision-serving-models-564530715752-v2"

echo "Uploading ViT-tiny ONNX..."
gsutil cp triton_repo/vit_fp32/1/model.onnx \
    gs://$BUCKET/triton_repo/vit_fp32/1/model.onnx

echo "Uploading CLIP visual encoder..."
gsutil cp triton_repo/clip_visual/1/model.onnx \
    gs://$BUCKET/triton_repo/clip_visual/1/model.onnx

echo "Uploading CLIP text encoder..."
gsutil cp triton_repo/clip_text/1/model.onnx \
    gs://$BUCKET/triton_repo/clip_text/1/model.onnx

echo "Uploading Triton configs..."
gsutil cp triton_repo/vit_fp32/config.pbtxt \
    gs://$BUCKET/triton_repo/vit_fp32/config.pbtxt
gsutil cp triton_repo/vit_int8/config.pbtxt \
    gs://$BUCKET/triton_repo/vit_int8/config.pbtxt
gsutil cp triton_repo/clip_visual/config.pbtxt \
    gs://$BUCKET/triton_repo/clip_visual/config.pbtxt
gsutil cp triton_repo/clip_text/config.pbtxt \
    gs://$BUCKET/triton_repo/clip_text/config.pbtxt

echo "✓ All models uploaded"
gsutil ls gs://$BUCKET/triton_repo/
