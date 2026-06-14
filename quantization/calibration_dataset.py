"""
Build INT8 calibration dataset.

INT8 quantisation needs ~500 representative images to measure
activation ranges per layer. This script downloads a small
ImageNet validation subset from HuggingFace datasets and saves
preprocessed numpy arrays to quantization/calibration_data/.

Runs on Vast.ai before TensorRT conversion.
"""

import argparse
from pathlib import Path

import numpy as np
from transformers import ViTImageProcessor

CHECKPOINT = "WinKawaks/vit-tiny-patch16-224"
OUTPUT_DIR = Path("quantization/calibration_data")
NUM_IMAGES = 500


def build(num_images: int, output_dir: Path) -> None:
    # Import here — datasets is in the quantization dep group
    from datasets import load_dataset  # type: ignore[import]

    print(f"Downloading {num_images} ImageNet validation images...")
    dataset = load_dataset(
        "imagenet-1k",
        split=f"validation[:{num_images}]",
        trust_remote_code=True,
    )

    processor = ViTImageProcessor.from_pretrained(CHECKPOINT)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(dataset):
        image = sample["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Preprocess exactly as the model expects
        inputs = processor(images=image, return_tensors="np")
        pixel_values = inputs["pixel_values"][0]  # [3, 224, 224]

        out_path = output_dir / f"calib_{i:04d}.npy"
        np.save(out_path, pixel_values)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{num_images}")

    print(f"✓ Saved {num_images} calibration arrays to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-images", type=int, default=NUM_IMAGES)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()
    build(args.num_images, Path(args.output_dir))
