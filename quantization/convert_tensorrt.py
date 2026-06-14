"""
Convert ONNX → TensorRT INT8 engine.

Runs on Vast.ai GPU instance ONLY — TensorRT is not available on M1.

Prerequisites on Vast.ai:
  pip install tensorrt pycuda
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

What this produces:
  triton_repo/vit_int8/1/model.plan   ← TensorRT INT8 engine (~6MB)
  triton_repo/vit_fp32/1/model.plan   ← TensorRT FP32 engine (~25MB)

Why two engines:
  FP32 is the baseline. INT8 is ~3-5x faster with <0.5% accuracy drop.
  Both get deployed to Triton — clients choose via model_version field.
  This is the A/B versioning story demonstrated in Grafana.
"""

import argparse
from pathlib import Path

import numpy as np

ONNX_PATH = Path("triton_repo/vit_fp32/1/model.onnx")
INT8_OUTPUT = Path("triton_repo/vit_int8/1/model.plan")
FP32_OUTPUT = Path("triton_repo/vit_fp32/1/model.plan")
CALIB_DIR = Path("quantization/calibration_data")


class Int8CalibrationCache:
    """
    TensorRT INT8 calibrator.
    Feeds calibration images to TensorRT so it can compute
    per-layer activation ranges and determine optimal INT8 scale factors.
    """

    def __init__(self, calib_dir: Path, cache_file: str = "calib.cache") -> None:
        self.calib_files = sorted(calib_dir.glob("*.npy"))
        self.cache_file = cache_file
        self.current_index = 0
        self.batch_size = 8

        assert len(self.calib_files) > 0, f"No calibration files in {calib_dir}"
        print(f"Calibrator: {len(self.calib_files)} images, batch size {self.batch_size}")

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names: list[str]) -> list | None:
        import pycuda.driver as cuda  # type: ignore[import]

        if self.current_index >= len(self.calib_files):
            return None

        batch_files = self.calib_files[self.current_index : self.current_index + self.batch_size]
        self.current_index += self.batch_size

        batch = np.stack([np.load(f) for f in batch_files]).astype(np.float32)
        # Pad last batch if smaller than batch_size
        if batch.shape[0] < self.batch_size:
            pad = np.zeros((self.batch_size - batch.shape[0], *batch.shape[1:]), dtype=np.float32)
            batch = np.concatenate([batch, pad], axis=0)

        device_mem = cuda.mem_alloc(batch.nbytes)
        cuda.memcpy_htod(device_mem, batch)
        return [device_mem]

    def read_calibration_cache(self) -> bytes | None:
        if Path(self.cache_file).exists():
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        print(f"✓ Calibration cache written to {self.cache_file}")


def build_engine(
    onnx_path: Path,
    output_path: Path,
    use_int8: bool,
    calib_dir: Path | None = None,
) -> None:
    import tensorrt as trt  # type: ignore[import]

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    # 4GB workspace — TensorRT uses this for layer fusion scratch space
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    if use_int8:
        assert calib_dir is not None
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = Int8CalibrationCache(calib_dir)
        print("Building INT8 engine (this takes ~5 minutes)...")
    else:
        config.set_flag(trt.BuilderFlag.FP16)
        print("Building FP16 engine...")

    # Dynamic batch profile — tells TensorRT the min/opt/max batch sizes
    # to optimise for. Matches Triton's dynamic batcher config.
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "pixel_values",
        min=(1, 3, 224, 224),
        opt=(8, 3, 224, 224),
        max=(32, 3, 224, 224),
    )
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    assert serialized is not None, "Engine build failed"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(serialized)

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"✓ Engine saved to {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-path", default=str(ONNX_PATH))
    parser.add_argument("--int8-output", default=str(INT8_OUTPUT))
    parser.add_argument("--fp32-output", default=str(FP32_OUTPUT))
    parser.add_argument("--calib-dir", default=str(CALIB_DIR))
    args = parser.parse_args()

    # Build FP32 engine first (baseline)
    build_engine(
        Path(args.onnx_path),
        Path(args.fp32_output),
        use_int8=False,
    )

    # Build INT8 engine (production)
    build_engine(
        Path(args.onnx_path),
        Path(args.int8_output),
        use_int8=True,
        calib_dir=Path(args.calib_dir),
    )
