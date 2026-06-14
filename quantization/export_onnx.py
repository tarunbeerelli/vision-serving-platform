import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path

from optimum.exporters.onnx import main_export

OUTPUT_DIR = Path("triton_repo/vit_fp32/1")
CHECKPOINT = "WinKawaks/vit-tiny-patch16-224"

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Exporting {CHECKPOINT} to ONNX via Optimum...")

    main_export(
        model_name_or_path=CHECKPOINT,
        output=OUTPUT_DIR,
        task="image-classification",
        device="cpu",
        opset=17,
    )
    print(f"✓ Exported to {OUTPUT_DIR}")
