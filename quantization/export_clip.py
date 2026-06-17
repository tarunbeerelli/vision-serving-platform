"""
Export CLIP visual and text encoders to ONNX.

CLIP has two separate encoders:
  - Visual encoder: image → embedding [1, 512]
  - Text encoder: tokenized text → embedding [1, 512]

We export them as separate ONNX models so Triton can run them
independently and we can batch visual and text requests separately.

Runs locally on M1 — no GPU needed.
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path

import torch
from transformers import CLIPModel

CHECKPOINT = "openai/clip-vit-base-patch32"
VISUAL_OUTPUT = Path("triton_repo/clip_visual/1/model.onnx")
TEXT_OUTPUT = Path("triton_repo/clip_text/1/model.onnx")


class CLIPVisualEncoder(torch.nn.Module):
    """Wraps CLIP visual encoder for clean ONNX export."""

    def __init__(self, model: CLIPModel) -> None:
        super().__init__()
        self.vision_model = model.vision_model
        self.visual_projection = model.visual_projection

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.vision_model(pixel_values=pixel_values)
        pooled = outputs.pooler_output
        return self.visual_projection(pooled)  # [batch, 512]


class CLIPTextEncoder(torch.nn.Module):
    """Wraps CLIP text encoder for clean ONNX export."""

    def __init__(self, model: CLIPModel) -> None:
        super().__init__()
        self.text_model = model.text_model
        self.text_projection = model.text_projection

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled = outputs.pooler_output
        return self.text_projection(pooled)  # [batch, 512]


def export_visual(model: CLIPModel, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoder = CLIPVisualEncoder(model).eval()
    dummy = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        encoder,
        dummy,
        str(output_path),
        opset_version=17,
        input_names=["pixel_values"],
        output_names=["image_embeddings"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "image_embeddings": {0: "batch_size"},
        },
        do_constant_folding=True,
    )
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"✓ Visual encoder exported to {output_path} ({size_mb:.1f} MB)")


def export_text(model: CLIPModel, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoder = CLIPTextEncoder(model).eval()

    # CLIP text encoder max sequence length is 77
    dummy_ids = torch.zeros(1, 77, dtype=torch.long)
    dummy_mask = torch.ones(1, 77, dtype=torch.long)

    torch.onnx.export(
        encoder,
        (dummy_ids, dummy_mask),
        str(output_path),
        opset_version=17,
        input_names=["input_ids", "attention_mask"],
        output_names=["text_embeddings"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "text_embeddings": {0: "batch_size"},
        },
        do_constant_folding=True,
    )
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"✓ Text encoder exported to {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    print(f"Loading {CHECKPOINT}...")
    model = CLIPModel.from_pretrained(CHECKPOINT)
    model.eval()

    export_visual(model, VISUAL_OUTPUT)
    export_text(model, TEXT_OUTPUT)
    print("\n✓ CLIP export complete")
