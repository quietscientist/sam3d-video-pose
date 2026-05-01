#!/usr/bin/env python3
"""Crop-level appearance embedder for person re-identification."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


class AppearanceEmbedder:
    """Extracts visual embeddings from person crops using DINOv2."""

    def __init__(self, model_id: str = "facebook/dinov2-small", device: str = "cuda"):
        print(f"  Loading appearance embedder ({model_id})...")
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device)
        self.model.eval()
        self.device = device
        print(f"  ✓ Appearance embedder loaded")

    @torch.no_grad()
    def embed_crop(self, frame_rgb: np.ndarray, box: tuple[float, float, float, float]) -> torch.Tensor | None:
        """Extract normalized CLS-token embedding from a bounding box crop."""
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        h, w = frame_rgb.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame_rgb[y1:y2, x1:x2]
        pil = Image.fromarray(crop)
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        cls_token = outputs.last_hidden_state[:, 0]  # (1, D)
        return F.normalize(cls_token, dim=-1).squeeze(0)  # (D,)

    @staticmethod
    def similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        return float((emb1 * emb2).sum().item())
