#!/usr/bin/env python3
"""
3D mesh estimation using SAM-3D-Body.
"""

import os
import sys

# Add parent directory to path for utils import
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from sam3dvideo.utils import setup_sam_3d_body, process_image_with_mask


class MeshEstimator:
    """Estimates 3D body meshes using SAM-3D-Body."""

    def __init__(self, hf_repo_id="facebook/sam-3d-body-dinov3", device="cuda:0"):
        """
        Initialize mesh estimator.

        Args:
            hf_repo_id: HuggingFace repository ID for SAM-3D-Body model
            device: Device to use (default: "cuda:1" to use second GPU)
        """
        print(f"\nLoading 3D body estimator on {device}...")
        self.estimator = setup_sam_3d_body(hf_repo_id=hf_repo_id, device=device)
        self.faces = self.estimator.faces
        print("✓ Loaded")

    def estimate_mesh(self, image_path, mask_path):
        """
        Generate 3D mesh from image and mask.

        Args:
            image_path: Path to input image
            mask_path: Path to segmentation mask

        Returns:
            Mesh outputs dict or None if failed
        """
        return process_image_with_mask(self.estimator, image_path, mask_path)

    def get_faces(self):
        """Get mesh face topology."""
        return self.faces
