#!/usr/bin/env python3
"""
Keypoint extraction using ViTPose and MHR parameters.
"""

import torch
import numpy as np
from transformers import AutoProcessor, VitPoseForPoseEstimation
from PIL import Image


class KeypointExtractor:
    """Extracts 2D (ViTPose) and 3D (MHR) keypoints."""

    def __init__(self, device=None):
        """
        Initialize keypoint extractor.

        Args:
            device: Torch device (will auto-detect if None)
        """
        from accelerate import Accelerator
        self.device = device or Accelerator().device
        self.vitpose_processor = None
        self.vitpose_model = None
        self._load_vitpose()

    def _load_vitpose(self):
        """Load ViTPose models."""
        print("Loading ViTPose model...")
        self.vitpose_processor = AutoProcessor.from_pretrained(
            "usyd-community/vitpose-plus-small", use_fast=True
        )
        self.vitpose_model = VitPoseForPoseEstimation.from_pretrained(
            "usyd-community/vitpose-plus-small", device_map=self.device
        )
        print("✓ ViTPose model loaded")

    def extract_2d_keypoints(self, image_path, bbox):
        """
        Extract 2D keypoints using ViTPose given a bounding box.

        Args:
            image_path: Path to image
            bbox: Bounding box [x1, y1, x2, y2] or [x, y, w, h]

        Returns:
            dict: {'keypoints': [x1, y1, v1, ...], 'num_keypoints': int}
            or None if extraction failed
        """
        try:
            image = Image.open(image_path).convert('RGB')

            # Convert bbox to list if numpy
            if isinstance(bbox, np.ndarray):
                bbox = bbox.tolist()

            # Convert xyxy to xywh if needed
            if bbox[2] > bbox[0] + 100:  # x2 > x1 + threshold means xyxy format
                x1, y1, x2, y2 = bbox
                bbox = [x1, y1, x2 - x1, y2 - y1]

            # Triple-nested list for ViTPose
            boxes = [[bbox]]

            inputs = self.vitpose_processor(image, boxes=boxes, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.vitpose_model(**inputs, dataset_index=torch.tensor(0).to(self.device))

            pose_results = self.vitpose_processor.post_process_pose_estimation(
                outputs, boxes=boxes, threshold=0.3
            )

            if not pose_results or not pose_results[0]:
                return None

            person = pose_results[0][0]
            if isinstance(person, list):  # Handle extra nesting
                person = person[0]

            keypoints_data = person['keypoints'].cpu().numpy()
            scores_data = person['scores'].cpu().numpy()

            coco_keypoints = []
            for kp_idx in range(len(keypoints_data)):
                x, y = keypoints_data[kp_idx]
                score = scores_data[kp_idx]
                v = 2 if score > 0.3 else 0
                coco_keypoints.extend([float(x), float(y), v])

            return {
                'keypoints': coco_keypoints,
                'num_keypoints': sum(1 for i in range(2, len(coco_keypoints), 3) if coco_keypoints[i] > 0)
            }

        except Exception as e:
            print(f"⚠ ViTPose failed: {e}")
            return None

    def extract_mhr_parameters(self, mesh_outputs):
        """
        Extract all MHR parameters from mesh outputs.

        Args:
            mesh_outputs: Mesh estimation outputs

        Returns:
            dict: MHR parameters including 3D keypoints, vertices, pose, shape, etc.
        """
        if not mesh_outputs or len(mesh_outputs) == 0:
            return None

        # mesh_outputs is a list, get first person
        person_data = mesh_outputs[0]

        mhr_params = {}

        # Extract all parameters
        param_keys = [
            'pred_keypoints_3d',      # 3D skeletal joints (70 points)
            'pred_keypoints_2d',      # 2D projected joints
            'pred_vertices',          # Mesh vertices
            'pred_cam_t',             # Camera translation
            'pred_pose_raw',          # Raw pose
            'global_rot',             # Global rotation
            'body_pose_params',       # Body pose parameters
            'hand_pose_params',       # Hand pose parameters
            'shape_params',           # Body shape parameters
            'scale_params',           # Scale parameters
            'pred_joint_coords',      # Joint coordinates
            'pred_global_rots',       # Global rotations
            'bbox',                   # Bounding box
            'focal_length',           # Focal length
            'lhand_bbox',             # Left hand bbox
            'rhand_bbox'              # Right hand bbox
        ]

        for key in param_keys:
            if key in person_data:
                val = person_data[key]
                if torch.is_tensor(val):
                    mhr_params[key] = val.cpu().numpy()
                elif isinstance(val, np.ndarray):
                    mhr_params[key] = val
                else:
                    mhr_params[key] = val

        return mhr_params
