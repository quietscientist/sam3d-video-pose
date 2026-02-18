#!/usr/bin/env python3
"""
SAM3 video segmentation processor.
"""

import torch
import cv2
import numpy as np
from transformers import Sam3VideoModel, Sam3VideoProcessor
from accelerate import Accelerator


class SAM3Segmenter:
    """Segments videos using SAM3 model."""

    ALLOWED_TRACKING_PARAMS = {
        'max_num_objects', 'new_det_thresh', 'score_threshold_detection',
        'assoc_iou_thresh', 'min_trk_keep_alive', 'max_trk_keep_alive',
        'hotstart_delay', 'recondition_every_nth_frame',
    }

    def __init__(self, device=None, chunk_size=50, tracking_params=None):
        """
        Initialize SAM3 segmenter.

        Args:
            device: Torch device (will auto-detect if None)
            chunk_size: Number of frames to process per chunk
            tracking_params: Dict of SAM3 tracking parameter overrides (e.g.
                {'max_num_objects': 1, 'new_det_thresh': 0.9})
        """
        self.device = device or Accelerator().device
        self.chunk_size = chunk_size
        self.tracking_params = tracking_params or {}
        self.sam_model = None
        self.sam_processor = None
        self._load_models()

    def _load_models(self):
        """Load SAM3 models."""
        print("\nLoading SAM3 Video model...")
        self.sam_model = Sam3VideoModel.from_pretrained("facebook/sam3").to(
            self.device, dtype=torch.bfloat16
        )
        self.sam_processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")

        if self.tracking_params:
            print("  Applying tracking parameter overrides:")
            for param, value in self.tracking_params.items():
                if param not in self.ALLOWED_TRACKING_PARAMS:
                    print(f"    ⚠ Unknown tracking parameter '{param}', skipping")
                    continue
                old_value = getattr(self.sam_model, param, None)
                setattr(self.sam_model, param, value)
                print(f"    {param}: {old_value} → {value}")

        print(f"✓ Loaded on {self.device}")

    def segment_video_chunks(self, video_path, text_prompt, max_frames, start_idx=0):
        """
        Segment video in chunks.

        Args:
            video_path: Path to video file
            text_prompt: Text prompt for segmentation (e.g., "a baby")
            max_frames: Maximum number of frames to process
            start_idx: Starting frame index

        Yields:
            tuple: (frame_idx, segmentation_outputs)
        """
        end_idx = start_idx + max_frames

        # Load this chunk using OpenCV
        chunk_frames = []
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

        for _ in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB and add to list
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            chunk_frames.append(frame_rgb)

        cap.release()

        if not chunk_frames:
            return

        # Convert to numpy array matching transformers format
        chunk_frames = np.array(chunk_frames)
        print(f"Loaded {len(chunk_frames)} frames")

        # Process with SAM3
        print("Running SAM3 segmentation...")
        inference_session = self.sam_processor.init_video_session(
            video=chunk_frames,
            inference_device=self.device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=torch.bfloat16,
        )

        inference_session = self.sam_processor.add_text_prompt(
            inference_session=inference_session,
            text=text_prompt,
        )

        chunk_outputs = {}
        for model_outputs in self.sam_model.propagate_in_video_iterator(
            inference_session=inference_session,
            max_frame_num_to_track=len(chunk_frames)
        ):
            processed = self.sam_processor.postprocess_outputs(inference_session, model_outputs)
            global_idx = start_idx + model_outputs.frame_idx
            chunk_outputs[global_idx] = processed

        print(f"✓ SAM3: {len(chunk_outputs)} frames")

        # Yield frame by frame with the loaded frame data
        for frame_idx, outputs in chunk_outputs.items():
            local_idx = frame_idx - start_idx
            frame = chunk_frames[local_idx]
            yield frame_idx, outputs, frame

    def cleanup(self):
        """Clean up GPU memory."""
        if self.sam_model is not None:
            del self.sam_model
        if self.sam_processor is not None:
            del self.sam_processor
        torch.cuda.empty_cache()
