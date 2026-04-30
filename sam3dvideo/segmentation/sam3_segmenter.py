#!/usr/bin/env python3
"""
SAM3 video segmentation processor.
"""

import traceback
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

    def segment_video_chunks(self, video_path, text_prompt, max_frames, start_idx=0,
                              prev_mask=None):
        """
        Segment video in chunks.

        Args:
            video_path: Path to video file
            text_prompt: Text prompt for segmentation (e.g., "a baby")
            max_frames: Maximum number of frames to process
            start_idx: Starting frame index
            prev_mask: Optional binary mask (H, W) bool tensor from the last frame of the
                previous chunk. When provided, seeds object 0 on frame 0 of this chunk so
                the tracker re-locks onto the same subject rather than re-running text
                detection from scratch.

        Yields:
            tuple: (frame_idx, segmentation_outputs, frame)
        """
        # Load this chunk using OpenCV
        chunk_frames = []
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

        for _ in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            chunk_frames.append(frame_rgb)

        cap.release()

        if not chunk_frames:
            return

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

        # Attempt to seed frame 0 with the previous chunk's last mask so the tracker
        # re-locks onto the same subject at chunk boundaries. This manipulates internal
        # SAM3 session state; if it fails we fall back to normal text-prompt detection.
        seeded = False
        if prev_mask is not None:
            try:
                obj_idx = inference_session.obj_id_to_idx(0)
                mask_input = prev_mask.float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
                inference_session.add_mask_inputs(obj_idx, 0, mask_input)
                inference_session.obj_with_new_inputs = [0]
                inference_session.obj_id_to_prompt_id[0] = 0
                inference_session.obj_id_to_score[0] = 1.0
                inference_session.max_obj_id = 0
                seeded = True
                print("  ↳ Seeded frame 0 with previous chunk mask")
            except Exception:
                print("  ⚠ Mask seeding failed, falling back to text-prompt detection:")
                traceback.print_exc()

        def _collect_outputs(session):
            result = {}
            for model_outputs in self.sam_model.propagate_in_video_iterator(
                inference_session=session,
                max_frame_num_to_track=len(chunk_frames)
            ):
                processed = self.sam_processor.postprocess_outputs(session, model_outputs)
                result[start_idx + model_outputs.frame_idx] = processed
            return result

        try:
            chunk_outputs = _collect_outputs(inference_session)
        except Exception:
            if seeded:
                print("  ✗ Propagation failed with seeded session — retrying without seed:")
                traceback.print_exc()
                # Rebuild a clean session without seeding
                fresh_session = self.sam_processor.init_video_session(
                    video=chunk_frames,
                    inference_device=self.device,
                    processing_device="cpu",
                    video_storage_device="cpu",
                    dtype=torch.bfloat16,
                )
                fresh_session = self.sam_processor.add_text_prompt(
                    inference_session=fresh_session,
                    text=text_prompt,
                )
                chunk_outputs = _collect_outputs(fresh_session)
            else:
                raise

        print(f"✓ SAM3: {len(chunk_outputs)} frames")

        # Warn if the first frame's mask doesn't overlap with prev_mask — indicates
        # the tracker locked onto a different subject at the chunk boundary.
        if prev_mask is not None and chunk_outputs:
            first_idx = min(chunk_outputs.keys())
            first_masks = chunk_outputs[first_idx].get('masks')
            if first_masks is not None and len(first_masks) > 0:
                m = first_masks[0].cpu().bool()
                ref = prev_mask.bool()
                iou = (m & ref).sum().float() / (m | ref).sum().float().clamp(min=1)
                if iou < 0.1:
                    print(f"  ⚠ Chunk boundary IoU={iou:.2f} — tracker may have switched subjects")
                else:
                    print(f"  ✓ Chunk boundary IoU={iou:.2f} — tracking appears continuous")

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
