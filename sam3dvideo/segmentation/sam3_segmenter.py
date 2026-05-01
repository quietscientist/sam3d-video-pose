#!/usr/bin/env python3
"""
SAM3 video segmentation processor.
"""

import traceback
import os
import torch
import cv2
import numpy as np
from transformers import Sam3VideoModel, Sam3VideoProcessor
from accelerate import Accelerator

from .appearance_embedder import AppearanceEmbedder
from .target_selector import TargetLockDecision, TargetSelector


class SAM3Segmenter:
    """Segments videos using SAM3 model."""

    ALLOWED_TRACKING_PARAMS = {
        'max_num_objects', 'new_det_thresh', 'score_threshold_detection',
        'assoc_iou_thresh', 'min_trk_keep_alive', 'max_trk_keep_alive',
        'hotstart_delay', 'recondition_every_nth_frame',
    }
    WRAPPER_TRACKING_PARAMS = {
        'target_lock', 'target_lock_candidate_count', 'target_output_object_id',
        'target_initial_selection', 'target_min_iou', 'target_max_center_jump',
        'target_max_area_ratio', 'target_reacquire_after',
        'target_center_prior_weight',
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
        self.target_lock_enabled = self._target_lock_default()
        self.target_candidate_count = int(
            self.tracking_params.get('target_lock_candidate_count', 5)
        )
        appearance_model_id = self.tracking_params.get('appearance_model_id', 'facebook/dinov2-small')
        embedder = None
        if self.target_lock_enabled:
            try:
                embedder = AppearanceEmbedder(model_id=appearance_model_id, device=str(device or 'cuda'))
            except Exception as e:
                print(f"  ⚠ Could not load appearance embedder: {e} — reacquire will use area proximity")

        self.target_selector = TargetSelector(
            enabled=self.target_lock_enabled,
            output_object_id=int(self.tracking_params.get('target_output_object_id', 0)),
            initial_selection=self.tracking_params.get('target_initial_selection', 'score_center'),
            min_iou=float(self.tracking_params.get('target_min_iou', 0.02)),
            max_center_jump=float(self.tracking_params.get('target_max_center_jump', 0.15)),
            max_area_ratio=float(self.tracking_params.get('target_max_area_ratio', 4.0)),
            reacquire_after=int(self.tracking_params.get('target_reacquire_after', 12)),
            center_prior_weight=float(self.tracking_params.get('target_center_prior_weight', 0.25)),
            embedder=embedder,
            similarity_threshold=float(self.tracking_params.get('target_similarity_threshold', 0.5)),
        )
        self.sam_model = None
        self.sam_processor = None
        self._load_models()

    def _target_lock_default(self):
        raw_value = self.tracking_params.get(
            'target_lock',
            self.tracking_params.get('max_num_objects') == 1,
        )
        if isinstance(raw_value, str):
            return raw_value.lower() in {'1', 'true', 'yes', 'on'}
        return bool(raw_value)

    def _effective_model_tracking_params(self):
        model_params = {}
        for param, value in self.tracking_params.items():
            if param in self.ALLOWED_TRACKING_PARAMS:
                model_params[param] = value

        if self.target_lock_enabled:
            requested_max = model_params.get('max_num_objects')
            if requested_max is None or int(requested_max) <= 1:
                model_params['max_num_objects'] = max(self.target_candidate_count, 1)

        return model_params

    def _load_models(self):
        """Load SAM3 models."""
        print("\nLoading SAM3 Video model...")
        model_id = os.environ.get("SAM3_MODEL_ID", "facebook/sam3.1")
        try:
            self.sam_model = Sam3VideoModel.from_pretrained(model_id).to(
                self.device, dtype=torch.bfloat16
            )
            self.sam_processor = Sam3VideoProcessor.from_pretrained(model_id)
        except OSError as exc:
            fallback_model_id = "facebook/sam3"
            if model_id == fallback_model_id:
                raise
            print(f"  ⚠ Could not load {model_id}: {exc}")
            print(f"  ↳ Falling back to {fallback_model_id}")
            self.sam_model = Sam3VideoModel.from_pretrained(fallback_model_id).to(
                self.device, dtype=torch.bfloat16
            )
            self.sam_processor = Sam3VideoProcessor.from_pretrained(fallback_model_id)

        model_tracking_params = self._effective_model_tracking_params()
        if self.tracking_params:
            print("  Applying tracking parameter overrides:")
            for param, value in self.tracking_params.items():
                if param not in self.ALLOWED_TRACKING_PARAMS and param not in self.WRAPPER_TRACKING_PARAMS:
                    print(f"    ⚠ Unknown tracking parameter '{param}', skipping")
            for param, value in model_tracking_params.items():
                old_value = getattr(self.sam_model, param, None)
                setattr(self.sam_model, param, value)
                print(f"    {param}: {old_value} → {value}")

        if self.target_lock_enabled:
            print(
                "  Target lock: enabled "
                f"(SAM3 candidates={model_tracking_params.get('max_num_objects')}, "
                f"output_object_id={self.target_selector.output_object_id})"
            )

        print(f"✓ Loaded on {self.device}")

    def _log_target_decision(self, decision: TargetLockDecision):
        if not decision.should_log:
            return

        if decision.selected:
            if decision.reason == "initial_lock":
                print(
                    f"  Target lock frame {decision.frame_idx}: "
                    f"locked source object {decision.source_object_id} "
                    f"(score={decision.score:.3f})"
                )
            elif decision.reason == "source_changed":
                print(
                    f"  Target lock frame {decision.frame_idx}: "
                    f"accepted source object {decision.source_object_id} "
                    f"(IoU={decision.iou:.3f}, center_jump={decision.center_dist_norm:.3f})"
                )
        else:
            details = ""
            if decision.iou is not None and decision.center_dist_norm is not None:
                details = (
                    f" (best source={decision.source_object_id}, "
                    f"IoU={decision.iou:.3f}, center_jump={decision.center_dist_norm:.3f})"
                )
            print(f"  Target lock frame {decision.frame_idx}: rejected {decision.reason}{details}")

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
                abs_frame_idx = start_idx + model_outputs.frame_idx
                local_idx = model_outputs.frame_idx
                frame_rgb = chunk_frames[local_idx] if local_idx < len(chunk_frames) else None
                processed, decision = self.target_selector.select(processed, abs_frame_idx, frame=frame_rgb)
                self._log_target_decision(decision)
                result[abs_frame_idx] = processed
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
