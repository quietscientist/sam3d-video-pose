#!/usr/bin/env python3
"""
Process video to extract 3D body meshes and keypoints with temporal smoothing.

Usage:
    python process_video.py VIDEO_PATH [--max-frames 300] [--output-dir "output"]
"""

import argparse
import csv
import os
import sys
import json
from collections import Counter
from pathlib import Path
import torch
import gc
from tqdm import tqdm
import numpy as np
import cv2
from scipy.signal import savgol_filter
from dotenv import load_dotenv
import trimesh

# Add project root directory to path for camt imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Parent of scripts/ is the project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import our modular processors
from sam3dvideo.utils import (
    patch_sam3,
    ConfigLoader,
    ExperimentLogger,
    setup_sam_3d_body,
    visualize_3d_mesh,
    save_mesh_results,
    download_and_convert_video
)
from sam3dvideo.segmentation import SAM3Segmenter
from sam3dvideo.reconstruction import MeshEstimator, KeypointExtractor
from sam3dvideo.processing import (
    QualityAnalyzer,
    TemporalSmoother,
    BundleAdjuster,
    MetricsLogger
)

# Environment setup
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'
load_dotenv()


class VideoProcessor:
    """Main orchestrator for video processing pipeline."""

    def __init__(self, video_path, text_prompt="a baby", max_frames=300, start_frame=0,
                 output_dir="output", enable_quality_filter=False, smoothing_sigma=2.0,
                 skip_mesh_generation=False, skip_mesh_saving=False,
                 export_coco_csv=False, bundle_adjustment=True, constrain_torso=False,
                 temporal_smooth_keypoints=True, temporal_smooth_window=11,
                 temporal_smooth_polyorder=3, quality_z_velocity_threshold=0.15,
                 quality_total_velocity_threshold=0.25, quality_vertex_displacement_threshold=0.08,
                 fast_2d_only=False,
                 enable_metrics=True, enable_plots=True,
                 tracking_params=None,
                 chunk_size=50):
        """
        Initialize video processor.

        Args:
            video_path: Path to input video
            text_prompt: Object to segment (default: "a baby")
            max_frames: Maximum frames to process
            output_dir: Output directory
            enable_quality_filter: If True, skip bad quality frames
            smoothing_sigma: Temporal smoothing strength for meshes
            skip_mesh_generation: If True, only extract 2D keypoints (no 3D meshes)
            skip_mesh_saving: If True, generate meshes for MHR params but don't save PLY files
            export_coco_csv: If True, export COCO 17-point CSV files
            bundle_adjustment: If True, apply bundle adjustment to COCO keypoints
            constrain_torso: If True, constrain torso length (not recommended for infants)
            temporal_smooth_keypoints: If True, apply Savitzky-Golay smoothing to keypoints
            temporal_smooth_window: Window length for SG filter (default: 11, must be odd)
            temporal_smooth_polyorder: Polynomial order for SG filter (default: 3)
            quality_z_velocity_threshold: Max Z-axis camera movement for quality (default: 0.15)
            quality_total_velocity_threshold: Max total camera movement for quality (default: 0.25)
            quality_vertex_displacement_threshold: Max vertex deformation for quality (default: 0.08)
            fast_2d_only: If True, run SAM3 + ViTPose only with temporal smoothing
            enable_metrics: If True, track and log comprehensive metrics (default: True)
            enable_plots: If True, generate visualization plots (default: True)
        """
        self.video_path = video_path
        self.text_prompt = text_prompt
        self.max_frames = max_frames
        self.start_frame = start_frame
        self.output_dir = output_dir
        self.enable_quality_filter = enable_quality_filter
        self.smoothing_sigma = smoothing_sigma
        self.skip_mesh_generation = skip_mesh_generation
        self.skip_mesh_saving = skip_mesh_saving
        self.export_coco_csv = export_coco_csv
        self.bundle_adjustment = bundle_adjustment
        self.constrain_torso = constrain_torso
        self.temporal_smooth_keypoints = temporal_smooth_keypoints
        self.temporal_smooth_window = temporal_smooth_window
        self.temporal_smooth_polyorder = temporal_smooth_polyorder
        self.fast_2d_only = fast_2d_only
        self.enable_metrics = enable_metrics
        self.enable_plots = enable_plots
        self.tracking_params = tracking_params or {}
        self.chunk_size = chunk_size
        self.video_name = Path(video_path).stem

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Initialize processors
        self.segmenter = None
        self.mesh_estimator = None
        self.keypoint_extractor = None
        self.quality_analyzer = QualityAnalyzer(
            z_velocity_threshold=quality_z_velocity_threshold,
            total_velocity_threshold=quality_total_velocity_threshold,
            vertex_displacement_threshold=quality_vertex_displacement_threshold
        )
        self.temporal_smoother = TemporalSmoother(smoothing_sigma=smoothing_sigma)
        self.bundle_adjuster = BundleAdjuster(
            constrain_torso=constrain_torso,
            temporal_smooth_window=temporal_smooth_window,
            temporal_smooth_polyorder=temporal_smooth_polyorder
        ) if export_coco_csv else None


        # Initialize metrics logger
        self.metrics_logger = MetricsLogger(
            output_dir=output_dir,
            video_name=self.video_name,
            enable_plots=enable_plots
        ) if enable_metrics else None

        # Get video info
        self.video_info = self._get_video_info()

    def _get_video_info(self):
        """Extract video metadata."""
        cap = cv2.VideoCapture(self.video_path)
        info = {
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        }
        cap.release()

        if self.max_frames == -1:
            self.max_frames = info['total_frames']
        else:
            self.max_frames = min(self.max_frames, info['total_frames'])

        info['processed_frames'] = self.max_frames
        return info

    def process_video(self):
        """
        Main processing pipeline.

        Returns:
            dict: Processing results including meshes, keypoints, quality log
        """
        print("\n" + "="*60)
        if self.fast_2d_only:
            print("FAST 2D ONLY VIDEO PROCESSING")
        elif self.skip_mesh_generation:
            print("VIDEO TO 2D KEYPOINT EXTRACTION (MESH GENERATION SKIPPED)")
        else:
            print("VIDEO TO 3D MESH PROCESSING")
        print("="*60)
        print(f"Video: {self.video_path}")
        print(f"Prompt: '{self.text_prompt}'")
        print(f"Max frames: {self.max_frames}")
        print(f"Output: {self.output_dir}")
        print(f"Quality filter: {'ENABLED' if self.enable_quality_filter else 'disabled'}")
        if self.fast_2d_only:
            print("Mode: SAM3 + ViTPose with temporal smoothing (no SAMBdy, no bundle adjustment)")
        elif self.skip_mesh_generation:
            print(f"Mode: 2D keypoints only (no 3D meshes)")
        elif self.skip_mesh_saving:
            print(f"Mode: Generate meshes but don't save PLY files")
        else:
            print(f"Mode: Full 3D mesh + keypoint extraction")
        print("="*60 + "\n")

        # Initialize models
        self.segmenter = SAM3Segmenter(tracking_params=self.tracking_params)

        # Only load mesh estimator if needed
        if not self.skip_mesh_generation:
            self.mesh_estimator = MeshEstimator()
        else:
            self.mesh_estimator = None

        self.keypoint_extractor = KeypointExtractor()

        # Log initial sample size
        if self.metrics_logger:
            self.metrics_logger.log_sample_size(
                'initial_frames',
                self.max_frames,
                "Total frames to process"
            )

        # Process frames
        all_mesh_results = []
        frame_to_mhr_params = {}
        quality_log = []
        prev_mhr_params = None
        faces = None

        masks_dir = os.path.join(self.output_dir, f"{self.video_name}_masks")
        meshes_dir = os.path.join(self.output_dir, f"{self.video_name}_meshes")
        os.makedirs(masks_dir, exist_ok=True)
        os.makedirs(meshes_dir, exist_ok=True)

        # Process in chunks
        chunk_size = self.chunk_size
        num_chunks = (self.max_frames + chunk_size - 1) // chunk_size
        prev_mask = None  # last-frame mask carried forward between chunks

        for chunk_idx in range(num_chunks):
            start_idx = self.start_frame + chunk_idx * chunk_size
            chunk_frames = min(chunk_size, self.max_frames - chunk_idx * chunk_size)

            print(f"\n{'='*60}")
            print(f"CHUNK {chunk_idx + 1}/{num_chunks}: Frames {start_idx}-{start_idx + chunk_frames - 1}")
            print(f"{'='*60}")

            chunk_last_mask = None  # will be updated as we process this chunk

            # Segment with SAM3
            for frame_idx, outputs, frame in self.segmenter.segment_video_chunks(
                self.video_path, self.text_prompt, chunk_frames, start_idx,
                prev_mask=prev_mask
            ):
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Save frame temporarily
                temp_frame_path = os.path.join(masks_dir, f"temp_frame_{frame_idx:04d}.jpg")
                cv2.imwrite(temp_frame_path, frame_bgr)

                # Track last mask for handoff to next chunk
                if len(outputs['masks']) > 0:
                    chunk_last_mask = outputs['masks'][0].cpu()

                # Process each detected object
                for i, (mask, obj_id, score) in enumerate(zip(
                    outputs['masks'],
                    outputs['object_ids'],
                    outputs['scores']
                )):
                    # Save mask
                    mask_path = os.path.join(masks_dir, f"frame_{frame_idx:04d}_obj_{obj_id.item()}.png")
                    mask_np = (mask.cpu().numpy() > 0.5).astype(np.uint8) * 255
                    cv2.imwrite(mask_path, mask_np)

                    box = outputs['boxes'][i].cpu().numpy()
                    mesh_ply_files = None
                    mhr_params = None
                    quality = None

                    # Generate 3D mesh if not skipped
                    if not self.skip_mesh_generation:
                        mesh_outputs = self.mesh_estimator.estimate_mesh(temp_frame_path, mask_path)

                        if mesh_outputs:
                            # Extract MHR parameters
                            mhr_params = self.keypoint_extractor.extract_mhr_parameters(mesh_outputs)

                            # Analyze quality (requires MHR params)
                            quality = self.quality_analyzer.analyze_frame_quality(
                                mhr_params, prev_mhr_params, sam_bbox=box, mask=mask
                            )

                            quality_log.append({
                                'frame_idx': frame_idx,
                                'object_id': obj_id.item(),
                                'sam_score': float(score.item()),
                                'quality': quality
                            })

                            if not quality['is_good'] and not self.enable_quality_filter:
                                print(f"\n  ⚠ Frame {frame_idx}: {quality['reason']}")

                            if self.enable_quality_filter and not quality['is_good']:
                                print(f"  ⏭ Skipping frame {frame_idx}: {quality['reason']}")
                                prev_mhr_params = mhr_params
                                continue

                            # Save mesh PLY files (unless skipped)
                            if not self.skip_mesh_saving:
                                mesh_output_dir = os.path.join(meshes_dir, f"frame_{frame_idx:04d}_obj_{obj_id.item()}")
                                mesh_ply_files = save_mesh_results(
                                    frame_bgr, mesh_outputs, self.mesh_estimator.get_faces(),
                                    mesh_output_dir, f"frame_{frame_idx}_obj_{obj_id.item()}"
                                )

                                # Get faces once
                                if faces is None:
                                    mesh = trimesh.load(mesh_ply_files[0])
                                    faces = mesh.faces
                            else:
                                # Get faces from estimator directly
                                if faces is None:
                                    faces = self.mesh_estimator.get_faces()

                            # Store MHR params
                            frame_to_mhr_params[frame_idx] = mhr_params
                            prev_mhr_params = mhr_params

                    # Extract 2D keypoints (always, independent of mesh)
                    keypoints_2d = self.keypoint_extractor.extract_2d_keypoints(temp_frame_path, box)

                    # Create output directory for keypoints
                    mesh_output_dir = os.path.join(meshes_dir, f"frame_{frame_idx:04d}_obj_{obj_id.item()}")
                    os.makedirs(mesh_output_dir, exist_ok=True)

                    # Save MHR parameters as NPZ (if generated)
                    if mhr_params is not None:
                        mhr_params_path = os.path.join(mesh_output_dir, 'mhr_parameters.npz')
                        np.savez(mhr_params_path, **mhr_params)

                    # Save keypoints JSON
                    keypoints_output = {
                        'frame_idx': frame_idx,
                        'object_id': obj_id.item(),
                        'keypoints_2d': keypoints_2d,
                        'keypoints_3d': mhr_params['pred_keypoints_3d'].tolist() if mhr_params and 'pred_keypoints_3d' in mhr_params else None,
                        'hand_poses': mhr_params['hand_pose_params'].tolist() if mhr_params and 'hand_pose_params' in mhr_params else None,
                        'body_pose': mhr_params['body_pose_params'].tolist() if mhr_params and 'body_pose_params' in mhr_params else None,
                        'quality': quality
                    }

                    keypoints_path = os.path.join(mesh_output_dir, 'keypoints.json')
                    with open(keypoints_path, 'w') as f:
                        json.dump(keypoints_output, f, indent=2)

                    all_mesh_results.append({
                        'frame_idx': frame_idx,
                        'object_id': obj_id.item(),
                        'ply_files': mesh_ply_files,
                        'mesh_output_dir': mesh_output_dir,
                        'score': score.item(),
                        'quality': quality
                    })

                # Clean up temp frame
                if os.path.exists(temp_frame_path):
                    os.remove(temp_frame_path)

            # Carry last mask forward for next chunk's frame-0 seed
            prev_mask = chunk_last_mask

            # Clean up chunk
            gc.collect()
            torch.cuda.empty_cache()

        # Clean up models
        self.segmenter.cleanup()
        gc.collect()
        torch.cuda.empty_cache()

        # Log metrics for processing stages
        if self.metrics_logger:
            # After segmentation: all results (including bad quality frames)
            self.metrics_logger.log_sample_size(
                'after_segmentation',
                len(all_mesh_results),
                "Frames with valid SAM3 masks"
            )

            # After quality filter: good frames only
            good_frames = [r for r in all_mesh_results if r['quality'] and r['quality']['is_good']]
            self.metrics_logger.log_sample_size(
                'after_quality_filter',
                len(good_frames),
                "Frames passing quality checks"
            )

            # Valid 3D keypoints: frames with MHR parameters
            self.metrics_logger.log_sample_size(
                'valid_3d_keypoints',
                len(frame_to_mhr_params),
                "Successful MHR extraction"
            )

            # Log quality statistics
            if quality_log:
                self.metrics_logger.log_quality_statistics(quality_log)

            # Extract and log raw keypoint statistics (if we have 3D data)
            if frame_to_mhr_params and not self.skip_mesh_generation:
                raw_keypoints_3d_list = []
                for frame_idx in sorted(frame_to_mhr_params.keys()):
                    mhr_params = frame_to_mhr_params[frame_idx]
                    if 'pred_keypoints_3d' in mhr_params:
                        raw_keypoints_3d_list.append(mhr_params['pred_keypoints_3d'])

                if raw_keypoints_3d_list:
                    self.metrics_logger.log_keypoint_statistics(
                        stage='raw',
                        keypoints_3d_list=raw_keypoints_3d_list
                    )

        # Save aggregate results
        self._save_results(all_mesh_results, quality_log)

        smoothed_2d_results = None
        if self.temporal_smooth_keypoints:
            smoothed_2d_results = self._smooth_and_export_2d_keypoints(all_mesh_results)

        # Apply bundle adjustment and export COCO CSV if requested
        csv_results = None
        if self.export_coco_csv and self.bundle_adjuster:
            print("\n" + "="*60)
            print("COCO KEYPOINT EXTRACTION & BUNDLE ADJUSTMENT")
            print("="*60)

            # Aggregate keypoints
            all_keypoints = self._aggregate_keypoints(all_mesh_results)

            # Process with bundle adjuster
            csv_results = self.bundle_adjuster.process_keypoints(
                all_keypoints,
                self.output_dir,
                self.video_name,
                apply_temporal_smoothing=self.temporal_smooth_keypoints,
                apply_bundle_adjustment=self.bundle_adjustment,
                export_both=True,
                metrics_logger=self.metrics_logger
            )

            if csv_results:
                print("\n✓ COCO CSV export complete")
                if 'unadjusted' in csv_results:
                    print(f"  Unadjusted: {csv_results['unadjusted']}")
                if 'adjusted' in csv_results:
                    print(f"  Adjusted: {csv_results['adjusted']}")
            print("="*60 + "\n")

        # Save per-frame head direction from MHR neck rotation params
        # neck joint (idx 112) col-0 negated gives face-forward in camera space
        if frame_to_mhr_params:
            head_dir_path = os.path.join(self.output_dir, f"{self.video_name}_head_direction.csv")
            with open(head_dir_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['frame', 'hx', 'hy', 'hz'])
                for frame_idx in sorted(frame_to_mhr_params.keys()):
                    rots = frame_to_mhr_params[frame_idx].get('pred_global_rots')
                    if rots is not None and rots.shape[0] > 112:
                        fwd = -rots[112, :, 0]
                        writer.writerow([frame_idx, fwd[0], fwd[1], fwd[2]])
            print(f"✓ Saved head directions to: {head_dir_path}")

        # Generate comprehensive metrics report
        if self.metrics_logger:
            print("\n" + "="*60)
            print("GENERATING METRICS REPORT & PLOTS")
            print("="*60)

            # Generate all plots
            if self.metrics_logger.enable_plots:
                self.metrics_logger.plot_sample_size_funnel()

                if quality_log:
                    self.metrics_logger.plot_quality_metrics(quality_log)
                    self.metrics_logger.plot_quality_distributions(quality_log)

                # Only plot keypoint jitter if we have multiple stages
                # if len(self.metrics_logger.metrics['keypoint_stats']) > 1:
                #     self.metrics_logger.plot_keypoint_jitter_comparison()

            # Generate final report (saves JSON, CSV, prints summary)
            self.metrics_logger.generate_final_report()
            print("="*60 + "\n")

        return {
            'results': all_mesh_results,
            'frame_mhr_params': frame_to_mhr_params,
            'faces': faces,
            'video_info': self.video_info,
            'quality_log': quality_log,
            'segments': self.quality_analyzer.identify_segments(quality_log),
            'keypoints': self._aggregate_keypoints(all_mesh_results),
            'csv_results': csv_results,
            'smoothed_2d_results': smoothed_2d_results
        }

    def _aggregate_keypoints(self, all_mesh_results):
        """Aggregate all keypoints from mesh results."""
        all_keypoints = {'frames': []}

        for result in all_mesh_results:
            keypoints_path = os.path.join(result['mesh_output_dir'], 'keypoints.json')
            if os.path.exists(keypoints_path):
                with open(keypoints_path, 'r') as f:
                    all_keypoints['frames'].append(json.load(f))

        return all_keypoints

    def _save_results(self, all_mesh_results, quality_log):
        """Save all results to JSON files."""
        # Quality log
        quality_log_path = os.path.join(self.output_dir, f"{self.video_name}_quality_log.json")
        with open(quality_log_path, 'w') as f:
            json.dump(quality_log, f, indent=2)
        print(f"\n✓ Saved quality log to: {quality_log_path}")

        # Segments
        segments = self.quality_analyzer.identify_segments(quality_log)
        segments_data = {
            'video_name': self.video_name,
            'total_frames_processed': len(quality_log),
            'good_frames': sum(1 for e in quality_log if e['quality']['is_good']),
            'bad_frames': sum(1 for e in quality_log if not e['quality']['is_good']),
            'num_segments': len(segments),
            'segments': segments
        }

        segments_path = os.path.join(self.output_dir, f"{self.video_name}_segments.json")
        with open(segments_path, 'w') as f:
            json.dump(segments_data, f, indent=2)

        print(f"\n{'='*60}")
        print("QUALITY ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"Total frames: {len(quality_log)}")
        print(f"Good frames: {segments_data['good_frames']}")
        print(f"Bad frames: {segments_data['bad_frames']}")
        print(f"Identified {len(segments)} continuous segments:")
        for i, seg in enumerate(segments):
            print(f"  Segment {i+1}: frames {seg['start_frame']}-{seg['end_frame']} ({seg['num_frames']} frames)")
        print(f"Saved segments info to: {segments_path}")
        print(f"{'='*60}\n")

        # All keypoints
        all_keypoints = self._aggregate_keypoints(all_mesh_results)
        keypoints_summary_path = os.path.join(self.output_dir, f"{self.video_name}_all_keypoints.json")
        with open(keypoints_summary_path, 'w') as f:
            json.dump(all_keypoints, f, indent=2)
        print(f"✓ Saved all keypoints to: {keypoints_summary_path}")

        # Mesh results
        results_summary_path = os.path.join(self.output_dir, f"{self.video_name}_mesh_results.json")
        with open(results_summary_path, 'w') as f:
            json.dump(all_mesh_results, f, indent=2)
        print(f"✓ Saved mesh results summary to: {results_summary_path}")

        # Video info (fps, dimensions, frame counts)
        video_info_path = os.path.join(self.output_dir, f"{self.video_name}_video_info.json")
        with open(video_info_path, 'w') as f:
            json.dump(self.video_info, f, indent=2)
        print(f"✓ Saved video info to: {video_info_path}")

    def _smooth_and_export_2d_keypoints(self, all_mesh_results):
        """Smooth 2D keypoints over time and export JSON/CSV outputs."""
        all_keypoints = self._aggregate_keypoints(all_mesh_results)
        frames = all_keypoints.get('frames', [])
        if not frames:
            print("⚠ No 2D keypoints available for smoothing")
            return None

        by_object = {}
        for frame_data in frames:
            keypoints_2d = frame_data.get('keypoints_2d')
            if not keypoints_2d or 'keypoints' not in keypoints_2d:
                continue
            by_object.setdefault(frame_data['object_id'], []).append(frame_data)

        if not by_object:
            print("⚠ No valid 2D keypoint detections available for smoothing")
            return None

        print("\n" + "="*60)
        print("FAST 2D TEMPORAL SMOOTHING")
        print("="*60)
        print(f"Window length: {self.temporal_smooth_window} frames")
        print(f"Polynomial order: {self.temporal_smooth_polyorder}")

        smoothed_count = 0
        for object_id, object_frames in by_object.items():
            object_frames.sort(key=lambda x: x['frame_idx'])
            parsed_frames = []
            for frame_data in object_frames:
                flat = frame_data.get('keypoints_2d', {}).get('keypoints')
                if not flat:
                    continue
                if len(flat) % 3 != 0:
                    print(
                        f"⚠ Object {object_id}, frame {frame_data['frame_idx']}: "
                        f"invalid keypoint length {len(flat)} (not divisible by 3), skipping"
                    )
                    continue
                kp_count = len(flat) // 3
                parsed_frames.append((frame_data, kp_count, np.array(flat, dtype=float).reshape(kp_count, 3)))

            if len(parsed_frames) < self.temporal_smooth_window:
                print(f"⚠ Object {object_id}: only {len(parsed_frames)} valid frames, skipping smoothing")
                continue

            # Smooth only the dominant keypoint layout for this object to avoid shape mismatches.
            kp_counter = Counter(kp_count for _, kp_count, _ in parsed_frames)
            target_kp_count = kp_counter.most_common(1)[0][0]
            matched = [(frame_data, arr) for frame_data, kp_count, arr in parsed_frames if kp_count == target_kp_count]

            if len(matched) < self.temporal_smooth_window:
                print(
                    f"⚠ Object {object_id}: dominant keypoint count ({target_kp_count}) has "
                    f"only {len(matched)} frames, skipping smoothing"
                )
                continue

            keypoints_array = np.array([arr for _, arr in matched])
            smoothed_xy = keypoints_array[:, :, :2].copy()

            for kp_idx in range(target_kp_count):
                for coord_idx in range(2):
                    smoothed_xy[:, kp_idx, coord_idx] = savgol_filter(
                        smoothed_xy[:, kp_idx, coord_idx],
                        window_length=self.temporal_smooth_window,
                        polyorder=self.temporal_smooth_polyorder,
                        mode='nearest'
                    )

            for i, (frame_data, _) in enumerate(matched):
                smoothed = keypoints_array[i].copy()
                smoothed[:, :2] = smoothed_xy[i]
                frame_data['keypoints_2d_smoothed'] = {
                    'keypoints': smoothed.reshape(-1).tolist(),
                    'num_keypoints': frame_data['keypoints_2d'].get('num_keypoints', 0)
                }
                smoothed_count += 1

            print(f"✓ Object {object_id}: smoothed {len(matched)} frames ({target_kp_count} keypoints)")

        # Persist per-frame files with smoothed keypoints when available
        for frame_data in frames:
            frame_idx = frame_data['frame_idx']
            object_id = frame_data['object_id']
            keypoints_path = os.path.join(
                self.output_dir,
                f"{self.video_name}_meshes",
                f"frame_{frame_idx:04d}_obj_{object_id}",
                "keypoints.json"
            )
            if os.path.exists(keypoints_path):
                with open(keypoints_path, 'w') as f:
                    json.dump(frame_data, f, indent=2)

        smoothed_json_path = os.path.join(self.output_dir, f"{self.video_name}_all_keypoints_2d_smoothed.json")
        with open(smoothed_json_path, 'w') as f:
            json.dump(all_keypoints, f, indent=2)

        smoothed_csv_path = os.path.join(self.output_dir, f"{self.video_name}_2D_smoothed.csv")
        with open(smoothed_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)

            max_kp_count = 0
            for frame_data in frames:
                payload = frame_data.get('keypoints_2d_smoothed') or frame_data.get('keypoints_2d')
                if payload and 'keypoints' in payload:
                    max_kp_count = max(max_kp_count, len(payload['keypoints']) // 3)

            header = ['frame', 'object_id']
            for kp_idx in range(max_kp_count):
                header.extend([f'kp{kp_idx}_x', f'kp{kp_idx}_y', f'kp{kp_idx}_v'])
            writer.writerow(header)

            for frame_data in sorted(frames, key=lambda x: (x['frame_idx'], x.get('object_id', 0))):
                payload = frame_data.get('keypoints_2d_smoothed') or frame_data.get('keypoints_2d')
                if not payload or 'keypoints' not in payload:
                    continue
                row_keypoints = payload['keypoints']
                expected_len = max_kp_count * 3
                if len(row_keypoints) < expected_len:
                    row_keypoints = row_keypoints + [float('nan')] * (expected_len - len(row_keypoints))
                writer.writerow([frame_data['frame_idx'], frame_data.get('object_id', 0), *row_keypoints])

        print(f"✓ Saved smoothed 2D keypoints JSON: {smoothed_json_path}")
        print(f"✓ Saved smoothed 2D keypoints CSV: {smoothed_csv_path}")

        video_path = self._save_2d_overlay_video(frames)
        print("="*60 + "\n")

        return {
            'json': smoothed_json_path,
            'csv': smoothed_csv_path,
            'video': video_path,
            'num_frames': smoothed_count
        }

    # COCO skeleton connections: (kp_idx_a, kp_idx_b, BGR_color)
    _SKELETON = [
        (0, 1, (255, 200, 100)), (0, 2, (255, 200, 100)),   # nose-eyes
        (1, 3, (255, 200, 100)), (2, 4, (255, 200, 100)),   # eyes-ears
        (5, 6, (0, 255, 255)),                               # shoulders
        (5, 7, (0, 255, 0)),   (7, 9, (0, 200, 100)),       # left arm
        (6, 8, (0, 255, 0)),   (8, 10, (0, 200, 100)),      # right arm
        (5, 11, (0, 255, 255)), (6, 12, (0, 255, 255)),     # torso sides
        (11, 12, (0, 255, 255)),                             # hips
        (11, 13, (255, 0, 255)), (13, 15, (200, 100, 255)), # left leg
        (12, 14, (255, 0, 255)), (14, 16, (200, 100, 255)), # right leg
    ]

    def _save_2d_overlay_video(self, frames):
        """Render 2D keypoint skeleton overlay onto the original video."""
        # Build frame_idx -> keypoints lookup from smoothed (or raw) 2D data
        kp_by_frame = {}
        for fd in frames:
            payload = fd.get('keypoints_2d_smoothed') or fd.get('keypoints_2d')
            if not payload or 'keypoints' not in payload:
                continue
            flat = payload['keypoints']
            if len(flat) % 3 != 0:
                continue
            n = len(flat) // 3
            kp_by_frame[fd['frame_idx']] = np.array(flat, dtype=float).reshape(n, 3)

        if not kp_by_frame:
            print("⚠ No keypoints for video overlay")
            return None

        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video_out_path = os.path.join(self.output_dir, f"{self.video_name}_2D_skeleton.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_out_path, fourcc, fps, (w, h))

        min_frame = min(kp_by_frame)
        max_frame = max(kp_by_frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, min_frame)

        print(f"Rendering 2D overlay video (frames {min_frame}–{max_frame})...")
        for frame_idx in range(min_frame, max_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in kp_by_frame:
                kps = kp_by_frame[frame_idx]  # (N, 3): x, y, visibility
                n = len(kps)
                for a, b, color in self._SKELETON:
                    if a >= n or b >= n:
                        continue
                    if kps[a, 2] < 0.1 or kps[b, 2] < 0.1:
                        continue
                    pt1 = (int(kps[a, 0]), int(kps[a, 1]))
                    pt2 = (int(kps[b, 0]), int(kps[b, 1]))
                    cv2.line(frame, pt1, pt2, color, 3, cv2.LINE_AA)
                for i in range(n):
                    if kps[i, 2] < 0.1:
                        continue
                    cv2.circle(frame, (int(kps[i, 0]), int(kps[i, 1])), 5, (255, 255, 255), -1)
            writer.write(frame)

        cap.release()
        writer.release()
        print(f"✓ Saved 2D overlay video: {video_out_path}")
        return video_out_path

    def smooth_and_export(self, processing_data):
        """Apply temporal smoothing and export smoothed meshes."""
        # Skip if mesh generation was disabled
        if self.skip_mesh_generation:
            print("\n⚠ Skipping temporal smoothing (mesh generation was disabled)")
            return None

        print("\n" + "="*60)
        print("TEMPORAL SMOOTHING & MESH EXPORT")
        print("="*60)

        results = processing_data['results']
        faces = processing_data['faces']

        if not results or faces is None:
            print("✗ No mesh data to smooth")
            return None

        # Load vertices from saved PLY files
        print("\nLoading meshes from PLY files...")
        frame_vertices = {}
        for result in results:
            frame_idx = result['frame_idx']
            if result['ply_files']:
                mesh = trimesh.load(result['ply_files'][0])
                frame_vertices[frame_idx] = mesh.vertices

        if not frame_vertices:
            print("✗ No vertices loaded from PLY files")
            return None

        frame_indices = sorted(frame_vertices.keys())
        print(f"Loaded {len(frame_indices)} frames")

        # Smooth vertices
        smoothed_vertices = self.temporal_smoother.smooth_mesh_vertices(frame_vertices, frame_indices)

        # Interpolate missing frames
        interpolated_vertices, full_range = self.temporal_smoother.interpolate_frames(
            smoothed_vertices, frame_indices
        )

        # Save smoothed meshes
        saved_meshes, export_dir, smoothed_dir = self.temporal_smoother.save_smoothed_meshes(
            interpolated_vertices, full_range, faces, self.output_dir, self.video_name
        )

        # Save metadata
        metadata_path = self.temporal_smoother.save_metadata(
            export_dir, self.video_info, frame_indices, full_range,
            smoothed_vertices.shape[1], len(faces), smoothed_dir
        )

        print("\n" + "="*60)
        print("EXPORT COMPLETE!")
        print("="*60)
        print(f"\nSmoothed meshes: {smoothed_dir}/")
        print(f"Metadata: {metadata_path}")
        print("="*60 + "\n")

        return {
            'meshes': saved_meshes,
            'metadata': metadata_path
        }


def main():
    parser = argparse.ArgumentParser(
        description="Process video to 3D meshes with temporal smoothing"
    )
    parser.add_argument("video_path", type=str, nargs='?', default=None,
                       help="Path to input video or folder (optional if using --config)")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to YAML configuration file")
    parser.add_argument("--experiment-name", type=str, default=None,
                       help="Experiment name for logging (auto-generated if not provided)")
    parser.add_argument("--text-prompt", type=str, default="a baby",
                       help="Object to segment (default: 'baby')")
    parser.add_argument("--max-frames", type=int, default=300,
                       help="Max frames to process (default: 300, -1 for all)")
    parser.add_argument("--start-frame", type=int, default=0,
                       help="Video frame index to start processing from (default: 0)")
    parser.add_argument("--output-dir", type=str, default="output",
                       help="Output directory (default: 'output')")
    parser.add_argument("--smoothing-sigma", type=float, default=2.0,
                       help="Smoothing strength (default: 2.0)")
    parser.add_argument("--enable-quality-filter", action="store_true",
                       help="Skip bad quality frames (rolling, being picked up)")
    parser.add_argument("--debug-quality", action="store_true",
                       help="Log detailed quality metrics for all frames")
    parser.add_argument("--skip-mesh-generation", action="store_true",
                       help="Skip 3D mesh generation (extract 2D keypoints only)")
    parser.add_argument("--fast-2d-only", action="store_true",
                       help="Fast mode: SAM3 + ViTPose with temporal smoothing only (no SAMBdy, no bundle adjustment)")
    parser.add_argument("--skip-mesh-saving", action="store_true",
                       help="Generate meshes for MHR params but don't save PLY files")
    parser.add_argument("--export-coco-csv", action="store_true",
                       help="Export COCO 17-point keypoints to CSV files")
    parser.add_argument("--no-bundle-adjustment", action="store_true",
                       help="Skip bundle adjustment when exporting COCO CSV")
    parser.add_argument("--constrain-torso", action="store_true",
                       help="Include torso in bone length constraints (not recommended for infants)")
    parser.add_argument("--no-temporal-smooth-keypoints", action="store_true",
                       help="Skip temporal smoothing of keypoints (enabled by default)")
    parser.add_argument("--temporal-smooth-window", type=int, default=11,
                       help="Savitzky-Golay filter window length for keypoints (must be odd, default: 11)")
    parser.add_argument("--temporal-smooth-polyorder", type=int, default=3,
                       help="Savitzky-Golay polynomial order for keypoints (default: 3)")
    parser.add_argument("--quality-z-velocity", type=float, default=0.15,
                       help="Quality: max Z-axis camera movement threshold (default: 0.15, lower=more sensitive)")
    parser.add_argument("--quality-total-velocity", type=float, default=0.25,
                       help="Quality: max total camera movement threshold (default: 0.25, lower=more sensitive)")
    parser.add_argument("--quality-vertex-displacement", type=float, default=0.08,
                       help="Quality: max vertex deformation threshold (default: 0.08, lower=more sensitive)")
    parser.add_argument("--no-metrics", action="store_true",
                       help="Disable comprehensive metrics logging and reports")
    parser.add_argument("--no-plots", action="store_true",
                       help="Disable plot generation (still generate metrics)")
    parser.add_argument("--tracking-params", type=str, default=None,
                       help='SAM3 tracking parameter overrides as JSON string, '
                            'e.g. \'{"max_num_objects": 1, "new_det_thresh": 0.9}\'')
    parser.add_argument("--max-detections", type=int, default=None,
                       help='Maximum number of objects to track (default: SAM3 model default of 10000). '
                            'Set to 1 for single-subject videos to prevent locking onto background objects.')
    parser.add_argument("--chunk-size", type=int, default=50,
                       help='Frames per SAM3 tracking chunk. Larger values reduce track-switch '
                            'artifacts at chunk boundaries but use more GPU memory. '
                            'Set to max-frames to process in one shot (default: 50)')

    args = parser.parse_args()

    # Load configuration
    config = {}
    if args.config:
        print(f"\nLoading configuration from: {args.config}")
        try:
            config = ConfigLoader.load(args.config)
            ConfigLoader.validate(config)
            config = ConfigLoader.merge_with_defaults(config)
            print("✓ Configuration loaded successfully\n")
        except Exception as e:
            print(f"✗ Error loading config: {e}")
            sys.exit(1)

    # Determine input path (CLI takes precedence over config)
    input_path_str = args.video_path if args.video_path else config.get('input')
    if not input_path_str:
        print("Error: No input specified. Provide video_path argument or use --config")
        sys.exit(1)

    # Check if input is a URL
    if input_path_str.startswith(('http://', 'https://')):
        print(f"\n{'='*60}")
        print("URL DETECTED - DOWNLOADING VIDEO")
        print(f"{'='*60}")
        try:
            # Download and convert to data/ directory
            data_dir = "data"
            print(f"Downloading from: {input_path_str}")
            local_path = download_and_convert_video(input_path_str, output_dir=data_dir)
            print(f"✓ Video downloaded and converted to: {local_path}")
            print(f"{'='*60}\n")
            input_path = Path(local_path)
        except Exception as e:
            print(f"✗ Failed to download video: {e}")
            sys.exit(1)
    else:
        input_path = Path(input_path_str)
        if not input_path.exists():
            print(f"Error: Input path not found: {input_path}")
            sys.exit(1)

    # Merge config with command-line args (CLI takes precedence)
    # First, check which CLI args were explicitly provided
    cli_args_provided = set()
    for i, arg in enumerate(sys.argv):
        if arg.startswith('--'):
            cli_args_provided.add(arg.lstrip('-').replace('-', '_'))

    def get_param(cli_value, cli_arg_name, config_keys, default=None):
        """Get parameter from CLI or config, CLI takes precedence."""
        # If CLI argument was explicitly provided, use it
        if cli_arg_name in cli_args_provided:
            return cli_value

        # Otherwise, try to get from config
        if isinstance(config_keys, list):
            for key in config_keys:
                val = config
                parts = key.split('.')
                for i, k in enumerate(parts):
                    if not isinstance(val, dict):
                        val = default
                        break
                    # Use default (not {}) for the final key so missing leaves
                    # don't return an empty dict
                    val = val.get(k, default if i == len(parts) - 1 else {})
                if val != default and val is not None and not isinstance(val, dict):
                    return val
        else:
            config_val = config.get(config_keys, default)
            if config_val != default and config_val is not None:
                return config_val

        # Fall back to default
        return default

    # Build parameters
    text_prompt = get_param(args.text_prompt, 'text_prompt', 'text_prompt', 'a baby')
    output_dir = get_param(args.output_dir, 'output_dir', 'output_dir', 'output')
    experiment_name = args.experiment_name if 'experiment_name' in cli_args_provided else config.get('experiment_name')

    # Processing parameters
    max_frames = get_param(args.max_frames, 'max_frames', ['processing.max_frames'], 300)
    skip_mesh_generation = get_param(args.skip_mesh_generation, 'skip_mesh_generation', ['processing.skip_mesh_generation'], False)
    fast_2d_only = get_param(args.fast_2d_only, 'fast_2d_only', ['processing.fast_2d_only'], False)
    skip_mesh_saving = get_param(args.skip_mesh_saving, 'skip_mesh_saving', ['processing.skip_mesh_saving'], False)
    export_coco_csv = get_param(args.export_coco_csv, 'export_coco_csv', ['processing.export_coco_csv'], False)
    smoothing_sigma = get_param(args.smoothing_sigma, 'smoothing_sigma', ['processing.smoothing_sigma'], 2.0)
    constrain_torso = get_param(args.constrain_torso, 'constrain_torso', ['processing.constrain_torso'], False)
    temporal_smooth_window = get_param(args.temporal_smooth_window, 'temporal_smooth_window', ['processing.temporal_smooth_window'], 11)
    temporal_smooth_polyorder = get_param(args.temporal_smooth_polyorder, 'temporal_smooth_polyorder', ['processing.temporal_smooth_polyorder'], 3)

    # Handle boolean flags (inverted) - these are special because the CLI flag is inverted
    if 'no_bundle_adjustment' in cli_args_provided:
        bundle_adjustment = not args.no_bundle_adjustment
    else:
        bundle_adjustment = config.get('processing', {}).get('bundle_adjustment', True)

    if 'no_temporal_smooth_keypoints' in cli_args_provided:
        temporal_smooth_keypoints = not args.no_temporal_smooth_keypoints
    else:
        temporal_smooth_keypoints = config.get('processing', {}).get('temporal_smooth_keypoints', True)

    # Fast 2D mode force-enables smoothing and disables mesh/BA paths
    if fast_2d_only:
        skip_mesh_generation = True
        export_coco_csv = False
        bundle_adjustment = False
        temporal_smooth_keypoints = True

    # Quality parameters
    enable_quality_filter = get_param(args.enable_quality_filter, 'enable_quality_filter', ['quality.enable_filter'], False)
    quality_z_velocity = get_param(args.quality_z_velocity, 'quality_z_velocity', ['quality.z_velocity_threshold'], 0.15)
    quality_total_velocity = get_param(args.quality_total_velocity, 'quality_total_velocity', ['quality.total_velocity_threshold'], 0.25)
    quality_vertex_displacement = get_param(args.quality_vertex_displacement, 'quality_vertex_displacement', ['quality.vertex_displacement_threshold'], 0.08)

    # Metrics parameters (inverted flags)
    if 'no_metrics' in cli_args_provided:
        enable_metrics = not args.no_metrics
    else:
        enable_metrics = config.get('metrics', {}).get('enable_metrics', True)

    if 'no_plots' in cli_args_provided:
        enable_plots = not args.no_plots
    else:
        enable_plots = config.get('metrics', {}).get('enable_plots', True)

    # Tracking parameters (JSON string from CLI, or dict from config)
    tracking_params = {}
    config_tracking = config.get('tracking', {})
    if config_tracking:
        tracking_params.update(config_tracking)
    if args.tracking_params:
        import json as json_mod
        tracking_params.update(json_mod.loads(args.tracking_params))
    if args.max_detections is not None:
        tracking_params['max_num_objects'] = args.max_detections

    # Get list of videos to process
    video_files = []
    if input_path.is_file():
        video_files = [input_path]
    elif input_path.is_dir():
        # Find all video files in directory
        for ext in ['*.mp4', '*.avi', '*.mov', '*.MP4', '*.AVI', '*.MOV']:
            video_files.extend(input_path.glob(ext))
        video_files = sorted(video_files)

        if not video_files:
            print(f"Error: No video files found in directory: {input_path}")
            sys.exit(1)

        print(f"\nFound {len(video_files)} video(s) to process:")
        for vf in video_files:
            print(f"  - {vf.name}")
        print()

    # Initialize experiment logger
    exp_logger = ExperimentLogger(experiments_dir="experiments")

    # Build config dict for logging
    run_config = {
        'input': str(input_path),
        'experiment_name': experiment_name,
        'text_prompt': text_prompt,
        'output_dir': output_dir,
        'processing': {
            'max_frames': max_frames,
            'skip_mesh_generation': skip_mesh_generation,
            'fast_2d_only': fast_2d_only,
            'skip_mesh_saving': skip_mesh_saving,
            'export_coco_csv': export_coco_csv,
            'bundle_adjustment': bundle_adjustment,
            'constrain_torso': constrain_torso,
            'temporal_smooth_keypoints': temporal_smooth_keypoints,
            'temporal_smooth_window': temporal_smooth_window,
            'temporal_smooth_polyorder': temporal_smooth_polyorder,
            'smoothing_sigma': smoothing_sigma,
        },
        'quality': {
            'enable_filter': enable_quality_filter,
            'z_velocity_threshold': quality_z_velocity,
            'total_velocity_threshold': quality_total_velocity,
            'vertex_displacement_threshold': quality_vertex_displacement,
        },
        'metrics': {
            'enable_metrics': enable_metrics,
            'enable_plots': enable_plots,
        },
        'tracking': tracking_params,
        'num_videos': len(video_files),
    }

    # Start experiment run
    run_id = exp_logger.start_run(run_config, experiment_name)

    # Patch SAM3
    patch_sam3()

    try:
        # Process each video
        for video_idx, video_path in enumerate(video_files, 1):
            if len(video_files) > 1:
                print("\n" + "="*70)
                print(f"PROCESSING VIDEO {video_idx}/{len(video_files)}: {video_path.name}")
                print("="*70)

            video_output_dir = os.path.join(output_dir, video_path.stem)
            os.makedirs(video_output_dir, exist_ok=True)


            # Initialize processor
            processor = VideoProcessor(
                str(video_path),
                text_prompt=text_prompt,
                max_frames=max_frames,
                output_dir=video_output_dir,
                enable_quality_filter=enable_quality_filter,
                smoothing_sigma=smoothing_sigma,
                skip_mesh_generation=skip_mesh_generation,
                skip_mesh_saving=skip_mesh_saving,
                export_coco_csv=export_coco_csv,
                bundle_adjustment=bundle_adjustment,
                constrain_torso=constrain_torso,
                temporal_smooth_keypoints=temporal_smooth_keypoints,
                temporal_smooth_window=temporal_smooth_window,
                temporal_smooth_polyorder=temporal_smooth_polyorder,
                quality_z_velocity_threshold=quality_z_velocity,
                quality_total_velocity_threshold=quality_total_velocity,
                quality_vertex_displacement_threshold=quality_vertex_displacement,
                fast_2d_only=fast_2d_only,
                enable_metrics=enable_metrics,
                enable_plots=enable_plots,
                tracking_params=tracking_params,
                chunk_size=args.chunk_size,
                start_frame=args.start_frame
            )

            # Step 1: Process video
            print("\n" + "="*60)
            if fast_2d_only:
                print("STEP 1: MASK GENERATION & FAST 2D KEYPOINT EXTRACTION")
            else:
                print("STEP 1: MASK GENERATION & 3D MESH ESTIMATION")
            print("="*60)

            processing_data = processor.process_video()
            if fast_2d_only:
                print(f"\n✓ Processed {len(processing_data['results'])} detections in fast 2D mode")
            else:
                print(f"\n✓ Generated {len(processing_data['results'])} 3D meshes")

            # Log results
            exp_logger.log_result(f'{video_path.stem}_num_frames', len(processing_data['results']))
            if processing_data.get('quality_log'):
                good_frames = sum(1 for e in processing_data['quality_log'] if e['quality']['is_good'])
                exp_logger.log_result(f'{video_path.stem}_good_frames', good_frames)

            # Step 2: Smooth and export
            print("\n" + "="*60)
            if fast_2d_only:
                print("STEP 2: FAST 2D EXPORT")
            else:
                print("STEP 2: TEMPORAL SMOOTHING & MESH EXPORT")
            print("="*60)

            exports = None if fast_2d_only else processor.smooth_and_export(processing_data)

            if fast_2d_only:
                smoothed_2d = processing_data.get('smoothed_2d_results')
                if smoothed_2d:
                    print("\n✓ VIDEO PROCESSING COMPLETE!")
                    print(f"Smoothed 2D CSV: {smoothed_2d['csv']}")
                    print(f"Smoothed 2D JSON: {smoothed_2d['json']}")
                    exp_logger.log_result(f'{video_path.stem}_status', 'completed')
                else:
                    exp_logger.log_result(f'{video_path.stem}_status', 'no_export')
            elif exports:
                print("\n✓ VIDEO PROCESSING COMPLETE!")
                print(f"Smoothed meshes: {len(exports['meshes'])} PLY files")
                print(f"Metadata: {exports['metadata']}")
                exp_logger.log_result(f'{video_path.stem}_status', 'completed')
            else:
                exp_logger.log_result(f'{video_path.stem}_status', 'no_export')

        # End experiment run
        exp_logger.end_run(status='completed')

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        exp_logger.log_error(str(e))
        exp_logger.end_run(status='failed')
        sys.exit(1)


if __name__ == "__main__":
    main()
