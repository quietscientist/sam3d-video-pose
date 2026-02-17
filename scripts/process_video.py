#!/usr/bin/env python3
"""
Process video to extract 3D body meshes and keypoints with temporal smoothing.

Usage:
    python process_video.py VIDEO_PATH [--max-frames 300] [--output-dir "output"]
"""

import argparse
import os
import sys
import json
from pathlib import Path
import torch
import gc
from tqdm import tqdm
import numpy as np
import cv2
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

    def __init__(self, video_path, text_prompt="a baby", max_frames=300,
                 output_dir="output", enable_quality_filter=False, smoothing_sigma=2.0,
                 skip_mesh_generation=False, skip_mesh_saving=False,
                 export_coco_csv=False, bundle_adjustment=True, constrain_torso=False,
                 temporal_smooth_keypoints=True, temporal_smooth_window=11,
                 temporal_smooth_polyorder=3, quality_z_velocity_threshold=0.15,
                 quality_total_velocity_threshold=0.25, quality_vertex_displacement_threshold=0.08,
                 enable_metrics=True, enable_plots=True):
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
            enable_metrics: If True, track and log comprehensive metrics (default: True)
            enable_plots: If True, generate visualization plots (default: True)
        """
        self.video_path = video_path
        self.text_prompt = text_prompt
        self.max_frames = max_frames
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
        self.enable_metrics = enable_metrics
        self.enable_plots = enable_plots
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
        if self.skip_mesh_generation:
            print("VIDEO TO 2D KEYPOINT EXTRACTION (MESH GENERATION SKIPPED)")
        else:
            print("VIDEO TO 3D MESH PROCESSING")
        print("="*60)
        print(f"Video: {self.video_path}")
        print(f"Prompt: '{self.text_prompt}'")
        print(f"Max frames: {self.max_frames}")
        print(f"Output: {self.output_dir}")
        print(f"Quality filter: {'ENABLED' if self.enable_quality_filter else 'disabled'}")
        if self.skip_mesh_generation:
            print(f"Mode: 2D keypoints only (no 3D meshes)")
        elif self.skip_mesh_saving:
            print(f"Mode: Generate meshes but don't save PLY files")
        else:
            print(f"Mode: Full 3D mesh + keypoint extraction")
        print("="*60 + "\n")

        # Initialize models
        self.segmenter = SAM3Segmenter()

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
        chunk_size = 50
        num_chunks = (self.max_frames + chunk_size - 1) // chunk_size

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            chunk_frames = min(chunk_size, self.max_frames - start_idx)

            print(f"\n{'='*60}")
            print(f"CHUNK {chunk_idx + 1}/{num_chunks}: Frames {start_idx}-{start_idx + chunk_frames - 1}")
            print(f"{'='*60}")

            # Segment with SAM3
            for frame_idx, outputs, frame in self.segmenter.segment_video_chunks(
                self.video_path, self.text_prompt, chunk_frames, start_idx
            ):
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Save frame temporarily
                temp_frame_path = os.path.join(masks_dir, f"temp_frame_{frame_idx:04d}.jpg")
                cv2.imwrite(temp_frame_path, frame_bgr)

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
            'csv_results': csv_results
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
                for k in key.split('.'):
                    val = val.get(k, {}) if isinstance(val, dict) else default
                if val != default and val is not None:
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
                enable_metrics=enable_metrics,
                enable_plots=enable_plots
            )

            # Step 1: Process video
            print("\n" + "="*60)
            print("STEP 1: MASK GENERATION & 3D MESH ESTIMATION")
            print("="*60)

            processing_data = processor.process_video()
            print(f"\n✓ Generated {len(processing_data['results'])} 3D meshes")

            # Log results
            exp_logger.log_result(f'{video_path.stem}_num_frames', len(processing_data['results']))
            if processing_data.get('quality_log'):
                good_frames = sum(1 for e in processing_data['quality_log'] if e['quality']['is_good'])
                exp_logger.log_result(f'{video_path.stem}_good_frames', good_frames)

            # Step 2: Smooth and export
            print("\n" + "="*60)
            print("STEP 2: TEMPORAL SMOOTHING & MESH EXPORT")
            print("="*60)

            exports = processor.smooth_and_export(processing_data)

            if exports:
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
