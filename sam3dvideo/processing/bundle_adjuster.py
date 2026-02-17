#!/usr/bin/env python3
"""
Bundle adjustment for 3D keypoints with COCO subset extraction.
Maps MHR 70-point skeleton to COCO 17-point skeleton and enforces fixed bone lengths.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.signal import savgol_filter
import csv
from pathlib import Path


class BundleAdjuster:
    """Handles COCO keypoint extraction and bundle adjustment."""

    # COCO to MHR index mapping
    COCO_TO_MHR_MAP = {
        0: 0,    # nose
        1: 1,    # left_eye
        2: 2,    # right_eye
        3: 3,    # left_ear
        4: 4,    # right_ear
        5: 5,    # left_shoulder
        6: 6,    # right_shoulder
        7: 7,    # left_elbow
        8: 8,    # right_elbow
        9: 62,   # left_wrist
        10: 41,  # right_wrist
        11: 9,   # left_hip
        12: 10,  # right_hip
        13: 11,  # left_knee
        14: 12,  # right_knee
        15: 13,  # left_ankle
        16: 14,  # right_ankle
    }

    COCO_KEYPOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]

    # Skeleton connectivity for COCO keypoints
    # Full skeleton including flexible torso
    COCO_SKELETON_FULL = {
        'torso': (11, 5),              # left_hip to left_shoulder (FLEXIBLE!)
        'right_torso': (12, 6),        # right_hip to right_shoulder (FLEXIBLE!)
        'left_thigh': (11, 13),        # left_hip to left_knee
        'right_thigh': (12, 14),       # right_hip to right_knee
        'left_shin': (13, 15),         # left_knee to left_ankle
        'right_shin': (14, 16),        # right_knee to right_ankle
        'left_upper_arm': (5, 7),      # left_shoulder to left_elbow
        'right_upper_arm': (6, 8),     # right_shoulder to right_elbow
        'left_forearm': (7, 9),        # left_elbow to left_wrist
        'right_forearm': (8, 10),      # right_elbow to right_wrist
        'shoulder_width': (5, 6),      # left to right shoulder
        'hip_width': (11, 12),         # left to right hip
    }

    # Rigid limbs only (recommended for infants - excludes flexible torso)
    COCO_SKELETON_LIMBS_ONLY = {
        'left_thigh': (11, 13),        # left_hip to left_knee
        'right_thigh': (12, 14),       # right_hip to right_knee
        'left_shin': (13, 15),         # left_knee to left_ankle
        'right_shin': (14, 16),        # right_knee to right_ankle
        'left_upper_arm': (5, 7),      # left_shoulder to left_elbow
        'right_upper_arm': (6, 8),     # right_shoulder to right_elbow
        'left_forearm': (7, 9),        # left_elbow to left_wrist
        'right_forearm': (8, 10),      # right_elbow to right_wrist
        'shoulder_width': (5, 6),      # left to right shoulder (relatively stable)
        'hip_width': (11, 12),         # left to right hip (relatively stable)
    }

    # Default: use limbs-only for infant motion (flexible spine)
    COCO_SKELETON = COCO_SKELETON_LIMBS_ONLY

    def __init__(self, constrain_torso=False, temporal_smooth_window=11, temporal_smooth_polyorder=3):
        """
        Initialize bundle adjuster.

        Args:
            constrain_torso: If True, include torso in bone length constraints.
                           If False (default), only constrain limbs to allow spinal flexibility.
            temporal_smooth_window: Window length for Savitzky-Golay filter (must be odd, default: 11)
            temporal_smooth_polyorder: Polynomial order for SG filter (default: 3)
        """
        if constrain_torso:
            self.skeleton = self.COCO_SKELETON_FULL
        else:
            self.skeleton = self.COCO_SKELETON_LIMBS_ONLY

        self.temporal_smooth_window = temporal_smooth_window
        self.temporal_smooth_polyorder = temporal_smooth_polyorder

    def extract_coco_keypoints(self, keypoints_data):
        """
        Extract COCO 17-point keypoints from all_keypoints.json data.

        Args:
            keypoints_data: Dict with 'frames' containing keypoint data

        Returns:
            List of dicts with 'frame_idx' and 'keypoints_3d_coco' (17, 3)
        """
        frames_with_keypoints = []

        for frame_data in keypoints_data['frames']:
            frame_idx = frame_data['frame_idx']
            keypoints_3d = frame_data.get('keypoints_3d')

            if keypoints_3d is None:
                print(f"Warning: Frame {frame_idx} has no 3D keypoints, skipping")
                continue

            # Map MHR to COCO format
            keypoints_coco = np.zeros((17, 3))
            valid = True

            for coco_idx in range(17):
                mhr_idx = self.COCO_TO_MHR_MAP[coco_idx]

                # Check if MHR index exists
                if mhr_idx >= len(keypoints_3d):
                    print(f"Warning: Frame {frame_idx}, COCO {coco_idx} ({self.COCO_KEYPOINT_NAMES[coco_idx]}) "
                          f"maps to MHR {mhr_idx} but only {len(keypoints_3d)} keypoints available")
                    valid = False
                    break

                keypoints_coco[coco_idx] = keypoints_3d[mhr_idx]

            if valid:
                frames_with_keypoints.append({
                    'frame_idx': frame_idx,
                    'keypoints_3d_coco': keypoints_coco
                })

        print(f"✓ Extracted COCO keypoints from {len(frames_with_keypoints)} valid frames")
        return frames_with_keypoints

    def apply_temporal_smoothing(self, frames_data, verbose=True):
        """
        Apply Savitzky-Golay temporal smoothing to keypoint positions.

        Args:
            frames_data: List of dicts with 'frame_idx' and 'keypoints_3d_coco' (17, 3)
            verbose: If True, print smoothing details

        Returns:
            List of dicts with temporally smoothed keypoints
        """
        if len(frames_data) < self.temporal_smooth_window:
            if verbose:
                print(f"⚠ Not enough frames ({len(frames_data)}) for temporal smoothing (need {self.temporal_smooth_window})")
                print("  Skipping temporal smoothing")
            return frames_data

        if verbose:
            print("\n" + "="*60)
            print("TEMPORAL SMOOTHING: Savitzky-Golay Filter")
            print("="*60)
            print(f"Window length: {self.temporal_smooth_window} frames")
            print(f"Polynomial order: {self.temporal_smooth_polyorder}")

        # Extract frame indices and keypoints
        frame_indices = [f['frame_idx'] for f in frames_data]

        # Build keypoint array (num_frames, 17, 3)
        keypoints_array = np.array([f['keypoints_3d_coco'] for f in frames_data])
        num_frames, num_keypoints, _ = keypoints_array.shape

        if verbose:
            print(f"\nSmoothing {num_frames} frames, {num_keypoints} keypoints each...")

        # Apply SG filter to each keypoint's x, y, z coordinates
        smoothed_keypoints = np.zeros_like(keypoints_array)

        for kp_idx in range(num_keypoints):
            for coord_idx in range(3):  # x, y, z
                # Extract time series for this keypoint's coordinate
                time_series = keypoints_array[:, kp_idx, coord_idx]

                # Apply Savitzky-Golay filter
                smoothed = savgol_filter(
                    time_series,
                    window_length=self.temporal_smooth_window,
                    polyorder=self.temporal_smooth_polyorder,
                    mode='nearest'  # Handle boundaries
                )

                smoothed_keypoints[:, kp_idx, coord_idx] = smoothed

        # Compute smoothing impact
        if verbose:
            displacement = np.linalg.norm(keypoints_array - smoothed_keypoints, axis=2)  # (num_frames, 17)
            mean_displacement_per_kp = np.mean(displacement, axis=0)
            overall_mean = np.mean(displacement)

            print(f"\n✓ Temporal smoothing complete")
            print(f"  Mean displacement per keypoint: {overall_mean:.4f} mm")
            print(f"  Max displacement: {np.max(displacement):.4f} mm")

            print("\nPer-keypoint smoothing displacement:")
            print(f"{'Keypoint':<20s} {'Mean Displacement (mm)':<25s}")
            print("-" * 50)
            for kp_idx, kp_name in enumerate(self.COCO_KEYPOINT_NAMES):
                print(f"{kp_name:<20s} {mean_displacement_per_kp[kp_idx]:<25.4f}")

            print("="*60 + "\n")

        # Package back into frames_data format
        smoothed_frames = []
        for i, frame_data in enumerate(frames_data):
            smoothed_frames.append({
                'frame_idx': frame_data['frame_idx'],
                'keypoints_3d_coco': smoothed_keypoints[i]
            })

        return smoothed_frames

    def estimate_canonical_bone_lengths(self, all_keypoints_list, percentile=50):
        """
        Estimate canonical bone lengths from all frames.

        Args:
            all_keypoints_list: List of (17, 3) numpy arrays
            percentile: Percentile to use (50 = median, robust to outliers)

        Returns:
            dict of bone_name -> canonical_length
        """
        bone_measurements = {bone: [] for bone in self.skeleton}

        for keypoints in all_keypoints_list:
            for bone_name, (idx1, idx2) in self.skeleton.items():
                length = np.linalg.norm(keypoints[idx1] - keypoints[idx2])
                bone_measurements[bone_name].append(length)

        canonical_lengths = {}
        for bone_name, lengths in bone_measurements.items():
            canonical_lengths[bone_name] = np.percentile(lengths, percentile)

        return canonical_lengths

    def bundle_adjust_frame(self, keypoints_3d, canonical_lengths, max_iter=500):
        """
        Adjust a single frame's 3D keypoints to satisfy bone length constraints.

        Args:
            keypoints_3d: (17, 3) array of 3D keypoint positions
            canonical_lengths: dict of bone_name -> desired_length
            max_iter: maximum optimization iterations

        Returns:
            optimized_keypoints: (17, 3) array
            residual: final optimization residual
        """
        # Flatten keypoints for optimization
        x0 = keypoints_3d.flatten()

        def objective(x):
            """Minimize deviation from original positions."""
            kp = x.reshape(17, 3)
            return np.sum((kp - keypoints_3d) ** 2)

        def bone_length_constraints(x):
            """Equality constraints: bone lengths must match canonical."""
            kp = x.reshape(17, 3)
            residuals = []

            for bone_name, (idx1, idx2) in self.skeleton.items():
                current_length = np.linalg.norm(kp[idx1] - kp[idx2])
                target_length = canonical_lengths[bone_name]
                # Normalized residual (unitless)
                residuals.append((current_length - target_length) / target_length)

            return np.array(residuals)

        # Set up constraints
        constraints = {
            'type': 'eq',
            'fun': bone_length_constraints
        }

        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            constraints=constraints,
            options={'maxiter': max_iter, 'ftol': 1e-9}
        )

        optimized_keypoints = result.x.reshape(17, 3)

        # Compute final residual
        residual = np.linalg.norm(bone_length_constraints(result.x))

        return optimized_keypoints, residual

    def apply_bundle_adjustment(self, frames_data, verbose=True):
        """
        Apply bundle adjustment to all frames to enforce consistent bone lengths.

        Args:
            frames_data: List of dicts with 'frame_idx' and 'keypoints_3d_coco' (17, 3)
            verbose: If True, print detailed statistics

        Returns:
            List of dicts with adjusted keypoints
        """
        if len(frames_data) == 0:
            return frames_data

        if verbose:
            print("\n" + "="*60)
            print("BUNDLE ADJUSTMENT: Enforcing Fixed Bone Lengths")
            print("="*60)

        # Extract all keypoints
        all_keypoints = [f['keypoints_3d_coco'] for f in frames_data]

        # Estimate canonical bone lengths
        if verbose:
            print("\nEstimating canonical bone lengths (median across all frames)...")
        canonical_lengths = self.estimate_canonical_bone_lengths(all_keypoints)

        if verbose:
            print("\nCanonical bone lengths:")
            for bone_name, length in sorted(canonical_lengths.items()):
                print(f"  {bone_name:<20s}: {length:>8.4f} mm")

        # Compute bone length variation BEFORE adjustment
        if verbose:
            print("\nBone length variation BEFORE adjustment:")
            bone_lengths_before = {bone: [] for bone in self.skeleton}
            for keypoints in all_keypoints:
                for bone_name, (idx1, idx2) in self.skeleton.items():
                    length = np.linalg.norm(keypoints[idx1] - keypoints[idx2])
                    bone_lengths_before[bone_name].append(length)

            print(f"{'Bone':<20s} {'Mean':<10s} {'Std':<10s} {'CV %':<10s}")
            print("-" * 50)
            for bone_name in sorted(self.skeleton.keys()):
                lengths = bone_lengths_before[bone_name]
                mean_val = np.mean(lengths)
                std_val = np.std(lengths)
                cv = (std_val / mean_val * 100) if mean_val > 0 else 0
                print(f"{bone_name:<20s} {mean_val:<10.4f} {std_val:<10.4f} {cv:<10.2f}")

        # Apply bundle adjustment to each frame
        if verbose:
            print(f"\nAdjusting {len(frames_data)} frames...")
        adjusted_frames = []
        residuals = []

        for i, frame_data in enumerate(frames_data):
            keypoints = frame_data['keypoints_3d_coco']
            adjusted_kp, residual = self.bundle_adjust_frame(keypoints, canonical_lengths)

            adjusted_frames.append({
                'frame_idx': frame_data['frame_idx'],
                'keypoints_3d_coco': adjusted_kp
            })
            residuals.append(residual)

            if verbose and (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(frames_data)} frames...")

        if verbose:
            print(f"✓ Adjusted all {len(frames_data)} frames")
            print(f"  Mean constraint residual: {np.mean(residuals):.6f}")
            print(f"  Max constraint residual: {np.max(residuals):.6f}")

        # Compute bone length variation AFTER adjustment
        if verbose:
            print("\nBone length variation AFTER adjustment:")
            bone_lengths_after = {bone: [] for bone in self.skeleton}
            for frame_data in adjusted_frames:
                keypoints = frame_data['keypoints_3d_coco']
                for bone_name, (idx1, idx2) in self.skeleton.items():
                    length = np.linalg.norm(keypoints[idx1] - keypoints[idx2])
                    bone_lengths_after[bone_name].append(length)

            print(f"{'Bone':<20s} {'Mean':<10s} {'Std':<10s} {'CV %':<10s}")
            print("-" * 50)
            for bone_name in sorted(self.skeleton.keys()):
                lengths = bone_lengths_after[bone_name]
                mean_val = np.mean(lengths)
                std_val = np.std(lengths)
                cv = (std_val / mean_val * 100) if mean_val > 0 else 0
                print(f"{bone_name:<20s} {mean_val:<10.4f} {std_val:<10.6f} {cv:<10.4f}")

            print("="*60 + "\n")

        return adjusted_frames

    def export_to_csv(self, frames_data, output_path):
        """
        Export keypoints to CSV format.

        Args:
            frames_data: List of dicts with 'frame_idx' and 'keypoints_3d_coco'
            output_path: Path to save CSV file

        Returns:
            Path: Output path
        """
        output_path = Path(output_path)

        # Write to CSV
        print(f"\nWriting to {output_path.name}...")
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            writer.writerow(['frame', 'x', 'y', 'z', 'part_idx'])

            # Write all frames
            for frame_data in frames_data:
                frame_idx = frame_data['frame_idx']
                keypoints = frame_data['keypoints_3d_coco']

                for part_idx in range(17):
                    x, y, z = keypoints[part_idx]
                    writer.writerow([frame_idx, x, y, z, part_idx])

        print(f"✓ Saved 3D CSV to: {output_path}")

        # Print statistics
        num_frames = len(frames_data)
        num_rows = num_frames * 17
        print(f"\nStatistics:")
        print(f"  Frames processed: {num_frames}")
        print(f"  Total rows: {num_rows}")
        print(f"  Keypoints per frame: 17 (COCO)")

        return output_path

    def process_keypoints(self, keypoints_data, output_dir, video_name,
                         apply_temporal_smoothing=True, apply_bundle_adjustment=True,
                         export_both=True, metrics_logger=None):
        """
        Complete processing: extract COCO keypoints, temporal smoothing, bundle adjustment, export CSV.

        Args:
            keypoints_data: Dict with 'frames' containing keypoint data
            output_dir: Output directory
            video_name: Video name for file naming
            apply_temporal_smoothing: If True, apply Savitzky-Golay temporal smoothing
            apply_bundle_adjustment: If True, enforce fixed bone lengths
            export_both: If True, export both adjusted and unadjusted CSVs
            metrics_logger: Optional MetricsLogger instance for tracking statistics

        Returns:
            dict: Paths to generated CSV files
        """
        output_dir = Path(output_dir)

        # Extract COCO keypoints
        print(f"\nExtracting COCO keypoints from {video_name}...")
        frames_coco = self.extract_coco_keypoints(keypoints_data)

        if len(frames_coco) == 0:
            print("⚠ No valid frames to process")
            return None

        results = {}

        # Export raw (no smoothing, no adjustment) if requested
        if export_both and apply_temporal_smoothing:
            raw_path = output_dir / f"{video_name}_3D_raw.csv"
            self.export_to_csv(frames_coco, raw_path)
            results['raw'] = raw_path

        # Apply temporal smoothing if requested
        if apply_temporal_smoothing:
            frames_smoothed = self.apply_temporal_smoothing(frames_coco, verbose=True)

            # Log smoothed keypoint statistics
            if metrics_logger:
                smoothed_keypoints_list = [f['keypoints_3d_coco'] for f in frames_smoothed]
                metrics_logger.log_keypoint_statistics(
                    stage='smoothed',
                    keypoints_3d_list=smoothed_keypoints_list,
                    keypoint_names=self.COCO_KEYPOINT_NAMES
                )
                metrics_logger.log_sample_size(
                    'after_temporal_smoothing',
                    len(frames_smoothed),
                    "After Savitzky-Golay temporal smoothing"
                )

            # Export smoothed (no bundle adjustment) if requested
            if export_both or not apply_bundle_adjustment:
                smoothed_path = output_dir / f"{video_name}_3D_smoothed.csv"
                self.export_to_csv(frames_smoothed, smoothed_path)
                results['smoothed'] = smoothed_path
        else:
            frames_smoothed = frames_coco

            # Export unadjusted if no smoothing
            if export_both or not apply_bundle_adjustment:
                unadjusted_path = output_dir / f"{video_name}_3D.csv"
                self.export_to_csv(frames_coco, unadjusted_path)
                results['unadjusted'] = unadjusted_path

        # Apply bundle adjustment if requested
        if apply_bundle_adjustment:
            adjusted_frames = self.apply_bundle_adjustment(frames_smoothed, verbose=True)

            # Log bundle adjusted keypoint statistics
            if metrics_logger:
                adjusted_keypoints_list = [f['keypoints_3d_coco'] for f in adjusted_frames]
                metrics_logger.log_keypoint_statistics(
                    stage='bundle_adjusted',
                    keypoints_3d_list=adjusted_keypoints_list,
                    keypoint_names=self.COCO_KEYPOINT_NAMES
                )
                metrics_logger.log_sample_size(
                    'after_bundle_adjustment',
                    len(adjusted_frames),
                    "After bundle adjustment (fixed bone lengths)"
                )

            # Naming: smoothed+adjusted vs just adjusted
            if apply_temporal_smoothing:
                adjusted_path = output_dir / f"{video_name}_3D_smoothed_adjusted.csv"
            else:
                adjusted_path = output_dir / f"{video_name}_3D_adjusted.csv"

            self.export_to_csv(adjusted_frames, adjusted_path)
            results['final'] = adjusted_path
        else:
            print("\n⚠ Skipping bundle adjustment (bone lengths NOT constrained)")

        return results
