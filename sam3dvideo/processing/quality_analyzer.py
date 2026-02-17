#!/usr/bin/env python3
"""
Frame quality analysis for detecting bad frames in baby videos.
"""

import numpy as np


class QualityAnalyzer:
    """Analyzes frame quality to detect motion artifacts and transitions."""

    def __init__(self, z_velocity_threshold=0.15, total_velocity_threshold=0.25,
                 vertex_displacement_threshold=0.08):
        """
        Initialize quality analyzer.

        Args:
            z_velocity_threshold: Max Z-axis camera movement (default: 0.15)
            total_velocity_threshold: Max total camera movement (default: 0.25)
            vertex_displacement_threshold: Max vertex deformation (default: 0.08)
        """
        # Ensure thresholds are floats (handle dict or other types)
        self.z_velocity_threshold = float(z_velocity_threshold) if not isinstance(z_velocity_threshold, dict) else 0.15
        self.total_velocity_threshold = float(total_velocity_threshold) if not isinstance(total_velocity_threshold, dict) else 0.25
        self.vertex_displacement_threshold = float(vertex_displacement_threshold) if not isinstance(vertex_displacement_threshold, dict) else 0.08

    def analyze_frame_quality(self, mhr_params, prev_mhr_params=None, sam_bbox=None, mask=None):
        """
        Detect bad frames for baby videos specifically.

        Args:
            mhr_params: Current frame MHR parameters
            prev_mhr_params: Previous frame MHR parameters
            sam_bbox: SAM bounding box
            mask: SAM mask

        Returns:
            dict: Quality information with 'is_good', 'flags', 'metrics', 'reason'
        """
        quality_info = {
            'is_good': True,
            'flags': [],
            'metrics': {},
            'reason': None
        }

        # Only check temporal changes, not static orientation
        if prev_mhr_params is None:
            return quality_info  # First frame is always good

        # Check 1: Rapid camera movement (being picked up)
        if 'pred_cam_t' in mhr_params and 'pred_cam_t' in prev_mhr_params:
            cam_t_curr = mhr_params['pred_cam_t']
            cam_t_prev = prev_mhr_params['pred_cam_t']

            # Check Z-axis movement specifically (toward/away from camera)
            z_velocity = float(abs(cam_t_curr[2] - cam_t_prev[2]))
            total_velocity = float(np.linalg.norm(cam_t_curr - cam_t_prev))

            quality_info['metrics']['z_velocity'] = z_velocity
            quality_info['metrics']['total_velocity'] = total_velocity

            # Check against configurable thresholds
            if z_velocity > self.z_velocity_threshold or total_velocity > self.total_velocity_threshold:
                quality_info['is_good'] = False
                quality_info['reason'] = f"Picked up (z_vel={z_velocity:.3f})"

        # Check 2: Large body deformation (mid-roll)
        if 'pred_vertices' in mhr_params and 'pred_vertices' in prev_mhr_params:
            vert_curr = mhr_params['pred_vertices']
            vert_prev = prev_mhr_params['pred_vertices']

            # Mean vertex displacement
            displacement = np.linalg.norm(vert_curr - vert_prev, axis=1)
            mean_disp = float(np.mean(displacement))

            quality_info['metrics']['vertex_displacement'] = mean_disp

            # Check against configurable threshold
            if mean_disp > self.vertex_displacement_threshold:
                quality_info['is_good'] = False
                quality_info['reason'] = f"Transitioning (disp={mean_disp:.3f})"

        return quality_info

    def identify_segments(self, quality_log, min_segment_length=10):
        """
        Identify continuous segments of good frames.

        Args:
            quality_log: List of quality info dicts
            min_segment_length: Minimum number of consecutive good frames to keep

        Returns:
            List of segment dicts with start/end frames
        """
        segments = []
        current_segment = None

        for entry in quality_log:
            frame_idx = entry['frame_idx']
            is_good = entry['quality']['is_good']

            if is_good:
                if current_segment is None:
                    # Start new segment
                    current_segment = {
                        'start_frame': frame_idx,
                        'end_frame': frame_idx,
                        'num_frames': 1
                    }
                else:
                    # Continue current segment
                    current_segment['end_frame'] = frame_idx
                    current_segment['num_frames'] += 1
            else:
                # Bad frame - end current segment if exists
                if current_segment is not None:
                    if current_segment['num_frames'] >= min_segment_length:
                        segments.append(current_segment)
                    current_segment = None

        # Add final segment if exists
        if current_segment is not None and current_segment['num_frames'] >= min_segment_length:
            segments.append(current_segment)

        return segments
