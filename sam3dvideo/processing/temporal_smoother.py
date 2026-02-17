#!/usr/bin/env python3
"""
Temporal smoothing for mesh vertices and keypoints.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from tqdm import tqdm
import trimesh
import json
import os


class TemporalSmoother:
    """Applies temporal smoothing to mesh sequences."""

    def __init__(self, smoothing_sigma=2.0):
        """
        Initialize temporal smoother.

        Args:
            smoothing_sigma: Smoothing strength (higher = more smoothing)
        """
        # Ensure sigma is a float (handle dict or other types)
        self.smoothing_sigma = float(smoothing_sigma) if not isinstance(smoothing_sigma, dict) else 2.0

    def smooth_mesh_vertices(self, frame_vertices, frame_indices):
        """
        Smooth mesh vertices over time using Gaussian filter.

        Args:
            frame_vertices: Dict mapping frame_idx -> vertices array
            frame_indices: Sorted list of frame indices

        Returns:
            np.ndarray: Smoothed vertices array (T, V, 3)
        """
        # Stack into array (T, V, 3)
        vertices_list = [frame_vertices[idx] for idx in frame_indices]
        vertices_array = np.array(vertices_list)
        print(f"Vertices shape: {vertices_array.shape}")

        # Apply Gaussian smoothing
        print(f"\nApplying temporal smoothing (sigma={self.smoothing_sigma})...")
        smoothed_vertices = np.zeros_like(vertices_array)
        num_vertices = vertices_array.shape[1]

        for v in tqdm(range(num_vertices), desc="Smoothing vertices"):
            for c in range(3):  # x, y, z
                smoothed_vertices[:, v, c] = gaussian_filter1d(
                    vertices_array[:, v, c],
                    sigma=self.smoothing_sigma
                )

        print("✓ Smoothing complete")
        return smoothed_vertices

    def interpolate_frames(self, smoothed_vertices, frame_indices):
        """
        Interpolate missing frames using cubic or linear interpolation.

        Args:
            smoothed_vertices: Smoothed vertices array (T, V, 3)
            frame_indices: Original frame indices

        Returns:
            tuple: (interpolated_vertices, full_range)
        """
        print("\nInterpolating missing frames...")
        min_frame = min(frame_indices)
        max_frame = max(frame_indices)
        full_range = np.arange(min_frame, max_frame + 1)

        # Choose interpolation method based on number of frames
        num_frames = len(frame_indices)
        if num_frames < 4:
            # Use linear interpolation for few frames (cubic needs 4+ points)
            interp_kind = 'linear'
            print(f"Using linear interpolation ({num_frames} frames)")
        else:
            interp_kind = 'cubic'
            print(f"Using cubic interpolation ({num_frames} frames)")

        num_vertices = smoothed_vertices.shape[1]
        interpolated_vertices = np.zeros((len(full_range), num_vertices, 3))

        for v in tqdm(range(num_vertices), desc="Interpolating"):
            for c in range(3):
                f = interp1d(
                    frame_indices,
                    smoothed_vertices[:, v, c],
                    kind=interp_kind,
                    fill_value='extrapolate'
                )
                interpolated_vertices[:, v, c] = f(full_range)

        print(f"✓ Interpolated to {len(full_range)} frames")
        return interpolated_vertices, full_range

    def save_smoothed_meshes(self, interpolated_vertices, full_range, faces,
                            output_dir, video_name):
        """
        Save smoothed meshes as PLY files.

        Args:
            interpolated_vertices: Interpolated vertices array
            full_range: Frame range
            faces: Mesh face topology
            output_dir: Output directory
            video_name: Video name for file naming

        Returns:
            list: Paths to saved PLY files
        """
        export_dir = os.path.join(output_dir, f"{video_name}_smoothed")
        smoothed_dir = os.path.join(export_dir, "meshes")
        os.makedirs(smoothed_dir, exist_ok=True)

        print("\nSaving smoothed meshes...")
        saved_meshes = []
        for i, frame_idx in enumerate(tqdm(full_range, desc="Saving PLY")):
            mesh = trimesh.Trimesh(vertices=interpolated_vertices[i], faces=faces)
            ply_path = os.path.join(smoothed_dir, f"smoothed_frame_{frame_idx:04d}.ply")
            mesh.export(ply_path)
            saved_meshes.append(ply_path)

        return saved_meshes, export_dir, smoothed_dir

    def save_metadata(self, export_dir, video_info, frame_indices, full_range,
                     num_vertices, num_faces, smoothed_dir):
        """
        Save processing metadata.

        Args:
            export_dir: Export directory
            video_info: Video information dict
            frame_indices: Original frame indices
            full_range: Full frame range
            num_vertices: Number of vertices
            num_faces: Number of faces
            smoothed_dir: Directory containing smoothed meshes

        Returns:
            str: Path to metadata file
        """
        metadata = {
            'video_info': video_info,
            'smoothing': {
                'sigma': self.smoothing_sigma,
                'original_frames': len(frame_indices),
                'interpolated_frames': len(full_range),
                'frame_range': [int(min(full_range)), int(max(full_range))]
            },
            'mesh_info': {
                'num_vertices': num_vertices,
                'num_faces': num_faces
            },
            'output_files': {
                'smoothed_meshes': smoothed_dir
            }
        }

        metadata_path = os.path.join(export_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Saved metadata: {metadata_path}")
        return metadata_path
