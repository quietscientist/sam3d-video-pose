#!/usr/bin/env python3
"""
Visualize 3D COCO keypoints time series with skeleton bones.
Excludes face keypoints (nose, eyes, ears).
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import pandas as pd


# COCO keypoint names and connections
COCO_KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',  # 0-4: FACE (exclude)
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',  # 5-8: UPPER BODY
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',  # 9-12: WRISTS & HIPS
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'  # 13-16: LEGS
]

# Body-only skeleton (excludes face, excludes flexible torso)
BODY_SKELETON = [
    ('left_hip', 'left_knee'),          # left thigh
    ('right_hip', 'right_knee'),        # right thigh
    ('left_knee', 'left_ankle'),        # left shin
    ('right_knee', 'right_ankle'),      # right shin
    ('left_shoulder', 'left_elbow'),    # left upper arm
    ('right_shoulder', 'right_elbow'),  # right upper arm
    ('left_elbow', 'left_wrist'),       # left forearm
    ('right_elbow', 'right_wrist'),     # right forearm
    ('left_shoulder', 'right_shoulder'), # shoulder width
    ('left_hip', 'right_hip'),          # hip width
]

# Map keypoint names to indices
KP_NAME_TO_IDX = {name: idx for idx, name in enumerate(COCO_KEYPOINT_NAMES)}

# Body keypoint indices (exclude face: 0-4)
BODY_KEYPOINT_INDICES = list(range(5, 17))  # indices 5-16

# Bone connections as index pairs
BONE_CONNECTIONS = [
    (KP_NAME_TO_IDX[start], KP_NAME_TO_IDX[end])
    for start, end in BODY_SKELETON
]


def load_keypoints_from_json(json_path):
    """
    Load 3D keypoints from all_keypoints.json file.

    Returns:
        List of dicts with 'frame_idx' and 'keypoints_3d' (17, 3) numpy arrays
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    frames = []
    for frame_data in data['frames']:
        if 'keypoints_3d' in frame_data and frame_data['keypoints_3d'] is not None:
            frames.append({
                'frame_idx': frame_data['frame_idx'],
                'keypoints_3d': np.array(frame_data['keypoints_3d'])
            })

    print(f"✓ Loaded {len(frames)} frames from {json_path}")
    return frames


def load_keypoints_from_csv(csv_path):
    """
    Load 3D keypoints from COCO CSV file.

    Returns:
        List of dicts with 'frame_idx' and 'keypoints_3d' (17, 3) numpy arrays
    """
    df = pd.read_csv(csv_path)

    frames = []
    for _, row in df.iterrows():
        keypoints_3d = np.zeros((17, 3))
        for i in range(17):
            keypoints_3d[i, 0] = row[f'{COCO_KEYPOINT_NAMES[i]}_x']
            keypoints_3d[i, 1] = row[f'{COCO_KEYPOINT_NAMES[i]}_y']
            keypoints_3d[i, 2] = row[f'{COCO_KEYPOINT_NAMES[i]}_z']

        frames.append({
            'frame_idx': int(row['frame']),
            'keypoints_3d': keypoints_3d
        })

    print(f"✓ Loaded {len(frames)} frames from {csv_path}")
    return frames


def plot_3d_skeleton(ax, keypoints_3d, show_face=False):
    """
    Plot 3D skeleton with bones.

    Args:
        ax: Matplotlib 3D axis
        keypoints_3d: (17, 3) numpy array of 3D keypoints
        show_face: If True, show face keypoints (default: False)
    """
    # Clear previous plot
    ax.clear()

    # Determine which keypoints to show
    if show_face:
        kp_indices = list(range(17))
    else:
        kp_indices = BODY_KEYPOINT_INDICES  # 5-16 (exclude face)

    # Plot keypoints
    for idx in kp_indices:
        x, y, z = keypoints_3d[idx]
        ax.scatter(x, y, z, c='red', marker='o', s=50)
        # Optionally label keypoints
        # ax.text(x, y, z, COCO_KEYPOINT_NAMES[idx], fontsize=8)

    # Plot bones (connections)
    for start_idx, end_idx in BONE_CONNECTIONS:
        # Skip if either endpoint is a face keypoint and we're not showing face
        if not show_face and (start_idx < 5 or end_idx < 5):
            continue

        start_pos = keypoints_3d[start_idx]
        end_pos = keypoints_3d[end_idx]

        xs = [start_pos[0], end_pos[0]]
        ys = [start_pos[1], end_pos[1]]
        zs = [start_pos[2], end_pos[2]]

        ax.plot(xs, ys, zs, 'b-', linewidth=2)

    # Set labels and aspect
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1])

    # Set consistent axis limits
    all_points = keypoints_3d[kp_indices]
    center = np.mean(all_points, axis=0)
    max_range = np.max(np.ptp(all_points, axis=0)) * 0.6

    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)


def visualize_static(frames_data, output_path=None, show_face=False, num_frames=4):
    """
    Create a static multi-panel visualization showing multiple frames.

    Args:
        frames_data: List of frame dicts with 'keypoints_3d'
        output_path: Path to save figure (optional)
        show_face: If True, show face keypoints
        num_frames: Number of frames to display
    """
    # Select evenly spaced frames
    indices = np.linspace(0, len(frames_data) - 1, num_frames, dtype=int)

    fig = plt.figure(figsize=(16, 4))

    for i, idx in enumerate(indices):
        frame = frames_data[idx]
        ax = fig.add_subplot(1, num_frames, i + 1, projection='3d')
        plot_3d_skeleton(ax, frame['keypoints_3d'], show_face=show_face)
        ax.set_title(f"Frame {frame['frame_idx']}")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to {output_path}")

    plt.show()


def visualize_animation(frames_data, output_path=None, show_face=False, fps=10):
    """
    Create an animated visualization of the skeleton over time.

    Args:
        frames_data: List of frame dicts with 'keypoints_3d'
        output_path: Path to save animation (optional, .gif or .mp4)
        show_face: If True, show face keypoints
        fps: Frames per second for animation
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame_idx):
        frame = frames_data[frame_idx]
        plot_3d_skeleton(ax, frame['keypoints_3d'], show_face=show_face)
        ax.set_title(f"Frame {frame['frame_idx']}", fontsize=14)
        return ax,

    anim = FuncAnimation(
        fig, update, frames=len(frames_data),
        interval=1000/fps, blit=False
    )

    if output_path:
        output_path = Path(output_path)
        if output_path.suffix == '.gif':
            anim.save(output_path, writer='pillow', fps=fps)
            print(f"✓ Saved animation to {output_path}")
        elif output_path.suffix == '.mp4':
            anim.save(output_path, writer='ffmpeg', fps=fps)
            print(f"✓ Saved animation to {output_path}")
        else:
            print(f"⚠ Unsupported format {output_path.suffix}, use .gif or .mp4")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize 3D COCO keypoints with skeleton bones (no face)"
    )
    parser.add_argument(
        'input_path',
        help='Path to keypoints file (all_keypoints.json or coco_keypoints_adjusted.csv)'
    )
    parser.add_argument(
        '--mode', choices=['static', 'animation'], default='static',
        help='Visualization mode (default: static)'
    )
    parser.add_argument(
        '--output', '-o', type=str,
        help='Output path for saving (PNG for static, GIF/MP4 for animation)'
    )
    parser.add_argument(
        '--show-face', action='store_true',
        help='Include face keypoints (nose, eyes, ears)'
    )
    parser.add_argument(
        '--num-frames', type=int, default=4,
        help='Number of frames to display in static mode (default: 4)'
    )
    parser.add_argument(
        '--fps', type=int, default=10,
        help='Frames per second for animation (default: 10)'
    )

    args = parser.parse_args()

    # Load keypoints
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return

    if input_path.suffix == '.json':
        frames_data = load_keypoints_from_json(input_path)
    elif input_path.suffix == '.csv':
        frames_data = load_keypoints_from_csv(input_path)
    else:
        print(f"Error: Unsupported file format: {input_path.suffix}")
        print("Supported formats: .json, .csv")
        return

    if not frames_data:
        print("Error: No frames with valid keypoints found")
        return

    # Visualize
    if args.mode == 'static':
        visualize_static(
            frames_data,
            output_path=args.output,
            show_face=args.show_face,
            num_frames=args.num_frames
        )
    elif args.mode == 'animation':
        visualize_animation(
            frames_data,
            output_path=args.output,
            show_face=args.show_face,
            fps=args.fps
        )


if __name__ == '__main__':
    main()
