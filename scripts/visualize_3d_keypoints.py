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
from scipy.spatial.transform import Rotation


# COCO keypoint names and connections
COCO_KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',  # 0-4: FACE (exclude)
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',  # 5-8: UPPER BODY
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',  # 9-12: WRISTS & HIPS
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'  # 13-16: LEGS
]

# Body-only skeleton (excludes face, includes sternum-to-hips connection)
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
    ('sternum', 'pelvis'),              # torso (sternum to pelvis center)
]

# Map keypoint names to indices
KP_NAME_TO_IDX = {name: idx for idx, name in enumerate(COCO_KEYPOINT_NAMES)}

# Body keypoint indices (exclude face: 0-4)
BODY_KEYPOINT_INDICES = list(range(5, 17))  # indices 5-16

# Bone connections as index pairs (excluding sternum/pelvis which are computed)
BONE_CONNECTIONS = [
    (KP_NAME_TO_IDX[start], KP_NAME_TO_IDX[end])
    for start, end in BODY_SKELETON
    if start in KP_NAME_TO_IDX and end in KP_NAME_TO_IDX
]


def compute_sternum_and_pelvis(keypoints_3d):
    """
    Compute sternum (center of shoulders) and pelvis (center of hips).

    Args:
        keypoints_3d: (17, 3) array of COCO keypoints

    Returns:
        sternum: (3,) array
        pelvis: (3,) array
    """
    left_shoulder = keypoints_3d[KP_NAME_TO_IDX['left_shoulder']]
    right_shoulder = keypoints_3d[KP_NAME_TO_IDX['right_shoulder']]
    left_hip = keypoints_3d[KP_NAME_TO_IDX['left_hip']]
    right_hip = keypoints_3d[KP_NAME_TO_IDX['right_hip']]

    sternum = (left_shoulder + right_shoulder) / 2.0
    pelvis = (left_hip + right_hip) / 2.0

    return sternum, pelvis


def load_keypoints_from_json(json_path, flip_z=False):
    """
    Load 3D keypoints from all_keypoints.json file.

    Args:
        json_path: Path to keypoints JSON
        flip_z: If True, flip the z-axis (multiply by -1)

    Returns:
        List of dicts with 'frame_idx' and 'keypoints_3d' (17, 3) numpy arrays
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    frames = []
    for frame_data in data['frames']:
        if 'keypoints_3d' in frame_data and frame_data['keypoints_3d'] is not None:
            keypoints_3d = np.array(frame_data['keypoints_3d'])
            if flip_z:
                keypoints_3d[:, 2] *= -1
            frames.append({
                'frame_idx': frame_data['frame_idx'],
                'keypoints_3d': keypoints_3d
            })

    print(f"✓ Loaded {len(frames)} frames from {json_path}")
    return frames


def load_keypoints_from_csv(csv_path, flip_z=False):
    """
    Load 3D keypoints from COCO CSV file.

    Args:
        csv_path: Path to CSV file
        flip_z: If True, flip the z-axis (multiply by -1)

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

        if flip_z:
            keypoints_3d[:, 2] *= -1

        frames.append({
            'frame_idx': int(row['frame']),
            'keypoints_3d': keypoints_3d
        })

    print(f"✓ Loaded {len(frames)} frames from {csv_path}")
    return frames


def plot_3d_skeleton(ax, keypoints_3d, show_face=False, clean_style=True):
    """
    Plot 3D skeleton with bones.

    Args:
        ax: Matplotlib 3D axis
        keypoints_3d: (17, 3) numpy array of 3D keypoints
        show_face: If True, show face keypoints (default: False)
        clean_style: If True, black skeleton on white background, no axes (default: True)
    """
    # Clear previous plot
    ax.clear()

    # Compute sternum and pelvis
    sternum, pelvis = compute_sternum_and_pelvis(keypoints_3d)

    # Determine which keypoints to show
    if show_face:
        kp_indices = list(range(17))
    else:
        kp_indices = BODY_KEYPOINT_INDICES  # 5-16 (exclude face)

    # Set colors
    if clean_style:
        kp_color = 'black'
        bone_color = 'black'
        linewidth = 2
    else:
        kp_color = 'red'
        bone_color = 'blue'
        linewidth = 2

    # Plot keypoints
    for idx in kp_indices:
        x, y, z = keypoints_3d[idx]
        ax.scatter(x, y, z, c=kp_color, marker='o', s=50, zorder=10)

    # Plot sternum and pelvis
    ax.scatter(*sternum, c=kp_color, marker='o', s=50, zorder=10)
    ax.scatter(*pelvis, c=kp_color, marker='o', s=50, zorder=10)

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

        ax.plot(xs, ys, zs, color=bone_color, linewidth=linewidth, zorder=5)

    # Plot sternum-to-pelvis connection
    ax.plot([sternum[0], pelvis[0]],
            [sternum[1], pelvis[1]],
            [sternum[2], pelvis[2]],
            color=bone_color, linewidth=linewidth, zorder=5)

    # Set consistent axis limits
    all_points = np.vstack([keypoints_3d[kp_indices], sternum, pelvis])
    center = np.mean(all_points, axis=0)
    max_range = np.max(np.ptp(all_points, axis=0)) * 0.6

    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)

    # Apply clean style if requested
    if clean_style:
        # Remove axes, grid, background
        ax.set_axis_off()
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('white')
        ax.yaxis.pane.set_edgecolor('white')
        ax.zaxis.pane.set_edgecolor('white')
    else:
        # Keep labels and aspect
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1, 1, 1])


def visualize_static(frames_data, output_path=None, show_face=False, num_frames=4, clean_style=True):
    """
    Create a static multi-panel visualization showing multiple frames.

    Args:
        frames_data: List of frame dicts with 'keypoints_3d'
        output_path: Path to save figure (optional)
        show_face: If True, show face keypoints
        num_frames: Number of frames to display
        clean_style: If True, black skeleton on white, no axes
    """
    # Select evenly spaced frames
    indices = np.linspace(0, len(frames_data) - 1, num_frames, dtype=int)

    fig = plt.figure(figsize=(16, 4), facecolor='white')

    for i, idx in enumerate(indices):
        frame = frames_data[idx]
        ax = fig.add_subplot(1, num_frames, i + 1, projection='3d')
        ax.set_facecolor('white')
        plot_3d_skeleton(ax, frame['keypoints_3d'], show_face=show_face, clean_style=clean_style)
        if not clean_style:
            ax.set_title(f"Frame {frame['frame_idx']}")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved visualization to {output_path}")
    else:
        plt.show()


def visualize_animation(frames_data, output_path=None, show_face=False, fps=10, clean_style=True):
    """
    Create an animated visualization of the skeleton over time.

    Args:
        frames_data: List of frame dicts with 'keypoints_3d'
        output_path: Path to save animation (optional, .gif or .mp4)
        show_face: If True, show face keypoints
        fps: Frames per second for animation
        clean_style: If True, black skeleton on white, no axes
    """
    fig = plt.figure(figsize=(10, 8), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')

    def update(frame_idx):
        frame = frames_data[frame_idx]
        plot_3d_skeleton(ax, frame['keypoints_3d'], show_face=show_face, clean_style=clean_style)
        if not clean_style:
            ax.set_title(f"Frame {frame['frame_idx']}", fontsize=14)
        return ax,

    anim = FuncAnimation(
        fig, update, frames=len(frames_data),
        interval=1000/fps, blit=False
    )

    if output_path:
        output_path = Path(output_path)
        if output_path.suffix == '.gif':
            anim.save(output_path, writer='pillow', fps=fps, savefig_kwargs={'facecolor': 'white'})
            print(f"✓ Saved animation to {output_path}")
        elif output_path.suffix == '.mp4':
            anim.save(output_path, writer='ffmpeg', fps=fps, savefig_kwargs={'facecolor': 'white'})
            print(f"✓ Saved animation to {output_path}")
        else:
            print(f"⚠ Unsupported format {output_path.suffix}, use .gif or .mp4")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize 3D COCO keypoints with skeleton bones"
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
    parser.add_argument(
        '--no-clean-style', action='store_true',
        help='Use colored skeleton with axes (default: black on white, no axes)'
    )
    parser.add_argument(
        '--flip-z', action='store_true',
        help='Flip the z-axis (multiply z-coordinates by -1)'
    )

    args = parser.parse_args()

    # Load keypoints
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return

    clean_style = not args.no_clean_style

    if input_path.suffix == '.json':
        frames_data = load_keypoints_from_json(input_path, flip_z=args.flip_z)
    elif input_path.suffix == '.csv':
        frames_data = load_keypoints_from_csv(input_path, flip_z=args.flip_z)
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
            num_frames=args.num_frames,
            clean_style=clean_style
        )
    elif args.mode == 'animation':
        visualize_animation(
            frames_data,
            output_path=args.output,
            show_face=args.show_face,
            fps=args.fps,
            clean_style=clean_style
        )


if __name__ == '__main__':
    main()
