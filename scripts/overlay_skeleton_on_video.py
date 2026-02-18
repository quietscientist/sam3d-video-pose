#!/usr/bin/env python3
"""
Overlay 3D COCO skeleton on original video with bright colors.
Projects 3D keypoints to 2D image space and draws skeleton with OpenPose-style colors.
"""

import argparse
import numpy as np
import cv2
import pandas as pd
from pathlib import Path


# COCO keypoint names
COCO_KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

KP_NAME_TO_IDX = {name: idx for idx, name in enumerate(COCO_KEYPOINT_NAMES)}

# Skeleton connections with OpenPose-style colors
SKELETON_CONNECTIONS = [
    # Face (light blue)
    (('nose', 'left_eye'), (100, 200, 255)),
    (('nose', 'right_eye'), (100, 200, 255)),
    (('left_eye', 'left_ear'), (100, 200, 255)),
    (('right_eye', 'right_ear'), (100, 200, 255)),

    # Torso (cyan)
    (('left_shoulder', 'right_shoulder'), (255, 255, 0)),
    (('left_shoulder', 'left_hip'), (0, 255, 255)),
    (('right_shoulder', 'right_hip'), (0, 255, 255)),
    (('left_hip', 'right_hip'), (255, 255, 0)),

    # Arms (green/yellow)
    (('left_shoulder', 'left_elbow'), (0, 255, 0)),
    (('left_elbow', 'left_wrist'), (0, 255, 128)),
    (('right_shoulder', 'right_elbow'), (0, 255, 0)),
    (('right_elbow', 'right_wrist'), (0, 255, 128)),

    # Legs (magenta/pink)
    (('left_hip', 'left_knee'), (255, 0, 255)),
    (('left_knee', 'left_ankle'), (255, 100, 255)),
    (('right_hip', 'right_knee'), (255, 0, 255)),
    (('right_knee', 'right_ankle'), (255, 100, 255)),

    # Spine (sternum to pelvis)
    (('sternum', 'pelvis'), (0, 255, 255)),
]



def load_keypoints_from_csv(csv_path):
    """Load 3D COCO keypoints from wide CSV format."""
    df = pd.read_csv(csv_path)

    frames = {}
    for _, row in df.iterrows():
        frame_idx = int(row['frame'])
        keypoints_3d = np.zeros((17, 3))

        for i, name in enumerate(COCO_KEYPOINT_NAMES):
            keypoints_3d[i, 0] = row[f'{name}_x']
            keypoints_3d[i, 1] = row[f'{name}_y']
            keypoints_3d[i, 2] = row[f'{name}_z']

        frames[frame_idx] = keypoints_3d

    print(f"✓ Loaded COCO 3D keypoints for {len(frames)} frames from {csv_path}")
    return frames


def load_camera_params(meshes_dir):
    """
    Load per-frame MHR camera parameters (pred_cam_t, focal_length) from NPZ files.
    """
    meshes_dir = Path(meshes_dir)
    camera_params = {}

    for npz_path in sorted(meshes_dir.glob('frame_*_obj_*/mhr_parameters.npz')):
        dir_name = npz_path.parent.name  # e.g., "frame_0000_obj_0"
        frame_idx = int(dir_name.split('_')[1])

        npz = np.load(npz_path, allow_pickle=True)
        camera_params[frame_idx] = {
            'pred_cam_t': npz['pred_cam_t'],
            'focal_length': float(npz['focal_length']),
        }

    print(f"✓ Loaded camera parameters for {len(camera_params)} frames")
    return camera_params


def project_3d_to_2d(keypoints_3d, pred_cam_t, focal_length, image_width, image_height):
    """
    Project 3D keypoints to 2D image space using MHR perspective camera model.
    Uses: x_2d = f * (X + tx) / (Z + tz) + cx
    """
    kp_cam = keypoints_3d + pred_cam_t[np.newaxis, :]

    cx = image_width / 2.0
    cy = image_height / 2.0

    x_2d = focal_length * kp_cam[:, 0] / kp_cam[:, 2] + cx
    y_2d = focal_length * kp_cam[:, 1] / kp_cam[:, 2] + cy

    return np.stack([x_2d, y_2d], axis=1)


def draw_skeleton(frame, keypoints_2d, bone_thickness=4, point_radius=6):
    """Draw skeleton with OpenPose-style colors and 3D-style rendering."""
    height, width = frame.shape[:2]

    # Compute sternum and pelvis in 2D
    ls = keypoints_2d[KP_NAME_TO_IDX['left_shoulder']]
    rs = keypoints_2d[KP_NAME_TO_IDX['right_shoulder']]
    lh = keypoints_2d[KP_NAME_TO_IDX['left_hip']]
    rh = keypoints_2d[KP_NAME_TO_IDX['right_hip']]
    sternum_2d = (ls + rs) / 2.0
    pelvis_2d = (lh + rh) / 2.0

    # Build lookup for all points including virtual ones
    all_points_2d = {}
    for i, name in enumerate(COCO_KEYPOINT_NAMES):
        all_points_2d[name] = keypoints_2d[i]
    all_points_2d['sternum'] = sternum_2d
    all_points_2d['pelvis'] = pelvis_2d

    # Draw bones (dark outline + bright color for 3D cylinder effect)
    for (start_name, end_name), color in SKELETON_CONNECTIONS:
        if start_name not in all_points_2d or end_name not in all_points_2d:
            continue

        start_pt = all_points_2d[start_name]
        end_pt = all_points_2d[end_name]

        # Skip if points are way out of bounds
        if (abs(start_pt[0]) > width * 2 or abs(start_pt[1]) > height * 2 or
            abs(end_pt[0]) > width * 2 or abs(end_pt[1]) > height * 2):
            continue

        p1 = (int(round(start_pt[0])), int(round(start_pt[1])))
        p2 = (int(round(end_pt[0])), int(round(end_pt[1])))

        cv2.line(frame, p1, p2, (0, 0, 0), bone_thickness + 2, cv2.LINE_AA)
        cv2.line(frame, p1, p2, color, bone_thickness, cv2.LINE_AA)

    # Draw keypoints (3D ball effect: dark outline, bright fill, white highlight)
    for name, pt in all_points_2d.items():
        if abs(pt[0]) > width * 2 or abs(pt[1]) > height * 2:
            continue

        center = (int(round(pt[0])), int(round(pt[1])))
        cv2.circle(frame, center, point_radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, center, point_radius, (255, 255, 255), -1, cv2.LINE_AA)
        highlight = (center[0] - max(1, point_radius // 3), center[1] - max(1, point_radius // 3))
        cv2.circle(frame, highlight, max(2, point_radius // 3), (255, 255, 255), -1, cv2.LINE_AA)


def overlay_skeleton_on_video(video_path, keypoints_data, camera_params, output_path, fps=None):
    """
    Overlay COCO skeleton on video using MHR camera projection.

    Args:
        video_path: Path to input video
        keypoints_data: Dict of frame_idx -> (17, 3) numpy array of 3D COCO keypoints
        camera_params: Dict of frame_idx -> {'pred_cam_t', 'focal_length'}
        output_path: Path for output video
        fps: Output FPS (default: same as input)
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"❌ Error: Could not open video: {video_path}")
        return False

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    output_fps = fps or input_fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))

    frame_idx = 0
    frames_with_skeleton = 0

    print(f"Processing video: {video_path}")
    print(f"Output: {output_path}")
    print(f"Resolution: {width}x{height} @ {output_fps:.1f} FPS")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in keypoints_data and frame_idx in camera_params:
            kp_3d = keypoints_data[frame_idx]
            cam = camera_params[frame_idx]

            kp_2d = project_3d_to_2d(
                kp_3d, cam['pred_cam_t'], cam['focal_length'],
                width, height
            )

            draw_skeleton(frame, kp_2d)
            frames_with_skeleton += 1

        out.write(frame)
        frame_idx += 1

        if frame_idx % 50 == 0:
            print(f"  Processed {frame_idx} frames ({frames_with_skeleton} with skeletons)...")

    cap.release()
    out.release()

    print(f"✓ Saved overlay video to: {output_path}")
    print(f"  Total frames: {frame_idx}, Frames with skeletons: {frames_with_skeleton}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Overlay COCO 3D skeleton on video with bright colors"
    )
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('keypoints_csv', help='Path to COCO 3D wide CSV file')
    parser.add_argument('meshes_dir', help='Path to meshes directory (for MHR camera parameters)')
    parser.add_argument('--output', '-o', help='Output video path (default: <video>_skeleton_overlay.mp4)')
    parser.add_argument('--fps', type=int, default=None, help='Output video FPS (default: same as input)')

    args = parser.parse_args()

    video_path = Path(args.video_path)
    keypoints_csv = Path(args.keypoints_csv)
    meshes_dir = Path(args.meshes_dir)

    for path, label in [(video_path, "Video"), (keypoints_csv, "CSV"), (meshes_dir, "Meshes dir")]:
        if not path.exists():
            print(f"❌ Error: {label} not found: {path}")
            return

    output_path = Path(args.output) if args.output else video_path.parent / f"{video_path.stem}_skeleton_overlay.mp4"

    keypoints_data = load_keypoints_from_csv(keypoints_csv)
    camera_params = load_camera_params(meshes_dir)

    overlay_skeleton_on_video(video_path, keypoints_data, camera_params, output_path, fps=args.fps)


if __name__ == '__main__':
    main()
