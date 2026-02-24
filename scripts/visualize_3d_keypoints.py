#!/usr/bin/env python3
"""
Visualize 3D COCO keypoints time series with skeleton bones.
Supports automatic world-alignment (gravity detection) and ground plane.
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
import pandas as pd


# ===================================================================
# COCO keypoint constants
# ===================================================================

COCO_KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',  # 0-4: FACE
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',  # 5-8
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',  # 9-12
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'  # 13-16
]

KP_NAME_TO_IDX = {name: idx for idx, name in enumerate(COCO_KEYPOINT_NAMES)}
BODY_KEYPOINT_INDICES = list(range(5, 17))  # exclude face

BODY_SKELETON = [
    ('left_hip', 'left_knee'), ('right_hip', 'right_knee'),
    ('left_knee', 'left_ankle'), ('right_knee', 'right_ankle'),
    ('left_shoulder', 'left_elbow'), ('right_shoulder', 'right_elbow'),
    ('left_elbow', 'left_wrist'), ('right_elbow', 'right_wrist'),
    ('left_shoulder', 'right_shoulder'), ('left_hip', 'right_hip'),
    ('sternum', 'pelvis'),
]

BONE_CONNECTIONS = [
    (KP_NAME_TO_IDX[s], KP_NAME_TO_IDX[e])
    for s, e in BODY_SKELETON
    if s in KP_NAME_TO_IDX and e in KP_NAME_TO_IDX
]

# Left/right bone coloring
LEFT_BONE_SET = {(5, 7), (7, 9), (11, 13), (13, 15)}
RIGHT_BONE_SET = {(6, 8), (8, 10), (12, 14), (14, 16)}


# ===================================================================
# Data loading
# ===================================================================
def compute_head_direction(kp):
    left_eye = kp[1]
    right_eye = kp[2]
    nose = kp[0]

    eye_mid = (left_eye + right_eye) / 2.0

    # --- Face plane normal ---
    v1 = right_eye - left_eye
    v2 = nose - eye_mid

    forward = np.cross(v1, v2)
    norm = np.linalg.norm(forward)
    if norm < 1e-6:
        return None, None
    forward /= norm

    # --- Compute torso forward direction ---
    sternum = (kp[5] + kp[6]) / 2.0
    pelvis = (kp[11] + kp[12]) / 2.0
    spine = sternum - pelvis
    spine_n = spine / (np.linalg.norm(spine) + 1e-8)

    shoulders = kp[5] - kp[6]
    chest = -np.cross(spine_n, shoulders)
    chest /= (np.linalg.norm(chest) + 1e-8)

    # --- Align gaze sign with torso ---
    if np.dot(forward, chest) < 0:
        forward = -forward

    # --- Remove spine component (nose sits below the eyes anatomically, which
    #     creates a spurious downward tilt in the cross-product face normal).
    #     Projecting onto the plane perpendicular to the spine gives a gaze
    #     direction that lies in the body's transverse plane.
    #     For standing subjects: removes downward tilt.
    #     For supine subjects: spine is horizontal so the upward gaze is preserved.
    forward = forward - np.dot(forward, spine_n) * spine_n
    norm = np.linalg.norm(forward)
    if norm < 1e-6:
        return None, None
    forward /= norm

    return eye_mid, forward


def compute_sternum_and_pelvis(kp):
    sternum = (kp[5] + kp[6]) / 2.0
    pelvis = (kp[11] + kp[12]) / 2.0
    return sternum, pelvis


def load_keypoints_from_json(json_path, flip_z=False):
    with open(json_path, 'r') as f:
        data = json.load(f)
    frames = []
    for fd in data['frames']:
        if 'keypoints_3d' in fd and fd['keypoints_3d'] is not None:
            kp = np.array(fd['keypoints_3d'])
            if flip_z:
                kp[:, 2] *= -1
            frames.append({'frame_idx': fd['frame_idx'], 'keypoints_3d': kp})
    print(f"Loaded {len(frames)} frames from {json_path}")
    return frames


def load_keypoints_from_csv(csv_path, flip_z=False):
    df = pd.read_csv(csv_path)
    frames = []

    # Auto-detect format: long (frame, x, y, z, part_idx) vs wide (frame, nose_x, ...)
    if 'part_idx' in df.columns:
        # Long format: one row per keypoint per frame
        for frame_idx, group in df.groupby('frame'):
            group = group.sort_values('part_idx')
            kp = np.zeros((17, 3))
            for _, row in group.iterrows():
                idx = int(row['part_idx'])
                if idx < 17:
                    kp[idx] = [row['x'], row['y'], row['z']]
            if flip_z:
                kp[:, 2] *= -1
            frames.append({'frame_idx': int(frame_idx), 'keypoints_3d': kp})
    else:
        # Wide format: one row per frame, columns like nose_x, nose_y, nose_z
        for _, row in df.iterrows():
            kp = np.zeros((17, 3))
            for i in range(17):
                kp[i, 0] = row[f'{COCO_KEYPOINT_NAMES[i]}_x']
                kp[i, 1] = row[f'{COCO_KEYPOINT_NAMES[i]}_y']
                kp[i, 2] = row[f'{COCO_KEYPOINT_NAMES[i]}_z']
            if flip_z:
                kp[:, 2] *= -1
            frames.append({'frame_idx': int(row['frame']), 'keypoints_3d': kp})

    print(f"Loaded {len(frames)} frames from {csv_path}")
    return frames


def load_head_direction_csv(csv_path):
    """Load per-frame head direction vectors (camera space) from _head_direction.csv.
    Returns dict mapping frame_idx -> (3,) np.array."""
    df = pd.read_csv(csv_path)
    return {int(row.frame): np.array([row.hx, row.hy, row.hz])
            for _, row in df.iterrows()}


# ===================================================================
# World alignment (gravity detection + rotation)
# ===================================================================

def compute_body_frame(kp):
    """
    Compute body-frame vectors from COCO keypoints.

    cross(spine, shoulders) gives the BACK-facing direction by right-hand rule.
    We negate to get chest-facing.

    Returns: (spine_n, chest_n, shoulder_n) as unit vectors.
    """
    sternum = (kp[5] + kp[6]) / 2
    pelvis = (kp[11] + kp[12]) / 2
    spine = sternum - pelvis
    spine_n = spine / (np.linalg.norm(spine) + 1e-8)

    shoulders = kp[5] - kp[6]
    shoulders_n = shoulders / (np.linalg.norm(shoulders) + 1e-8)

    back_normal = np.cross(spine_n, shoulders_n)
    chest_n = -back_normal
    chest_n = chest_n / (np.linalg.norm(chest_n) + 1e-8)

    return spine_n, chest_n, shoulders_n


def detect_camera_orientation(frames_data):
    """
    Analyze body geometry across frames to detect camera viewing angle.
    MHR camera coords: X=right, Y=down, Z=forward (into screen).

    Returns: (orientation, gravity_up, stats)
    """
    spine_dots_z, spine_dots_y, normal_dots_z = [], [], []

    for fd in frames_data:
        kp = fd['keypoints_3d']
        spine_n, chest_n, _ = compute_body_frame(kp)
        spine_dots_z.append(np.dot(spine_n, [0, 0, 1]))
        spine_dots_y.append(np.dot(spine_n, [0, 1, 0]))
        normal_dots_z.append(np.dot(chest_n, [0, 0, 1]))

    stats = {
        'spine_dot_camZ': np.mean(spine_dots_z),
        'spine_dot_camY': np.mean(spine_dots_y),
        'normal_dot_camZ': np.mean(normal_dots_z),
    }

    abs_sz = abs(stats['spine_dot_camZ'])
    abs_sy = abs(stats['spine_dot_camY'])
    abs_nz = abs(stats['normal_dot_camZ'])

    if abs_sz > 0.7:
        orientation = 'top-down-axial'
        gravity_up = np.median([compute_body_frame(fd['keypoints_3d'])[1]
                                for fd in frames_data], axis=0)
    elif abs_sy > 0.6:
        orientation = 'front-on-standing'
        gravity_up = np.array([0.0, -1.0, 0.0])
    elif abs_nz > 0.5:
        orientation = 'top-down-supine' if abs_sy < 0.3 else 'front-on-standing'
        gravity_up = np.median([compute_body_frame(fd['keypoints_3d'])[1]
                                for fd in frames_data], axis=0)
        if gravity_up[2] > 0:
            gravity_up = -gravity_up
    else:
        orientation = 'oblique'
        gravity_up = np.median([compute_body_frame(fd['keypoints_3d'])[1]
                                for fd in frames_data], axis=0)
        if gravity_up[2] > 0:
            gravity_up = -gravity_up

    gravity_up = gravity_up / (np.linalg.norm(gravity_up) + 1e-8)
    return orientation, gravity_up, stats


def build_world_rotation(gravity_up, spine_hint=None):
    """
    Build 3x3 rotation: gravity_up -> +Z, spine_hint -> +X.
    """
    target_up = np.array([0.0, 0.0, 1.0])
    rot_axis = np.cross(gravity_up, target_up)
    rot_axis_norm = np.linalg.norm(rot_axis)

    if rot_axis_norm < 1e-6:
        R = np.eye(3) if np.dot(gravity_up, target_up) > 0 else np.diag([1., -1., -1.])
    else:
        rot_axis /= rot_axis_norm
        cos_a = np.clip(np.dot(gravity_up, target_up), -1, 1)
        sin_a = np.sqrt(1 - cos_a ** 2)
        K = np.array([[0, -rot_axis[2], rot_axis[1]],
                       [rot_axis[2], 0, -rot_axis[0]],
                       [-rot_axis[1], rot_axis[0], 0]])
        R = np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)

    if spine_hint is not None:
        sr = R @ spine_hint
        sx, sy = sr[0], sr[1]
        if abs(sx) + abs(sy) > 1e-6:
            angle_z = -np.arctan2(sy, sx)
            cz, sz_ = np.cos(angle_z), np.sin(angle_z)
            Rz = np.array([[cz, -sz_, 0], [sz_, cz, 0], [0, 0, 1]])
            R = Rz @ R

    return R


def compute_ground_z_offset(kp, orientation):
    """Z value to subtract so the ground contact sits at Z=0."""
    if orientation == 'front-on-standing':
        return np.min(kp[[15, 16], 2])
    elif orientation in ('top-down-supine', 'top-down-axial'):
        return np.min(kp[[5, 6, 11, 12], 2])
    else:
        body_kp = kp[BODY_KEYPOINT_INDICES]
        return np.min(body_kp[:, 2]) - 0.05 * np.ptp(body_kp[:, 2])


def world_align_frames(frames_data):
    """
    Detect camera orientation, compute world rotation, and transform all
    frames so that gravity is +Z, ground is at Z=0, and body is XY-centered.

    Returns:
        frames_data: same list, with 'keypoints_3d' replaced by world-aligned coords
        orientation: detected orientation string
        global_bounds: dict with fixed axis limits and ground plane extent
    """
    orientation, gravity_up, stats = detect_camera_orientation(frames_data)

    median_spine = np.median([compute_body_frame(fd['keypoints_3d'])[0]
                              for fd in frames_data], axis=0)
    median_spine /= np.linalg.norm(median_spine)

    R_world = build_world_rotation(gravity_up, spine_hint=median_spine)

    print(f"  Camera orientation: {orientation}")
    print(f"  spine.camY={stats['spine_dot_camY']:+.2f}  "
          f"chest.camZ={stats['normal_dot_camZ']:+.2f}")

    # Transform all frames
    all_body_pts = []
    for fd in frames_data:
        kp = fd['keypoints_3d'].copy()
        kp = (R_world @ kp.T).T

        # Center XY on torso midpoint
        sternum, pelvis = compute_sternum_and_pelvis(kp)
        torso_center_xy = (sternum[:2] + pelvis[:2]) / 2
        kp[:, 0] -= torso_center_xy[0]
        kp[:, 1] -= torso_center_xy[1]

        # Shift Z so ground = 0
        ground_z = compute_ground_z_offset(kp, orientation)
        kp[:, 2] -= ground_z

        fd['keypoints_3d'] = kp

        # Collect body points for global bounds
        sternum, pelvis = compute_sternum_and_pelvis(kp)
        body_pts = np.vstack([kp[BODY_KEYPOINT_INDICES], [sternum], [pelvis]])
        all_body_pts.append(body_pts)

    # Compute fixed global bounds across all frames
    all_pts = np.vstack(all_body_pts)
    x_range = np.ptp(all_pts[:, 0])
    y_range = np.ptp(all_pts[:, 1])
    z_range = np.ptp(all_pts[:, 2])
    max_range = max(x_range, y_range, z_range) * 0.7

    cx = np.mean(all_pts[:, 0])
    cy = np.mean(all_pts[:, 1])
    z_lo = min(0.0, all_pts[:, 2].min()) - 0.05 * max_range
    z_hi = max(0.0, all_pts[:, 2].max()) + 0.05 * max_range
    z_center = (z_lo + z_hi) / 2
    z_half = max(max_range, (z_hi - z_lo) / 2)

    global_bounds = {
        'xlim': (cx - max_range, cx + max_range),
        'ylim': (cy - max_range, cy + max_range),
        'zlim': (z_center - z_half, z_center + z_half),
        'ground_extent': max_range,  # fixed ground plane half-size
    }

    return frames_data, orientation, global_bounds, R_world


# ===================================================================
# Plotting
# ===================================================================
def draw_cone(ax, origin, direction, length=0.2, radius=0.05, resolution=30):
    direction = direction / (np.linalg.norm(direction) + 1e-8)

    tip = origin
    base_center = origin + direction * length

    # Create orthonormal basis
    v = direction
    if abs(v[0]) < 0.9:
        ref = np.array([1, 0, 0])
    else:
        ref = np.array([0, 1, 0])

    u = np.cross(v, ref)
    u /= np.linalg.norm(u)
    w = np.cross(v, u)

    # Create circular base
    theta = np.linspace(0, 2*np.pi, resolution)
    r = np.linspace(0, radius, 2)  # 2 rows (base to tip)

    T, R = np.meshgrid(theta, r)

    # Cone in local coordinates: base (wide) at origin, tip (narrow) at front
    X = R * np.cos(T)
    Y = R * np.sin(T)
    Z = (1 - R / radius) * length

    # Stack
    points = np.stack([X.flatten(), Y.flatten(), Z.flatten()])

    # Build rotation matrix
    Rmat = np.column_stack((u, w, v))

    # Transform to world coords
    world_pts = origin[:, None] + Rmat @ points

    Xw = world_pts[0].reshape(X.shape)
    Yw = world_pts[1].reshape(Y.shape)
    Zw = world_pts[2].reshape(Z.shape)

    ax.plot_surface(
        Xw, Yw, Zw,
        color='#B3EAFC',
        alpha=0.25,
        linewidth=0,
        shade=True
    )



def plot_3d_skeleton(ax, keypoints_3d, show_face=False, clean_style=True,
                     global_bounds=None, show_ground=False, head_direction=None):
    """
    Plot 3D skeleton with bones and optional ground plane.

    Args:
        ax: Matplotlib 3D axis
        keypoints_3d: (17, 3) numpy array of 3D keypoints
        show_face: If True, show face keypoints
        clean_style: If True, minimal style with no axes
        global_bounds: If provided, use fixed axis limits and ground plane size
        show_ground: If True, draw green ground plane at Z=0
    """
    ax.clear()
        
    sternum, pelvis = compute_sternum_and_pelvis(keypoints_3d)

    kp_indices = list(range(17)) if show_face else BODY_KEYPOINT_INDICES

    # Draw ground plane first (behind skeleton)
    if show_ground and global_bounds:
        ext = global_bounds['ground_extent']
        xx = [-ext, ext, ext, -ext]
        yy = [-ext, -ext, ext, ext]
        zz = [0.0] * 4
        verts = [list(zip(xx, yy, zz))]
        ground = Poly3DCollection(verts, alpha=0.10, facecolor='#27AE60',
                                   edgecolor='#27AE60', linewidth=0.5)
        ax.add_collection3d(ground)

    # Draw bones with left/right coloring
    for start_idx, end_idx in BONE_CONNECTIONS:
        if not show_face and (start_idx < 5 or end_idx < 5):
            continue
        pair = (start_idx, end_idx)

        if pair in LEFT_BONE_SET:
            color = '#444444'   # darker grey
        elif pair in RIGHT_BONE_SET:
            color = '#888888'   # lighter grey
        else:
            color = '#222222'   # torso / centerline darkest
        # if pair in LEFT_BONE_SET:
        #     color = '#E74C3C'
        # elif pair in RIGHT_BONE_SET:
        #     color = '#3498DB'
        # else:
        #     color = '#2C3E50'
        ax.plot([keypoints_3d[start_idx, 0], keypoints_3d[end_idx, 0]],
                [keypoints_3d[start_idx, 1], keypoints_3d[end_idx, 1]],
                [keypoints_3d[start_idx, 2], keypoints_3d[end_idx, 2]],
                color=color, linewidth=2.5, zorder=5)

    # Sternum-pelvis spine
    ax.plot([sternum[0], pelvis[0]], [sternum[1], pelvis[1]],
            [sternum[2], pelvis[2]], color='#2C3E50', linewidth=2,
            linestyle='--', zorder=5)

    origin, fallback_forward = compute_head_direction(keypoints_3d)
    forward = head_direction if head_direction is not None else fallback_forward

    if origin is not None:
        cone_length = 0.2 * global_bounds['ground_extent']
        draw_cone(ax, origin, forward,
                length=cone_length,
                radius=0.08 * global_bounds['ground_extent'])
        # Marker at cone tip (narrow end, in gaze direction)
        tip = origin + forward * cone_length
        ax.scatter(tip[0], tip[1], tip[2],
            c="#1F2DA5", marker='D', s=35, zorder=11)

    # Draw joints
    for idx in kp_indices:
        ax.scatter(*keypoints_3d[idx], c='black', marker='o', s=30, zorder=10)
    ax.scatter(*sternum, c="#222222", s=35, zorder=11)
    ax.scatter(*pelvis, c='#222222', marker='D', s=35, zorder=11)
    # ax.scatter(*sternum, c='#E74C3C', marker='D', s=35, zorder=11)
    # ax.scatter(*pelvis, c='#3498DB', marker='D', s=35, zorder=11)

    # Axis limits
    if global_bounds:
        ax.set_xlim(*global_bounds['xlim'])
        ax.set_ylim(*global_bounds['ylim'])
        ax.set_zlim(*global_bounds['zlim'])
    else:
        all_points = np.vstack([keypoints_3d[kp_indices], sternum, pelvis])
        center = np.mean(all_points, axis=0)
        max_range = np.max(np.ptp(all_points, axis=0)) * 0.6
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)

    if clean_style:
        ax.set_axis_off()
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('white')
        ax.yaxis.pane.set_edgecolor('white')
        ax.zaxis.pane.set_edgecolor('white')
    else:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1, 1, 1])


# ===================================================================
# Visualization modes
# ===================================================================

def visualize_static(frames_data, output_path=None, show_face=False,
                     num_frames=4, clean_style=True, global_bounds=None,
                     show_ground=False):
    indices = np.linspace(0, len(frames_data) - 1, num_frames, dtype=int)
    fig = plt.figure(figsize=(16, 4), facecolor='white')

    for i, idx in enumerate(indices):
        frame = frames_data[idx]
        ax = fig.add_subplot(1, num_frames, i + 1, projection='3d')
        ax.set_facecolor('white')
        plot_3d_skeleton(ax, frame['keypoints_3d'], show_face=show_face,
                         clean_style=clean_style, global_bounds=global_bounds,
                         show_ground=show_ground,
                         head_direction=frame.get('head_direction'))
        if not clean_style:
            ax.set_title(f"Frame {frame['frame_idx']}")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()


def visualize_animation(frames_data, output_path=None, show_face=False,
                        fps=10, clean_style=True, global_bounds=None,
                        show_ground=False):
    fig = plt.figure(figsize=(10, 8), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')

    def update(frame_idx):
        frame = frames_data[frame_idx]
        plot_3d_skeleton(ax, frame['keypoints_3d'], show_face=show_face,
                         clean_style=clean_style, global_bounds=global_bounds,
                         show_ground=show_ground,
                         head_direction=frame.get('head_direction'))
        if not clean_style:
            ax.set_title(f"Frame {frame['frame_idx']}", fontsize=14)
        return ax,

    anim = FuncAnimation(
        fig, update, frames=len(frames_data),
        interval=1000 / fps, blit=False
    )

    if output_path:
        output_path = Path(output_path)
        if output_path.suffix == '.gif':
            anim.save(output_path, writer='pillow', fps=fps,
                      savefig_kwargs={'facecolor': 'white'})
            print(f"Saved animation to {output_path}")
        elif output_path.suffix == '.mp4':
            anim.save(output_path, writer='ffmpeg', fps=fps,
                      savefig_kwargs={'facecolor': 'white'})
            print(f"Saved animation to {output_path}")
        else:
            print(f"Unsupported format {output_path.suffix}, use .gif or .mp4")
    else:
        plt.show()


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize 3D COCO keypoints with skeleton bones"
    )
    parser.add_argument('input_path',
                        help='Path to keypoints file (.json or .csv)')
    parser.add_argument('--mode', choices=['static', 'animation'],
                        default='static', help='Visualization mode')
    parser.add_argument('--output', '-o', type=str,
                        help='Output path (PNG for static, GIF/MP4 for animation)')
    parser.add_argument('--show-face', action='store_true',
                        help='Include face keypoints')
    parser.add_argument('--num-frames', type=int, default=4,
                        help='Frames to display in static mode (default: 4)')
    parser.add_argument('--fps', type=int, default=10,
                        help='Animation FPS (default: 10)')
    parser.add_argument('--no-clean-style', action='store_true',
                        help='Show axes and colored style instead of minimal')
    parser.add_argument('--flip-z', action='store_true',
                        help='Flip z-axis before processing')
    parser.add_argument('--no-world-align', action='store_true',
                        help='Disable automatic world alignment (use raw camera coords)')

    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return

    clean_style = not args.no_clean_style

    # Load keypoints
    if input_path.suffix == '.json':
        frames_data = load_keypoints_from_json(input_path, flip_z=args.flip_z)
    elif input_path.suffix == '.csv':
        frames_data = load_keypoints_from_csv(input_path, flip_z=args.flip_z)
    else:
        print(f"Error: Unsupported format: {input_path.suffix}")
        return

    if not frames_data:
        print("Error: No frames with valid keypoints found")
        return

    # Auto-detect sibling _head_direction.csv from process_video.py output
    head_directions_cam = {}
    stem = input_path.stem
    for suffix in ['_3D_smoothed_adjusted', '_3D_smoothed', '_3D_raw']:
        if stem.endswith(suffix):
            video_name = stem[:-len(suffix)]
            break
    else:
        video_name = stem
    head_dir_csv = input_path.parent / f"{video_name}_head_direction.csv"
    if head_dir_csv.exists():
        print(f"Loading MHR head directions from {head_dir_csv}")
        head_directions_cam = load_head_direction_csv(head_dir_csv)

    # World alignment
    global_bounds = None
    show_ground = False
    if not args.no_world_align:
        print("Detecting camera orientation and aligning to world frame...")
        frames_data, orientation, global_bounds, R_world = world_align_frames(frames_data)
        show_ground = True
        # Rotate camera-space head directions into world frame and store per frame
        if head_directions_cam:
            for fd in frames_data:
                cam_dir = head_directions_cam.get(fd['frame_idx'])
                if cam_dir is not None:
                    fd['head_direction'] = R_world @ cam_dir

    # Visualize
    if args.mode == 'static':
        visualize_static(frames_data, output_path=args.output,
                         show_face=args.show_face, num_frames=args.num_frames,
                         clean_style=clean_style, global_bounds=global_bounds,
                         show_ground=show_ground)
    elif args.mode == 'animation':
        visualize_animation(frames_data, output_path=args.output,
                            show_face=args.show_face, fps=args.fps,
                            clean_style=clean_style, global_bounds=global_bounds,
                            show_ground=show_ground)


if __name__ == '__main__':
    main()
