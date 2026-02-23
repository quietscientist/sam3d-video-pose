#!/usr/bin/env python3
"""
Debug script: inspect camera orientation from MHR parameters and 3D keypoint geometry.
Goal: detect top-down vs front-on camera, orient output so "floor" is at bottom.

Produces 3 debug images:
  1. 4 frames in original camera coords (3/4 view)
  2. Same 4 frames after world-alignment rotation (3/4 view)
  3. Same 4 frames world-aligned, viewed from the side (Z=up on screen)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
import csv

# ===================================================================
# Constants
# ===================================================================

COCO_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

BODY_INDICES = list(range(5, 17))  # exclude face

BONE_PAIRS = [
    (5, 7), (7, 9),      # left arm
    (6, 8), (8, 10),     # right arm
    (5, 6),              # shoulders
    (11, 12),            # hips
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]

# Left-side bones (warm colors) vs right-side (cool colors)
LEFT_BONES = [(5, 7), (7, 9), (11, 13), (13, 15)]
RIGHT_BONES = [(6, 8), (8, 10), (12, 14), (14, 16)]

# ===================================================================
# Data loading
# ===================================================================

def load_raw_csv(csv_path):
    """Load long-format CSV into dict of {frame: (17,3) array}."""
    frames = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            fi = int(row['frame'])
            pi = int(row['part_idx'])
            if fi not in frames:
                frames[fi] = np.zeros((17, 3))
            frames[fi][pi] = [float(row['x']), float(row['y']), float(row['z'])]
    return frames

# ===================================================================
# Body geometry analysis
# ===================================================================

def compute_body_frame(kp):
    """
    From COCO keypoints, compute body-frame vectors.

    Convention: cross(spine, shoulders) yields the BACK-facing direction
    (by right-hand rule: fingers along spine toward head, curl toward left
    shoulder, thumb points toward the back). We negate to get chest-facing.

    Returns:
        spine_n: unit vector from pelvis -> sternum (body "head" direction)
        chest_n: unit vector perpendicular to torso plane (chest-facing, away from back)
        shoulder_n: unit vector right_shoulder -> left_shoulder
    """
    sternum = (kp[5] + kp[6]) / 2
    pelvis = (kp[11] + kp[12]) / 2

    spine = sternum - pelvis
    spine_n = spine / (np.linalg.norm(spine) + 1e-8)

    shoulders = kp[5] - kp[6]  # right_shoulder -> left_shoulder
    shoulders_n = shoulders / (np.linalg.norm(shoulders) + 1e-8)

    back_normal = np.cross(spine_n, shoulders_n)
    chest_n = -back_normal  # flip: back -> chest
    chest_n = chest_n / (np.linalg.norm(chest_n) + 1e-8)

    return spine_n, chest_n, shoulders_n


def detect_camera_orientation(all_frames):
    """
    Analyze body geometry across frames to detect camera viewing angle.

    MHR camera coords: X=right, Y=down, Z=forward (into screen).

    Returns:
        orientation: str - 'top-down', 'front-on-standing', 'front-on-lying', 'oblique'
        gravity_up: (3,) unit vector pointing "up" (away from floor) in camera coords
        stats: dict of diagnostic values
    """
    spine_dots_z = []
    spine_dots_y = []
    normal_dots_z = []
    normal_dots_y = []

    for fi in sorted(all_frames.keys()):
        kp = all_frames[fi]
        spine_n, normal_n, _ = compute_body_frame(kp)

        spine_dots_z.append(np.dot(spine_n, [0, 0, 1]))
        spine_dots_y.append(np.dot(spine_n, [0, 1, 0]))
        normal_dots_z.append(np.dot(normal_n, [0, 0, 1]))
        normal_dots_y.append(np.dot(normal_n, [0, 1, 0]))

    stats = {
        'spine_dot_camZ': np.mean(spine_dots_z),
        'spine_dot_camY': np.mean(spine_dots_y),
        'normal_dot_camZ': np.mean(normal_dots_z),
        'normal_dot_camY': np.mean(normal_dots_y),
        'n_frames': len(all_frames),
    }

    abs_sz = abs(stats['spine_dot_camZ'])
    abs_sy = abs(stats['spine_dot_camY'])
    abs_nz = abs(stats['normal_dot_camZ'])
    abs_ny = abs(stats['normal_dot_camY'])

    # Decision logic using chest_n (chest-facing normal, away from back):
    #
    # SPINE aligned with cam_Z (into screen):
    #   -> camera is looking along the body axis = top-of-head or feet view
    #
    # SPINE aligned with cam_Y (image vertical):
    #   -> person is standing/upright in frame
    #   -> gravity_up = -cam_Y (image up)
    #
    # CHEST NORMAL aligned with cam_Z (negative = toward camera):
    #   -> For supine baby: chest faces camera, so chest_n points toward
    #      camera (-Z). gravity_up = chest_n (up = away from floor = chest)
    #   -> For prone: chest faces away from camera (+Z), gravity_up = -chest_n
    #
    # Key insight for infant-in-crib: spine is ~horizontal in camera X,
    # chest normal points toward camera (-Z). Camera is above looking down
    # at baby lying on back. gravity_up = chest_n (toward camera = real up).

    if abs_sz > 0.7:
        orientation = 'top-down-axial'
        gravity_up = np.median([compute_body_frame(all_frames[fi])[1]
                                for fi in sorted(all_frames.keys())], axis=0)
    elif abs_sy > 0.6:
        orientation = 'front-on-standing'
        gravity_up = np.array([0.0, -1.0, 0.0])
    elif abs_nz > 0.5:
        if abs_sy < 0.3:
            orientation = 'top-down-supine'
        else:
            orientation = 'front-on-standing'
        # chest_n is the "up" direction (away from floor/back)
        gravity_up = np.median([compute_body_frame(all_frames[fi])[1]
                                for fi in sorted(all_frames.keys())], axis=0)
        # For supine: chest faces toward camera (-Z), so chest_n has -Z component.
        # Ensure gravity_up points toward camera (negative Z = toward camera = real up)
        if gravity_up[2] > 0:
            gravity_up = -gravity_up
    else:
        orientation = 'oblique'
        gravity_up = np.median([compute_body_frame(all_frames[fi])[1]
                                for fi in sorted(all_frames.keys())], axis=0)
        if gravity_up[2] > 0:
            gravity_up = -gravity_up

    gravity_up = gravity_up / (np.linalg.norm(gravity_up) + 1e-8)

    return orientation, gravity_up, stats


def build_world_rotation(gravity_up, spine_hint=None):
    """
    Build a 3x3 rotation matrix that maps:
        gravity_up  -> +Z  (up in plot)
        spine_hint  -> +X  (horizontal, head-right in plot)

    This puts the floor at z=min and the body horizontal or vertical
    depending on the original pose.
    """
    target_up = np.array([0.0, 0.0, 1.0])

    # Step 1: rotate gravity_up to +Z
    rot_axis = np.cross(gravity_up, target_up)
    rot_axis_norm = np.linalg.norm(rot_axis)

    if rot_axis_norm < 1e-6:
        if np.dot(gravity_up, target_up) > 0:
            R = np.eye(3)
        else:
            R = np.diag([1.0, -1.0, -1.0])
    else:
        rot_axis /= rot_axis_norm
        cos_a = np.clip(np.dot(gravity_up, target_up), -1, 1)
        sin_a = np.sqrt(1 - cos_a**2)
        K = np.array([
            [0, -rot_axis[2], rot_axis[1]],
            [rot_axis[2], 0, -rot_axis[0]],
            [-rot_axis[1], rot_axis[0], 0]
        ])
        R = np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)

    # Step 2 (optional): rotate around Z so spine aligns with +X
    if spine_hint is not None:
        spine_rotated = R @ spine_hint
        # Project onto XY plane
        sx, sy = spine_rotated[0], spine_rotated[1]
        if abs(sx) + abs(sy) > 1e-6:
            angle_z = -np.arctan2(sy, sx)  # rotate to align with +X
            cz, sz_ = np.cos(angle_z), np.sin(angle_z)
            Rz = np.array([[cz, -sz_, 0], [sz_, cz, 0], [0, 0, 1]])
            R = Rz @ R

    return R


# ===================================================================
# Plotting
# ===================================================================

def compute_ground_z_offset(kp, orientation):
    """
    Compute the Z offset to apply so that the ground contact sits at Z=0.

    After rotating into world frame, this returns the Z value of the "floor"
    which we subtract from all keypoints to place the ground at Z=0.

    Args:
        kp: (17, 3) keypoints already in world frame (after R_world rotation)
        orientation: detection string

    Returns:
        ground_z: the Z value to subtract (body shifts so floor = 0)
    """
    if orientation == 'front-on-standing':
        # Floor = feet. Use min Z of ankles (indices 15, 16)
        ankles = kp[[15, 16], 2]
        return np.min(ankles)
    elif orientation in ('top-down-supine', 'top-down-axial'):
        # Floor = back of torso. Use min Z of shoulder + hip keypoints
        torso = kp[[5, 6, 11, 12], 2]
        return np.min(torso)
    else:
        # Oblique / flying: no real contact. Floor = below everything
        body_kp = kp[BODY_INDICES]
        body_height = np.ptp(body_kp[:, 2])
        return np.min(body_kp[:, 2]) - 0.05 * body_height


def plot_skeleton_with_ground(ax, kp, title="", R=None, orientation='oblique'):
    """Plot skeleton with ground plane at Z=0. If R given, rotate keypoints first."""
    kp = kp.copy()
    if R is not None:
        kp = (R @ kp.T).T

    # Center XY on torso midpoint, set Z so ground = 0
    sternum = (kp[5] + kp[6]) / 2
    pelvis = (kp[11] + kp[12]) / 2
    torso_center_xy = (sternum[:2] + pelvis[:2]) / 2
    ground_z = compute_ground_z_offset(kp, orientation)

    kp[:, 0] -= torso_center_xy[0]
    kp[:, 1] -= torso_center_xy[1]
    kp[:, 2] -= ground_z  # shift so floor = 0

    body_kp = kp[BODY_INDICES]
    sternum = (kp[5] + kp[6]) / 2
    pelvis = (kp[11] + kp[12]) / 2

    # Plot bones with left/right coloring
    for i, j in BONE_PAIRS:
        if (i, j) in LEFT_BONES:
            color = '#E74C3C'  # red
        elif (i, j) in RIGHT_BONES:
            color = '#3498DB'  # blue
        else:
            color = '#2C3E50'  # dark
        ax.plot([kp[i, 0], kp[j, 0]],
                [kp[i, 1], kp[j, 1]],
                [kp[i, 2], kp[j, 2]],
                c=color, linewidth=2.5, zorder=5)

    # Sternum-pelvis spine
    ax.plot([sternum[0], pelvis[0]],
            [sternum[1], pelvis[1]],
            [sternum[2], pelvis[2]],
            c='#2C3E50', linewidth=2, linestyle='--', zorder=5)

    # Plot joints
    ax.scatter(body_kp[:, 0], body_kp[:, 1], body_kp[:, 2],
               c='black', s=30, zorder=10)
    ax.scatter(*sternum, c='#E74C3C', s=40, zorder=11, marker='D', label='sternum')
    ax.scatter(*pelvis, c='#3498DB', s=40, zorder=11, marker='D', label='pelvis')

    # Ground plane at Z=0
    all_pts = np.vstack([body_kp, [sternum], [pelvis]])
    center_xy = np.mean(all_pts[:, :2], axis=0)
    extent = np.max(np.ptp(all_pts, axis=0)) * 0.7

    xx = [center_xy[0] - extent, center_xy[0] + extent,
          center_xy[0] + extent, center_xy[0] - extent]
    yy = [center_xy[1] - extent, center_xy[1] - extent,
          center_xy[1] + extent, center_xy[1] + extent]
    zz = [0.0] * 4  # ground is always at Z=0
    verts = [list(zip(xx, yy, zz))]
    ground = Poly3DCollection(verts, alpha=0.12, facecolor='#27AE60',
                               edgecolor='#27AE60', linewidth=0.5)
    ax.add_collection3d(ground)

    # Equal aspect ratio, ensuring ground plane is visible
    max_range = extent * 1.2
    cx, cy = np.mean(all_pts[:, :2], axis=0)
    z_lo = min(0.0, all_pts[:, 2].min()) - 0.05 * max_range
    z_hi = max(0.0, all_pts[:, 2].max()) + 0.05 * max_range
    z_center = (z_lo + z_hi) / 2
    z_half = max(max_range, (z_hi - z_lo) / 2)

    ax.set_xlim(cx - max_range, cx + max_range)
    ax.set_ylim(cy - max_range, cy + max_range)
    ax.set_zlim(z_center - z_half, z_center + z_half)

    ax.set_xlabel('X', fontsize=8)
    ax.set_ylabel('Y', fontsize=8)
    ax.set_zlabel('Z (up)', fontsize=8)
    ax.set_title(title, fontsize=9, pad=0)
    ax.tick_params(labelsize=6)


# ===================================================================
# Main
# ===================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Debug camera orientation detection")
    parser.add_argument('csv_path', nargs='?',
                        default="output/Infant_babbling_in_crib/Infant_babbling_in_crib_3D_raw.csv",
                        help="Path to _3D_raw.csv file")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    output_dir = csv_path.parent
    label = csv_path.stem.replace('_3D_raw', '')

    print(f"Loading keypoints from {csv_path}...")
    all_frames = load_raw_csv(csv_path)
    print(f"  {len(all_frames)} frames")

    # --- Detect camera orientation ---
    print("\nDetecting camera orientation...")
    orientation, gravity_up, stats = detect_camera_orientation(all_frames)

    print(f"\n  Camera stats (averaged over {stats['n_frames']} frames):")
    print(f"    spine . cam_Z = {stats['spine_dot_camZ']:+.3f}  (|1| = looking along body axis)")
    print(f"    spine . cam_Y = {stats['spine_dot_camY']:+.3f}  (|-1| = standing upright in frame)")
    print(f"    normal . cam_Z = {stats['normal_dot_camZ']:+.3f}  (|1| = chest faces camera)")
    print(f"    normal . cam_Y = {stats['normal_dot_camY']:+.3f}  (|1| = chest faces down in frame)")
    print(f"\n  >>> Detected: {orientation}")
    print(f"  >>> gravity_up in camera coords: ({gravity_up[0]:+.3f}, {gravity_up[1]:+.3f}, {gravity_up[2]:+.3f})")

    # --- Build world rotation ---
    median_spine = np.median([compute_body_frame(all_frames[fi])[0]
                              for fi in sorted(all_frames.keys())], axis=0)
    median_spine /= np.linalg.norm(median_spine)

    R_world = build_world_rotation(gravity_up, spine_hint=median_spine)

    # Verify
    check_up = R_world @ gravity_up
    check_spine = R_world @ median_spine
    print(f"\n  After rotation:")
    print(f"    gravity_up  -> ({check_up[0]:+.3f}, {check_up[1]:+.3f}, {check_up[2]:+.3f})  (should be ~[0,0,1])")
    print(f"    median_spine -> ({check_spine[0]:+.3f}, {check_spine[1]:+.3f}, {check_spine[2]:+.3f})  (should be ~[1,0,0])")

    # --- Debug plots ---
    sorted_frames = sorted(all_frames.keys())
    n = len(sorted_frames)
    debug_frames = [sorted_frames[int(i * (n - 1) / 3)] for i in range(4)]
    print(f"\n  Debug frames: {debug_frames}")

    # Helper to render a row of 4 frames
    def render_figure(title, figpath, elev, azim, R=None, orient='oblique'):
        fig = plt.figure(figsize=(16, 4.5))
        fig.suptitle(title, fontsize=12, y=1.02)
        for i, fi in enumerate(debug_frames):
            if fi not in all_frames:
                continue
            ax = fig.add_subplot(1, 4, i + 1, projection='3d')
            plot_skeleton_with_ground(ax, all_frames[fi], title=f"Frame {fi}",
                                      R=R, orientation=orient)
            ax.view_init(elev=elev, azim=azim)
        plt.tight_layout()
        fig.savefig(figpath, dpi=150, bbox_inches='tight')
        print(f"Saved: {figpath}")
        plt.close(fig)

    render_figure(
        f"{label}: Original camera coordinates  [detected: {orientation}]",
        output_dir / "debug_01_camera_coords.png",
        elev=25, azim=-60, R=None, orient='oblique')

    render_figure(
        f"{label}: World-aligned ({orientation})  —  3/4 view",
        output_dir / "debug_02_world_aligned_3quarter.png",
        elev=25, azim=-60, R=R_world, orient=orientation)

    render_figure(
        f"{label}: World-aligned  —  side view (floor=Z=0, Z=up)",
        output_dir / "debug_03_world_aligned_side.png",
        elev=0, azim=-90, R=R_world, orient=orientation)

    render_figure(
        f"{label}: World-aligned  —  top view (looking down from above)",
        output_dir / "debug_04_world_aligned_top.png",
        elev=90, azim=0, R=R_world, orient=orientation)

    print("\nDone!")


if __name__ == '__main__':
    main()
