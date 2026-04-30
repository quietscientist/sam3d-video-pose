#!/usr/bin/env python3
"""
render_mesh_video.py — renders PLY meshes to video (EGL offscreen).

Two modes:
  overlay  (default): Assembles per-frame overlay PNGs saved by process_video.py
                       into a video. Fast; requires skip_mesh_saving=False.
  smoothed           : Renders the smoothed PLY sequence overlaid on source video
                       frames using pyrender EGL offscreen. Better quality.

Usage:
    python render_mesh_video.py VIDEO_OUTPUT_DIR VIDEO_STEM
    python render_mesh_video.py output/C001_T1_R1_masks C001_T1_R1_masks
    python render_mesh_video.py output/C001_T1_R1_masks C001_T1_R1_masks --mode smoothed
    python render_mesh_video.py output/C001_T1_R1_masks C001_T1_R1_masks \\
        --source-video /path/to/C001_T1_R1_masks.mp4

Arguments:
    video_output_dir  Directory created by process_video.py for this video
    video_stem        Stem of the original video filename (no extension)
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import imageio
import imageio.v3 as iio
import numpy as np
import trimesh

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def _get_sorted_frame_dirs(meshes_dir: str):
    """Return subdirs sorted by frame index."""
    dirs = []
    for d in Path(meshes_dir).iterdir():
        if not d.is_dir():
            continue
        parts = d.name.split("_")
        # Expected: frame_XXXX_obj_Y
        if len(parts) >= 2 and parts[0] == "frame":
            try:
                frame_idx = int(parts[1])
                dirs.append((frame_idx, d))
            except ValueError:
                continue
    return sorted(dirs, key=lambda x: x[0])


def assemble_overlay_video(video_output_dir: str, video_stem: str,
                           output_path: str, fps: float = 15.0) -> str:
    """
    Assemble per-frame overlay PNGs (saved by process_video.py / save_mesh_results)
    into a single MP4 video.
    """
    meshes_dir = os.path.join(video_output_dir, f"{video_stem}_meshes")
    if not os.path.exists(meshes_dir):
        raise FileNotFoundError(f"Meshes directory not found: {meshes_dir}")

    frame_dirs = _get_sorted_frame_dirs(meshes_dir)
    if not frame_dirs:
        raise RuntimeError(f"No frame directories found in {meshes_dir}")

    # Collect overlay PNGs
    overlay_frames = []
    for frame_idx, frame_dir in frame_dirs:
        pngs = sorted(frame_dir.glob("*_overlay_000.png"))
        if pngs:
            overlay_frames.append((frame_idx, str(pngs[0])))

    if not overlay_frames:
        raise RuntimeError(
            f"No overlay PNGs found in {meshes_dir}. "
            "Re-run process_video.py without --skip-mesh-saving."
        )

    print(f"Assembling {len(overlay_frames)} overlay frames → {output_path}")

    writer = imageio.get_writer(output_path, fps=fps, codec="libx264",
                                pixelformat="yuv420p", quality=None,
                                output_params=["-crf", "18"])
    for frame_idx, png_path in overlay_frames:
        frame_bgr = cv2.imread(png_path)
        if frame_bgr is None:
            print(f"  Warning: could not read {png_path}")
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
    writer.close()

    print(f"✓ Overlay video saved: {output_path}")
    return output_path


def render_smoothed_video(video_output_dir: str, video_stem: str,
                          source_video: str | None,
                          output_path: str, fps: float = 15.0) -> str:
    """
    Render smoothed PLY sequence overlaid on source video frames using
    pyrender EGL offscreen rendering.
    """
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    # Set up sam-3d-body path for Renderer import
    sam3d_path = os.path.join(project_root, "external", "sam-3d-body")
    if sam3d_path not in sys.path:
        sys.path.insert(0, sam3d_path)

    from sam3dvideo.utils import patch_sam3
    patch_sam3()

    from sam_3d_body.visualization.renderer import Renderer

    smoothed_dir = Path(video_output_dir) / f"{video_stem}_smoothed" / "meshes"
    meshes_dir = Path(video_output_dir) / f"{video_stem}_meshes"

    if not smoothed_dir.exists():
        raise FileNotFoundError(
            f"Smoothed meshes not found: {smoothed_dir}\n"
            "Run process_video.py with skip_mesh_saving=False to generate them."
        )

    ply_files = sorted(smoothed_dir.glob("smoothed_frame_*.ply"))
    if not ply_files:
        raise RuntimeError(f"No smoothed PLY files in {smoothed_dir}")

    print(f"Found {len(ply_files)} smoothed PLY files")

    # Parse frame indices from PLY filenames (smoothed_frame_XXXX.ply)
    def _ply_frame_idx(p: Path) -> int:
        return int(p.stem.split("_")[-1])

    ply_by_frame = {_ply_frame_idx(p): p for p in ply_files}

    # Load focal_length, cam_t per raw frame, and faces
    focal_length = 5000.0
    faces = None
    cam_t_by_frame: dict[int, np.ndarray] = {}

    if meshes_dir.exists():
        frame_dirs = _get_sorted_frame_dirs(str(meshes_dir))
        for frame_idx, frame_dir in frame_dirs:
            npz_path = frame_dir / "mhr_parameters.npz"
            if npz_path.exists():
                params = np.load(str(npz_path), allow_pickle=True)
                if "focal_length" in params:
                    focal_length = float(params["focal_length"])
                if "pred_cam_t" in params:
                    cam_t_by_frame[frame_idx] = params["pred_cam_t"].ravel()
                if "pred_vertices" in params and faces is None:
                    # Get faces from a raw PLY
                    for ply in frame_dir.glob("*.ply"):
                        m = trimesh.load(str(ply))
                        faces = np.array(m.faces)
                        break

    if faces is None:
        m = trimesh.load(str(ply_files[0]))
        faces = np.array(m.faces)

    print(f"focal_length={focal_length:.1f}, faces={faces.shape}")
    renderer = Renderer(focal_length=focal_length, faces=faces)

    # Map smoothed frame → nearest raw cam_t
    raw_frames = sorted(cam_t_by_frame.keys())

    def _nearest_cam_t(frame_idx: int) -> np.ndarray:
        if not raw_frames:
            return np.array([0.0, 0.0, 3.0])
        nearest = min(raw_frames, key=lambda f: abs(f - frame_idx))
        return cam_t_by_frame[nearest]

    # R_180 matrix (its own inverse)
    R_180 = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1],
    ], dtype=np.float64)

    # Read source video frames (for background)
    source_frames: dict[int, np.ndarray] = {}
    if source_video and os.path.exists(source_video):
        cap = cv2.VideoCapture(source_video)
        fi = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            source_frames[fi] = frame  # BGR
            fi += 1
        cap.release()
        print(f"Loaded {len(source_frames)} source video frames")

    sorted_frame_idxs = sorted(ply_by_frame.keys())
    if not sorted_frame_idxs:
        raise RuntimeError("No frames to render")

    # Determine output resolution from first source frame or first PLY
    if source_frames:
        first_src = source_frames.get(sorted_frame_idxs[0], list(source_frames.values())[0])
        H, W = first_src.shape[:2]
    else:
        H, W = 512, 512

    print(f"Rendering {len(sorted_frame_idxs)} frames at {W}x{H}...")

    writer = imageio.get_writer(output_path, fps=fps, codec="libx264",
                                pixelformat="yuv420p", quality=None,
                                output_params=["-crf", "18"])

    for frame_idx in sorted_frame_idxs:
        ply_path = ply_by_frame[frame_idx]
        mesh = trimesh.load(str(ply_path))
        ply_verts = np.array(mesh.vertices, dtype=np.float64)

        # PLY vertices = R_180 @ (pred_vertices + pred_cam_t)
        # Invert R_180 (R_180 = R_180^-1) → pred_vertices + pred_cam_t
        verts_plus_t = R_180 @ ply_verts.T  # (3, N)
        verts_plus_t = verts_plus_t.T        # (N, 3)

        cam_t = _nearest_cam_t(frame_idx)
        # Recover pred_vertices ≈ verts_plus_t - cam_t
        pred_verts = verts_plus_t - cam_t

        # Get source frame (fallback: black)
        if frame_idx in source_frames:
            bg_bgr = source_frames[frame_idx]
        elif source_frames:
            closest = min(source_frames.keys(), key=lambda f: abs(f - frame_idx))
            bg_bgr = source_frames[closest]
        else:
            bg_bgr = np.zeros((H, W, 3), dtype=np.uint8)

        # Resize background to match desired resolution if needed
        if bg_bgr.shape[:2] != (H, W):
            bg_bgr = cv2.resize(bg_bgr, (W, H))

        rendered = renderer(
            pred_verts.astype(np.float32),
            cam_t.astype(np.float32),
            bg_bgr,
            mesh_base_color=(0.65, 0.74, 0.86),
        )
        frame_out = (rendered * 255).astype(np.uint8)
        # renderer returns BGR-ish; convert to RGB for imageio
        frame_rgb = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)

    writer.close()
    print(f"✓ Smoothed mesh video saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Render PLY meshes to video (EGL offscreen)."
    )
    parser.add_argument("video_output_dir",
                        help="Per-video output directory from process_video.py")
    parser.add_argument("video_stem",
                        help="Stem of the original video filename (no extension)")
    parser.add_argument("--mode", choices=["overlay", "smoothed"], default="overlay",
                        help="overlay: assemble overlay PNGs; smoothed: render PLY sequence")
    parser.add_argument("--source-video", default=None,
                        help="Path to original video (used as background in smoothed mode)")
    parser.add_argument("--fps", type=float, default=15.0,
                        help="Output video FPS (default: 15)")
    parser.add_argument("--output", default=None,
                        help="Output video path (default: VIDEO_OUTPUT_DIR/VIDEO_STEM_mesh_video.mp4)")
    args = parser.parse_args()

    if args.output is None:
        suffix = "overlay" if args.mode == "overlay" else "smoothed"
        args.output = os.path.join(args.video_output_dir,
                                   f"{args.video_stem}_mesh_video_{suffix}.mp4")

    if args.mode == "overlay":
        assemble_overlay_video(args.video_output_dir, args.video_stem,
                               args.output, fps=args.fps)
    else:
        render_smoothed_video(args.video_output_dir, args.video_stem,
                              args.source_video, args.output, fps=args.fps)


if __name__ == "__main__":
    main()
