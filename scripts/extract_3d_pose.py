# sam3d_simple.py
import json
import cv2
import numpy as np
import argparse
from pathlib import Path
from notebook.utils import setup_sam_3d_body
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib
matplotlib.use('Agg')

# Skeleton connections for baby pose
SKELETON = [
    (0, 1), (0, 4),  # head to shoulders
    (4, 5), (5, 6),  # left arm
    (7, 8), (8, 9), (9, 10),  # right arm
    (4, 11), (7, 11),  # shoulders to hips
    (11, 12), (12, 13),  # left leg
    (14, 15), (15, 16),  # right leg
]

def extract_3d_keypoints(video_path, estimator):
    """Extract 3D keypoints from video using SAM 3D Body"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    keypoints_3d_sequence = []
    frame_idx = 0
    
    print("Processing video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            # Run SAM 3D Body (no 2D prompts, just use image)
            outputs = estimator.process_one_image(frame_rgb)
            
            # Extract 3D keypoints only
            joints_3d = outputs['joints_3d']  # Shape: (num_keypoints, 3)
            keypoints_3d_sequence.append(joints_3d)
            
        except Exception as e:
            print(f"Error on frame {frame_idx}: {e}")
            keypoints_3d_sequence.append(None)
        
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  Processed {frame_idx} frames...")
    
    cap.release()
    print(f"Done! Processed {frame_idx} frames total")
    
    return np.array(keypoints_3d_sequence), fps

def plot_3d_skeleton(ax, joints_3d, frame_num, elev=20, azim=45):
    """Plot 3D skeleton on axis"""
    ax.clear()
    
    if joints_3d is None or len(joints_3d) == 0:
        ax.text(0.5, 0.5, 0.5, 'No data', ha='center', va='center')
        return
    
    # Plot keypoints
    xs, ys, zs = joints_3d[:, 0], joints_3d[:, 1], joints_3d[:, 2]
    ax.scatter(xs, ys, zs, c='red', s=50, marker='o')
    
    # Plot skeleton connections
    for i, j in SKELETON:
        if i < len(joints_3d) and j < len(joints_3d):
            pts = joints_3d[[i, j]]
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'b-', linewidth=2)
    
    # Set axis properties
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Pose - Frame {frame_num}', fontsize=14, fontweight='bold')
    
    # Equal aspect ratio
    max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]).max() / 2.0
    mid = np.array([xs.mean(), ys.mean(), zs.mean()])
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    # Set view angle
    ax.view_init(elev=elev, azim=azim)

def create_3d_video(keypoints_3d_sequence, output_path, fps=30, rotate=True):
    """Create video of 3D keypoints"""
    print("Creating 3D visualization video...")
    
    # Setup video writer
    frame_width, frame_height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Create figure
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    num_frames = len(keypoints_3d_sequence)
    
    for i, joints_3d in enumerate(keypoints_3d_sequence):
        # Optionally rotate view
        azim = 45 + (i * 360 / num_frames) if rotate else 45
        
        plot_3d_skeleton(ax, joints_3d, i, elev=20, azim=azim)
        
        # Convert plot to image
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.frombuffer(buf, dtype=np.uint8).reshape(frame_height, frame_width, 4)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
        out.write(img_bgr)
        
        if (i + 1) % 30 == 0:
            print(f"  Rendered {i + 1}/{num_frames} frames...")
    
    out.release()
    plt.close(fig)
    print(f"Video saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Extract 3D keypoints from video using SAM 3D Body')
    parser.add_argument('--video', type=str, required=True, help='Input video path')
    parser.add_argument('--output_keypoints', type=str, default='keypoints_3d.npy',
                       help='Output path for 3D keypoints (.npy)')
    parser.add_argument('--output_video', type=str, default='pose_3d_visualization.mp4',
                       help='Output path for 3D visualization video')
    parser.add_argument('--no_rotate', action='store_true', 
                       help='Disable camera rotation in visualization')
    args = parser.parse_args()
    
    # Load SAM 3D Body model
    print("Loading SAM 3D Body model...")
    estimator = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3")
    print("Model loaded!\n")
    
    # Extract 3D keypoints from video
    keypoints_3d, fps = extract_3d_keypoints(args.video, estimator)
    
    # Save keypoints
    np.save(args.output_keypoints, keypoints_3d)
    print(f"\nKeypoints saved: {args.output_keypoints}")
    print(f"Shape: {keypoints_3d.shape}")
    
    # Create 3D visualization video
    create_3d_video(keypoints_3d, args.output_video, fps=fps, rotate=not args.no_rotate)
    
    print("\n✓ Done!")

if __name__ == '__main__':
    main()