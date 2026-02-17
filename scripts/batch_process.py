#!/usr/bin/env python3
"""
Multi-GPU Video Processing Pipeline for SAM 3D Body
Processes multiple videos in parallel across available GPUs
"""

import sys
import os
import argparse
from pathlib import Path
import multiprocessing

# CRITICAL: Set multiprocessing start method to 'spawn' for CUDA compatibility
# Must be done before any other imports that might use multiprocessing
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

# Add sam-3d-body directory to path for utils import
script_dir = os.path.dirname(os.path.abspath(__file__))
sam_3d_body_dir = os.path.join(script_dir, 'sam-3d-body/notebook')

sys.path.insert(0, sam_3d_body_dir)
#utils located in sam-3d-body/notebook/utils.py

from sam3dvideo.utils import (
    setup_sam_3d_body, setup_visualizer,
    visualize_2d_results, visualize_3d_mesh, save_mesh_results,
    display_results_grid, process_image_with_mask
)

import cv2
import json
import numpy as np
from tqdm import tqdm
import gc
import torch
from contextlib import redirect_stdout, redirect_stderr
import time
import psutil
import threading
from multiprocessing import Process, Manager, Queue
import glob
from dotenv import load_dotenv
from scipy.optimize import linear_sum_assignment

# --- OKS Tracking Helper Functions ---

# COCO keypoint sigmas for OKS calculation
# These define the falloff for each keypoint type
COCO_KEYPOINT_SIGMAS = np.array([
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062,
    0.107, 0.107, 0.087, 0.087, 0.089, 0.089
]) * 2  # 17 COCO keypoints

def compute_oks(keypoints1, keypoints2, area, sigmas=None):
    """
    Compute Object Keypoint Similarity between two sets of keypoints.
    
    Args:
        keypoints1, keypoints2: Arrays of shape (n_keypoints, 3) where each row is (x, y, visibility)
        area: Bounding box area (used for scale normalization)
        sigmas: Per-keypoint falloff constants (defaults to COCO sigmas)
    
    Returns:
        OKS score between 0 and 1 (higher is better match)
    """
    if sigmas is None:
        sigmas = COCO_KEYPOINT_SIGMAS
    
    # Handle different keypoint formats
    if not isinstance(keypoints1, np.ndarray):
        keypoints1 = np.array(keypoints1)
    if not isinstance(keypoints2, np.ndarray):
        keypoints2 = np.array(keypoints2)
    
    # Ensure we have at least x, y coordinates
    if keypoints1.ndim == 1:
        keypoints1 = keypoints1.reshape(-1, 3 if len(keypoints1) % 3 == 0 else 2)
    if keypoints2.ndim == 1:
        keypoints2 = keypoints2.reshape(-1, 3 if len(keypoints2) % 3 == 0 else 2)
    
    # Adjust sigmas if we have different number of keypoints
    n_kpts = min(len(keypoints1), len(keypoints2))
    if n_kpts != len(sigmas):
        # Use default sigma if mismatch
        sigmas = np.ones(n_kpts) * 0.05
    
    sigmas = sigmas[:n_kpts]
    keypoints1 = keypoints1[:n_kpts]
    keypoints2 = keypoints2[:n_kpts]
    
    # Extract coordinates
    x1, y1 = keypoints1[:, 0], keypoints1[:, 1]
    x2, y2 = keypoints2[:, 0], keypoints2[:, 1]
    
    # Extract visibility if available (1 = visible, 0 = not visible)
    if keypoints1.shape[1] >= 3 and keypoints2.shape[1] >= 3:
        v1 = keypoints1[:, 2] > 0
        v2 = keypoints2[:, 2] > 0
        visible = v1 & v2  # Both must be visible
    else:
        # Assume all visible if no visibility info
        visible = np.ones(n_kpts, dtype=bool)
    
    # Compute squared distances
    dx = x1 - x2
    dy = y1 - y2
    d_squared = dx**2 + dy**2
    
    # Compute OKS per keypoint
    # OKS_i = exp(-d_i^2 / (2 * area * sigma_i^2))
    s = area if area > 0 else 1.0  # Prevent division by zero
    variance = (sigmas * 2) ** 2
    oks_per_kpt = np.exp(-d_squared / (2 * s * variance))
    
    # Only count visible keypoints
    if visible.sum() == 0:
        return 0.0
    
    # Average OKS over visible keypoints
    oks = oks_per_kpt[visible].sum() / visible.sum()
    
    return oks

def extract_keypoints_and_bbox(instance):
    """Extract keypoints and bbox from detection instance"""
    keypoints = None
    bbox = None
    area = 1.0
    
    # Extract bbox
    if 'bbox' in instance:
        bbox = instance['bbox']
        if isinstance(bbox, list) and len(bbox) > 0:
            if isinstance(bbox[0], list):
                bbox = bbox[0]
            if len(bbox) >= 4:
                # Calculate area for OKS normalization
                area = bbox[2] * bbox[3] if len(bbox) == 4 else (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                area = max(area, 1.0)  # Prevent zero area
    
    # Extract keypoints (check common field names)
    for key in ['keypoints', 'keypoint', 'pose', 'skeleton']:
        if key in instance:
            keypoints = instance[key]
            break
    
    return keypoints, bbox, area

def match_detections(prev_tracks, curr_detections, oks_threshold=0.5):
    """
    Match current detections to previous tracks using OKS and Hungarian algorithm.
    
    Args:
        prev_tracks: dict of track_id -> {'keypoints': array, 'bbox': array, 'area': float}
        curr_detections: list of detection instances
        oks_threshold: Minimum OKS score to accept a match
    
    Returns:
        dict mapping curr_detection_idx -> track_id
    """
    if not prev_tracks or not curr_detections:
        return {}
    
    # Build cost matrix (1 - OKS, lower is better)
    n_tracks = len(prev_tracks)
    n_detections = len(curr_detections)
    cost_matrix = np.ones((n_tracks, n_detections))
    
    track_ids = list(prev_tracks.keys())
    
    for i, track_id in enumerate(track_ids):
        track_data = prev_tracks[track_id]
        track_kpts = track_data.get('keypoints')
        track_area = track_data.get('area', 1.0)
        
        if track_kpts is None:
            continue
        
        for j, det in enumerate(curr_detections):
            det_kpts, det_bbox, det_area = extract_keypoints_and_bbox(det)
            
            if det_kpts is not None:
                # Use average area for scale normalization
                avg_area = (track_area + det_area) / 2
                oks = compute_oks(track_kpts, det_kpts, avg_area)
                cost_matrix[i, j] = 1 - oks
    
    # Hungarian algorithm for optimal assignment
    track_indices, det_indices = linear_sum_assignment(cost_matrix)
    
    # Build mapping, only accept matches above threshold
    matches = {}
    
    for t_idx, d_idx in zip(track_indices, det_indices):
        oks = 1 - cost_matrix[t_idx, d_idx]
        if oks >= oks_threshold:
            track_id = track_ids[t_idx]
            matches[d_idx] = track_id
    
    return matches

# --- Configuration Loading ---
def load_config(config_path='config.ini'):
    """Load configuration from INI file"""
    import configparser
    config = configparser.ConfigParser()
    
    # Create default config if it doesn't exist
    if not os.path.exists(config_path):
        config['DEFAULT'] = {
            'input_video_dir': '/workspaces/sam-3d-infant/data/videos/pickleball/',
            'output_dir': '/workspaces/sam-3d-infant/output/',
            'detection_json_dir': '',  # Empty = skip detections
            'num_gpus': '2',
            'process_every_n_frames': '1',
            'clear_cache_every': '10',
            'hf_repo_id': 'facebook/sam-3d-body-vith'
        }
        
        with open(config_path, 'w') as f:
            config.write(f)
        
        print(f"Created default config at {config_path}")
    else:
        config.read(config_path)
    
    return config

def load_environment():
    """Load environment variables from .env file"""
    # Try to load .env from current directory or parent
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    env_paths = ['.env', '../.env', os.path.join(parent_dir, '.env')]
    
    env_loaded = False
    for env_path in env_paths:
        if os.path.exists(env_path):
            load_dotenv(env_path)
            print(f"Loaded environment from: {env_path}")
            env_loaded = True
            break
    
    if not env_loaded:
        # Create example .env file
        example_env = """# HuggingFace Token (required for some models)
HF_TOKEN=your_huggingface_token_here

# Optional: Override config settings
# INPUT_VIDEO_DIR=/path/to/videos
# OUTPUT_DIR=/path/to/output
# DETECTION_JSON_DIR=/path/to/detections
# NUM_GPUS=2
"""
        with open('.env.example', 'w') as f:
            f.write(example_env)
        
        print("No .env file found. Created .env.example")
        print("Copy .env.example to .env and add your HuggingFace token")
    
    return env_loaded

def get_hf_token():
    """Get HuggingFace token from environment"""
    token = os.environ.get('HF_TOKEN', '').strip()
    
    if token:
        os.environ['HF_TOKEN'] = token
        return token
    
    return None

# --- Video Quality Checks ---
def validate_video(video_path):
    """Validate video file before processing"""
    assert os.path.exists(video_path), f"Video file does not exist: {video_path}"
    assert os.path.getsize(video_path) > 0, f"Video file is empty: {video_path}"
    
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Cannot open video file: {video_path}"
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Quality assertions
    assert fps > 0, f"Invalid FPS: {fps}"
    assert width > 0 and height > 0, f"Invalid resolution: {width}x{height}"
    assert frame_count > 0, f"Video has no frames: {frame_count}"
    assert width >= 224 and height >= 224, f"Resolution too low (minimum 224x224): {width}x{height}"
    assert fps >= 1, f"FPS too low (minimum 1): {fps}"
    
    # Test read first frame
    ret, frame = cap.read()
    assert ret, "Cannot read first frame from video"
    assert frame is not None, "First frame is None"
    assert frame.size > 0, "First frame is empty"
    
    cap.release()
    
    return {
        'fps': fps,
        'width': width,
        'height': height,
        'frame_count': frame_count,
        'valid': True
    }

def validate_json(json_path):
    """Validate detection JSON file"""
    if not json_path or not os.path.exists(json_path):
        return 0
    
    assert os.path.getsize(json_path) > 0, f"JSON file is empty: {json_path}"
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    assert isinstance(data, list), "JSON should contain a list"
    assert len(data) > 0, "JSON contains no frames"
    
    return len(data)

# --- Resource Monitoring ---
class ResourceMonitor:
    def __init__(self, gpu_id=None):
        self.gpu_id = gpu_id
        self.cpu_usage = []
        self.gpu_usage = []
        self.gpu_memory = []
        self.timestamps = []
        self.monitoring = False
        self.monitor_thread = None
        
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            self.timestamps.append(time.time())
            self.cpu_usage.append(psutil.cpu_percent(interval=0.1))
            
            if torch.cuda.is_available() and self.gpu_id is not None:
                try:
                    import pynvml
                    handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    self.gpu_usage.append(util.gpu)
                    self.gpu_memory.append(mem_info.used / 1024**3)
                except:
                    self.gpu_usage.append(0)
                    mem = torch.cuda.memory_allocated(self.gpu_id) / 1024**3
                    self.gpu_memory.append(mem)
            else:
                self.gpu_usage.append(0)
                self.gpu_memory.append(0)
            
            time.sleep(1)
    
    def start(self):
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop(self):
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
    
    def get_stats(self):
        if not self.cpu_usage:
            return {
                'cpu_avg': 0, 'cpu_max': 0,
                'gpu_avg': 0, 'gpu_max': 0,
                'gpu_mem_avg': 0, 'gpu_mem_max': 0
            }
        return {
            'cpu_avg': np.mean(self.cpu_usage),
            'cpu_max': np.max(self.cpu_usage),
            'gpu_avg': np.mean(self.gpu_usage) if self.gpu_usage else 0,
            'gpu_max': np.max(self.gpu_usage) if self.gpu_usage else 0,
            'gpu_mem_avg': np.mean(self.gpu_memory) if self.gpu_memory else 0,
            'gpu_mem_max': np.max(self.gpu_memory) if self.gpu_memory else 0,
        }

# --- Video Processing Function ---
def process_video_on_gpu(video_path, json_path, output_path, gpu_id, hf_repo_id, 
                        process_every_n, clear_cache_every, stats_queue):
    """Process a single video on a specific GPU"""
    
    try:
        # Set GPU device
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        torch.cuda.set_device(0)
        device = torch.device('cuda:0')
        
        print(f"[GPU {gpu_id}] Initialized on device: {torch.cuda.get_device_name(0)}")
        print(f"[GPU {gpu_id}] Processing: {Path(video_path).name}")
        
        # Initialize pynvml for monitoring
        try:
            import pynvml
            pynvml.nvmlInit()
        except:
            pass
        
        # Initialize monitor
        monitor = ResourceMonitor(gpu_id)
        monitor.start()
        
        # Validate video
        print(f"[GPU {gpu_id}] Validating video...")
        video_info = validate_video(video_path)
        print(f"[GPU {gpu_id}] Video OK - {video_info['width']}x{video_info['height']} @ {video_info['fps']:.2f}fps, {video_info['frame_count']} frames")
        
        # Check if using detections
        use_detections = json_path and os.path.exists(json_path)
        det_data = []
        
        if use_detections:
            print(f"[GPU {gpu_id}] Validating detections...")
            json_frames = validate_json(json_path)
            print(f"[GPU {gpu_id}] JSON OK - {json_frames} frames")
            
            # Load detection data
            load_start = time.time()
            with open(json_path, "r") as f:
                det_data = json.load(f)
            load_time = time.time() - load_start
        else:
            print(f"[GPU {gpu_id}] No detections provided - processing all frames")
            load_time = 0
        
        # Load models
        start_time = time.time()
        print(f"[GPU {gpu_id}] Loading SAM 3D Body model...")
        estimator = setup_sam_3d_body(hf_repo_id=hf_repo_id)
        
        if hasattr(estimator, 'to'):
            estimator = estimator.to(device)
        
        print(f"[GPU {gpu_id}] Loading visualizer...")
        visualizer = setup_visualizer()
        
        model_load_time = time.time() - start_time
        print(f"[GPU {gpu_id}] Models loaded in {model_load_time:.2f}s")
        print(f"[GPU {gpu_id}] GPU Memory allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise RuntimeError(f"Could not open video writer for {output_path}")
        
        # Processing variables
        processed_count = 0
        skipped_count = 0
        error_count = 0
        total_detections = 0
        frame_idx = 0
        frame_times = []
        mesh_gen_times = []
        
        # OKS-based tracking variables (only if using detections)
        tracks = {}
        next_track_id = 0
        track_colors = {}
        
        # Temp directory
        temp_dir = f"/tmp/mesh_temp_gpu{gpu_id}_{os.getpid()}"
        os.makedirs(temp_dir, exist_ok=True)
        
        processing_start = time.time()
        
        # Process frames
        pbar = tqdm(total=total_frames, desc=f"GPU {gpu_id}", position=gpu_id, leave=True)
        
        while True:
            frame_start = time.time()
            
            ret, frame_bgr = cap.read()
            if not ret:
                break
            
            # Skip frames if configured
            if frame_idx % process_every_n != 0:
                out.write(frame_bgr)
                frame_idx += 1
                pbar.update(1)
                continue
            
            # Determine if we should process this frame
            should_process = True
            instances = []
            
            if use_detections:
                # Check detection data
                has_detection = frame_idx < len(det_data) and det_data[frame_idx] is not None
                has_instances = has_detection and det_data[frame_idx].get('instances', [])
                
                if has_detection and has_instances:
                    instances = det_data[frame_idx]['instances']
                    total_detections += len(instances)
                else:
                    should_process = False
            
            if should_process:
                try:
                    # If using detections with tracking
                    if use_detections and instances:
                        # Match current detections to existing tracks using OKS
                        matches = match_detections(tracks, instances, oks_threshold=0.5)
                        
                        # Update tracks and assign new IDs
                        new_tracks = {}
                        detection_track_ids = {}
                        
                        for det_idx, instance in enumerate(instances):
                            if det_idx in matches:
                                track_id = matches[det_idx]
                            else:
                                track_id = next_track_id
                                next_track_id += 1
                                np.random.seed(track_id)
                                track_colors[track_id] = tuple(np.random.randint(50, 255, 3).tolist())
                            
                            detection_track_ids[det_idx] = track_id
                            
                            keypoints, bbox, area = extract_keypoints_and_bbox(instance)
                            new_tracks[track_id] = {
                                'keypoints': keypoints,
                                'bbox': bbox,
                                'area': area
                            }
                        
                        tracks = new_tracks
                        
                        # Process ALL instances
                        composite_frame = frame_bgr.copy()
                        mesh_rendered_count = 0
                        
                        sorted_detections = sorted(enumerate(instances), 
                                                 key=lambda x: detection_track_ids.get(x[0], 999))
                        
                        for det_idx, instance in sorted_detections:
                            track_id = detection_track_ids[det_idx]
                            
                            temp_path = f"{temp_dir}/frame_{frame_idx}_track_{track_id}.png"
                            cv2.imwrite(temp_path, frame_bgr)
                            
                            with open(os.devnull, 'w') as devnull:
                                with redirect_stdout(devnull), redirect_stderr(devnull):
                                    outputs = estimator.process_one_image(temp_path)
                            
                            os.remove(temp_path)
                            
                            if outputs:
                                mesh_start = time.time()
                                mesh_results = visualize_3d_mesh(frame_bgr, outputs, estimator.faces)
                                mesh_time = time.time() - mesh_start
                                mesh_gen_times.append(mesh_time)
                                
                                if mesh_results:
                                    combined_img = mesh_results[0]
                                    quarter_width = combined_img.shape[1] // 4
                                    mesh_overlay = combined_img[:, quarter_width:2*quarter_width, :]
                                    
                                    if mesh_overlay.shape[:2] != (height, width):
                                        mesh_overlay = cv2.resize(mesh_overlay, (width, height))
                                    
                                    mask = np.any(mesh_overlay > 10, axis=2)
                                    composite_frame[mask] = mesh_overlay[mask]
                                    
                                    mesh_rendered_count += 1
                                    
                                    del mesh_results, combined_img, mesh_overlay, mask
                                
                                del outputs
                        
                        if mesh_rendered_count > 0:
                            out.write(composite_frame)
                            processed_count += 1
                        else:
                            out.write(frame_bgr)
                            skipped_count += 1
                        
                        del composite_frame
                    
                    else:
                        # No detections JSON - process entire frame and extract all people from SAM output
                        temp_path = f"{temp_dir}/frame_{frame_idx}.png"
                        cv2.imwrite(temp_path, frame_bgr)
                        
                        with open(os.devnull, 'w') as devnull:
                            with redirect_stdout(devnull), redirect_stderr(devnull):
                                outputs = estimator.process_one_image(temp_path)
                        
                        os.remove(temp_path)
                        
                        if outputs is not None:
                            # outputs is a list of dicts, one per detected person
                            if not isinstance(outputs, (list, tuple)):
                                print(f"[GPU {gpu_id}] Warning: Unexpected outputs type {type(outputs)} on frame {frame_idx}, skipping")
                                out.write(frame_bgr)
                                skipped_count += 1
                                del outputs
                                frame_time = time.time() - frame_start
                                frame_times.append(frame_time)
                                del frame_bgr
                                frame_idx += 1
                                pbar.update(1)
                                continue
                            
                            num_people = len(outputs)
                            
                            if num_people == 0:
                                out.write(frame_bgr)
                                skipped_count += 1
                                del outputs
                                frame_time = time.time() - frame_start
                                frame_times.append(frame_time)
                                del frame_bgr
                                frame_idx += 1
                                pbar.update(1)
                                continue
                            
                            total_detections += num_people
                            
                            # Create synthetic instances from SAM outputs for tracking
                            synthetic_instances = []
                            for person_idx, person_output in enumerate(outputs):
                                if not isinstance(person_output, dict):
                                    print(f"[GPU {gpu_id}] Warning: Person {person_idx} output is not a dict on frame {frame_idx}, skipping")
                                    continue
                                
                                # Extract bbox (format: [x1, y1, x2, y2])
                                bbox = None
                                area = 1.0
                                if 'bbox' in person_output:
                                    bbox_array = person_output['bbox']
                                    if hasattr(bbox_array, '__len__') and len(bbox_array) >= 4:
                                        # Convert [x1, y1, x2, y2] to [x, y, w, h] for tracking
                                        bbox = [float(bbox_array[0]), float(bbox_array[1]), 
                                               float(bbox_array[2] - bbox_array[0]), 
                                               float(bbox_array[3] - bbox_array[1])]
                                        area = float((bbox_array[2] - bbox_array[0]) * (bbox_array[3] - bbox_array[1]))
                                        area = max(area, 1.0)  # Prevent zero area
                                
                                # Extract keypoints for OKS tracking
                                # Priority: pred_keypoints_3d > pred_keypoints_2d > pred_joint_coords
                                keypoints = None
                                if 'pred_keypoints_3d' in person_output:
                                    kpts = person_output['pred_keypoints_3d']
                                    # Convert to numpy if tensor
                                    if torch.is_tensor(kpts):
                                        kpts = kpts.detach().cpu().numpy()
                                    keypoints = kpts
                                elif 'pred_keypoints_2d' in person_output:
                                    kpts = person_output['pred_keypoints_2d']
                                    if torch.is_tensor(kpts):
                                        kpts = kpts.detach().cpu().numpy()
                                    keypoints = kpts
                                elif 'pred_joint_coords' in person_output:
                                    kpts = person_output['pred_joint_coords']
                                    if torch.is_tensor(kpts):
                                        kpts = kpts.detach().cpu().numpy()
                                    keypoints = kpts
                                
                                # If no bbox but we have keypoints, compute bbox from keypoints
                                if bbox is None and keypoints is not None and len(keypoints) > 0:
                                    # Use x, y coordinates (first 2 columns)
                                    kpts_xy = keypoints[:, :2] if keypoints.ndim > 1 else keypoints.reshape(-1, 2)
                                    if len(kpts_xy) > 0:
                                        x_min, y_min = kpts_xy.min(axis=0)
                                        x_max, y_max = kpts_xy.max(axis=0)
                                        bbox = [float(x_min), float(y_min), 
                                               float(x_max - x_min), float(y_max - y_min)]
                                        area = float((x_max - x_min) * (y_max - y_min))
                                        area = max(area, 1.0)
                                
                                if bbox is None:
                                    # Fallback: use full frame area divided by number of people
                                    area = float(width * height / num_people)
                                
                                synthetic_instances.append({
                                    'keypoints': keypoints,
                                    'bbox': bbox,
                                    'area': area,
                                    'person_idx': person_idx,
                                    'output': person_output  # Store the full dict with all keys
                                })
                            
                            if not synthetic_instances:
                                out.write(frame_bgr)
                                skipped_count += 1
                                del outputs
                                frame_time = time.time() - frame_start
                                frame_times.append(frame_time)
                                del frame_bgr
                                frame_idx += 1
                                pbar.update(1)
                                continue
                            
                            # Match to existing tracks
                            matches = match_detections(tracks, synthetic_instances, oks_threshold=0.5)
                            
                            # Update tracks
                            new_tracks = {}
                            detection_track_ids = {}
                            
                            for det_idx, instance in enumerate(synthetic_instances):
                                if det_idx in matches:
                                    track_id = matches[det_idx]
                                else:
                                    track_id = next_track_id
                                    next_track_id += 1
                                    np.random.seed(track_id)
                                    track_colors[track_id] = tuple(np.random.randint(50, 255, 3).tolist())
                                
                                detection_track_ids[det_idx] = track_id
                                
                                new_tracks[track_id] = {
                                    'keypoints': instance['keypoints'],
                                    'bbox': instance['bbox'],
                                    'area': instance['area']
                                }
                            
                            tracks = new_tracks
                            
                            # Process and composite all people
                            composite_frame = frame_bgr.copy()
                            mesh_rendered_count = 0
                            
                            try:
                                mesh_start = time.time()
                                # visualize_3d_mesh takes the ENTIRE outputs list, not individual dicts
                                # It returns one combined image per person
                                mesh_results = visualize_3d_mesh(frame_bgr, outputs, estimator.faces)
                                mesh_time = time.time() - mesh_start
                                
                                if mesh_results:
                                    # mesh_results is a list of combined images, one per person
                                    # Sort by track_id for consistent ordering
                                    sorted_detections = sorted(enumerate(synthetic_instances),
                                                             key=lambda x: detection_track_ids.get(x[0], 999))
                                    
                                    for det_idx, instance in sorted_detections:
                                        if det_idx >= len(mesh_results):
                                            continue
                                        
                                        track_id = detection_track_ids[det_idx]
                                        combined_img = mesh_results[det_idx]
                                        
                                        # Extract mesh overlay from the 4-panel layout
                                        # Layout: [Original | Mesh Overlay | Front View | Side View]
                                        quarter_width = combined_img.shape[1] // 4
                                        mesh_overlay = combined_img[:, quarter_width:2*quarter_width, :]
                                        
                                        if mesh_overlay.shape[:2] != (height, width):
                                            mesh_overlay = cv2.resize(mesh_overlay, (width, height))
                                        
                                        # Composite onto frame
                                        mask = np.any(mesh_overlay > 10, axis=2)
                                        composite_frame[mask] = mesh_overlay[mask]
                                        
                                        mesh_rendered_count += 1
                                        mesh_gen_times.append(mesh_time / len(mesh_results))  # Avg time per person
                                    
                                    if mesh_rendered_count > 0:
                                        out.write(composite_frame)
                                        processed_count += 1
                                    else:
                                        out.write(frame_bgr)
                                        skipped_count += 1
                                    
                                    del mesh_results
                                else:
                                    out.write(frame_bgr)
                                    skipped_count += 1
                                    
                            except Exception as e:
                                print(f"[GPU {gpu_id}] Warning: Could not render meshes on frame {frame_idx}: {e}")
                                if frame_idx < 10:  # Print traceback for first few frames
                                    import traceback
                                    traceback.print_exc()
                                out.write(frame_bgr)
                                skipped_count += 1
                            
                            del composite_frame, outputs
                        else:
                            out.write(frame_bgr)
                            skipped_count += 1
                    
                    # Periodic cleanup
                    if frame_idx % clear_cache_every == 0:
                        torch.cuda.empty_cache()
                        gc.collect()
                        
                except Exception as e:
                    print(f"[GPU {gpu_id}] Error on frame {frame_idx}: {e}")
                    error_count += 1
                    out.write(frame_bgr)
                    skipped_count += 1
            else:
                out.write(frame_bgr)
                skipped_count += 1
            
            frame_time = time.time() - frame_start
            frame_times.append(frame_time)
            
            if frame_idx > 0 and frame_idx % 30 == 0:
                avg_frame_time = np.mean(frame_times[-30:])
                gpu_mem = torch.cuda.memory_allocated(0) / 1024**3
                pbar.set_postfix({
                    'fps': f'{1/avg_frame_time:.1f}',
                    'gpu_mem': f'{gpu_mem:.2f}GB'
                })
            
            del frame_bgr
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        processing_time = time.time() - processing_start
        
        # Cleanup
        cap.release()
        out.release()
        monitor.stop()
        
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        torch.cuda.empty_cache()
        gc.collect()
        
        # Collect statistics
        resource_stats = monitor.get_stats()
        total_time = time.time() - start_time
        
        stats = {
            'video': video_path,
            'output': output_path,
            'gpu_id': gpu_id,
            'total_frames': total_frames,
            'processed_count': processed_count,
            'skipped_count': skipped_count,
            'error_count': error_count,
            'total_detections': total_detections,
            'total_time': total_time,
            'model_load_time': model_load_time,
            'load_time': load_time,
            'processing_time': processing_time,
            'avg_frame_time': np.mean(frame_times) if frame_times else 0,
            'min_frame_time': np.min(frame_times) if frame_times else 0,
            'max_frame_time': np.max(frame_times) if frame_times else 0,
            'processing_fps': len(frame_times) / processing_time if processing_time > 0 else 0,
            'avg_mesh_time': np.mean(mesh_gen_times) if mesh_gen_times else 0,
            'min_mesh_time': np.min(mesh_gen_times) if mesh_gen_times else 0,
            'max_mesh_time': np.max(mesh_gen_times) if mesh_gen_times else 0,
            'cpu_avg': resource_stats.get('cpu_avg', 0),
            'cpu_max': resource_stats.get('cpu_max', 0),
            'gpu_avg': resource_stats.get('gpu_avg', 0),
            'gpu_max': resource_stats.get('gpu_max', 0),
            'gpu_mem_avg': resource_stats.get('gpu_mem_avg', 0),
            'gpu_mem_max': resource_stats.get('gpu_mem_max', 0),
        }
        
        stats_queue.put(stats)
        print(f"[GPU {gpu_id}] Completed: {Path(video_path).name}")
        print(f"[GPU {gpu_id}] Processed {total_detections} detections across {processed_count} frames")
        
    except Exception as e:
        print(f"[GPU {gpu_id}] ERROR processing {video_path}: {e}")
        import traceback
        traceback.print_exc()
        stats_queue.put({'video': video_path, 'error': str(e), 'gpu_id': gpu_id})

# --- Report Generation ---
def generate_report(all_stats, pipeline_time, output_dir):
    """Generate comprehensive processing report"""
    
    report_path = os.path.join(output_dir, "processing_report.txt")
    
    successful = [s for s in all_stats if 'error' not in s]
    failed = [s for s in all_stats if 'error' in s]
    
    report = f"""
{'='*80}
BATCH PROCESSING REPORT
{'='*80}

SUMMARY:
  Total videos processed:     {len(all_stats)}
  Successful:                 {len(successful)}
  Failed:                     {len(failed)}
  Total pipeline time:        {pipeline_time:.2f}s ({pipeline_time/60:.2f} min)

"""
    
    if successful:
        total_frames = sum(s['total_frames'] for s in successful)
        total_processed = sum(s['processed_count'] for s in successful)
        total_detections = sum(s.get('total_detections', 0) for s in successful)
        avg_processing_time = np.mean([s['processing_time'] for s in successful])
        avg_fps = np.mean([s['processing_fps'] for s in successful])
        
        report += f"""AGGREGATE STATISTICS:
  Total frames processed:     {total_frames}
  Frames with mesh:           {total_processed}
  Total detections:           {total_detections}
  Average processing time:    {avg_processing_time:.2f}s
  Average processing FPS:     {avg_fps:.2f}
  Effective throughput:       {total_frames/pipeline_time:.2f} frames/sec

RESOURCE USAGE (AVERAGE ACROSS ALL VIDEOS):
  CPU Usage (avg):            {np.mean([s['cpu_avg'] for s in successful]):.1f}%
  CPU Usage (max):            {np.max([s['cpu_max'] for s in successful]):.1f}%
  GPU Usage (avg):            {np.mean([s['gpu_avg'] for s in successful]):.1f}%
  GPU Usage (max):            {np.max([s['gpu_max'] for s in successful]):.1f}%
  GPU Memory (avg):           {np.mean([s['gpu_mem_avg'] for s in successful]):.2f} GB
  GPU Memory (max):           {np.max([s['gpu_mem_max'] for s in successful]):.2f} GB

"""
    
    report += "\nDETAILED RESULTS:\n" + "="*80 + "\n"
    
    for i, stats in enumerate(successful, 1):
        video_name = Path(stats['video']).name
        report += f"""
Video {i}: {video_name}
  GPU:                        {stats['gpu_id']}
  Output:                     {Path(stats['output']).name}
  Total time:                 {stats['total_time']:.2f}s
  Model load time:            {stats['model_load_time']:.2f}s
  Processing time:            {stats['processing_time']:.2f}s
  Frames:                     {stats['total_frames']}
  Frames with mesh:           {stats['processed_count']}
  Total detections:           {stats.get('total_detections', 0)}
  Processing FPS:             {stats['processing_fps']:.2f}
  Avg frame time:             {stats['avg_frame_time']:.3f}s
  Avg mesh gen time:          {stats['avg_mesh_time']:.3f}s
  CPU usage:                  {stats['cpu_avg']:.1f}% (max: {stats['cpu_max']:.1f}%)
  GPU usage:                  {stats['gpu_avg']:.1f}% (max: {stats['gpu_max']:.1f}%)
  GPU memory:                 {stats['gpu_mem_avg']:.2f} GB (max: {stats['gpu_mem_max']:.2f} GB)
"""
    
    if failed:
        report += "\nFAILED VIDEOS:\n" + "="*80 + "\n"
        for stats in failed:
            video_name = Path(stats['video']).name
            report += f"  {video_name}: {stats.get('error', 'Unknown error')}\n"
    
    report += "\n" + "="*80 + "\n"
    
    print(report)
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Report saved to: {report_path}")

# --- Main Pipeline ---
def main():
    parser = argparse.ArgumentParser(description='Multi-GPU SAM 3D Body Video Processor')
    parser.add_argument('--config', type=str, default='config.ini', 
                       help='Path to configuration file')
    parser.add_argument('--input-dir', type=str, help='Override input video directory')
    parser.add_argument('--output-dir', type=str, help='Override output directory')
    parser.add_argument('--gpus', type=int, help='Override number of GPUs to use')
    parser.add_argument('--no-detections', action='store_true',
                       help='Process videos without detection JSON files')
    args = parser.parse_args()
    
    print("="*80)
    print("MULTI-GPU VIDEO PROCESSING PIPELINE")
    print("="*80)
    
    # Load environment variables from .env
    load_environment()
    
    # Load configuration
    config = load_config(args.config)
    
    # Get HuggingFace token from environment
    hf_token = get_hf_token()
    if hf_token:
        print("✓ HuggingFace token loaded successfully")
    else:
        print("⚠ WARNING: No HuggingFace token found. Some models may not load.")
        print("  Set HF_TOKEN in your .env file")
    
    # Get settings (priority: command line > environment > config)
    INPUT_VIDEO_DIR = (args.input_dir or 
                       os.environ.get('INPUT_VIDEO_DIR') or 
                       config.get('DEFAULT', 'input_video_dir'))
    OUTPUT_DIR = (args.output_dir or 
                  os.environ.get('OUTPUT_DIR') or 
                  config.get('DEFAULT', 'output_dir'))
    DETECTION_JSON_DIR = (os.environ.get('DETECTION_JSON_DIR') or 
                          config.get('DEFAULT', 'detection_json_dir', fallback=''))
    
    # Skip detections if flag set or directory not specified
    if args.no_detections or not DETECTION_JSON_DIR:
        DETECTION_JSON_DIR = None
        print("Running without detection JSON files")
    
    NUM_GPUS = (args.gpus or 
                int(os.environ.get('NUM_GPUS', 0)) or 
                config.getint('DEFAULT', 'num_gpus'))
    PROCESS_EVERY_N_FRAMES = config.getint('DEFAULT', 'process_every_n_frames')
    CLEAR_CACHE_EVERY = config.getint('DEFAULT', 'clear_cache_every')
    HF_REPO_ID = config.get('DEFAULT', 'hf_repo_id')
    
    # Validate GPU availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return 1
    
    available_gpus = torch.cuda.device_count()
    print(f"\nDetected {available_gpus} GPU(s):")
    for i in range(available_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    if NUM_GPUS > available_gpus:
        print(f"WARNING: Requested {NUM_GPUS} GPUs but only {available_gpus} available. Using {available_gpus}.")
        NUM_GPUS = available_gpus
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Find video files
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(INPUT_VIDEO_DIR, ext)))
    
    video_files = sorted(video_files)
    
    if not video_files:
        print(f"ERROR: No video files found in {INPUT_VIDEO_DIR}")
        return 1
    
    print(f"\nFound {len(video_files)} video(s)")
    print(f"Using {NUM_GPUS} GPU(s)")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    # Match videos with JSON files (if using detections)
    video_tasks = []
    for video_path in video_files:
        video_name = Path(video_path).stem
        
        if DETECTION_JSON_DIR:
            json_path = os.path.join(DETECTION_JSON_DIR, f"{video_name}_detections.json")
            
            if not os.path.exists(json_path):
                print(f"WARNING: No detection JSON for {video_name}, will process entire frame")
                json_path = None
        else:
            json_path = None
        
        output_path = os.path.join(OUTPUT_DIR, f"{video_name}_mesh.mp4")
        video_tasks.append((video_path, json_path, output_path))
    
    if not video_tasks:
        print("ERROR: No valid videos found!")
        return 1
    
    print(f"Processing {len(video_tasks)} video(s)...\n")
    
    # Process videos
    pipeline_start = time.time()
    manager = Manager()
    stats_queue = manager.Queue()
    processes = []
    
    for i, (video_path, json_path, output_path) in enumerate(video_tasks):
        gpu_id = i % NUM_GPUS
        
        p = Process(target=process_video_on_gpu, 
                   args=(video_path, json_path, output_path, gpu_id, HF_REPO_ID,
                        PROCESS_EVERY_N_FRAMES, CLEAR_CACHE_EVERY, stats_queue))
        p.start()
        processes.append(p)
        
        if len(processes) >= NUM_GPUS:
            for proc in processes:
                proc.join()
            processes = []
    
    # Wait for remaining
    for proc in processes:
        proc.join()
    
    pipeline_time = time.time() - pipeline_start
    
    # Collect statistics
    all_stats = []
    while not stats_queue.empty():
        all_stats.append(stats_queue.get())
    
    # Generate report
    generate_report(all_stats, pipeline_time, OUTPUT_DIR)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())