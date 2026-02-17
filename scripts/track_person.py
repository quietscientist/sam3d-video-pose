"""
SOTA Person Re-Identification Pipeline for Pickleball
Components:
- YOLOX: Object detection
- TransReID: Feature extraction
- ByteTrack: Multi-object tracking with ReID embeddings
- Output: Player ID, bbox, court position per frame
"""

# Suppress NumPy warnings for compiled modules
import warnings
warnings.filterwarnings('ignore', message='.*NumPy.*')

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataclasses import dataclass, asdict
import json
from scipy.spatial.distance import cdist


@dataclass
class Detection:
    """Single detection result"""
    frame_id: int
    track_id: int
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    embedding: Optional[np.ndarray] = None
    court_position: Optional[Tuple[float, float]] = None
    
    def to_dict(self):
        d = asdict(self)
        if self.embedding is not None:
            d['embedding'] = self.embedding.tolist()
        return d


class YOLOXDetector:
    """
    YOLOX object detector for person detection
    Uses torchvision's FasterRCNN as fallback (NumPy 2.x compatible)
    """
    def __init__(self, model_size='x', device='cuda', conf_threshold=0.5):
        print(f"\n[DEBUG] YOLOXDetector.__init__ called with device={device}")
        print(f"[DEBUG] torch.cuda.is_available() = {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[DEBUG] torch.cuda.device_count() = {torch.cuda.device_count()}")
            print(f"[DEBUG] torch.cuda.current_device() = {torch.cuda.current_device()}")
            print(f"[DEBUG] torch.cuda.get_device_name(0) = {torch.cuda.get_device_name(0)}")
        
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.conf_threshold = conf_threshold
        print(f"[DEBUG] Using device: {self.device}")
        
        # Load detector (NumPy 2.x compatible)
        try:
            print("[DEBUG] Loading torchvision FasterRCNN (NumPy 2.x compatible)...")
            import torchvision
            from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
            
            weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            self.model = fasterrcnn_resnet50_fpn_v2(weights=weights)
            self.model.to(self.device)
            self.model.eval()
            
            self.transforms = weights.transforms()
            self.use_torchvision = True
            
            print(f"[DEBUG] FasterRCNN loaded successfully (COCO pretrained)")
            
        except Exception as e:
            print(f"[DEBUG] Failed to load detector: {e}")
            print(f"[DEBUG] Exception type: {type(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect persons in frame
        
        Returns:
            bboxes: [N, 4] array of [x1, y1, x2, y2]
            scores: [N] array of confidence scores
            class_ids: [N] array of class IDs (all 0 for person)
        """
        return self._detect_torchvision(frame)
    
    def _detect_torchvision(self, frame: np.ndarray):
        """FasterRCNN detection (NumPy 2.x compatible)"""
        # Convert to tensor
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(img_tensor)[0]
        
        # Filter for person class (label 1 in COCO) and confidence threshold
        person_mask = (predictions['labels'] == 1) & (predictions['scores'] >= self.conf_threshold)
        
        if person_mask.sum() == 0:
            return np.array([]), np.array([]), np.array([])
        
        bboxes = predictions['boxes'][person_mask].cpu().numpy()
        scores = predictions['scores'][person_mask].cpu().numpy()
        class_ids = np.zeros(len(bboxes), dtype=int)
        
        return bboxes, scores, class_ids


class TransReIDExtractor:
    """
    TransReID feature extractor
    """
    def __init__(self, model_name='vit_transreid_384', device='cuda'):
        print(f"\n[DEBUG] TransReIDExtractor.__init__ called with device={device}")
        print(f"[DEBUG] torch.cuda.is_available() = {torch.cuda.is_available()}")
        
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"[DEBUG] Using device: {self.device}")
        
        print("[DEBUG] Loading ReID model...")
        self.model = self._load_model(model_name)
        self.model.eval()
        
        print(f"[DEBUG] Moving model to device: {self.device}")
        try:
            self.model.to(self.device)
            print(f"[DEBUG] Model successfully moved to {self.device}")
        except Exception as e:
            print(f"[DEBUG] Error moving model to device: {e}")
            print(f"[DEBUG] Exception type: {type(e)}")
            import traceback
            traceback.print_exc()
            raise
        
        # TransReID preprocessing (384x128 for ViT models)
        self.transform = A.Compose([
            A.Resize(384, 128),  # height x width
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        print(f"[DEBUG] TransReID model loaded on {self.device}")
    
    def _load_model(self, model_name):
        """Load TransReID model"""
        print(f"[DEBUG] _load_model called with model_name={model_name}")
        
        try:
            # Try loading from fastreid
            print("[DEBUG] Attempting to import fastreid...")
            from fastreid.config import get_cfg
            from fastreid.modeling.meta_arch import build_model
            from fastreid.config import CfgNode as CN
            
            print("[DEBUG] FastReID imported successfully")
            print("[DEBUG] Configuring ResNeSt-IBN backbone...")
            
            cfg = get_cfg()
            
            # Use ResNeSt backbone (available in FastReID, SOTA performance)
            cfg.MODEL.BACKBONE.NAME = 'build_resnest_backbone'
            cfg.MODEL.BACKBONE.DEPTH = '50x'
            cfg.MODEL.BACKBONE.WITH_IBN = True  # IBN improves cross-domain performance
            cfg.MODEL.BACKBONE.PRETRAIN = True
            cfg.MODEL.BACKBONE.PRETRAIN_PATH = ''
            
            # Feature extraction settings
            cfg.MODEL.HEADS.NAME = 'EmbeddingHead'
            cfg.MODEL.HEADS.POOL_LAYER = 'GeneralizedMeanPoolingP'  # GeM pooling (correct name)
            cfg.MODEL.HEADS.NECK_FEAT = 'after'
            cfg.MODEL.HEADS.NUM_CLASSES = 0  # Feature extraction mode
            cfg.MODEL.HEADS.EMBEDDING_DIM = 2048
            
            print("[DEBUG] Building FastReID model...")
            model = build_model(cfg)
            print("[DEBUG] ResNeSt-IBN loaded from FastReID (SOTA CNN-based ReID)")
            return model
            
        except (ImportError, KeyError) as e:
            print(f"[DEBUG] FastReID load failed: {e}")
            print(f"[DEBUG] Exception type: {type(e)}")
            
            # Fallback to torchreid
            try:
                print("[DEBUG] Attempting to import torchreid...")
                import torchreid
                print("[DEBUG] TorchReID imported successfully")
                
                print("[DEBUG] Building OSNet model...")
                model = torchreid.models.build_model(
                    name='osnet_x1_0',
                    num_classes=1000,
                    pretrained=True,
                    use_gpu=self.device == 'cuda' or self.device.startswith('cuda:')
                )
                print("[DEBUG] Fallback: Loaded OSNet from TorchReID")
                return model
                
            except ImportError as e2:
                print(f"[DEBUG] TorchReID also failed: {e2}")
                raise ImportError(
                    "Please install: pip install fastreid OR pip install torchreid"
                )
        except Exception as e:
            print(f"[DEBUG] Unexpected error in _load_model: {e}")
            print(f"[DEBUG] Exception type: {type(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    @torch.no_grad()
    def extract(self, crop: np.ndarray) -> np.ndarray:
        """
        Extract ReID feature embedding
        
        Args:
            crop: RGB image numpy array [H, W, 3]
        
        Returns:
            Normalized embedding vector [D]
        """
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return np.zeros(2048)  # Return zero vector for invalid crops
        
        try:
            # Preprocess
            transformed = self.transform(image=crop)
            img_tensor = transformed['image'].unsqueeze(0).to(self.device)
            
            # Extract features
            features = self.model(img_tensor)
            
            # Handle different output formats
            if isinstance(features, dict):
                # FastReID returns dict
                features = features.get('features', features.get('embeddings', features))
            
            # L2 normalize
            features = F.normalize(features, p=2, dim=1)
            
            return features.cpu().numpy().flatten()
            
        except Exception as e:
            print(f"[DEBUG] Error in extract(): {e}")
            print(f"[DEBUG] Crop shape: {crop.shape}")
            print(f"[DEBUG] Device: {self.device}")
            import traceback
            traceback.print_exc()
            raise


class ByteTrackReID:
    """
    ByteTrack with ReID embeddings for robust multi-object tracking
    """
    def __init__(
        self,
        track_buffer: int = 30,
        match_threshold: float = 0.6,
        second_match_threshold: float = 0.4,
        new_track_threshold: float = 0.7,
        embedding_distance: str = 'cosine'
    ):
        self.track_buffer = track_buffer
        self.match_threshold = match_threshold
        self.second_match_threshold = second_match_threshold
        self.new_track_threshold = new_track_threshold
        self.embedding_distance = embedding_distance
        
        self.tracks = {}  # track_id -> Track object
        self.next_id = 1
        self.frame_id = 0
    
    def update(
        self,
        bboxes: np.ndarray,
        scores: np.ndarray,
        embeddings: np.ndarray
    ) -> List[Dict]:
        """
        Update tracks with new detections
        
        Args:
            bboxes: [N, 4] array of [x1, y1, x2, y2]
            scores: [N] array of detection confidences
            embeddings: [N, D] array of ReID embeddings
        
        Returns:
            List of active tracks with track_id and bbox
        """
        self.frame_id += 1
        
        # Separate high and low score detections
        high_score_mask = scores >= self.new_track_threshold
        low_score_mask = (scores >= 0.1) & (scores < self.new_track_threshold)
        
        high_dets = {
            'bboxes': bboxes[high_score_mask],
            'scores': scores[high_score_mask],
            'embeddings': embeddings[high_score_mask]
        }
        
        low_dets = {
            'bboxes': bboxes[low_score_mask],
            'scores': scores[low_score_mask],
            'embeddings': embeddings[low_score_mask]
        }
        
        # Get active tracks
        active_tracks = [t for t in self.tracks.values() if not t.is_lost()]
        lost_tracks = [t for t in self.tracks.values() if t.is_lost()]
        
        # First association: high score detections with active tracks
        matched, unmatched_tracks, unmatched_dets = self._match(
            active_tracks,
            high_dets,
            self.match_threshold
        )
        
        # Update matched tracks
        for track_idx, det_idx in matched:
            track = active_tracks[track_idx]
            track.update(
                high_dets['bboxes'][det_idx],
                high_dets['embeddings'][det_idx],
                self.frame_id
            )
        
        # Second association: lost tracks with remaining high score detections
        if len(unmatched_dets) > 0 and len(lost_tracks) > 0:
            remaining_high_dets = {
                'bboxes': high_dets['bboxes'][unmatched_dets],
                'scores': high_dets['scores'][unmatched_dets],
                'embeddings': high_dets['embeddings'][unmatched_dets]
            }
            
            matched2, unmatched_lost, unmatched_dets2 = self._match(
                lost_tracks,
                remaining_high_dets,
                self.second_match_threshold
            )
            
            for track_idx, det_idx in matched2:
                track = lost_tracks[track_idx]
                track.reactivate(
                    remaining_high_dets['bboxes'][det_idx],
                    remaining_high_dets['embeddings'][det_idx],
                    self.frame_id
                )
            
            unmatched_dets = [unmatched_dets[i] for i in unmatched_dets2]
        
        # Third association: unmatched tracks with low score detections
        if len(low_dets['bboxes']) > 0 and len(unmatched_tracks) > 0:
            unmatched_track_objs = [active_tracks[i] for i in unmatched_tracks]
            matched3, _, _ = self._match(
                unmatched_track_objs,
                low_dets,
                self.second_match_threshold
            )
            
            for track_idx, det_idx in matched3:
                track = unmatched_track_objs[track_idx]
                track.update(
                    low_dets['bboxes'][det_idx],
                    low_dets['embeddings'][det_idx],
                    self.frame_id
                )
        
        # Create new tracks for unmatched high score detections
        for det_idx in unmatched_dets:
            new_track = Track(
                self.next_id,
                high_dets['bboxes'][det_idx],
                high_dets['embeddings'][det_idx],
                self.frame_id
            )
            self.tracks[self.next_id] = new_track
            self.next_id += 1
        
        # Remove old lost tracks
        self.tracks = {
            tid: t for tid, t in self.tracks.items()
            if not t.should_remove(self.frame_id, self.track_buffer)
        }
        
        # Return active tracks
        output = []
        for track in self.tracks.values():
            if track.is_active():
                output.append({
                    'track_id': track.track_id,
                    'bbox': track.bbox.tolist(),
                    'score': track.score,
                    'embedding': track.smooth_embedding
                })
        
        return output
    
    def _match(self, tracks, detections, threshold):
        """Match tracks to detections using ReID embeddings + IoU"""
        if len(tracks) == 0 or len(detections['bboxes']) == 0:
            return [], list(range(len(tracks))), list(range(len(detections['bboxes'])))
        
        # Compute embedding distance matrix
        track_embeddings = np.array([t.smooth_embedding for t in tracks])
        det_embeddings = detections['embeddings']
        
        if self.embedding_distance == 'cosine':
            # Cosine similarity (higher is better)
            emb_cost = 1 - np.dot(track_embeddings, det_embeddings.T)
        else:
            # Euclidean distance
            emb_cost = cdist(track_embeddings, det_embeddings, metric='euclidean')
        
        # Compute IoU distance matrix
        track_bboxes = np.array([t.bbox for t in tracks])
        det_bboxes = detections['bboxes']
        iou_cost = 1 - self._compute_iou_matrix(track_bboxes, det_bboxes)
        
        # Combined cost: weighted sum
        cost = 0.7 * emb_cost + 0.3 * iou_cost
        
        # Hungarian matching
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost)
        
        # Filter matches by threshold
        matched = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_dets = list(range(len(det_bboxes)))
        
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < (1 - threshold):  # Convert threshold to cost
                matched.append((r, c))
                unmatched_tracks.remove(r)
                unmatched_dets.remove(c)
        
        return matched, unmatched_tracks, unmatched_dets
    
    @staticmethod
    def _compute_iou_matrix(bboxes1, bboxes2):
        """Compute IoU matrix between two sets of bboxes"""
        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
        area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
        
        ious = np.zeros((len(bboxes1), len(bboxes2)))
        
        for i, bbox1 in enumerate(bboxes1):
            for j, bbox2 in enumerate(bboxes2):
                x1 = max(bbox1[0], bbox2[0])
                y1 = max(bbox1[1], bbox2[1])
                x2 = min(bbox1[2], bbox2[2])
                y2 = min(bbox1[3], bbox2[3])
                
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                union = area1[i] + area2[j] - inter
                
                ious[i, j] = inter / (union + 1e-6)
        
        return ious


class Track:
    """Individual track for one person"""
    def __init__(self, track_id, bbox, embedding, frame_id):
        self.track_id = track_id
        self.bbox = bbox
        self.embedding = embedding
        self.smooth_embedding = embedding.copy()
        self.score = 1.0
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.lost_frames = 0
        self.state = 'active'
        
        # Embedding smoothing
        self.embedding_history = deque(maxlen=30)
        self.embedding_history.append(embedding)
    
    def update(self, bbox, embedding, frame_id):
        """Update track with new detection"""
        self.bbox = bbox
        self.embedding = embedding
        self.frame_id = frame_id
        self.lost_frames = 0
        self.state = 'active'
        
        # Update smooth embedding with exponential moving average
        self.embedding_history.append(embedding)
        self.smooth_embedding = np.mean(self.embedding_history, axis=0)
        self.smooth_embedding /= (np.linalg.norm(self.smooth_embedding) + 1e-8)
    
    def reactivate(self, bbox, embedding, frame_id):
        """Reactivate lost track"""
        self.update(bbox, embedding, frame_id)
        self.state = 'active'
    
    def mark_lost(self):
        """Mark track as lost"""
        self.state = 'lost'
        self.lost_frames += 1
    
    def is_active(self):
        return self.state == 'active'
    
    def is_lost(self):
        return self.state == 'lost'
    
    def should_remove(self, current_frame, buffer):
        """Check if track should be removed"""
        return (current_frame - self.frame_id) > buffer


class CourtPositionEstimator:
    """
    Estimate court position from bbox (simple bottom-center heuristic)
    Can be replaced with homography if court keypoints available
    """
    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
    
    def estimate(self, bbox: np.ndarray) -> Tuple[float, float]:
        """
        Estimate court position from bbox
        Returns normalized position (x, y) in [0, 1]
        """
        x1, y1, x2, y2 = bbox
        
        # Use bottom center of bbox as court position
        center_x = (x1 + x2) / 2
        bottom_y = y2
        
        # Normalize to [0, 1]
        norm_x = center_x / self.frame_width
        norm_y = bottom_y / self.frame_height
        
        return (norm_x, norm_y)


class PickleballTrackingPipeline:
    """
    Complete pipeline: Detection -> ReID -> Tracking -> Output
    """
    def __init__(
        self,
        detector_model='x',
        reid_model='vit_transreid_384',
        device='cuda',
        conf_threshold=0.5,
        track_threshold=0.6
    ):
        print("Initializing Pickleball Tracking Pipeline...")
        print("="*60)
        
        # Initialize components
        self.detector = YOLOXDetector(detector_model, device, conf_threshold)
        self.reid_extractor = TransReIDExtractor(reid_model, device)
        self.tracker = ByteTrackReID(
            match_threshold=track_threshold,
            second_match_threshold=track_threshold - 0.2
        )
        
        self.court_estimator = None  # Initialize when video loaded
        
        print("="*60)
        print("Pipeline ready!\n")
    
    def process_video(
        self,
        video_path: str,
        output_video_path: Optional[str] = None,
        output_json_path: Optional[str] = None,
        visualize: bool = True,
        show_confidence: bool = True,
        show_court_position: bool = False,
        bbox_thickness: int = 3,
        font_scale: float = 0.7
    ) -> List[Detection]:
        """
        Process video end-to-end
        
        Args:
            video_path: Input video path
            output_video_path: Optional path to save annotated video
            output_json_path: Optional path to save tracking results
            visualize: Draw bboxes and IDs on frames
            show_confidence: Display confidence scores
            show_court_position: Display court position coordinates
            bbox_thickness: Thickness of bounding box lines
            font_scale: Scale of text labels
        
        Returns:
            List of Detection objects
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize court estimator
        self.court_estimator = CourtPositionEstimator(width, height)
        
        # Video writer
        writer = None
        if output_video_path and visualize:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        print(f"Processing: {video_path}")
        print(f"Resolution: {width}x{height} @ {fps}fps")
        print(f"Total frames: {total_frames}")
        print("-"*60)
        
        all_detections = []
        frame_id = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Step 1: Detection
            bboxes, scores, _ = self.detector.detect(frame)
            
            # Step 2: Extract ReID embeddings
            embeddings = []
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                
                crop = frame[y1:y2, x1:x2]
                embedding = self.reid_extractor.extract(crop)
                embeddings.append(embedding)
            
            embeddings = np.array(embeddings) if len(embeddings) > 0 else np.array([])
            
            # Step 3: Multi-object tracking with ReID
            if len(bboxes) > 0:
                tracks = self.tracker.update(bboxes, scores, embeddings)
            else:
                tracks = []
            
            # Step 4: Store results with court position
            for track in tracks:
                bbox = np.array(track['bbox'])
                court_pos = self.court_estimator.estimate(bbox)
                
                detection = Detection(
                    frame_id=frame_id,
                    track_id=track['track_id'],
                    bbox=track['bbox'],
                    confidence=track['score'],
                    embedding=track['embedding'],
                    court_position=court_pos
                )
                all_detections.append(detection)
            
            # Step 5: Visualization
            if visualize:
                for track in tracks:
                    bbox = track['bbox']
                    track_id = track['track_id']
                    
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    
                    # Get consistent color for this player
                    color = self._get_color(track_id)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, bbox_thickness)
                    
                    # Build label text
                    label = f"ID {track_id}"
                    if show_confidence:
                        label += f" ({track['score']:.2f})"
                    
                    # Draw label background
                    label_size, baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
                    )
                    y1_label = max(y1, label_size[1] + 10)
                    
                    # Filled rectangle for label background
                    cv2.rectangle(
                        frame,
                        (x1, y1_label - label_size[1] - 10),
                        (x1 + label_size[0] + 5, y1_label),
                        color,
                        -1
                    )
                    
                    # Draw label text
                    cv2.putText(
                        frame, label, (x1 + 2, y1_label - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2
                    )
                    
                    # Optional: Draw court position
                    if show_court_position:
                        court_pos = self.court_estimator.estimate(np.array(bbox))
                        pos_text = f"({court_pos[0]:.2f}, {court_pos[1]:.2f})"
                        cv2.putText(
                            frame, pos_text, (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                        )
                    
                    # Optional: Draw center point
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    cv2.circle(frame, (center_x, center_y), 5, color, -1)
                
                # Draw frame info overlay
                info_text = f"Frame: {frame_id}/{total_frames} | Players: {len(tracks)}"
                
                # Background for info text
                info_size, _ = cv2.getTextSize(
                    info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                )
                cv2.rectangle(
                    frame, (5, 5), (15 + info_size[0], 40), (0, 0, 0), -1
                )
                
                cv2.putText(
                    frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )
            
            if writer is not None:
                writer.write(frame)
            
            frame_id += 1
            
            if frame_id % 100 == 0:
                print(f"Processed {frame_id}/{total_frames} frames ({len(tracks)} players)")
        
        cap.release()
        if writer is not None:
            writer.release()
        
        print("-"*60)
        print(f"Processing complete!")
        print(f"Total detections: {len(all_detections)}")
        print(f"Unique players: {len(set(d.track_id for d in all_detections))}")
        
        # Save to JSON
        if output_json_path:
            self._save_to_json(all_detections, output_json_path)
            print(f"Saved results to: {output_json_path}")
        
        return all_detections
    
    def _save_to_json(self, detections: List[Detection], path: str):
        """Save detections to JSON file"""
        data = {
            'detections': [d.to_dict() for d in detections],
            'metadata': {
                'total_detections': len(detections),
                'unique_tracks': len(set(d.track_id for d in detections)),
                'frames': max(d.frame_id for d in detections) + 1
            }
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def _get_color(track_id: int) -> Tuple[int, int, int]:
        """Get consistent color for track ID"""
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
        ]
        return colors[(track_id - 1) % len(colors)]


def main():
    """
    Example usage with different visualization options
    """
    # Configuration
    VIDEO_PATH = "/workspaces/sam-3d-infant/data/videos/pickleball/gHK_nObpBQw_trimmed.mp4"
    OUTPUT_VIDEO = "pickleball_game_tracked.mp4"
    OUTPUT_JSON = "tracking_results.json"
    
    # Initialize pipeline
    pipeline = PickleballTrackingPipeline(
        detector_model='x',  # YOLOX-X (largest, most accurate)
        reid_model='vit_transreid_384',  # TransReID ViT
        device='cuda',
        conf_threshold=0.5,
        track_threshold=0.6
    )
    
    # Process video with visualization
    print("\n[DEBUG] Starting video processing...")
    try:
        detections = pipeline.process_video(
            video_path=VIDEO_PATH,
            output_video_path=OUTPUT_VIDEO,
            output_json_path=OUTPUT_JSON,
            visualize=True,              # Enable visualization
            show_confidence=True,         # Show confidence scores
            show_court_position=False,    # Show court coordinates (optional)
            bbox_thickness=3,             # Bounding box line thickness
            font_scale=0.7                # Text size
        )
        print("[DEBUG] Video processing completed successfully")
    except Exception as e:
        print(f"[DEBUG] Video processing failed: {e}")
        print(f"[DEBUG] Exception type: {type(e)}")
        import traceback
        traceback.print_exc()
        raise
    
    # Print summary statistics
    print("\n" + "="*60)
    print("TRACKING SUMMARY")
    print("="*60)
    
    track_ids = set(d.track_id for d in detections)
    for track_id in sorted(track_ids):
        track_detections = [d for d in detections if d.track_id == track_id]
        frames = [d.frame_id for d in track_detections]
        
        print(f"\nPlayer {track_id}:")
        print(f"  Detections: {len(track_detections)}")
        print(f"  First frame: {min(frames)}")
        print(f"  Last frame: {max(frames)}")
        print(f"  Frame coverage: {len(set(frames))}/{max(frames)-min(frames)+1}")
        
        # Average confidence
        avg_conf = np.mean([d.confidence for d in track_detections])
        print(f"  Avg confidence: {avg_conf:.3f}")


# Advanced usage examples
def visualize_minimal():
    """Minimal visualization - just boxes and IDs"""
    pipeline = PickleballTrackingPipeline()
    pipeline.process_video(
        "game.mp4",
        "game_minimal.mp4",
        visualize=True,
        show_confidence=False,
        bbox_thickness=2,
        font_scale=0.6
    )


def visualize_detailed():
    """Detailed visualization with all info"""
    pipeline = PickleballTrackingPipeline()
    pipeline.process_video(
        "game.mp4",
        "game_detailed.mp4",
        visualize=True,
        show_confidence=True,
        show_court_position=True,
        bbox_thickness=4,
        font_scale=0.8
    )


def tracking_only():
    """Track without visualization (faster)"""
    pipeline = PickleballTrackingPipeline()
    detections = pipeline.process_video(
        "game.mp4",
        output_json_path="results.json",
        visualize=False  # No video output, only JSON
    )
    return detections


if __name__ == "__main__":
    main()