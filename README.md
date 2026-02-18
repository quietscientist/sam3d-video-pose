# SAM3D Video Pose

3D pose estimation from video using [SAM3](https://github.com/facebookresearch/segment-anything-3) segmentation and [SAM-3D-Body](https://github.com/facebookresearch/sam-3d-body) mesh reconstruction. Works on humans, animals, and other subjects -- point it at any video with a text prompt and get 3D COCO keypoints, skeleton overlays, and animated visualizations.

## Getting Started

The fastest way to see the pipeline in action is with the built-in demos. Each downloads a short video from Wikimedia Commons, processes it end-to-end, and produces a skeleton overlay video + animated GIF.

```bash
# Pick a demo
python scripts/demo.py --demo-toddler   # 18-month-old walking
python scripts/demo.py --demo-infant     # baby in crib
python scripts/demo.py --demo-nhp        # crab-eating macaque
python scripts/demo.py --demo-raptor     # marsh harrier in flight
```

Each demo runs the full pipeline:
1. Download video from Wikimedia Commons
2. Segment subject with SAM3 (text-prompted)
3. Estimate 3D mesh + COCO keypoints with SAM-3D-Body
4. Bundle-adjust bone lengths
5. Generate 3D skeleton GIF
6. Overlay skeleton on original video with OpenPose-style colors

Output lands in `output/demo_<name>/` with the overlay video, skeleton GIF, and CSV keypoint files.

### Process Your Own Video

```bash
# Local file
python scripts/process_video.py path/to/video.mp4 \
    --text-prompt "a person" \
    --export-coco-csv \
    --max-frames 300

# URL (auto-downloads and converts)
python scripts/process_video.py "https://example.com/video.webm" \
    --text-prompt "a dancer" \
    --export-coco-csv
```

### Visualize Results

```bash
# Animated 3D skeleton GIF
python scripts/visualize_3d_keypoints.py output/video_name/video_name_3D_wide.csv \
    --mode animation --output skeleton.gif --flip-z

# Overlay skeleton on original video
python scripts/overlay_skeleton_on_video.py \
    data/video.mp4 \
    output/video_name/video_name_3D_wide.csv \
    output/video_name/video_name_meshes
```

## Installation

### Prerequisites
- Python 3.10
- CUDA-capable GPU with 16GB+ VRAM
- Git with submodule support

### Setup

```bash
git clone --recursive https://github.com/yourusername/sam3d-video-pose.git
cd sam3d-video-pose

python3.10 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Set up HuggingFace token for model downloads:
```bash
echo "HF_TOKEN=your_token_here" > .env
```

Get your token from https://huggingface.co/settings/tokens.

## Configuration

### YAML Config Files

Demo configs live in `configs/sam3d/`. Use them as templates for your own:

```yaml
experiment_name: my_analysis
text_prompt: "a person"
output_dir: "output"

processing:
  max_frames: 300
  export_coco_csv: true
  bundle_adjustment: true
  constrain_torso: false        # keep false for infants
  temporal_smooth_window: 11
  temporal_smooth_polyorder: 3
  smoothing_sigma: 2.0

tracking:
  max_num_objects: 1             # single-subject lock-on
  new_det_thresh: 0.9            # high bar for spawning new tracks

quality:
  enable_filter: false

visualization:
  flip_z: true
  fps: 10
```

### Tracking Parameters

SAM3's tracking behavior is tunable via the `tracking:` section in config or `--tracking-params` on the CLI:

| Parameter | Default | Effect |
|---|---|---|
| `max_num_objects` | 10000 | Max tracked objects (set to 1 for single subject) |
| `new_det_thresh` | 0.7 | Score needed to spawn a new track (higher = stickier) |
| `score_threshold_detection` | 0.5 | Min detection confidence |
| `assoc_iou_thresh` | 0.1 | Det-to-track matching threshold (lower = stickier) |
| `min_trk_keep_alive` | -1 | How long tracks survive without matches |
| `max_trk_keep_alive` | 30 | Budget for matched frames |
| `hotstart_delay` | 15 | Frames before output starts |
| `recondition_every_nth_frame` | 16 | Re-init from detections frequency |

CLI override example:
```bash
python scripts/process_video.py video.mp4 \
    --tracking-params '{"max_num_objects": 1, "new_det_thresh": 0.9}'
```

### Text Prompts

SAM3 uses text prompts for segmentation. More specific prompts help with tracking:

- `"a baby"` -- general infant
- `"a toddler walking in the center"` -- spatial + action hint
- `"a monkey"` -- non-human primate
- `"a bird"` -- avian subject

## Output Structure

```
output/video_name/
    video_name_3D_raw.csv                    # Raw COCO keypoints (long format)
    video_name_3D_smoothed_adjusted.csv      # Bundle-adjusted (long format)
    video_name_3D_wide.csv                   # Bundle-adjusted (wide format)
    video_name_skeleton.gif                  # 3D skeleton animation
    video_name_skeleton_overlay.mp4          # Skeleton overlaid on video
    video_name_meshes/                       # Per-frame MHR parameters
        frame_NNNN_obj_0/
            mhr_parameters.npz              # Camera params + 70-point skeleton
            keypoints.json                  # COCO keypoint subset
    video_name_all_keypoints.json           # All keypoints (JSON)
    video_name_mesh_results.json            # Mesh metadata
```

### CSV Formats

**Long format** (`_3D_raw.csv`, `_3D_smoothed_adjusted.csv`): one row per keypoint per frame.
```
frame,x,y,z,part_idx
0,0.123,0.456,0.789,0
0,0.234,0.567,0.890,1
...
```

**Wide format** (`_3D_wide.csv`): one row per frame, columns for each keypoint axis.
```
frame,nose_x,nose_y,nose_z,left_eye_x,...
0,0.123,0.456,0.789,0.234,...
```

## Pipeline Architecture

```
Video + Text Prompt
    |
    v
SAM3 Segmentation (text-prompted tracking across frames)
    |
    v
SAM-3D-Body (per-frame 3D mesh + 70-point MHR keypoints)
    |
    v
COCO 17-point extraction + temporal smoothing
    |
    v
Bundle adjustment (fixed bone lengths)
    |
    v
Visualization (3D GIF + video overlay)
```

## GPU Memory

The pipeline uses two GPUs when available:
- **GPU 0**: SAM3 segmentation (~15GB)
- **GPU 1**: SAM-3D-Body mesh estimation (~8GB)

For single GPU, reduce `--max-frames` or use `--skip-mesh-saving`.

## Troubleshooting

**CUDA Out of Memory**: Reduce `--max-frames` or use `--skip-mesh-saving`.

**Dark video after conversion**: The pipeline validates brightness after webm-to-mp4 conversion. If you see a warning, the retry with explicit color range normalization should fix it.

**Tracking wrong subject**: Add `tracking: {max_num_objects: 1}` to your config, use a more specific text prompt (e.g. `"a toddler walking in the center"` instead of `"a person"`).

**Module import errors**: Run `pip install -e .` and `git submodule update --init --recursive`.

## Acknowledgments

- [SAM3](https://github.com/facebookresearch/segment-anything-3) for video segmentation
- [SAM-3D-Body](https://github.com/facebookresearch/sam-3d-body) for 3D body reconstruction
