#!/usr/bin/env python3
"""
Demo CLI for SAM3D video pose estimation.
Runs the full pipeline: download → process → visualize.
Uses config files from configs/sam3d/demo_*.yaml
"""

import argparse
import subprocess
import sys
import yaml
from pathlib import Path

# Demo video URLs (not stored in config to avoid ConfigLoader URL validation issues)
DEMO_VIDEOS = {
    'nhp': 'https://upload.wikimedia.org/wikipedia/commons/f/f0/Crab_eating_macaque_walking.webm',
    'infant': 'https://upload.wikimedia.org/wikipedia/commons/8/84/Infant_babbling_in_crib.ogv',
    'raptor': 'https://upload.wikimedia.org/wikipedia/commons/b/bb/2022-10-08_Naravni_rezervat_ORMO%C5%A0KE_LAGUNE_Circus_aeruginosus_UJEDA.webm',
    'toddler': 'https://upload.wikimedia.org/wikipedia/commons/f/f9/18_meses_-_Camina_solo.webm'
}


def run_command(cmd, description):
    """Run a shell command and handle errors."""
    print(f"\n{'='*80}")
    print(f"🔧 {description}")
    print(f"{'='*80}")
    print(f"$ {cmd}\n")

    result = subprocess.run(cmd, shell=True, executable='/bin/bash')
    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed with exit code {result.returncode}")
        sys.exit(1)

    print(f"\n✓ {description} completed successfully")
    return result


def run_demo(config_path, demo_name, output_dir=None, fps=None, overlay_video=True):
    """Run a demo using the specified config file."""
    # Load config
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"❌ Error: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract parameters
    experiment_name = config.get('experiment_name', 'demo')
    video_url = DEMO_VIDEOS[demo_name]  # Get URL from dict instead of config
    text_prompt = config['text_prompt']
    vis_config = config.get('visualization', {})
    flip_z = vis_config.get('flip_z', False)
    default_fps = vis_config.get('fps', 10)
    fps = fps or default_fps

    # Extract video name from URL for finding output files
    # process_video.py creates subdirectory based on video filename stem
    from urllib.parse import urlparse
    url_path = urlparse(video_url).path
    video_filename = Path(url_path).stem  # e.g., "Crab_eating_macaque_walking"

    # Use provided output_dir or default from config
    if output_dir is None:
        output_dir = config.get('output_dir', 'output')

    output_path = Path(output_dir)

    # Display demo info
    demo_name = config_path.stem.replace('demo_', '').replace('_', ' ').title()
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  SAM3D Video Pose Estimation Demo: {demo_name:^45} ║
╚══════════════════════════════════════════════════════════════════════════════╝

This demo will:
  1. Download video from source
  2. Process video with SAM3D + SAM-3D-Body to extract COCO keypoints
  3. Apply bundle adjustment to enforce fixed bone lengths
  4. Convert keypoints to wide CSV format
  5. Generate animated GIF visualization
  6. Overlay skeleton on original video with bright colors

Config: {config_path}
Video: {video_url}
Output: {output_dir}/
""")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Process video
    # Pass all parameters explicitly (don't use --config to avoid ConfigLoader validation issues with URLs)
    processing = config.get('processing', {})
    max_frames = processing.get('max_frames', -1)
    constrain_torso = '--constrain-torso' if processing.get('constrain_torso', False) else ''
    temporal_smooth_window = processing.get('temporal_smooth_window', 11)
    temporal_smooth_polyorder = processing.get('temporal_smooth_polyorder', 3)
    smoothing_sigma = processing.get('smoothing_sigma', 2.0)

    process_cmd = (
        f"source ~/.venv-camt/bin/activate && "
        f"python scripts/process_video.py "
        f"'{video_url}' "
        f"--text-prompt '{text_prompt}' "
        f"--output-dir {output_dir} "
        f"--max-frames {max_frames} "
        f"--export-coco-csv "
        f"--temporal-smooth-window {temporal_smooth_window} "
        f"--temporal-smooth-polyorder {temporal_smooth_polyorder} "
        f"--smoothing-sigma {smoothing_sigma} "
        f"{constrain_torso}"
    ).strip()

    run_command(process_cmd, "Step 1/3: Processing video and extracting 3D COCO keypoints")

    # Rename output directory to demo name for easier access
    # process_video.py creates output_dir/video_filename/, we want output_dir/demo_<name>/
    demo_folder_name = f"demo_{demo_name}"
    demo_output_path = output_path / demo_folder_name
    video_output_path = output_path / video_filename

    # Move the video_filename directory to demo_folder_name
    if video_output_path.exists() and not demo_output_path.exists():
        import shutil
        shutil.move(str(video_output_path), str(demo_output_path))
        print(f"✓ Moved output to {demo_output_path}")

    # Step 2: Convert to wide format (use bundle-adjusted version for fixed bone lengths)
    # Files are now in demo_output_path
    adjusted_csv = demo_output_path / f"{video_filename}_3D_smoothed_adjusted.csv"
    wide_csv = demo_output_path / f"{video_filename}_3D_wide.csv"

    convert_cmd = (
        f"source ~/.venv-camt/bin/activate && "
        f"python scripts/convert_coco_csv.py "
        f"{adjusted_csv} "
        f"{wide_csv}"
    )
    run_command(convert_cmd, "Step 2/3: Converting bundle-adjusted CSV to wide format")

    # Step 3: Visualize
    output_gif = demo_output_path / f"{video_filename}_skeleton.gif"
    flip_z_flag = "--flip-z" if flip_z else ""

    visualize_cmd = (
        f"source ~/.venv-camt/bin/activate && "
        f"python scripts/visualize_3d_keypoints.py "
        f"{wide_csv} "
        f"--mode animation "
        f"--output {output_gif} "
        f"--fps {fps} "
        f"{flip_z_flag}"
    )
    run_command(visualize_cmd, "Step 3/4: Generating animated skeleton visualization")

    # Step 4: Overlay skeleton on video (optional)
    overlay_video_path = None
    if overlay_video:
        # Find the downloaded/converted video
        video_path = Path("data") / f"{video_filename}.mp4"
        if not video_path.exists():
            video_path = Path("data") / f"{video_filename}.webm"

        meshes_dir = demo_output_path / f"{video_filename}_meshes"

        if video_path.exists() and meshes_dir.exists():
            overlay_video_path = demo_output_path / f"{video_filename}_skeleton_overlay.mp4"

            overlay_cmd = (
                f"source ~/.venv-camt/bin/activate && "
                f"python scripts/overlay_skeleton_on_video.py "
                f"{video_path} "
                f"{wide_csv} "
                f"{meshes_dir} "
                f"--output {overlay_video_path}"
            )
            run_command(overlay_cmd, "Step 4/4: Overlaying skeleton on original video")
        else:
            if not video_path.exists():
                print(f"⚠ Warning: Could not find video file at {video_path}")
            if not meshes_dir.exists():
                print(f"⚠ Warning: Could not find meshes directory at {meshes_dir}")

    # Print summary
    total_steps = "4/4" if overlay_video else "3/3"
    overlay_line = f"  🎥 Overlay video:       {overlay_video_path}\n" if overlay_video_path else ""

    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Demo Complete! 🎉                                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

Output files:
  📁 Demo directory:      {demo_output_path}/
  📊 Adjusted CSV (long): {adjusted_csv}
  📊 Adjusted CSV (wide): {wide_csv}
  🎬 Skeleton GIF:        {output_gif}
{overlay_line}
Note: The skeleton visualization uses bundle-adjusted keypoints with fixed bone lengths.

View the skeleton animation:
  $ xdg-open {output_gif}
{"  $ xdg-open " + str(overlay_video_path) if overlay_video_path else ""}
""")


def main():
    parser = argparse.ArgumentParser(
        description="SAM3D Video Pose Estimation Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run non-human primate demo (macaque walking)
  python scripts/demo.py --demo-nhp

  # Run infant demo (baby babbling in crib)
  python scripts/demo.py --demo-infant

  # Run raptor demo (marsh harrier bird of prey)
  python scripts/demo.py --demo-raptor

  # Run toddler demo (18-month-old walking)
  python scripts/demo.py --demo-toddler

  # Run with custom output directory and FPS
  python scripts/demo.py --demo-nhp --output-dir output/my_demo --fps 15

  # Skip video overlay
  python scripts/demo.py --demo-nhp --no-overlay-video

Available demos:
  --demo-nhp       Non-human primate (crab-eating macaque walking)
  --demo-infant    Human infant (baby babbling in crib)
  --demo-raptor    Raptor/bird of prey (Western Marsh Harrier)
  --demo-toddler   Toddler (18-month-old walking)
        """
    )

    parser.add_argument(
        '--demo-nhp',
        action='store_true',
        help='Run non-human primate demo (crab-eating macaque)'
    )
    parser.add_argument(
        '--demo-infant',
        action='store_true',
        help='Run infant demo (baby babbling in crib)'
    )
    parser.add_argument(
        '--demo-raptor',
        action='store_true',
        help='Run raptor demo (marsh harrier bird of prey)'
    )
    parser.add_argument(
        '--demo-toddler',
        action='store_true',
        help='Run toddler demo (18-month-old walking)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: from config file)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=None,
        help='Frames per second for animation (default: from config file, usually 10)'
    )
    parser.add_argument(
        '--no-overlay-video',
        action='store_true',
        help='Skip video overlay with skeleton (overlay is on by default)'
    )

    args = parser.parse_args()

    # Determine which demo to run
    project_root = Path(__file__).parent.parent

    if args.demo_nhp:
        config_path = project_root / "configs/sam3d/demo_nhp.yaml"
        run_demo(config_path, demo_name='nhp', output_dir=args.output_dir, fps=args.fps, overlay_video=not args.no_overlay_video)
    elif args.demo_infant:
        config_path = project_root / "configs/sam3d/demo_infant.yaml"
        run_demo(config_path, demo_name='infant', output_dir=args.output_dir, fps=args.fps, overlay_video=not args.no_overlay_video)
    elif args.demo_raptor:
        config_path = project_root / "configs/sam3d/demo_raptor.yaml"
        run_demo(config_path, demo_name='raptor', output_dir=args.output_dir, fps=args.fps, overlay_video=not args.no_overlay_video)
    elif args.demo_toddler:
        config_path = project_root / "configs/sam3d/demo_toddler.yaml"
        run_demo(config_path, demo_name='toddler', output_dir=args.output_dir, fps=args.fps, overlay_video=not args.no_overlay_video)
    else:
        parser.print_help()
        print("\n❌ Error: Please specify a demo to run (e.g., --demo-nhp, --demo-infant, or --demo-raptor)")
        sys.exit(1)


if __name__ == '__main__':
    main()
