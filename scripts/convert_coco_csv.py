#!/usr/bin/env python3
"""
Convert COCO keypoints CSV from long format to wide format.

Long format:  frame, x, y, z, part_idx
Wide format:  frame, nose_x, nose_y, nose_z, left_eye_x, left_eye_y, left_eye_z, ...
"""

import argparse
import pandas as pd
from pathlib import Path


COCO_KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]


def convert_long_to_wide(input_csv, output_csv):
    """
    Convert COCO keypoints from long format to wide format.

    Args:
        input_csv: Path to input CSV in long format (frame, x, y, z, part_idx)
        output_csv: Path to output CSV in wide format
    """
    # Read long format CSV
    df_long = pd.read_csv(input_csv)

    # Group by frame
    frames = []
    for frame_idx, group in df_long.groupby('frame'):
        row = {'frame': frame_idx}

        # Sort by part_idx to ensure correct order
        group = group.sort_values('part_idx')

        # Add each keypoint's x, y, z
        for _, kp_row in group.iterrows():
            part_idx = int(kp_row['part_idx'])
            if part_idx < len(COCO_KEYPOINT_NAMES):
                kp_name = COCO_KEYPOINT_NAMES[part_idx]
                row[f'{kp_name}_x'] = kp_row['x']
                row[f'{kp_name}_y'] = kp_row['y']
                row[f'{kp_name}_z'] = kp_row['z']

        frames.append(row)

    # Create wide format dataframe
    df_wide = pd.DataFrame(frames)

    # Sort by frame
    df_wide = df_wide.sort_values('frame')

    # Save to CSV
    df_wide.to_csv(output_csv, index=False)
    print(f"✓ Converted {len(df_long)} rows ({len(frames)} frames) from long to wide format")
    print(f"✓ Saved to {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert COCO keypoints CSV from long to wide format"
    )
    parser.add_argument(
        'input_csv',
        help='Input CSV file in long format (frame, x, y, z, part_idx)'
    )
    parser.add_argument(
        'output_csv',
        help='Output CSV file in wide format (frame, nose_x, nose_y, ...)'
    )

    args = parser.parse_args()

    input_path = Path(args.input_csv)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_path}")
        return

    convert_long_to_wide(args.input_csv, args.output_csv)


if __name__ == '__main__':
    main()
