#!/usr/bin/env python3
"""
Configuration loader for YAML config files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """Simple YAML configuration loader."""

    @staticmethod
    def load(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to .yaml config file

        Returns:
            dict: Configuration parameters

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        if config_path.suffix not in ['.yaml']:
            raise ValueError(f"Config file must be .yaml (got: {config_path.suffix})")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        if config is None:
            raise ValueError(f"Empty or invalid YAML file: {config_path}")

        return config

    @staticmethod
    def validate(config: Dict[str, Any]) -> None:
        """
        Validate configuration parameters.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        # Check for input (required)
        if 'input' not in config:
            raise ValueError("Config must specify 'input' (video file or folder)")

        # Validate input path exists
        input_path = Path(config['input'])
        if not input_path.exists():
            raise ValueError(f"Input path does not exist: {input_path}")

        # Validate temporal smoothing window is odd
        if 'temporal_smooth_window' in config.get('processing', {}):
            window = config['processing']['temporal_smooth_window']
            if window % 2 == 0:
                raise ValueError(f"temporal_smooth_window must be odd (got: {window})")

    @staticmethod
    def merge_with_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge config with default values.

        Args:
            config: User configuration

        Returns:
            Complete configuration with defaults filled in
        """
        defaults = {
            'experiment_name': None,  # Auto-generated if not provided
            'text_prompt': 'a baby',
            'output_dir': 'output',
            'processing': {
                'max_frames': 300,
                'skip_mesh_generation': False,
                'skip_mesh_saving': False,
                'export_coco_csv': False,
                'bundle_adjustment': True,
                'constrain_torso': False,
                'temporal_smooth_keypoints': True,
                'temporal_smooth_window': 11,
                'temporal_smooth_polyorder': 3,
                'smoothing_sigma': 2.0,
            },
            'quality': {
                'enable_filter': False,
                'z_velocity_threshold': 0.15,
                'total_velocity_threshold': 0.25,
                'vertex_displacement_threshold': 0.08,
            },
            'metrics': {
                'enable_metrics': True,
                'enable_plots': True,
            },
            'tracking': {},
        }

        # Deep merge
        merged = defaults.copy()

        for key, value in config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # Merge nested dicts
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value

        return merged
