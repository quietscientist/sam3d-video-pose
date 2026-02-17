"""Utility modules for video processing."""

import os
import sys
import importlib.util

from .patches import patch_sam3
from .config_loader import ConfigLoader
from .experiment_logger import ExperimentLogger
from .video_download import download_video, convert_video_to_mp4, download_and_convert_video

# Import functions from external/sam-3d-body/notebook/utils.py
# This avoids code duplication and stays in sync with sam-3d-body updates
_camt_dir = os.path.dirname(__file__)
_project_root = os.path.dirname(os.path.dirname(_camt_dir))
_sam3d_notebook_utils = os.path.join(_project_root, 'external', 'sam-3d-body', 'notebook', 'utils.py')

if os.path.exists(_sam3d_notebook_utils):
    # Add external/sam-3d-body to path for tools imports
    _sam3d_path = os.path.join(_project_root, 'external', 'sam-3d-body')
    if _sam3d_path not in sys.path:
        sys.path.insert(0, _sam3d_path)

    _spec = importlib.util.spec_from_file_location("_sam3d_utils", _sam3d_notebook_utils)
    if _spec and _spec.loader:
        _sam3d_utils = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_sam3d_utils)

        # Re-export the functions
        setup_sam_3d_body = _sam3d_utils.setup_sam_3d_body
        visualize_3d_mesh = _sam3d_utils.visualize_3d_mesh
        save_mesh_results = _sam3d_utils.save_mesh_results
        process_image_with_mask = _sam3d_utils.process_image_with_mask
else:
    raise ImportError(
        "Could not find external/sam-3d-body/notebook/utils.py. "
        "Make sure the sam-3d-body submodule is initialized."
    )

__all__ = [
    'patch_sam3',
    'ConfigLoader',
    'ExperimentLogger',
    'setup_sam_3d_body',
    'visualize_3d_mesh',
    'save_mesh_results',
    'process_image_with_mask',
    'download_video',
    'convert_video_to_mp4',
    'download_and_convert_video',
]

# SAM3D utilities
from .patches import *
from .config_loader import ConfigLoader
from .experiment_logger import ExperimentLogger
