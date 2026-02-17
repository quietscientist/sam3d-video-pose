#!/usr/bin/env python3
"""
Experiment logging for tracking processing runs.
"""

import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


class ExperimentLogger:
    """Tracks and logs experiment runs."""

    def __init__(self, experiments_dir: str = "experiments"):
        """
        Initialize experiment logger.

        Args:
            experiments_dir: Directory to store experiment logs
        """
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(exist_ok=True)

        self.runs_log_path = self.experiments_dir / "runs_log.json"
        self.current_run = None

    def start_run(self, config: Dict[str, Any], experiment_name: Optional[str] = None) -> str:
        """
        Start a new experiment run.

        Args:
            config: Configuration dictionary
            experiment_name: Optional experiment name (auto-generated if None)

        Returns:
            run_id: Unique run identifier
        """
        # Generate run ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if experiment_name:
            run_id = f"{experiment_name}_{timestamp}"
        else:
            # Auto-generate from input path
            input_path = Path(config['input'])
            if input_path.is_file():
                run_id = f"{input_path.stem}_{timestamp}"
            else:
                run_id = f"{input_path.name}_{timestamp}"

        # Create run directory
        run_dir = self.experiments_dir / run_id
        run_dir.mkdir(exist_ok=True)

        # Save config
        config_path = run_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        # Initialize run metadata
        self.current_run = {
            'run_id': run_id,
            'run_dir': str(run_dir),
            'start_time': datetime.now().isoformat(),
            'config': config,
            'status': 'running',
            'results': {},
            'errors': []
        }

        return run_id

    def log_result(self, key: str, value: Any):
        """
        Log a result value.

        Args:
            key: Result key
            value: Result value
        """
        if self.current_run is None:
            raise RuntimeError("No active run. Call start_run() first.")

        self.current_run['results'][key] = value

    def log_error(self, error: str):
        """
        Log an error.

        Args:
            error: Error message
        """
        if self.current_run is None:
            raise RuntimeError("No active run. Call start_run() first.")

        self.current_run['errors'].append({
            'timestamp': datetime.now().isoformat(),
            'error': error
        })

    def end_run(self, status: str = 'completed'):
        """
        End the current run.

        Args:
            status: Final status ('completed', 'failed', 'interrupted')
        """
        if self.current_run is None:
            raise RuntimeError("No active run. Call start_run() first.")

        self.current_run['end_time'] = datetime.now().isoformat()
        self.current_run['status'] = status

        # Calculate duration
        start = datetime.fromisoformat(self.current_run['start_time'])
        end = datetime.fromisoformat(self.current_run['end_time'])
        duration = (end - start).total_seconds()
        self.current_run['duration_seconds'] = duration

        # Save run metadata
        run_dir = Path(self.current_run['run_dir'])
        run_metadata_path = run_dir / "run_metadata.json"
        with open(run_metadata_path, 'w') as f:
            json.dump(self.current_run, f, indent=2)

        # Append to runs log
        self._append_to_runs_log(self.current_run)

        print(f"\n{'='*60}")
        print(f"EXPERIMENT RUN COMPLETE")
        print(f"{'='*60}")
        print(f"Run ID: {self.current_run['run_id']}")
        print(f"Status: {status}")
        print(f"Duration: {duration:.1f}s")
        print(f"Run directory: {run_dir}")
        print(f"{'='*60}\n")

        self.current_run = None

    def _append_to_runs_log(self, run_data: Dict[str, Any]):
        """
        Append run to the global runs log.

        Args:
            run_data: Run metadata
        """
        # Load existing runs
        if self.runs_log_path.exists():
            with open(self.runs_log_path, 'r') as f:
                runs_log = json.load(f)
        else:
            runs_log = {'runs': []}

        # Append new run (summary only, not full config)
        run_summary = {
            'run_id': run_data['run_id'],
            'start_time': run_data['start_time'],
            'end_time': run_data.get('end_time'),
            'duration_seconds': run_data.get('duration_seconds'),
            'status': run_data['status'],
            'input': run_data['config'].get('input'),
            'experiment_name': run_data['config'].get('experiment_name'),
            'num_errors': len(run_data.get('errors', [])),
            'run_dir': run_data['run_dir']
        }

        runs_log['runs'].append(run_summary)

        # Save
        with open(self.runs_log_path, 'w') as f:
            json.dump(runs_log, f, indent=2)

    def get_run_dir(self) -> Optional[Path]:
        """Get the current run directory."""
        if self.current_run is None:
            return None
        return Path(self.current_run['run_dir'])
