#!/usr/bin/env python3
"""
Comprehensive metrics logging and visualization for the video processing pipeline.
Tracks sample sizes, keypoint statistics, and generates plots at every stage.
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


class MetricsLogger:
    """Tracks and visualizes metrics throughout the processing pipeline."""

    def __init__(self, output_dir, video_name, enable_plots=True):
        """
        Initialize metrics logger.

        Args:
            output_dir: Output directory for metrics and plots
            video_name: Video name for file naming
            enable_plots: If True, generate plots (default: True)
        """
        self.output_dir = Path(output_dir)
        self.video_name = video_name
        self.enable_plots = enable_plots

        # Create metrics directory
        self.metrics_dir = self.output_dir / f"{video_name}_metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Create plots directory
        self.plots_dir = self.metrics_dir / "plots"
        if self.enable_plots:
            self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Metrics storage
        self.metrics = {
            'video_name': video_name,
            'timestamp': datetime.now().isoformat(),
            'sample_sizes': {},
            'keypoint_stats': {},
            'quality_stats': {},
            'processing_stages': []
        }

    def log_sample_size(self, stage, count, description=None):
        """
        Log sample size at a processing stage.

        Args:
            stage: Stage name (e.g., 'initial', 'after_quality_filter')
            count: Number of frames
            description: Optional description
        """
        self.metrics['sample_sizes'][stage] = {
            'count': count,
            'description': description or stage
        }

        stage_info = {
            'stage': stage,
            'timestamp': datetime.now().isoformat(),
            'sample_size': count
        }
        self.metrics['processing_stages'].append(stage_info)

        print(f"[Metrics] {stage}: {count} frames" + (f" ({description})" if description else ""))

    def log_keypoint_statistics(self, stage, keypoints_3d_list, keypoint_names=None):
        """
        Compute and log keypoint statistics.

        Args:
            stage: Stage name
            keypoints_3d_list: List of (N, 3) numpy arrays
            keypoint_names: Optional list of keypoint names
        """
        if len(keypoints_3d_list) == 0:
            return

        # Stack all keypoints (num_frames, num_keypoints, 3)
        keypoints_array = np.array(keypoints_3d_list)
        num_frames, num_keypoints, _ = keypoints_array.shape

        # Compute statistics per keypoint
        stats = {}
        for kp_idx in range(num_keypoints):
            kp_name = keypoint_names[kp_idx] if keypoint_names else f"keypoint_{kp_idx}"

            # Extract this keypoint across all frames (num_frames, 3)
            kp_positions = keypoints_array[:, kp_idx, :]

            # Compute statistics
            mean_pos = np.mean(kp_positions, axis=0)
            std_pos = np.std(kp_positions, axis=0)

            # Temporal jitter (frame-to-frame displacement)
            if num_frames > 1:
                displacements = np.linalg.norm(np.diff(kp_positions, axis=0), axis=1)
                mean_jitter = np.mean(displacements)
                max_jitter = np.max(displacements)
            else:
                mean_jitter = 0.0
                max_jitter = 0.0

            stats[kp_name] = {
                'mean_position': mean_pos.tolist(),
                'std_position': std_pos.tolist(),
                'mean_jitter': float(mean_jitter),
                'max_jitter': float(max_jitter)
            }

        # Overall statistics
        all_jitters = [stats[kp]['mean_jitter'] for kp in stats]
        overall_stats = {
            'mean_jitter_all_keypoints': float(np.mean(all_jitters)),
            'max_jitter_all_keypoints': float(np.max(all_jitters)),
            'num_keypoints': num_keypoints,
            'num_frames': num_frames
        }

        self.metrics['keypoint_stats'][stage] = {
            'per_keypoint': stats,
            'overall': overall_stats
        }

        print(f"[Metrics] {stage} keypoint stats: mean_jitter={overall_stats['mean_jitter_all_keypoints']:.4f} mm")

    def log_quality_statistics(self, quality_log):
        """
        Analyze and log quality statistics.

        Args:
            quality_log: List of quality info dicts
        """
        if len(quality_log) == 0:
            return

        good_frames = sum(1 for entry in quality_log if entry['quality']['is_good'])
        bad_frames = len(quality_log) - good_frames

        # Extract metrics
        z_velocities = [entry['quality']['metrics'].get('z_velocity', 0) for entry in quality_log]
        total_velocities = [entry['quality']['metrics'].get('total_velocity', 0) for entry in quality_log]
        vertex_displacements = [entry['quality']['metrics'].get('vertex_displacement', 0) for entry in quality_log]

        # Compute statistics
        self.metrics['quality_stats'] = {
            'total_frames': len(quality_log),
            'good_frames': good_frames,
            'bad_frames': bad_frames,
            'good_frame_percentage': 100 * good_frames / len(quality_log) if len(quality_log) > 0 else 0,
            'z_velocity': {
                'mean': float(np.mean(z_velocities)),
                'std': float(np.std(z_velocities)),
                'min': float(np.min(z_velocities)),
                'max': float(np.max(z_velocities)),
                'percentiles': {
                    '25': float(np.percentile(z_velocities, 25)),
                    '50': float(np.percentile(z_velocities, 50)),
                    '75': float(np.percentile(z_velocities, 75)),
                    '95': float(np.percentile(z_velocities, 95))
                }
            },
            'total_velocity': {
                'mean': float(np.mean(total_velocities)),
                'std': float(np.std(total_velocities)),
                'min': float(np.min(total_velocities)),
                'max': float(np.max(total_velocities)),
                'percentiles': {
                    '25': float(np.percentile(total_velocities, 25)),
                    '50': float(np.percentile(total_velocities, 50)),
                    '75': float(np.percentile(total_velocities, 75)),
                    '95': float(np.percentile(total_velocities, 95))
                }
            },
            'vertex_displacement': {
                'mean': float(np.mean(vertex_displacements)),
                'std': float(np.std(vertex_displacements)),
                'min': float(np.min(vertex_displacements)),
                'max': float(np.max(vertex_displacements)),
                'percentiles': {
                    '25': float(np.percentile(vertex_displacements, 25)),
                    '50': float(np.percentile(vertex_displacements, 50)),
                    '75': float(np.percentile(vertex_displacements, 75)),
                    '95': float(np.percentile(vertex_displacements, 95))
                }
            }
        }

        print(f"[Metrics] Quality: {good_frames}/{len(quality_log)} good frames ({self.metrics['quality_stats']['good_frame_percentage']:.1f}%)")

    def plot_sample_size_funnel(self):
        """Generate funnel plot showing sample size changes through pipeline."""
        if not self.enable_plots:
            return None

        stages = list(self.metrics['sample_sizes'].keys())
        counts = [self.metrics['sample_sizes'][s]['count'] for s in stages]

        if len(stages) == 0:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        # Horizontal bar chart (funnel style)
        y_pos = np.arange(len(stages))
        bars = ax.barh(y_pos, counts, color='steelblue', alpha=0.7)

        # Add count labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(count + max(counts)*0.02, i, f'{count:,}',
                   va='center', fontsize=10, fontweight='bold')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(stages)
        ax.set_xlabel('Number of Frames', fontsize=12)
        ax.set_title(f'Sample Size Through Pipeline: {self.video_name}', fontsize=14, fontweight='bold')
        ax.invert_yaxis()

        # Add grid
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plot_path = self.plots_dir / "sample_size_funnel.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[Metrics] Saved funnel plot: {plot_path}")
        return plot_path

    def plot_quality_metrics(self, quality_log):
        """Generate plots for quality metrics over time."""
        if not self.enable_plots or len(quality_log) == 0:
            return None

        frames = [entry['frame_idx'] for entry in quality_log]
        z_vels = [entry['quality']['metrics'].get('z_velocity', 0) for entry in quality_log]
        total_vels = [entry['quality']['metrics'].get('total_velocity', 0) for entry in quality_log]
        vertex_disps = [entry['quality']['metrics'].get('vertex_displacement', 0) for entry in quality_log]
        is_good = [entry['quality']['is_good'] for entry in quality_log]

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        # Z-velocity
        axes[0].plot(frames, z_vels, alpha=0.7, linewidth=1, label='Z-velocity', color='blue')
        axes[0].axhline(y=0.15, color='red', linestyle='--', linewidth=2, label='Threshold (0.15)')
        bad_frames = [f for f, g in zip(frames, is_good) if not g]
        bad_z_vels = [z for f, z, g in zip(frames, z_vels, is_good) if not g]
        axes[0].scatter(bad_frames, bad_z_vels, color='red', s=20, zorder=5, label='Bad frames', alpha=0.6)
        axes[0].set_ylabel('Z-velocity', fontsize=11)
        axes[0].set_title(f'Quality Metrics Over Time: {self.video_name}', fontsize=14, fontweight='bold')
        axes[0].legend(loc='upper right')
        axes[0].grid(alpha=0.3)

        # Total velocity
        axes[1].plot(frames, total_vels, alpha=0.7, linewidth=1, label='Total velocity', color='green')
        axes[1].axhline(y=0.25, color='red', linestyle='--', linewidth=2, label='Threshold (0.25)')
        bad_total_vels = [v for f, v, g in zip(frames, total_vels, is_good) if not g]
        axes[1].scatter(bad_frames, bad_total_vels, color='red', s=20, zorder=5, label='Bad frames', alpha=0.6)
        axes[1].set_ylabel('Total velocity', fontsize=11)
        axes[1].legend(loc='upper right')
        axes[1].grid(alpha=0.3)

        # Vertex displacement
        axes[2].plot(frames, vertex_disps, alpha=0.7, linewidth=1, label='Vertex displacement', color='purple')
        axes[2].axhline(y=0.08, color='red', linestyle='--', linewidth=2, label='Threshold (0.08)')
        bad_disps = [d for f, d, g in zip(frames, vertex_disps, is_good) if not g]
        axes[2].scatter(bad_frames, bad_disps, color='red', s=20, zorder=5, label='Bad frames', alpha=0.6)
        axes[2].set_ylabel('Vertex displacement', fontsize=11)
        axes[2].set_xlabel('Frame', fontsize=11)
        axes[2].legend(loc='upper right')
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        plot_path = self.plots_dir / "quality_metrics_timeline.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[Metrics] Saved quality timeline plot: {plot_path}")
        return plot_path

    def plot_quality_distributions(self, quality_log):
        """Generate distribution plots for quality metrics."""
        if not self.enable_plots or len(quality_log) == 0:
            return None

        z_vels = [entry['quality']['metrics'].get('z_velocity', 0) for entry in quality_log]
        total_vels = [entry['quality']['metrics'].get('total_velocity', 0) for entry in quality_log]
        vertex_disps = [entry['quality']['metrics'].get('vertex_displacement', 0) for entry in quality_log]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Z-velocity histogram
        axes[0].hist(z_vels, bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[0].axvline(x=0.15, color='red', linestyle='--', linewidth=2, label='Threshold')
        axes[0].set_xlabel('Z-velocity', fontsize=11)
        axes[0].set_ylabel('Frequency', fontsize=11)
        axes[0].set_title('Z-velocity Distribution', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Total velocity histogram
        axes[1].hist(total_vels, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[1].axvline(x=0.25, color='red', linestyle='--', linewidth=2, label='Threshold')
        axes[1].set_xlabel('Total velocity', fontsize=11)
        axes[1].set_ylabel('Frequency', fontsize=11)
        axes[1].set_title('Total Velocity Distribution', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        # Vertex displacement histogram
        axes[2].hist(vertex_disps, bins=30, alpha=0.7, color='purple', edgecolor='black')
        axes[2].axvline(x=0.08, color='red', linestyle='--', linewidth=2, label='Threshold')
        axes[2].set_xlabel('Vertex displacement', fontsize=11)
        axes[2].set_ylabel('Frequency', fontsize=11)
        axes[2].set_title('Vertex Displacement Distribution', fontsize=12, fontweight='bold')
        axes[2].legend()
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        plot_path = self.plots_dir / "quality_distributions.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[Metrics] Saved quality distributions plot: {plot_path}")
        return plot_path

    def plot_keypoint_jitter_comparison(self):
        return self
        

    def save_metrics_json(self):
        """Save all metrics to JSON file."""
        metrics_path = self.metrics_dir / "metrics_summary.json"

        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

        print(f"[Metrics] Saved metrics JSON: {metrics_path}")
        return metrics_path

    def save_metrics_csv(self):
        """Save sample size tracking to CSV."""
        import csv

        csv_path = self.metrics_dir / "sample_sizes.csv"

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Stage', 'Frame Count', 'Description'])

            for stage, info in self.metrics['sample_sizes'].items():
                writer.writerow([stage, info['count'], info.get('description', '')])

        print(f"[Metrics] Saved sample sizes CSV: {csv_path}")
        return csv_path

    def generate_final_report(self):
        """Generate comprehensive final report with all metrics and plots."""
        print("\n" + "="*60)
        print("GENERATING METRICS REPORT")
        print("="*60)

        # Save JSON and CSV
        self.save_metrics_json()
        self.save_metrics_csv()

        # Generate plots
        if self.enable_plots:
            self.plot_sample_size_funnel()
            self.plot_keypoint_jitter_comparison()

        # Print summary
        print("\n" + "="*60)
        print("METRICS SUMMARY")
        print("="*60)

        print("\nSample Sizes:")
        for stage, info in self.metrics['sample_sizes'].items():
            print(f"  {stage:<30s}: {info['count']:>6,} frames")

        if 'quality_stats' in self.metrics and self.metrics['quality_stats']:
            qstats = self.metrics['quality_stats']
            print(f"\nQuality Statistics:")
            print(f"  Good frames: {qstats['good_frames']:,} ({qstats['good_frame_percentage']:.1f}%)")
            print(f"  Bad frames:  {qstats['bad_frames']:,}")

        if self.metrics['keypoint_stats']:
            print(f"\nKeypoint Statistics:")
            for stage, stats in self.metrics['keypoint_stats'].items():
                overall = stats['overall']
                print(f"  {stage}:")
                print(f"    Mean jitter: {overall['mean_jitter_all_keypoints']:.4f} mm")
                print(f"    Max jitter:  {overall['max_jitter_all_keypoints']:.4f} mm")

        print(f"\nMetrics saved to: {self.metrics_dir}")
        print("="*60 + "\n")
