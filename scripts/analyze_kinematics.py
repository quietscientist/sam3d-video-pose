#!/usr/bin/env python3
"""Kinematics analysis of a 3D COCO-17 keypoint timeseries.

Computes and saves visualizations of:
  - Joint angle timeseries and distributions (knee, elbow, hip, shoulder)
  - Endpoint speed and acceleration timeseries and distributions
  - Movement smoothness (dimensionless jerk)
  - Spectral content (power vs frequency heatmap)
  - Bilateral symmetry (left vs right limb correlation)

Smoothing and derivatives use scipy.signal.savgol_filter (Savitzky-Golay).
Spectral analysis uses scipy.signal.welch (Welch's method).

Accepts the same CSV format as visualize_3d_keypoints.py:
  long format:  frame, x, y, z, part_idx
  wide format:  frame, nose_x, nose_y, nose_z, left_eye_x, ...

Usage:
    python scripts/analyze_kinematics.py input.csv --fps 30 -o output_dir/
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter, welch
from scipy.stats import pearsonr

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ── COCO-17 constants ─────────────────────────────────────────────────────────

JOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',          # 0-4
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',     # 5-8
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',               # 9-12 (note: 9=l_wrist,10=r_wrist,11=l_hip,12=r_hip)
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',             # 13-16
]

J = {name: i for i, name in enumerate(JOINT_NAMES)}

# Hinge joint angles: (label, proximal_idx, vertex_idx, distal_idx)
# angle = arccos( dot(prox-vertex, distal-vertex) ) in [0°, 180°]
# 180° = fully extended, 0° = fully flexed
ANGLE_DEFS = [
    ('L knee',     J['left_hip'],       J['left_knee'],    J['left_ankle']),
    ('R knee',     J['right_hip'],      J['right_knee'],   J['right_ankle']),
    ('L elbow',    J['left_shoulder'],  J['left_elbow'],   J['left_wrist']),
    ('R elbow',    J['right_shoulder'], J['right_elbow'],  J['right_wrist']),
    ('L hip',      J['left_shoulder'],  J['left_hip'],     J['left_knee']),
    ('R hip',      J['right_shoulder'], J['right_hip'],    J['right_knee']),
    ('L shoulder', J['left_hip'],       J['left_shoulder'],  J['left_elbow']),
    ('R shoulder', J['right_hip'],      J['right_shoulder'], J['right_elbow']),
]

# Bilateral angle pairs for symmetry analysis
BILATERAL_ANGLE_PAIRS = [
    ('L knee',     'R knee'),
    ('L elbow',    'R elbow'),
    ('L hip',      'R hip'),
    ('L shoulder', 'R shoulder'),
]

# Endpoint joints to track speed / acceleration
SPEED_JOINTS = {
    'L wrist':  J['left_wrist'],
    'R wrist':  J['right_wrist'],
    'L ankle':  J['left_ankle'],
    'R ankle':  J['right_ankle'],
    'L elbow':  J['left_elbow'],
    'R elbow':  J['right_elbow'],
    'L knee':   J['left_knee'],
    'R knee':   J['right_knee'],
}

# Colour palette
LEFT_COLOUR  = '#2980B9'   # blue
RIGHT_COLOUR = '#E74C3C'   # red
NEUTRAL_COLOUR = '#27AE60' # green


# ── Data loading ──────────────────────────────────────────────────────────────

def load_csv(path):
    """Return (frames, positions) where positions is (F, 17, 3) float64, NaN for missing."""
    df = pd.read_csv(path)

    if 'part_idx' in df.columns:
        # Long format
        frames = sorted(df['frame'].unique())
        F = len(frames)
        positions = np.full((F, 17, 3), np.nan)
        frame_map = {f: i for i, f in enumerate(frames)}
        for _, row in df.iterrows():
            fi = frame_map[int(row['frame'])]
            ji = int(row['part_idx'])
            if ji < 17:
                positions[fi, ji] = [row['x'], row['y'], row['z']]
    else:
        # Wide format
        frames = list(df['frame'].astype(int))
        F = len(frames)
        positions = np.full((F, 17, 3), np.nan)
        for fi, (_, row) in enumerate(df.iterrows()):
            for ji, name in enumerate(JOINT_NAMES):
                positions[fi, ji] = [row[f'{name}_x'], row[f'{name}_y'], row[f'{name}_z']]

    return np.array(frames, dtype=int), positions


# ── Signal processing ─────────────────────────────────────────────────────────

def _odd(n):
    """Round n up to nearest odd integer (required by savgol_filter)."""
    return int(n) | 1


def interpolate_gaps(positions):
    """Linear interpolation across NaN frames per joint per axis."""
    out = positions.copy()
    F, J_n, _ = out.shape
    t = np.arange(F)
    for j in range(J_n):
        for ax in range(3):
            y = out[:, j, ax]
            valid = ~np.isnan(y)
            if valid.sum() >= 2:
                out[:, j, ax] = np.interp(t, t[valid], y[valid])
            elif valid.sum() == 1:
                out[:, j, ax] = y[valid][0]
            # else all NaN → leave as NaN
    return out


def sg_smooth(positions, fps, window_s=0.5, polyorder=3):
    """Apply Savitzky-Golay smoothing to position timeseries."""
    wl = max(_odd(window_s * fps), polyorder + 2)
    wl = _odd(wl)
    return savgol_filter(positions, window_length=wl, polyorder=polyorder, axis=0)


def sg_derivative(positions, fps, deriv, window_s=0.5, polyorder=4):
    """Compute n-th time derivative of positions via Savitzky-Golay.

    Returns values in SI units (m/s^n).
    """
    # Need polyorder >= deriv + 1
    po = max(polyorder, deriv + 1)
    wl = max(_odd(window_s * fps), po + 2)
    wl = _odd(wl)
    dt = 1.0 / fps
    return savgol_filter(positions, window_length=wl, polyorder=po,
                         deriv=deriv, delta=dt, axis=0)


# ── Joint angle computation ───────────────────────────────────────────────────

def compute_joint_angle(pos, prox_idx, vertex_idx, distal_idx):
    """Compute included angle at vertex joint (degrees) across all frames.

    Returns (F,) array in degrees. NaN where any of the three joints is NaN.
    """
    v1 = pos[:, prox_idx]   - pos[:, vertex_idx]   # (F, 3)
    v2 = pos[:, distal_idx] - pos[:, vertex_idx]

    n1 = np.linalg.norm(v1, axis=1, keepdims=True)
    n2 = np.linalg.norm(v2, axis=1, keepdims=True)

    with np.errstate(invalid='ignore', divide='ignore'):
        cos_a = np.einsum('fi,fi->f', v1, v2) / (n1[:, 0] * n2[:, 0])

    cos_a = np.clip(cos_a, -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(cos_a))

    # Mask frames where any joint is missing
    missing = (np.isnan(pos[:, prox_idx, 0]) |
               np.isnan(pos[:, vertex_idx, 0]) |
               np.isnan(pos[:, distal_idx, 0]))
    angles_deg[missing] = np.nan
    return angles_deg


def compute_all_angles(pos_smooth):
    """Return dict {label: (F,) angle array in degrees}."""
    angles = {}
    for label, prox, vertex, distal in ANGLE_DEFS:
        angles[label] = compute_joint_angle(pos_smooth, prox, vertex, distal)
    return angles


# ── Smoothness metric ─────────────────────────────────────────────────────────

def dimensionless_jerk(pos, fps, window_s=0.5):
    """Compute Dimensionless Jerk (DLMJ) for each joint.

    DLMJ = (mean_jerk² × duration⁵) / (peak_speed²)
    Lower is smoother.  Returns (J,) array.
    """
    vel  = sg_derivative(pos, fps, deriv=1, window_s=window_s)  # (F, J, 3)
    jerk = sg_derivative(pos, fps, deriv=3, window_s=window_s)  # (F, J, 3)

    speed     = np.linalg.norm(vel,  axis=2)   # (F, J)
    jerk_mag  = np.linalg.norm(jerk, axis=2)   # (F, J)

    duration   = pos.shape[0] / fps
    peak_speed = np.nanmax(speed, axis=0)      # (J,)
    mean_jerk2 = np.nanmean(jerk_mag ** 2, axis=0)

    with np.errstate(invalid='ignore', divide='ignore'):
        dlmj = np.sqrt(mean_jerk2 * duration**5 / (peak_speed ** 2))

    return dlmj   # (J,)


# ── Spectral analysis ─────────────────────────────────────────────────────────

def compute_spectra(signals_dict, fps, nperseg_s=4.0):
    """Welch power spectra for a dict of (F,) signals.

    Returns: freqs (N_freq,), psd_matrix (N_signals, N_freq), keys list.
    """
    nperseg = min(_odd(nperseg_s * fps), list(signals_dict.values())[0].shape[0])
    freqs, psds, keys = None, [], []

    for key, sig in signals_dict.items():
        valid = ~np.isnan(sig)
        if valid.sum() < nperseg:
            continue
        sig_filled = np.interp(np.arange(len(sig)),
                               np.where(valid)[0], sig[valid])
        f, p = welch(sig_filled, fs=fps, nperseg=nperseg, scaling='density')
        if freqs is None:
            freqs = f
        psds.append(p)
        keys.append(key)

    return freqs, np.array(psds), keys


# ── Summary statistics ────────────────────────────────────────────────────────

def summarise(data_dict, kind):
    """Build a DataFrame of summary stats from a dict of (F,) arrays."""
    rows = []
    for name, arr in data_dict.items():
        a = arr[~np.isnan(arr)]
        if len(a) == 0:
            continue
        rows.append({
            'name': name, 'kind': kind,
            'mean':   np.mean(a),
            'std':    np.std(a),
            'median': np.median(a),
            'q25':    np.percentile(a, 25),
            'q75':    np.percentile(a, 75),
            'min':    np.min(a),
            'max':    np.max(a),
            'range':  np.max(a) - np.min(a),
        })
    return pd.DataFrame(rows)


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _setup_style():
    sns.set_style('whitegrid')
    plt.rcParams.update({
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'figure.dpi': 100,
    })


def _frame_time_axis(frames, fps):
    return (frames - frames[0]) / fps   # seconds from start


def _violin(ax, data_dict, unit, colour=None, palette=None):
    """Seaborn violin plot from a dict of arrays."""
    rows = []
    for name, arr in data_dict.items():
        vals = arr[~np.isnan(arr)]
        for v in vals:
            rows.append({'Joint': name, 'value': v})
    df = pd.DataFrame(rows)
    if df.empty:
        return
    if palette:
        df['_colour'] = df['Joint'].map(palette)
        sns.violinplot(data=df, x='Joint', y='value', hue='Joint', ax=ax,
                       palette=palette, inner='quartile', linewidth=0.8,
                       cut=0, legend=False)
    else:
        sns.violinplot(data=df, x='Joint', y='value', ax=ax,
                       color=colour or '#444', inner='quartile',
                       linewidth=0.8, cut=0)
    ax.set_ylabel(unit)
    ax.tick_params(axis='x', rotation=35)


def _ts_panel(ax, frames, fps, data_dict, unit, title, left_keys=None, right_keys=None):
    """Plot time series for a dict of arrays, colouring L/R."""
    t = _frame_time_axis(frames, fps)
    for name, arr in data_dict.items():
        if left_keys and name in left_keys:
            c = LEFT_COLOUR
        elif right_keys and name in right_keys:
            c = RIGHT_COLOUR
        else:
            c = NEUTRAL_COLOUR
        ax.plot(t, arr, lw=0.9, alpha=0.85, label=name, color=c)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(unit)
    ax.set_title(title)


# ── Figure 1: Timeseries ──────────────────────────────────────────────────────

def plot_timeseries(frames, fps, angles, speeds, accels, out_path):
    _setup_style()
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    left_ang  = {k: v for k, v in angles.items() if k.startswith('L')}
    right_ang = {k: v for k, v in angles.items() if k.startswith('R')}
    left_spd  = {k: v for k, v in speeds.items()  if k.startswith('L')}
    right_spd = {k: v for k, v in speeds.items()  if k.startswith('R')}
    left_acc  = {k: v for k, v in accels.items()  if k.startswith('L')}
    right_acc = {k: v for k, v in accels.items()  if k.startswith('R')}

    # Panel 1: Lower-limb angles
    lower_ang = {k: angles[k] for k in ['L knee', 'R knee', 'L hip', 'R hip']}
    _ts_panel(axes[0], frames, fps, lower_ang, 'Angle (°)', 'Lower-limb joint angles',
              left_keys={'L knee', 'L hip'}, right_keys={'R knee', 'R hip'})
    axes[0].legend(loc='upper right', ncol=4, fontsize=8)

    # Panel 2: Upper-limb angles
    upper_ang = {k: angles[k] for k in ['L elbow', 'R elbow', 'L shoulder', 'R shoulder']}
    _ts_panel(axes[1], frames, fps, upper_ang, 'Angle (°)', 'Upper-limb joint angles',
              left_keys={'L elbow', 'L shoulder'}, right_keys={'R elbow', 'R shoulder'})
    axes[1].legend(loc='upper right', ncol=4, fontsize=8)

    # Panel 3: Endpoint speeds
    _ts_panel(axes[2], frames, fps, speeds, 'Speed (m/s)', 'Endpoint speed',
              left_keys={k for k in speeds if k.startswith('L')},
              right_keys={k for k in speeds if k.startswith('R')})
    axes[2].legend(loc='upper right', ncol=4, fontsize=8)

    # Panel 4: Endpoint accelerations
    _ts_panel(axes[3], frames, fps, accels, 'Accel (m/s²)', 'Endpoint acceleration',
              left_keys={k for k in accels if k.startswith('L')},
              right_keys={k for k in accels if k.startswith('R')})
    axes[3].legend(loc='upper right', ncol=4, fontsize=8)

    # Blue/red legend patch
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=LEFT_COLOUR, label='Left'),
                        Patch(color=RIGHT_COLOUR, label='Right')],
               loc='lower center', ncol=2, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle('Kinematics Timeseries', fontsize=12, fontweight='bold', y=1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved → {out_path}')


# ── Figure 2: Distributions ───────────────────────────────────────────────────

def plot_distributions(angles, speeds, accels, jerk_scores, out_path):
    _setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Build left/right colour palette for angles
    angle_palette = {k: LEFT_COLOUR if k.startswith('L') else RIGHT_COLOUR
                     for k in angles}

    # Panel A: joint angle distributions
    _violin(axes[0, 0], angles, 'Angle (°)', palette=angle_palette)
    axes[0, 0].set_title('Joint Angle Distributions')

    # Panel B: endpoint speed distributions
    speed_palette = {k: LEFT_COLOUR if k.startswith('L') else RIGHT_COLOUR
                     for k in speeds}
    _violin(axes[0, 1], speeds, 'Speed (m/s)', palette=speed_palette)
    axes[0, 1].set_title('Endpoint Speed Distributions')

    # Panel C: acceleration distributions
    accel_palette = {k: LEFT_COLOUR if k.startswith('L') else RIGHT_COLOUR
                     for k in accels}
    _violin(axes[1, 0], accels, 'Accel (m/s²)', palette=accel_palette)
    axes[1, 0].set_title('Endpoint Acceleration Distributions')

    # Panel D: dimensionless jerk bar chart (log scale)
    jerk_names = list(SPEED_JOINTS.keys())
    jerk_vals  = [jerk_scores[i] for i in range(len(jerk_names))]
    bar_colors = [LEFT_COLOUR if n.startswith('L') else RIGHT_COLOUR for n in jerk_names]
    axes[1, 1].bar(jerk_names, jerk_vals, color=bar_colors, edgecolor='white', linewidth=0.5)
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_ylabel('Dimensionless Jerk (log scale)')
    axes[1, 1].set_title('Movement Smoothness\n(lower = smoother)')
    axes[1, 1].tick_params(axis='x', rotation=35)
    # Annotate bars with value
    for i, (name, val) in enumerate(zip(jerk_names, jerk_vals)):
        if np.isfinite(val):
            axes[1, 1].text(i, val * 1.05, f'{val:.1f}', ha='center', va='bottom',
                            fontsize=7)

    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=LEFT_COLOUR, label='Left'),
                        Patch(color=RIGHT_COLOUR, label='Right')],
               loc='lower center', ncol=2, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle('Kinematics Distributions', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved → {out_path}')


# ── Figure 3: Spectral content ────────────────────────────────────────────────

def plot_spectra(angles, speeds, fps, out_path, max_freq=5.0):
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, sig_dict, title in [
        (axes[0], angles, 'Joint Angle Power Spectra (deg²/Hz)'),
        (axes[1], speeds, 'Endpoint Speed Power Spectra (m²/s²/Hz)'),
    ]:
        freqs, psd_matrix, keys = compute_spectra(sig_dict, fps)
        if freqs is None:
            continue
        freq_mask = freqs <= max_freq
        f_plot = freqs[freq_mask]
        p_plot = psd_matrix[:, freq_mask]

        # Normalise each row to [0, 1] for visual comparison
        row_max = p_plot.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1
        p_norm = p_plot / row_max

        im = ax.imshow(p_norm, aspect='auto', origin='lower',
                       extent=[f_plot[0], f_plot[-1], 0, len(keys)],
                       cmap='viridis', vmin=0, vmax=1)
        ax.set_yticks(np.arange(len(keys)) + 0.5)
        ax.set_yticklabels(keys, fontsize=8)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='Normalised power', shrink=0.8)

        # Mark dominant frequency per signal
        peak_freqs = f_plot[np.argmax(p_plot[:, :], axis=1)]
        for i, pf in enumerate(peak_freqs):
            ax.axvline(pf, ymin=i / len(keys), ymax=(i + 1) / len(keys),
                       color='white', lw=0.8, alpha=0.6)

    fig.suptitle('Spectral Content', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved → {out_path}')


# ── Figure 4: Bilateral symmetry ──────────────────────────────────────────────

def plot_bilateral(frames, fps, angles, out_path):
    _setup_style()
    n = len(BILATERAL_ANGLE_PAIRS)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))

    t = _frame_time_axis(frames, fps)

    for col, (left_key, right_key) in enumerate(BILATERAL_ANGLE_PAIRS):
        la = angles[left_key]
        ra = angles[right_key]
        joint = left_key.split(' ')[1]

        valid = ~(np.isnan(la) | np.isnan(ra))
        la_v, ra_v = la[valid], ra[valid]

        # Top row: time series overlay
        ax_ts = axes[0, col]
        ax_ts.plot(t, la, color=LEFT_COLOUR,  lw=0.9, label='Left',  alpha=0.85)
        ax_ts.plot(t, ra, color=RIGHT_COLOUR, lw=0.9, label='Right', alpha=0.85)
        if la_v.size > 1:
            r, p = pearsonr(la_v, ra_v)
            ax_ts.set_title(f'{joint.capitalize()}\nr = {r:.2f}, p = {p:.3f}')
        else:
            ax_ts.set_title(joint.capitalize())
        ax_ts.set_xlabel('Time (s)')
        ax_ts.set_ylabel('Angle (°)')
        ax_ts.legend(fontsize=8)

        # Bottom row: scatter left vs right
        ax_sc = axes[1, col]
        if la_v.size > 1:
            ax_sc.scatter(la_v, ra_v, alpha=0.3, s=6,
                          c=np.arange(len(la_v)), cmap='plasma')
            lim = [min(la_v.min(), ra_v.min()) - 5,
                   max(la_v.max(), ra_v.max()) + 5]
            ax_sc.plot(lim, lim, 'k--', lw=0.8, alpha=0.5, label='y=x')
            ax_sc.set_xlim(lim)
            ax_sc.set_ylim(lim)
        ax_sc.set_xlabel('Left angle (°)')
        ax_sc.set_ylabel('Right angle (°)')
        ax_sc.set_title(f'{joint.capitalize()} symmetry\n(color = time)')
        ax_sc.set_aspect('equal')

    fig.suptitle('Bilateral Symmetry', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved → {out_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Kinematics analysis of 3D COCO-17 keypoint timeseries',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument('input', help='CSV file (long or wide format)')
    ap.add_argument('--fps',      type=float, default=30.0,
                    help='Recording frame rate (Hz)')
    ap.add_argument('--smooth-s', type=float, default=0.5,
                    help='Savitzky-Golay window half-width in seconds')
    ap.add_argument('--max-freq', type=float, default=5.0,
                    help='Upper frequency limit for spectral plot (Hz)')
    ap.add_argument('-o', '--output', default=None,
                    help='Output directory (default: same dir as input)')
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output) if args.output else in_path.parent / f'{in_path.stem}_kinematics'
    out_dir.mkdir(parents=True, exist_ok=True)

    fps = args.fps

    # ── Load ─────────────────────────────────────────────────────────────────
    print(f'Loading {in_path} …')
    frames, positions = load_csv(in_path)
    F = len(frames)
    print(f'  {F} frames, {(frames[-1] - frames[0]) / fps:.1f}s at {fps:.0f} fps')

    # ── Preprocess ───────────────────────────────────────────────────────────
    positions = interpolate_gaps(positions)       # fill short NaN gaps
    pos_sm    = sg_smooth(positions, fps, window_s=args.smooth_s)  # smooth

    # ── Joint angles ─────────────────────────────────────────────────────────
    print('Computing joint angles …')
    angles = compute_all_angles(pos_sm)

    # Smooth angles (second SG pass to clean numerical noise)
    wl = max(_odd(args.smooth_s * fps), 5)
    for k in angles:
        valid = ~np.isnan(angles[k])
        if valid.sum() >= wl:
            tmp = angles[k].copy()
            tmp_filled = np.interp(np.arange(F), np.where(valid)[0], angles[k][valid])
            tmp_filled = savgol_filter(tmp_filled, window_length=wl, polyorder=3)
            tmp[valid] = tmp_filled[valid]
            angles[k] = tmp

    # ── Velocity / acceleration ───────────────────────────────────────────────
    print('Computing derivatives …')
    vel  = sg_derivative(positions, fps, deriv=1, window_s=args.smooth_s)   # (F, 17, 3) m/s
    accel = sg_derivative(positions, fps, deriv=2, window_s=args.smooth_s)  # (F, 17, 3) m/s²

    # Endpoint speed and acceleration magnitudes
    speeds = {name: np.linalg.norm(vel[:, idx], axis=1)
              for name, idx in SPEED_JOINTS.items()}
    accels = {name: np.linalg.norm(accel[:, idx], axis=1)
              for name, idx in SPEED_JOINTS.items()}

    # ── Dimensionless jerk ────────────────────────────────────────────────────
    print('Computing smoothness (dimensionless jerk) …')
    ep_indices = list(SPEED_JOINTS.values())
    jerk_all = dimensionless_jerk(positions[:, ep_indices], fps, window_s=args.smooth_s)
    # jerk_all is (len(SPEED_JOINTS),)

    # ── Spectral analysis ─────────────────────────────────────────────────────
    print('Computing power spectra …')

    # ── Figures ───────────────────────────────────────────────────────────────
    print('Saving figures …')
    plot_timeseries(frames, fps, angles, speeds, accels,
                    out_dir / 'timeseries.png')
    plot_distributions(angles, speeds, accels, jerk_all,
                       out_dir / 'distributions.png')
    plot_spectra(angles, speeds, fps,
                 out_dir / 'spectra.png', max_freq=args.max_freq)
    plot_bilateral(frames, fps, angles,
                   out_dir / 'bilateral.png')

    # ── Summary CSV ───────────────────────────────────────────────────────────
    print('Writing summary …')
    summary = pd.concat([
        summarise(angles, 'angle_deg'),
        summarise(speeds, 'speed_m_s'),
        summarise(accels, 'accel_m_s2'),
    ], ignore_index=True)

    # Add spectral peak frequencies — computed per signal kind, matched on (kind, name)
    for sig_dict, kind_label in [(angles, 'angle_deg'), (speeds, 'speed_m_s')]:
        freqs_s, psd_s, keys_s = compute_spectra(sig_dict, fps)
        if freqs_s is None:
            continue
        # Exclude DC (index 0) so slowly-drifting offsets don't dominate
        fm = (freqs_s > 0) & (freqs_s <= args.max_freq)
        for k, p_row in zip(keys_s, psd_s):
            peak_f = freqs_s[fm][np.argmax(p_row[fm])] if fm.any() else np.nan
            row_mask = (summary['kind'] == kind_label) & (summary['name'] == k)
            summary.loc[row_mask, 'peak_freq_hz'] = round(float(peak_f), 4)

    # Add bilateral correlation r-values — angles only, matched on (kind, name)
    for left_k, right_k in BILATERAL_ANGLE_PAIRS:
        la, ra = angles[left_k], angles[right_k]
        valid = ~(np.isnan(la) | np.isnan(ra))
        if valid.sum() > 2:
            r, p = pearsonr(la[valid], ra[valid])
            for name in (left_k, right_k):
                row_mask = (summary['kind'] == 'angle_deg') & (summary['name'] == name)
                summary.loc[row_mask, 'bilateral_r'] = round(r, 3)
                # Store p as string in scientific notation to avoid rounding to 0.0
                summary.loc[row_mask, 'bilateral_p'] = float(f'{p:.2e}')

    csv_path = out_dir / 'summary.csv'
    # Round numeric columns but preserve bilateral_p precision (values can be ~1e-25)
    round_cols = [c for c in summary.columns if c != 'bilateral_p']
    summary[round_cols] = summary[round_cols].round(4)
    summary.to_csv(csv_path, index=False)
    print(f'  Saved → {csv_path}')

    # Print concise terminal summary
    print('\n── Joint Angle Summary ──────────────────────────────')
    print(summary[summary.kind == 'angle_deg'][
        ['name', 'mean', 'std', 'range', 'peak_freq_hz']
    ].to_string(index=False))

    print('\n── Bilateral Correlation ────────────────────────────')
    bil_rows = (
        summary[(summary['kind'] == 'angle_deg') &
                summary['name'].isin([k for pair in BILATERAL_ANGLE_PAIRS for k in pair])]
        [['name', 'bilateral_r', 'bilateral_p']].dropna().drop_duplicates('name')
    )
    if not bil_rows.empty:
        print(bil_rows.to_string(index=False))

    print(f'\nAll outputs in {out_dir}/')


if __name__ == '__main__':
    main()
