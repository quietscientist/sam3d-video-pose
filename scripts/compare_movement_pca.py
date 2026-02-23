#!/usr/bin/env python3
"""Compare movement kinematics across populations via PCA.

Reads one or more 3D COCO-17 keypoint CSV files per population group
(e.g. infant, adult, nhp), extracts a per-frame kinematic feature vector,
runs joint PCA across all groups, and produces interpretable visualisations
of how groups differ in movement space.

Feature vector (per frame, per recording):
    - 8 joint angles in degrees (knees × 2, elbows × 2, hips × 2, shoulders × 2)
    - 8 endpoint speeds in m/s (wrists × 2, ankles × 2, elbows × 2, knees × 2)
    - 1 relative limb activity scalar (upper / lower limb speed ratio)
    Total: 17 features per frame

PCA is fitted on mean-centered, unit-variance features from all frames of
all groups combined (StandardScaler), then each group's frames are projected.

Outputs (all in --output directory):
    pca_scatter.png        — PC1 vs PC2 scatter, coloured by group
    pca_trajectories.png   — PC1/PC2 time trajectories per recording
    pca_loadings.png       — feature loadings for PC1…PC3 (bar chart)
    pca_variance.png       — cumulative explained variance
    pca_density.png        — 2D KDE per group in PC1/PC2 space
    pca_summary.csv        — per-frame PCA coordinates + metadata

Data directory layout expected (--data-dir):

    data/movement_pca/
        infant/
            recording_01.csv
            recording_02.csv
            ...
        adult/
            recording_01.csv
            ...
        nhp/
            recording_01.csv
            ...

Each CSV must be in the same long-format or wide-format used by
visualize_3d_keypoints.py and analyze_kinematics.py.

Alternatively, pass files directly:
    python scripts/compare_movement_pca.py \\
        --files infant:data/infant1.csv infant:data/infant2.csv \\
                adult:data/adult1.csv \\
                nhp:data/nhp1.csv \\
        --fps 30 -o output/pca/

Usage (directory-based):
    python scripts/compare_movement_pca.py \\
        --data-dir data/movement_pca/ \\
        --fps 30 -o output/movement_pca/
"""

import argparse
import warnings
from pathlib import Path
from itertools import cycle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ── COCO-17 joint definitions (shared with analyze_kinematics.py) ─────────────

JOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
]
J = {name: i for i, name in enumerate(JOINT_NAMES)}

# Joint angles: (label, proximal, vertex, distal)
ANGLE_DEFS = [
    ('L knee',     J['left_hip'],       J['left_knee'],    J['left_ankle']),
    ('R knee',     J['right_hip'],      J['right_knee'],   J['right_ankle']),
    ('L elbow',    J['left_shoulder'],  J['left_elbow'],   J['left_wrist']),
    ('R elbow',    J['right_shoulder'], J['right_elbow'],  J['right_wrist']),
    ('L hip',      J['left_shoulder'],  J['left_hip'],     J['left_knee']),
    ('R hip',      J['right_shoulder'], J['right_hip'],    J['right_knee']),
    ('L shoulder', J['left_hip'],       J['left_shoulder'], J['left_elbow']),
    ('R shoulder', J['right_hip'],      J['right_shoulder'], J['right_elbow']),
]

SPEED_JOINTS = {
    'L wrist': J['left_wrist'],   'R wrist': J['right_wrist'],
    'L ankle': J['left_ankle'],   'R ankle': J['right_ankle'],
    'L elbow': J['left_elbow'],   'R elbow': J['right_elbow'],
    'L knee':  J['left_knee'],    'R knee':  J['right_knee'],
}

FEATURE_NAMES = (
    [label for label, *_ in ANGLE_DEFS] +
    [f'{name} speed' for name in SPEED_JOINTS] +
    ['upper/lower speed ratio']
)

# Group colour palette
GROUP_COLOURS = {
    'infant': '#E74C3C',
    'adult':  '#2980B9',
    'nhp':    '#27AE60',
}
DEFAULT_COLOURS = ['#8E44AD', '#F39C12', '#1ABC9C', '#E67E22']


# ── Signal processing helpers ─────────────────────────────────────────────────

def _odd(n):
    return int(n) | 1


def _sg(arr, fps, window_s=0.4, deriv=0, polyorder=3):
    """Savitzky-Golay filter/derivative along axis 0."""
    wl = max(_odd(window_s * fps), polyorder + 2)
    wl = _odd(wl)
    dt = 1.0 / fps
    return savgol_filter(arr, window_length=wl, polyorder=polyorder,
                         deriv=deriv, delta=dt, axis=0)


def _interpolate(arr):
    """Linear interpolation of NaN values per column."""
    out = arr.copy()
    t = np.arange(len(arr))
    for j in range(arr.shape[1]):
        valid = ~np.isnan(arr[:, j])
        if valid.sum() >= 2:
            out[:, j] = np.interp(t, t[valid], arr[valid, j])
    return out


# ── Data loading ──────────────────────────────────────────────────────────────

def load_csv(path):
    """Return (frames, positions (F,17,3)) from long- or wide-format CSV."""
    df = pd.read_csv(path)
    if 'part_idx' in df.columns:
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
        frames = list(df['frame'].astype(int))
        F = len(frames)
        positions = np.full((F, 17, 3), np.nan)
        for fi, (_, row) in enumerate(df.iterrows()):
            for ji, name in enumerate(JOINT_NAMES):
                positions[fi, ji] = [row[f'{name}_x'],
                                     row[f'{name}_y'], row[f'{name}_z']]
    return np.array(frames, dtype=int), positions


# ── Feature extraction ────────────────────────────────────────────────────────

def _angle(pos, prox, vertex, distal):
    """(F,) joint angle in degrees."""
    v1 = pos[:, prox]   - pos[:, vertex]
    v2 = pos[:, distal] - pos[:, vertex]
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        cos_a = np.einsum('fi,fi->f', v1, v2) / (n1 * n2)
    deg = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
    missing = (np.isnan(pos[:, prox, 0]) |
               np.isnan(pos[:, vertex, 0]) |
               np.isnan(pos[:, distal, 0]))
    deg[missing] = np.nan
    return deg


def extract_features(positions, fps):
    """Return (F, n_features) float32 feature matrix for one recording.

    NaN rows are dropped before returning; original frame count may reduce.
    """
    F = positions.shape[0]

    # Interpolate gaps then smooth
    flat = positions.reshape(F, -1)
    flat = _interpolate(flat)
    pos_sm = flat.reshape(F, 17, 3)
    pos_sm = _sg(pos_sm, fps, window_s=0.4)

    # Angles
    angles = np.stack([_angle(pos_sm, p, v, d)
                       for _, p, v, d in ANGLE_DEFS], axis=1)   # (F, 8)

    # Speeds
    vel = _sg(pos_sm, fps, window_s=0.4, deriv=1)               # (F, 17, 3) m/s
    speeds = np.stack([np.linalg.norm(vel[:, idx], axis=1)
                       for idx in SPEED_JOINTS.values()], axis=1)  # (F, 8)

    # Upper / lower limb speed ratio
    upper_spd = speeds[:, [2, 3, 4, 5]].mean(axis=1)  # wrists + elbows
    lower_spd = speeds[:, [0, 1, 6, 7]].mean(axis=1)  # ankles + knees
    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = upper_spd / (lower_spd + 1e-6)         # (F,)

    feat = np.concatenate([angles, speeds, ratio[:, None]], axis=1)  # (F, 17)

    # Drop frames with any NaN
    valid = ~np.isnan(feat).any(axis=1)
    return feat[valid].astype(np.float32)


# ── Collect all recordings ────────────────────────────────────────────────────

def collect_recordings(data_dir=None, file_specs=None, fps=30.0):
    """Return list of dicts: {group, name, features (F, 17), frames}.

    data_dir: Path — scan <data_dir>/<group>/*.csv
    file_specs: list of "group:path" strings
    """
    records = []

    if data_dir is not None:
        data_dir = Path(data_dir)
        for group_dir in sorted(data_dir.iterdir()):
            if not group_dir.is_dir():
                continue
            group = group_dir.name
            for csv_path in sorted(group_dir.glob('*.csv')):
                _, pos = load_csv(csv_path)
                feat = extract_features(pos, fps)
                if len(feat) > 0:
                    records.append({'group': group, 'name': csv_path.stem,
                                    'features': feat, 'n_frames': len(feat)})
                    print(f'  [{group}] {csv_path.name}: {len(feat)} frames')

    if file_specs:
        for spec in file_specs:
            group, path = spec.split(':', 1)
            _, pos = load_csv(path)
            feat = extract_features(pos, fps)
            if len(feat) > 0:
                records.append({'group': group, 'name': Path(path).stem,
                                'features': feat, 'n_frames': len(feat)})
                print(f'  [{group}] {Path(path).name}: {len(feat)} frames')

    return records


# ── PCA ───────────────────────────────────────────────────────────────────────

def run_pca(records, n_components=10):
    """Fit PCA on all frames (all groups combined).

    Returns:
        pca: fitted sklearn PCA
        scaler: fitted StandardScaler
        df_proj: DataFrame with columns pc1…pcN, group, recording, frame_local
    """
    all_feat = np.concatenate([r['features'] for r in records], axis=0)

    scaler = StandardScaler()
    all_scaled = scaler.fit_transform(all_feat)

    n_components = min(n_components, all_scaled.shape[1], all_scaled.shape[0])
    pca = PCA(n_components=n_components, random_state=42)
    all_proj = pca.fit_transform(all_scaled)

    # Build long-form DataFrame
    rows = []
    offset = 0
    for r in records:
        n = r['n_frames']
        proj = all_proj[offset:offset + n]
        for i, row in enumerate(proj):
            d = {'group': r['group'], 'recording': r['name'], 'frame_local': i}
            for k, val in enumerate(row):
                d[f'pc{k+1}'] = val
            rows.append(d)
        offset += n

    df = pd.DataFrame(rows)
    return pca, scaler, df


# ── Plotting ──────────────────────────────────────────────────────────────────

def _group_colour(group, extra_colours):
    return GROUP_COLOURS.get(group, next(extra_colours))


def plot_scatter(df, pca, out_path):
    """PC1 vs PC2 scatter, coloured by group."""
    groups = df['group'].unique()
    extra = cycle(DEFAULT_COLOURS)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter (sub-sample large groups for readability)
    ax = axes[0]
    for group in sorted(groups):
        sub = df[df['group'] == group]
        # Sub-sample to ≤2000 points to avoid overplotting
        if len(sub) > 2000:
            sub = sub.sample(2000, random_state=42)
        c = _group_colour(group, extra)
        ax.scatter(sub['pc1'], sub['pc2'], alpha=0.25, s=8,
                   color=c, label=group, rasterized=True)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('PC1 vs PC2 — all frames')
    ax.legend(markerscale=3, framealpha=0.8)

    # 2D KDE contours per group
    ax2 = axes[1]
    extra2 = cycle(DEFAULT_COLOURS)
    for group in sorted(groups):
        sub = df[df['group'] == group]
        c = _group_colour(group, extra2)
        sns.kdeplot(data=sub, x='pc1', y='pc2', ax=ax2,
                    color=c, fill=True, alpha=0.25, levels=5, label=group)
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax2.set_title('PC1 vs PC2 — density (KDE)')
    ax2.legend(framealpha=0.8)

    plt.suptitle('Movement PCA — Group Comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved → {out_path}')


def plot_trajectories(df, pca, out_path):
    """PC1 and PC2 time trajectories per recording, faceted by group."""
    groups = sorted(df['group'].unique())
    n_groups = len(groups)

    fig, axes = plt.subplots(n_groups, 2, figsize=(14, 4 * n_groups),
                             squeeze=False, sharex=False)

    extra = cycle(DEFAULT_COLOURS)
    for row_i, group in enumerate(groups):
        c = _group_colour(group, extra)
        sub_g = df[df['group'] == group]
        for col_i, pc in enumerate(['pc1', 'pc2']):
            ax = axes[row_i, col_i]
            for rec in sub_g['recording'].unique():
                sub_r = sub_g[sub_g['recording'] == rec]
                ax.plot(sub_r['frame_local'].values,
                        sub_r[pc].values,
                        lw=0.8, alpha=0.7, color=c, label=rec)
            ax.set_xlabel('Frame')
            ax.set_ylabel(f'{pc.upper()} ({pca.explained_variance_ratio_[col_i]*100:.1f}%)')
            ax.set_title(f'{group} — {pc.upper()}')
            if sub_g['recording'].nunique() <= 6:
                ax.legend(fontsize=7, loc='upper right')

    plt.suptitle('PC Trajectories per Recording', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved → {out_path}')


def plot_loadings(pca, out_path, n_pcs=3):
    """Feature loadings for the first n_pcs components."""
    n_pcs = min(n_pcs, pca.n_components_)
    components = pca.components_[:n_pcs]   # (n_pcs, n_features)

    fig, axes = plt.subplots(1, n_pcs, figsize=(5 * n_pcs, 5), sharey=False)
    if n_pcs == 1:
        axes = [axes]

    colours = ['#E74C3C', '#2980B9', '#27AE60']
    for i, (ax, comp, col) in enumerate(zip(axes, components, colours)):
        sorted_idx = np.argsort(np.abs(comp))[::-1]
        names_sorted = [FEATURE_NAMES[j] for j in sorted_idx]
        vals_sorted  = comp[sorted_idx]
        bar_colours  = [col if v >= 0 else '#BDC3C7' for v in vals_sorted]
        ax.barh(names_sorted, vals_sorted, color=bar_colours, edgecolor='white')
        ax.axvline(0, color='black', lw=0.8)
        ax.set_xlabel('Loading')
        ax.set_title(f'PC{i+1} ({pca.explained_variance_ratio_[i]*100:.1f}%)\n'
                     f'loadings (sorted by |magnitude|)')
        ax.tick_params(axis='y', labelsize=8)

    plt.suptitle('PCA Feature Loadings', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved → {out_path}')


def plot_variance(pca, out_path):
    """Scree plot + cumulative explained variance."""
    ev = pca.explained_variance_ratio_
    cum_ev = np.cumsum(ev)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(range(1, len(ev) + 1), ev * 100, color='#2980B9', edgecolor='white')
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Explained variance (%)')
    axes[0].set_title('Scree Plot')

    axes[1].plot(range(1, len(cum_ev) + 1), cum_ev * 100, 'o-', color='#E74C3C')
    axes[1].axhline(90, color='gray', lw=0.8, linestyle='--', label='90%')
    axes[1].axhline(95, color='gray', lw=0.8, linestyle=':', label='95%')
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Cumulative explained variance (%)')
    axes[1].set_title('Cumulative Explained Variance')
    axes[1].legend()
    axes[1].set_ylim(0, 102)

    plt.suptitle('PCA Explained Variance', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved → {out_path}')


def plot_group_means(df, pca, out_path, n_pcs=5):
    """Bar chart of mean PC scores per group (shows group-level differences)."""
    n_pcs = min(n_pcs, pca.n_components_)
    groups = sorted(df['group'].unique())
    pc_cols = [f'pc{i+1}' for i in range(n_pcs)]

    group_means = df.groupby('group')[pc_cols].mean()
    group_sems  = df.groupby('group')[pc_cols].sem()

    fig, axes = plt.subplots(1, n_pcs, figsize=(4 * n_pcs, 4), sharey=False)
    if n_pcs == 1:
        axes = [axes]

    extra = cycle(DEFAULT_COLOURS)
    colours = {g: _group_colour(g, extra) for g in groups}

    for i, (ax, pc) in enumerate(zip(axes, pc_cols)):
        means = group_means[pc]
        sems  = group_sems[pc]
        bars = ax.bar(groups, means, yerr=sems, color=[colours[g] for g in groups],
                      capsize=5, edgecolor='white')
        ax.axhline(0, color='black', lw=0.8)
        ax.set_title(f'{pc.upper()}\n({pca.explained_variance_ratio_[i]*100:.1f}%)')
        ax.set_ylabel('Mean PC score' if i == 0 else '')
        ax.tick_params(axis='x', rotation=20)

    plt.suptitle('Mean PC Scores by Group\n(error bars = SEM)', fontsize=12,
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved → {out_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Compare movement kinematics across populations via PCA',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument('--data-dir', default=None,
                    help='Root directory containing <group>/<recording>.csv sub-dirs')
    ap.add_argument('--files', nargs='+', default=None,
                    metavar='GROUP:PATH',
                    help='Explicit files as "group:path" pairs')
    ap.add_argument('--fps', type=float, default=30.0,
                    help='Recording frame rate (Hz) — applied to all files')
    ap.add_argument('--n-components', type=int, default=10,
                    help='Number of PCA components to retain')
    ap.add_argument('-o', '--output', default='output/movement_pca',
                    help='Output directory for figures and CSV')
    args = ap.parse_args()

    if args.data_dir is None and args.files is None:
        ap.error('Provide --data-dir or --files')

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load recordings ───────────────────────────────────────────────────────
    print('Loading recordings …')
    records = collect_recordings(
        data_dir=args.data_dir,
        file_specs=args.files,
        fps=args.fps,
    )

    if not records:
        print('ERROR: No valid recordings found.')
        return

    groups = sorted({r['group'] for r in records})
    print(f'\n{len(records)} recordings across {len(groups)} groups: '
          f'{", ".join(groups)}')
    for g in groups:
        g_recs = [r for r in records if r['group'] == g]
        total_frames = sum(r['n_frames'] for r in g_recs)
        print(f'  {g}: {len(g_recs)} recording(s), {total_frames} frames')

    # ── PCA ───────────────────────────────────────────────────────────────────
    print('\nFitting PCA …')
    pca, scaler, df_proj = run_pca(records, n_components=args.n_components)

    print(f'  PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% of variance')
    print(f'  PC1+PC2 explain {pca.explained_variance_ratio_[:2].sum()*100:.1f}%')
    print(f'  PC1–{args.n_components} explain '
          f'{pca.explained_variance_ratio_.sum()*100:.1f}%')

    # ── Figures ───────────────────────────────────────────────────────────────
    print('\nSaving figures …')
    sns.set_style('whitegrid')
    plt.rcParams.update({'font.size': 9})

    plot_scatter(df_proj, pca, out_dir / 'pca_scatter.png')
    plot_trajectories(df_proj, pca, out_dir / 'pca_trajectories.png')
    plot_loadings(pca, out_dir / 'pca_loadings.png')
    plot_variance(pca, out_dir / 'pca_variance.png')
    plot_group_means(df_proj, pca, out_dir / 'pca_group_means.png')

    # ── Summary CSV ───────────────────────────────────────────────────────────
    csv_path = out_dir / 'pca_summary.csv'
    df_proj.round(4).to_csv(csv_path, index=False)
    print(f'  Saved → {csv_path}')

    # ── Per-group feature means (in original units, un-scaled) ───────────────
    all_feat  = np.concatenate([r['features'] for r in records])
    all_labels = np.concatenate([[r['group']] * r['n_frames'] for r in records])
    feat_df = pd.DataFrame(all_feat, columns=FEATURE_NAMES)
    feat_df['group'] = all_labels
    feat_means = feat_df.groupby('group').mean().round(3)
    feat_means_path = out_dir / 'feature_means_by_group.csv'
    feat_means.to_csv(feat_means_path)
    print(f'  Saved → {feat_means_path}')

    print(f'\nAll outputs in {out_dir}/')


if __name__ == '__main__':
    main()
