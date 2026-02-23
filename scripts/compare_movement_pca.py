#!/usr/bin/env python3
"""Compare movement kinematics across populations via PCA.

Reads one or more 3D COCO-17 keypoint CSV files per population group
(e.g. infant, adult, nhp), extracts a per-frame kinematic feature vector,
runs joint PCA across all groups, and produces interpretable visualisations
of how groups differ in movement space.

Feature selection (--features):
    angles  — 8 joint angles in degrees; geometry-invariant (recommended
              for cross-population comparison — unaffected by the skeletal prior)
    speeds  — 8 endpoint speeds in m/s + upper/lower ratio (9 features);
              body-scale dependent, use --prior to correct proportions
    all     — all 17 features (default)

Skeletal prior (--prior):
    SAM-3D-Body fits the MHR (adult human) body model to every input video,
    so keypoint *positions* are expressed in adult MHR skeleton units regardless of the
    subject's true body proportions.  The skeletal prior rescales each limb segment
    along its detected direction to match literature body proportions for that group.
    This corrects endpoint speeds — joint angles are invariant to this rescaling
    (directions are preserved, only segment lengths change).

    none    — no correction (default, backward-compatible)
    auto    — built-in defaults: infant→infant, adult→adult, nhp→macaque
    GROUP:PRIOR_NAME [...]  — explicit per-group mapping (e.g. "nhp:chimp")

Available built-in priors and their segment scale factors
(each factor = group_segment/height ÷ adult_segment/height):

    adult    — reference (all factors = 1.0); Winter (2009)
    infant   — 3–6 month old; Schneider et al. (2007); Heineman et al. (2008);
               WHO Child Growth Standards (2006)
    macaque  — Macaca mulatta; Schultz (1956); Turnquist & Kessler (1989)
    chimp    — Pan troglodytes; Schultz (1956); Kimura et al. (1979)

Outputs (all in --output directory):
    pca_scatter.png        — PC1 vs PC2 scatter, coloured by group
    pca_trajectories.png   — PC1/PC2 time trajectories per recording
    pca_loadings.png       — feature loadings for PC1…PC3 (bar chart)
    pca_variance.png       — cumulative explained variance
    pca_group_means.png    — mean PC scores per group with SEM error bars
    pca_summary.csv        — per-frame PCA coordinates + metadata
    feature_means_by_group.csv

Data directory layout (--data-dir):
    data/movement_pca/
        infant/recording_01.csv ...
        adult/recording_01.csv  ...
        nhp/recording_01.csv    ...

Each CSV must be the long-format COCO-17 output from this pipeline
(columns: frame, x, y, z, part_idx).

Usage:
    python scripts/compare_movement_pca.py \\
        --data-dir data/movement_pca/ --fps 30 \\
        --features angles --prior auto \\
        -o output/movement_pca/

    python scripts/compare_movement_pca.py \\
        --files "infant:path/infant.csv" "adult:path/adult.csv" "nhp:path/nhp.csv" \\
        --fps 30 --features speeds --prior "infant:infant" "nhp:chimp" \\
        -o output/movement_pca/
"""

import argparse
import sys
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

# ── COCO-17 joint definitions ──────────────────────────────────────────────────

JOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
]
J = {name: i for i, name in enumerate(JOINT_NAMES)}

# Joint angles: (label, proximal, vertex, distal)
ANGLE_DEFS = [
    ('L knee',     J['left_hip'],        J['left_knee'],      J['left_ankle']),
    ('R knee',     J['right_hip'],       J['right_knee'],     J['right_ankle']),
    ('L elbow',    J['left_shoulder'],   J['left_elbow'],     J['left_wrist']),
    ('R elbow',    J['right_shoulder'],  J['right_elbow'],    J['right_wrist']),
    ('L hip',      J['left_shoulder'],   J['left_hip'],       J['left_knee']),
    ('R hip',      J['right_shoulder'],  J['right_hip'],      J['right_knee']),
    ('L shoulder', J['left_hip'],        J['left_shoulder'],  J['left_elbow']),
    ('R shoulder', J['right_hip'],       J['right_shoulder'], J['right_elbow']),
]

SPEED_JOINTS = {
    'L wrist': J['left_wrist'],   'R wrist': J['right_wrist'],
    'L ankle': J['left_ankle'],   'R ankle': J['right_ankle'],
    'L elbow': J['left_elbow'],   'R elbow': J['right_elbow'],
    'L knee':  J['left_knee'],    'R knee':  J['right_knee'],
}

# All 17 feature names in canonical order
FEATURE_NAMES_ALL = (
    [label for label, *_ in ANGLE_DEFS] +             # 0-7:  8 angles
    [f'{name} speed' for name in SPEED_JOINTS] +      # 8-15: 8 speeds
    ['upper/lower speed ratio']                        # 16:   ratio
)

# Indices into the 17-feature vector for each --features mode
FEATURE_GROUPS = {
    'angles': list(range(8)),        # geometry-invariant, unaffected by prior
    'speeds': list(range(8, 17)),    # body-scale dependent (8 speeds + ratio)
    'all':    list(range(17)),
}

# Group colour palette
GROUP_COLOURS = {
    'infant': '#E74C3C',
    'adult':  '#2980B9',
    'nhp':    '#27AE60',
}
DEFAULT_COLOURS = ['#8E44AD', '#F39C12', '#1ABC9C', '#E67E22']


# ── Skeletal priors ────────────────────────────────────────────────────────────
#
# Each prior maps segment name → scale factor relative to the adult MHR reference.
# Scale factor = (group segment / group height) ÷ (adult segment / adult height)
#
# Segment definitions (COCO-17):
#   upper_arm: shoulder → elbow
#   forearm:   elbow    → wrist
#   thigh:     hip      → knee
#   shank:     knee     → ankle
#
# Adult reference segment proportions (% of standing height), Winter (2009):
#   upper_arm 18.6%,  forearm 14.6%,  thigh 24.5%,  shank 24.6%
#
# The correction rescales each segment vector along its detected direction,
# propagating distally from the trunk outward.  Joint ANGLES are invariant
# to this operation (directions preserved); speeds scale accordingly.

SKELETAL_PRIORS = {
    # ── Adult (reference) ────────────────────────────────────────────────────
    # Winter, D.A. (2009). Biomechanics and Motor Control of Human Movement.
    'adult': {
        'upper_arm': 1.0,
        'forearm':   1.0,
        'thigh':     1.0,
        'shank':     1.0,
    },

    # ── Human infant, 3–6 months (~67 cm standing height) ──────────────────
    # Infant proportions (% of height):
    #   upper_arm 12.7% → ratio 12.7/18.6 = 0.68
    #   forearm   11.9% → ratio 11.9/14.6 = 0.82
    #   thigh     21.6% → ratio 21.6/24.5 = 0.88
    #   shank     16.4% → ratio 16.4/24.6 = 0.67
    # Sources: Schneider, K. et al. (2007) J Biomech; Heineman, K.R. et al.
    # (2008) Dev Med Child Neurol; WHO Child Growth Standards (2006).
    'infant': {
        'upper_arm': 0.68,
        'forearm':   0.82,
        'thigh':     0.88,
        'shank':     0.67,
    },

    # ── Rhesus macaque, Macaca mulatta (~50 cm standing height) ────────────
    # Macaque proportions (% of height):
    #   upper_arm 21.0% → ratio 21.0/18.6 = 1.13  (arms longer relative to body)
    #   forearm   21.6% → ratio 21.6/14.6 = 1.48  (macaque forearm ≈ humerus length)
    #   thigh     25.4% → ratio 25.4/24.5 = 1.04
    #   shank     23.0% → ratio 23.0/24.6 = 0.94
    # Sources: Schultz, A.H. (1956) Primatologia; Turnquist, J.E. & Kessler,
    # M.J. (1989) Am J Phys Anthropol; Jenkins, F.A. & Weijs, W.A. (1979).
    'macaque': {
        'upper_arm': 1.13,
        'forearm':   1.48,
        'thigh':     1.04,
        'shank':     0.94,
    },

    # ── Common chimpanzee, Pan troglodytes (~130 cm standing height) ────────
    # Chimp proportions (% of height); IMI ≈ 101 (arms longer than legs):
    #   upper_arm 23.1% → ratio 23.1/18.6 = 1.24
    #   forearm   20.9% → ratio 20.9/14.6 = 1.43
    #   thigh     18.5% → ratio 18.5/24.5 = 0.75
    #   shank     16.2% → ratio 16.2/24.6 = 0.66
    # Sources: Schultz, A.H. (1956) Primatologia; Kimura, T. et al. (1979)
    # Primates; Goodall, J. (1986) The Chimpanzees of Gombe.
    'chimp': {
        'upper_arm': 1.24,
        'forearm':   1.43,
        'thigh':     0.75,
        'shank':     0.66,
    },
}

# Default group → prior mapping for --prior auto
DEFAULT_PRIOR_MAP = {
    'infant': 'infant',
    'adult':  'adult',
    'nhp':    'macaque',
}


def apply_skeletal_prior(positions, prior_name):
    """Rescale limb segments to correct for body proportion differences.

    Args:
        positions: (F, 17, 3) float array of COCO-17 keypoint positions
        prior_name: key into SKELETAL_PRIORS, or 'adult' / None for no-op

    Returns:
        corrected (F, 17, 3) array — trunk unchanged, limbs rescaled distally
    """
    if prior_name is None or prior_name == 'adult':
        return positions
    if prior_name not in SKELETAL_PRIORS:
        raise ValueError(f"Unknown prior '{prior_name}'. "
                         f"Available: {list(SKELETAL_PRIORS)}")
    s = SKELETAL_PRIORS[prior_name]
    pos = positions.copy()

    # Left arm — proximal to distal
    dir_ua = pos[:, J['left_elbow']]   - pos[:, J['left_shoulder']]
    pos[:, J['left_elbow']]  = pos[:, J['left_shoulder']] + s['upper_arm'] * dir_ua
    dir_fa = pos[:, J['left_wrist']]   - pos[:, J['left_elbow']]   # uses updated elbow
    pos[:, J['left_wrist']]  = pos[:, J['left_elbow']]  + s['forearm']   * dir_fa

    # Right arm
    dir_ua = pos[:, J['right_elbow']]  - pos[:, J['right_shoulder']]
    pos[:, J['right_elbow']] = pos[:, J['right_shoulder']] + s['upper_arm'] * dir_ua
    dir_fa = pos[:, J['right_wrist']]  - pos[:, J['right_elbow']]
    pos[:, J['right_wrist']] = pos[:, J['right_elbow']] + s['forearm']   * dir_fa

    # Left leg
    dir_th = pos[:, J['left_knee']]    - pos[:, J['left_hip']]
    pos[:, J['left_knee']]   = pos[:, J['left_hip']]    + s['thigh'] * dir_th
    dir_sh = pos[:, J['left_ankle']]   - pos[:, J['left_knee']]
    pos[:, J['left_ankle']]  = pos[:, J['left_knee']]   + s['shank'] * dir_sh

    # Right leg
    dir_th = pos[:, J['right_knee']]   - pos[:, J['right_hip']]
    pos[:, J['right_knee']]  = pos[:, J['right_hip']]   + s['thigh'] * dir_th
    dir_sh = pos[:, J['right_ankle']]  - pos[:, J['right_knee']]
    pos[:, J['right_ankle']] = pos[:, J['right_knee']]  + s['shank'] * dir_sh

    return pos


def _parse_prior_arg(prior_tokens, known_groups):
    """Parse --prior argument tokens into a {group: prior_name} dict.

    Accepts:
        ['none']          → {}  (no correction)
        ['auto']          → DEFAULT_PRIOR_MAP filtered to known_groups
        ['G1:P1', 'G2:P2'] → {G1: P1, G2: P2}
    """
    if not prior_tokens or prior_tokens == ['none']:
        return {}
    if prior_tokens == ['auto']:
        return {g: DEFAULT_PRIOR_MAP[g]
                for g in known_groups if g in DEFAULT_PRIOR_MAP}
    result = {}
    for tok in prior_tokens:
        if ':' not in tok:
            print(f"WARNING: --prior token '{tok}' is not in GROUP:PRIOR format; skipping.",
                  file=sys.stderr)
            continue
        group, pname = tok.split(':', 1)
        if pname not in SKELETAL_PRIORS:
            print(f"WARNING: unknown prior '{pname}' for group '{group}'. "
                  f"Available: {list(SKELETAL_PRIORS)}. Skipping.", file=sys.stderr)
            continue
        result[group] = pname
    return result


# ── Signal processing helpers ──────────────────────────────────────────────────

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


# ── Data loading ───────────────────────────────────────────────────────────────

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


# ── Feature extraction ─────────────────────────────────────────────────────────

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


def extract_features(positions, fps, feature_idxs=None):
    """Return (F, n_features) float32 feature matrix for one recording.

    Args:
        positions:    (F, 17, 3) raw positions (after prior correction if any)
        fps:          frame rate in Hz
        feature_idxs: list of indices into the 17-feature vector to return;
                      None → all 17 features

    NaN rows are dropped before returning.
    """
    F = positions.shape[0]

    # Interpolate gaps, then smooth positions
    flat = positions.reshape(F, -1)
    flat = _interpolate(flat)
    pos_sm = flat.reshape(F, 17, 3)
    pos_sm = _sg(pos_sm, fps, window_s=0.4)

    # 8 joint angles
    angles = np.stack([_angle(pos_sm, p, v, d)
                       for _, p, v, d in ANGLE_DEFS], axis=1)   # (F, 8)

    # 8 endpoint speeds
    vel = _sg(pos_sm, fps, window_s=0.4, deriv=1)               # (F, 17, 3) m/s
    speeds = np.stack([np.linalg.norm(vel[:, idx], axis=1)
                       for idx in SPEED_JOINTS.values()], axis=1)  # (F, 8)

    # Upper / lower limb speed ratio
    upper_spd = speeds[:, [0, 1, 4, 5]].mean(axis=1)  # wrists + elbows
    lower_spd = speeds[:, [2, 3, 6, 7]].mean(axis=1)  # ankles + knees
    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = upper_spd / (lower_spd + 1e-6)

    feat = np.concatenate([angles, speeds, ratio[:, None]], axis=1)  # (F, 17)

    if feature_idxs is not None:
        feat = feat[:, feature_idxs]

    # Drop frames with any NaN
    valid = ~np.isnan(feat).any(axis=1)
    return feat[valid].astype(np.float32)


# ── Collect all recordings ─────────────────────────────────────────────────────

def collect_recordings(data_dir=None, file_specs=None, fps=30.0,
                       prior_map=None, feature_idxs=None):
    """Return list of dicts: {group, name, features (F, n_feat), n_frames}.

    Args:
        data_dir:     Path — scan <data_dir>/<group>/*.csv
        file_specs:   list of "group:path" strings
        fps:          frame rate in Hz
        prior_map:    {group: prior_name} — skeletal prior per group
        feature_idxs: list of feature indices to retain (None = all 17)
    """
    prior_map = prior_map or {}
    records = []

    def _process(group, csv_path):
        _, pos = load_csv(csv_path)
        prior_name = prior_map.get(group)
        if prior_name:
            pos = apply_skeletal_prior(pos, prior_name)
        feat = extract_features(pos, fps, feature_idxs=feature_idxs)
        if len(feat) > 0:
            prior_tag = f' [{prior_name} prior]' if prior_name else ''
            print(f'  [{group}] {Path(csv_path).name}: {len(feat)} frames{prior_tag}')
            records.append({'group': group, 'name': Path(csv_path).stem,
                            'features': feat, 'n_frames': len(feat)})

    if data_dir is not None:
        for group_dir in sorted(Path(data_dir).iterdir()):
            if not group_dir.is_dir():
                continue
            for csv_path in sorted(group_dir.glob('*.csv')):
                _process(group_dir.name, csv_path)

    if file_specs:
        for spec in file_specs:
            group, path = spec.split(':', 1)
            _process(group, path)

    return records


# ── PCA ────────────────────────────────────────────────────────────────────────

def run_pca(records, n_components=10):
    """Fit PCA on all frames combined.

    Returns:
        pca:     fitted sklearn PCA
        scaler:  fitted StandardScaler
        df_proj: DataFrame with columns pc1…pcN, group, recording, frame_local
    """
    all_feat = np.concatenate([r['features'] for r in records], axis=0)

    scaler = StandardScaler()
    all_scaled = scaler.fit_transform(all_feat)

    n_components = min(n_components, all_scaled.shape[1], all_scaled.shape[0])
    pca = PCA(n_components=n_components, random_state=42)
    all_proj = pca.fit_transform(all_scaled)

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

    return pca, scaler, pd.DataFrame(rows)


# ── Plotting ───────────────────────────────────────────────────────────────────

def _group_colour(group, extra_colours):
    return GROUP_COLOURS.get(group, next(extra_colours))


def plot_scatter(df, pca, out_path):
    """PC1 vs PC2 scatter + KDE, coloured by group."""
    groups = df['group'].unique()
    extra = cycle(DEFAULT_COLOURS)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    for group in sorted(groups):
        sub = df[df['group'] == group]
        if len(sub) > 2000:
            sub = sub.sample(2000, random_state=42)
        c = _group_colour(group, extra)
        ax.scatter(sub['pc1'], sub['pc2'], alpha=0.25, s=8,
                   color=c, label=group, rasterized=True)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('PC1 vs PC2 — all frames')
    ax.legend(markerscale=3, framealpha=0.8)

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
    if ax2.get_legend_handles_labels()[0]:
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
                ax.plot(sub_r['frame_local'].values, sub_r[pc].values,
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


def plot_loadings(pca, out_path, feature_names, n_pcs=3):
    """Feature loadings for the first n_pcs components."""
    n_pcs = min(n_pcs, pca.n_components_)
    components = pca.components_[:n_pcs]   # (n_pcs, n_features)

    fig, axes = plt.subplots(1, n_pcs, figsize=(5 * n_pcs, 5), sharey=False)
    if n_pcs == 1:
        axes = [axes]

    colours = ['#E74C3C', '#2980B9', '#27AE60']
    for i, (ax, comp, col) in enumerate(zip(axes, components, colours)):
        sorted_idx = np.argsort(np.abs(comp))[::-1]
        names_sorted = [feature_names[j] for j in sorted_idx]
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
    """Bar chart of mean PC scores per group with SEM error bars."""
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
        ax.bar(groups, means, yerr=sems, color=[colours[g] for g in groups],
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


# ── Main ───────────────────────────────────────────────────────────────────────

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
    ap.add_argument('--features', choices=['angles', 'speeds', 'all'], default='all',
                    help=('"angles": 8 joint angles only (geometry-invariant, '
                          'unaffected by --prior); '
                          '"speeds": 8 speeds + upper/lower ratio; '
                          '"all": all 17 features'))
    ap.add_argument('--prior', nargs='+', default=['none'],
                    metavar='KEYWORD_OR_GROUP:PRIOR',
                    help=('Skeletal prior for body proportion correction. '
                          '"none": no correction; "auto": use defaults '
                          f'({DEFAULT_PRIOR_MAP}); '
                          'or explicit "GROUP:PRIOR" pairs (e.g. "nhp:chimp"). '
                          f'Available priors: {list(SKELETAL_PRIORS)}'))
    ap.add_argument('--list-priors', action='store_true',
                    help='Print available skeletal priors and exit')
    ap.add_argument('--n-components', type=int, default=10,
                    help='Number of PCA components to retain')
    ap.add_argument('-o', '--output', default='output/movement_pca',
                    help='Output directory for figures and CSV')
    args = ap.parse_args()

    if args.list_priors:
        print('Available skeletal priors:')
        for name, scales in SKELETAL_PRIORS.items():
            print(f'  {name:10s}  upper_arm={scales["upper_arm"]:.2f}  '
                  f'forearm={scales["forearm"]:.2f}  '
                  f'thigh={scales["thigh"]:.2f}  shank={scales["shank"]:.2f}')
        print(f'\nDefault auto mapping: {DEFAULT_PRIOR_MAP}')
        sys.exit(0)

    if args.data_dir is None and args.files is None:
        ap.error('Provide --data-dir or --files')

    # Resolve feature indices and names
    feature_idxs = FEATURE_GROUPS[args.features]
    active_feature_names = [FEATURE_NAMES_ALL[i] for i in feature_idxs]
    print(f'Feature set: {args.features} ({len(feature_idxs)} features)')
    if args.features == 'angles':
        print('  Note: joint angles are geometry-invariant — skeletal prior '
              'does not affect angle values.')

    # Resolve prior map (need group names first for 'auto')
    # We'll parse after loading so we know the groups; do a two-pass approach
    # by pre-scanning group names from file specs / data_dir
    raw_groups: set[str] = set()
    if args.data_dir:
        for d in Path(args.data_dir).iterdir():
            if d.is_dir():
                raw_groups.add(d.name)
    if args.files:
        for spec in args.files:
            raw_groups.add(spec.split(':', 1)[0])

    prior_map = _parse_prior_arg(args.prior, raw_groups)
    if prior_map:
        print('Skeletal priors:')
        for g in sorted(raw_groups):
            pname = prior_map.get(g, 'none (adult reference)')
            print(f'  {g} → {pname}')
    else:
        print('Skeletal prior: none')

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load recordings ────────────────────────────────────────────────────────
    print('\nLoading recordings …')
    records = collect_recordings(
        data_dir=args.data_dir,
        file_specs=args.files,
        fps=args.fps,
        prior_map=prior_map,
        feature_idxs=feature_idxs,
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

    # ── PCA ────────────────────────────────────────────────────────────────────
    print('\nFitting PCA …')
    pca, scaler, df_proj = run_pca(records, n_components=args.n_components)

    print(f'  PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% of variance')
    print(f'  PC1+PC2 explain {pca.explained_variance_ratio_[:2].sum()*100:.1f}%')
    print(f'  PC1–{args.n_components} explain '
          f'{pca.explained_variance_ratio_.sum()*100:.1f}%')

    # ── Figures ────────────────────────────────────────────────────────────────
    print('\nSaving figures …')
    sns.set_style('whitegrid')
    plt.rcParams.update({'font.size': 9})

    plot_scatter(df_proj, pca, out_dir / 'pca_scatter.png')
    plot_trajectories(df_proj, pca, out_dir / 'pca_trajectories.png')
    plot_loadings(pca, out_dir / 'pca_loadings.png',
                  feature_names=active_feature_names)
    plot_variance(pca, out_dir / 'pca_variance.png')
    plot_group_means(df_proj, pca, out_dir / 'pca_group_means.png')

    # ── Summary CSV ────────────────────────────────────────────────────────────
    csv_path = out_dir / 'pca_summary.csv'
    df_proj.round(4).to_csv(csv_path, index=False)
    print(f'  Saved → {csv_path}')

    # ── Per-group feature means ────────────────────────────────────────────────
    all_feat   = np.concatenate([r['features'] for r in records])
    all_labels = np.concatenate([[r['group']] * r['n_frames'] for r in records])
    feat_df = pd.DataFrame(all_feat, columns=active_feature_names)
    feat_df['group'] = all_labels
    feat_means = feat_df.groupby('group').mean().round(3)
    feat_means_path = out_dir / 'feature_means_by_group.csv'
    feat_means.to_csv(feat_means_path)
    print(f'  Saved → {feat_means_path}')

    print(f'\nAll outputs in {out_dir}/')


if __name__ == '__main__':
    main()
