#!/usr/bin/env python3
"""
Cluster video frames based on primary metrics.

This module performs k-means and GMM clustering on primary metrics from video analysis.
It operates on the CSV output from process_video.py.

Philosophy:
- Fail fast: No extensive error handling
- Let failures reveal problems
- Scientific code: Clear, simple, reproducible

Usage:
    python process_clusters.py <csv_file> [options]

Example:
    python process_clusters.py data/output/N29_3M2pM6dispA7_default/N29_3M2pM6dispA7_default_basic.csv

Output:
    - clusters.csv: Cluster assignments for each frame
    - cluster_scores.json: Quality metrics (silhouette, BIC, etc.)
    - cluster_exemplars.json: Representative frames per cluster
    - MIDI files: One per k_value with tracks for each boxcar period
"""

import numpy as np
import pandas as pd
import json
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import argparse
import os
import mido


def load_and_prepare_data(csv_path, metrics_to_exclude=None):
    """
    Load primary metrics CSV and prepare data for clustering.

    Drops constant columns (std = 0) automatically.

    Parameters:
    - csv_path: Path to basic metrics CSV
    - metrics_to_exclude: List of column names to exclude (e.g., motion metrics)

    Returns:
    - df: Full dataframe
    - metric_cols: List of column names used for clustering
    - data: numpy array of metric values (frames x metrics)
    """
    df = pd.read_csv(csv_path, index_col=0)

    if metrics_to_exclude is None:
        metrics_to_exclude = []

    # Get all numeric columns except frame_count
    all_cols = [col for col in df.columns if col != 'frame_count']

    # Exclude specified metrics
    metric_cols = [col for col in all_cols if col not in metrics_to_exclude]

    # Extract data
    data = df[metric_cols].values

    # Drop constant columns (std = 0)
    stds = np.std(data, axis=0)
    non_constant = stds > 0

    dropped_cols = [metric_cols[i] for i in range(len(metric_cols)) if not non_constant[i]]
    if dropped_cols:
        print(f"Dropped {len(dropped_cols)} constant columns: {dropped_cols[:5]}{'...' if len(dropped_cols) > 5 else ''}")

    metric_cols = [metric_cols[i] for i in range(len(metric_cols)) if non_constant[i]]
    data = data[:, non_constant]

    print(f"Loaded {len(df)} frames with {len(metric_cols)} metrics")

    return df, metric_cols, data


def normalize_data(data, method='rank'):
    """
    Normalize data to make metrics commensurate.

    Parameters:
    - data: numpy array (frames x metrics)
    - method: 'rank' or 'zscore'

    Returns:
    - normalized_data: numpy array with same shape
    """
    if method == 'rank':
        # Rank transformation: each column becomes ranks / (n+1)
        # This maps to [0, 1] and is robust to outliers
        normalized_data = np.zeros_like(data, dtype=float)
        n = data.shape[0]
        for i in range(data.shape[1]):
            # rankdata handles ties by averaging ranks
            normalized_data[:, i] = rankdata(data[:, i], method='average') / (n + 1)
        print(f"Normalized data using rank transformation")

    elif method == 'zscore':
        # Z-score: (x - mean) / std
        means = np.mean(data, axis=0)
        stds = np.std(data, axis=0)
        # Avoid division by zero (already handled by dropping constants, but safety)
        stds[stds == 0] = 1.0
        normalized_data = (data - means) / stds
        print(f"Normalized data using z-score transformation")

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized_data


def cluster_kmeans(data, k_values, random_state=42):
    """
    Perform K-Means clustering for multiple k values.

    Parameters:
    - data: normalized data (frames x metrics)
    - k_values: list of k values to try
    - random_state: for reproducibility

    Returns:
    - results: dict mapping k -> {'labels': array, 'model': KMeans object, 'centroids': array}
    """
    results = {}

    for k in k_values:
        if k >= len(data):
            print(f"Skipping k={k}: fewer frames ({len(data)}) than clusters")
            continue

        print(f"Running K-Means with k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        labels = kmeans.fit_predict(data)

        results[k] = {
            'labels': labels,
            'model': kmeans,
            'centroids': kmeans.cluster_centers_
        }

        # Show cluster sizes
        counts = np.bincount(labels)
        print(f"  K-Means k={k}: cluster sizes = {counts.tolist()}")

    return results


def cluster_gmm(data, k_values, random_state=42):
    """
    Perform Gaussian Mixture Model clustering for multiple k values.

    Parameters:
    - data: normalized data (frames x metrics)
    - k_values: list of k values to try
    - random_state: for reproducibility

    Returns:
    - results: dict mapping k -> {'labels': array, 'model': GMM object, 'bic': float, 'aic': float}
    """
    results = {}

    for k in k_values:
        if k >= len(data):
            print(f"Skipping k={k}: fewer frames ({len(data)}) than components")
            continue

        print(f"Running GMM with k={k}...")
        gmm = GaussianMixture(n_components=k, covariance_type='full',
                             random_state=random_state, n_init=3)
        labels = gmm.fit_predict(data)

        results[k] = {
            'labels': labels,
            'model': gmm,
            'bic': gmm.bic(data),
            'aic': gmm.aic(data)
        }

        # Show cluster sizes
        counts = np.bincount(labels)
        print(f"  GMM k={k}: cluster sizes = {counts.tolist()}, BIC = {gmm.bic(data):.2f}")

    return results


def compute_quality_metrics(data, labels_dict, algorithm):
    """
    Compute clustering quality metrics.

    Parameters:
    - data: normalized data
    - labels_dict: dict mapping k -> results dict with 'labels' key
    - algorithm: 'kmeans' or 'gmm'

    Returns:
    - scores: dict mapping k -> quality metrics dict
    """
    scores = {}

    for k, result in labels_dict.items():
        labels = result['labels']

        # Silhouette score (higher is better, range [-1, 1])
        if k > 1 and len(np.unique(labels)) > 1:
            silhouette = silhouette_score(data, labels)
        else:
            silhouette = None

        # Calinski-Harabasz score (higher is better)
        if k > 1 and len(np.unique(labels)) > 1:
            ch_score = calinski_harabasz_score(data, labels)
        else:
            ch_score = None

        # Davies-Bouldin score (lower is better)
        if k > 1 and len(np.unique(labels)) > 1:
            db_score = davies_bouldin_score(data, labels)
        else:
            db_score = None

        scores[k] = {
            'silhouette': silhouette,
            'calinski_harabasz': ch_score,
            'davies_bouldin': db_score
        }

        # Add algorithm-specific metrics
        if algorithm == 'kmeans':
            scores[k]['inertia'] = result['model'].inertia_
        elif algorithm == 'gmm':
            scores[k]['bic'] = result['bic']
            scores[k]['aic'] = result['aic']

        sil_str = f"{silhouette:.3f}" if silhouette is not None else "N/A"
        print(f"  {algorithm.upper()} k={k}: silhouette={sil_str}")

    return scores


def find_exemplars(data, labels_dict, df, n_exemplars=3):
    """
    Find exemplar frames closest to cluster centroids.

    Parameters:
    - data: normalized data
    - labels_dict: dict mapping k -> results dict
    - df: original dataframe (to get frame IDs)
    - n_exemplars: number of exemplars per cluster

    Returns:
    - exemplars: dict mapping k -> list of dicts with cluster info
    """
    exemplars = {}

    for k, result in labels_dict.items():
        labels = result['labels']

        if 'centroids' in result:
            # K-Means has explicit centroids
            centroids = result['centroids']
        else:
            # GMM: compute cluster means
            centroids = []
            for cluster_id in range(k):
                cluster_mask = labels == cluster_id
                cluster_mean = np.mean(data[cluster_mask], axis=0)
                centroids.append(cluster_mean)
            centroids = np.array(centroids)

        exemplars[k] = []

        for cluster_id in range(k):
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_indices) == 0:
                continue

            # Find frames closest to centroid
            centroid = centroids[cluster_id]
            distances = np.linalg.norm(data[cluster_indices] - centroid, axis=1)

            # Get top n_exemplars
            n_to_get = min(n_exemplars, len(distances))
            closest_idx = np.argsort(distances)[:n_to_get]
            frame_ids = df.index[cluster_indices[closest_idx]].tolist()

            exemplars[k].append({
                'cluster_id': int(cluster_id),
                'size': int(len(cluster_indices)),
                'exemplar_frames': frame_ids
            })

    return exemplars


def apply_boxcar_filter(assignments, width):
    """
    Apply boxcar filter to cluster assignments, returning most common cluster ID in window.

    Parameters:
    - assignments: numpy array of cluster IDs
    - width: odd integer, width of boxcar filter

    Returns:
    - smoothed: numpy array of smoothed cluster IDs
    """
    n = len(assignments)
    smoothed = np.zeros(n, dtype=int)
    half_width = width // 2

    for i in range(n):
        # Determine window bounds (truncate at edges)
        start = max(0, i - half_width)
        end = min(n, i + half_width + 1)

        # Get window and find mode (most common value)
        window = assignments[start:end]
        values, counts = np.unique(window, return_counts=True)
        mode_idx = np.argmax(counts)
        smoothed[i] = values[mode_idx]

    return smoothed


def apply_boxcar_to_clusters(cluster_df, boxcar_periods):
    """
    Apply boxcar filtering to all cluster assignment columns.

    Parameters:
    - cluster_df: DataFrame with cluster assignments
    - boxcar_periods: list of boxcar widths (odd integers)

    Returns:
    - cluster_df: DataFrame with added boxcar-filtered columns
    """
    # Get all cluster assignment columns (those starting with 'kmeans_' or 'gmm_')
    cluster_cols = [col for col in cluster_df.columns if col.startswith(('kmeans_', 'gmm_'))]

    print(f"\nApplying boxcar filtering with periods: {boxcar_periods}")

    for col in cluster_cols:
        assignments = cluster_df[col].values

        for period in boxcar_periods:
            # Apply boxcar filter
            smoothed = apply_boxcar_filter(assignments, period)

            # Add column with naming convention: {original}_b{period:03d}
            new_col = f"{col}_b{period:03d}"
            cluster_df[new_col] = smoothed

            # Count changes
            n_changes = np.sum(smoothed != assignments)
            print(f"  {col} -> {new_col}: {n_changes}/{len(assignments)} assignments changed")

    return cluster_df


def generate_cluster_midi(cluster_df, output_dir, base_name,
                         beats_per_minute=64, frames_per_second=30, ticks_per_beat=480):
    """
    Generate MIDI files from cluster assignments.

    Creates one MIDI file per k_value, with each boxcar period as a separate track.

    Parameters:
    - cluster_df: DataFrame with cluster assignments
    - output_dir: directory to save MIDI files
    - base_name: base filename for outputs
    - beats_per_minute: tempo for MIDI timing
    - frames_per_second: video frame rate
    - ticks_per_beat: MIDI resolution
    """
    print(f"\nGenerating MIDI files from cluster assignments")

    # Calculate ticks per frame
    ticks_per_frame = ticks_per_beat * beats_per_minute / (60 * frames_per_second)

    # Get frame counts (assuming index contains frame numbers)
    frame_count_list = cluster_df.index.tolist()

    # Find all cluster assignment columns
    cluster_cols = [col for col in cluster_df.columns
                   if col.startswith(('kmeans_', 'gmm_'))]

    # Group columns by algorithm and k_value
    # Format: kmeans_k2, kmeans_k2_b017, gmm_k3, gmm_k3_b065, etc.
    grouped_cols = {}

    for col in cluster_cols:
        # Parse column name to extract algorithm, k_value, and boxcar period
        parts = col.split('_')
        algorithm = parts[0]  # 'kmeans' or 'gmm'
        k_part = parts[1]     # 'k2', 'k3', etc.

        # Create base key (algorithm + k_value)
        base_key = f"{algorithm}_{k_part}"

        # Determine boxcar period
        if len(parts) > 2 and parts[2].startswith('b'):
            boxcar_period = parts[2]  # e.g., 'b017', 'b065'
        else:
            boxcar_period = 'b001'  # Original (unfiltered) assignments

        # Group by base_key
        if base_key not in grouped_cols:
            grouped_cols[base_key] = {}

        grouped_cols[base_key][boxcar_period] = col

    # Generate one MIDI file per k_value
    for base_key, tracks_dict in sorted(grouped_cols.items()):
        midi_file = mido.MidiFile()

        # Sort tracks by boxcar period to ensure consistent ordering
        sorted_periods = sorted(tracks_dict.keys())

        for boxcar_period in sorted_periods:
            col_name = tracks_dict[boxcar_period]

            # Get cluster assignments
            cluster_assignments = cluster_df[col_name].values

            # Calculate scaling factor: midi_value = round(127.0 / max_cluster) * cluster
            max_cluster = cluster_assignments.max()
            if max_cluster == 0:
                scale_factor = 127.0
            else:
                scale_factor = 127.0 / max_cluster

            # Create MIDI track
            midi_track = mido.MidiTrack()
            midi_file.tracks.append(midi_track)

            # Set track name
            track_name = f"{col_name}"
            midi_track.append(mido.MetaMessage('track_name', name=track_name, time=0))

            # Add MIDI events for each frame
            for i, cluster_id in enumerate(cluster_assignments):
                # Scale cluster ID to MIDI range
                midi_value = int(round(scale_factor * cluster_id))
                # Ensure value is in valid MIDI range [0, 127]
                midi_value = max(0, min(127, midi_value))

                # Calculate time delta
                time_tick = 0 if i == 0 else int(ticks_per_frame * (frame_count_list[i] - frame_count_list[i - 1]))

                # Add control change message (using CC 1 by default)
                midi_track.append(
                    mido.Message('control_change',
                               control=1,
                               value=midi_value,
                               channel=7,
                               time=time_tick))

        # Save MIDI file
        midi_filename = os.path.join(output_dir, f"{base_name}_{base_key}.mid")
        midi_file.save(midi_filename)
        print(f"  Saved {midi_filename} with {len(midi_file.tracks)} tracks")


def save_results(df, kmeans_results, gmm_results, kmeans_scores, gmm_scores,
                kmeans_exemplars, gmm_exemplars, output_dir, base_name, boxcar_periods=None,
                beats_per_minute=64, frames_per_second=30, ticks_per_beat=480):
    """
    Save clustering results to CSV, JSON, and MIDI files.

    Parameters:
    - df: original dataframe
    - kmeans_results: K-Means results dict
    - gmm_results: GMM results dict
    - kmeans_scores: K-Means quality scores
    - gmm_scores: GMM quality scores
    - kmeans_exemplars: K-Means exemplars
    - gmm_exemplars: GMM exemplars
    - output_dir: directory to save outputs
    - base_name: base filename (e.g., "N29_3M2pM6dispA7_default")
    - boxcar_periods: list of boxcar filter periods (optional)
    - beats_per_minute: tempo for MIDI timing (default: 64)
    - frames_per_second: video frame rate (default: 30)
    - ticks_per_beat: MIDI resolution (default: 480)
    """
    # Create output dataframe with cluster assignments
    cluster_df = df[['frame_count']].copy() if 'frame_count' in df.columns else pd.DataFrame(index=df.index)

    # Add K-Means cluster assignments
    for k, result in kmeans_results.items():
        cluster_df[f'kmeans_k{k}'] = result['labels']

    # Add GMM cluster assignments
    for k, result in gmm_results.items():
        cluster_df[f'gmm_k{k}'] = result['labels']

    # Apply boxcar filtering if requested
    if boxcar_periods:
        cluster_df = apply_boxcar_to_clusters(cluster_df, boxcar_periods)

    # Save cluster assignments CSV
    cluster_csv_path = os.path.join(output_dir, f"{base_name}_clusters.csv")
    cluster_df.to_csv(cluster_csv_path)
    print(f"\nSaved cluster assignments to: {cluster_csv_path}")

    # Save quality scores JSON
    scores = {
        'kmeans': {str(k): v for k, v in kmeans_scores.items()},
        'gmm': {str(k): v for k, v in gmm_scores.items()}
    }
    scores_json_path = os.path.join(output_dir, f"{base_name}_cluster_scores.json")
    with open(scores_json_path, 'w') as f:
        json.dump(scores, f, indent=2)
    print(f"Saved quality scores to: {scores_json_path}")

    # Save exemplars JSON
    exemplars = {
        'kmeans': {str(k): v for k, v in kmeans_exemplars.items()},
        'gmm': {str(k): v for k, v in gmm_exemplars.items()}
    }
    exemplars_json_path = os.path.join(output_dir, f"{base_name}_cluster_exemplars.json")
    with open(exemplars_json_path, 'w') as f:
        json.dump(exemplars, f, indent=2)
    print(f"Saved exemplars to: {exemplars_json_path}")

    # Generate MIDI files from cluster assignments
    generate_cluster_midi(cluster_df, output_dir, base_name,
                         beats_per_minute=beats_per_minute,
                         frames_per_second=frames_per_second,
                         ticks_per_beat=ticks_per_beat)


def cluster_primary_metrics(csv_path, k_values=None, normalization='rank',
                           metrics_to_exclude=None, random_state=42, boxcar_periods=None,
                           beats_per_minute=64, frames_per_second=30, ticks_per_beat=480):
    """
    Main entry point for clustering primary metrics.

    Parameters:
    - csv_path: Path to basic metrics CSV
    - k_values: List of k values to try (default: [2,3,4,5,6,8,10,12])
    - normalization: 'rank' or 'zscore' (default: 'rank')
    - metrics_to_exclude: List of metric names to exclude
    - random_state: Random seed for reproducibility
    - boxcar_periods: List of boxcar filter widths (default: None)
    - beats_per_minute: tempo for MIDI timing (default: 64)
    - frames_per_second: video frame rate (default: 30)
    - ticks_per_beat: MIDI resolution (default: 480)
    """
    if k_values is None:
        k_values = [2, 3, 4, 5, 6, 8, 10, 12]

    print(f"="*60)
    print(f"CLUSTERING PRIMARY METRICS")
    print(f"="*60)
    print(f"CSV: {csv_path}")
    print(f"K values: {k_values}")
    print(f"Normalization: {normalization}")
    print(f"Random state: {random_state}")
    if boxcar_periods:
        print(f"Boxcar periods: {boxcar_periods}")
    print()

    # Load and prepare data
    df, metric_cols, data = load_and_prepare_data(csv_path, metrics_to_exclude)

    # Normalize data
    normalized_data = normalize_data(data, method=normalization)

    # Run K-Means clustering
    print(f"\n{'='*60}")
    print(f"K-MEANS CLUSTERING")
    print(f"{'='*60}")
    kmeans_results = cluster_kmeans(normalized_data, k_values, random_state)
    kmeans_scores = compute_quality_metrics(normalized_data, kmeans_results, 'kmeans')
    kmeans_exemplars = find_exemplars(normalized_data, kmeans_results, df)

    # Run GMM clustering
    print(f"\n{'='*60}")
    print(f"GAUSSIAN MIXTURE MODEL CLUSTERING")
    print(f"{'='*60}")
    gmm_results = cluster_gmm(normalized_data, k_values, random_state)
    gmm_scores = compute_quality_metrics(normalized_data, gmm_results, 'gmm')
    gmm_exemplars = find_exemplars(normalized_data, gmm_results, df)

    # Save results
    print(f"\n{'='*60}")
    print(f"SAVING RESULTS")
    print(f"{'='*60}")
    output_dir = os.path.dirname(csv_path)
    base_name = os.path.basename(csv_path).replace('_basic.csv', '')

    save_results(df, kmeans_results, gmm_results, kmeans_scores, gmm_scores,
                kmeans_exemplars, gmm_exemplars, output_dir, base_name, boxcar_periods,
                beats_per_minute, frames_per_second, ticks_per_beat)

    print(f"\n{'='*60}")
    print(f"CLUSTERING COMPLETE")
    print(f"{'='*60}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Cluster video frames based on primary metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_clusters.py data/output/N29_*/N29_*_basic.csv
  python process_clusters.py data/output/N29_*/N29_*_basic.csv --k-values 2 3 4 5
  python process_clusters.py data/output/N29_*/N29_*_basic.csv --normalization zscore
        """
    )

    parser.add_argument('csv_path', help='Path to basic metrics CSV file')
    parser.add_argument('--k-values', nargs='+', type=int, default=[2, 3, 4, 5, 6, 8, 10, 12],
                       help='K values to try (default: 2 3 4 5 6 8 10 12)')
    parser.add_argument('--normalization', choices=['rank', 'zscore'], default='rank',
                       help='Normalization method (default: rank)')
    parser.add_argument('--exclude', nargs='*', default=None,
                       help='Metric names to exclude from clustering')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--boxcar-periods', nargs='+', type=int, default=None,
                       help='Boxcar filter periods (odd integers, e.g., 1 17 65 257)')
    parser.add_argument('--beats-per-minute', type=float, default=64,
                       help='Tempo for MIDI timing (default: 64)')
    parser.add_argument('--frames-per-second', type=float, default=30,
                       help='Video frame rate (default: 30)')
    parser.add_argument('--ticks-per-beat', type=int, default=480,
                       help='MIDI resolution (default: 480)')

    args = parser.parse_args()

    cluster_primary_metrics(
        args.csv_path,
        k_values=args.k_values,
        normalization=args.normalization,
        metrics_to_exclude=args.exclude,
        random_state=args.random_state,
        boxcar_periods=args.boxcar_periods,
        beats_per_minute=args.beats_per_minute,
        frames_per_second=args.frames_per_second,
        ticks_per_beat=args.ticks_per_beat
    )


if __name__ == "__main__":
    main()
