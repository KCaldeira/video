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
    python cluster_primary.py <csv_file> [options]

Example:
    python cluster_primary.py data/output/N29_3M2pM6dispA7_default/N29_3M2pM6dispA7_default_basic.csv

Output:
    - clusters_primary.csv: Cluster assignments for each frame
    - cluster_scores.json: Quality metrics (silhouette, BIC, etc.)
    - cluster_exemplars.json: Representative frames per cluster
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


def save_results(df, kmeans_results, gmm_results, kmeans_scores, gmm_scores,
                kmeans_exemplars, gmm_exemplars, output_dir, base_name):
    """
    Save clustering results to CSV and JSON files.

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
    """
    # Create output dataframe with cluster assignments
    cluster_df = df[['frame_count']].copy() if 'frame_count' in df.columns else pd.DataFrame(index=df.index)

    # Add K-Means cluster assignments
    for k, result in kmeans_results.items():
        cluster_df[f'kmeans_k{k}'] = result['labels']

    # Add GMM cluster assignments
    for k, result in gmm_results.items():
        cluster_df[f'gmm_k{k}'] = result['labels']

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


def cluster_primary_metrics(csv_path, k_values=None, normalization='rank',
                           metrics_to_exclude=None, random_state=42):
    """
    Main entry point for clustering primary metrics.

    Parameters:
    - csv_path: Path to basic metrics CSV
    - k_values: List of k values to try (default: [2,3,4,5,6,8,10,12])
    - normalization: 'rank' or 'zscore' (default: 'rank')
    - metrics_to_exclude: List of metric names to exclude
    - random_state: Random seed for reproducibility
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
                kmeans_exemplars, gmm_exemplars, output_dir, base_name)

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
  python cluster_primary.py data/output/N29_*/N29_*_basic.csv
  python cluster_primary.py data/output/N29_*/N29_*_basic.csv --k-values 2 3 4 5
  python cluster_primary.py data/output/N29_*/N29_*_basic.csv --normalization zscore
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

    args = parser.parse_args()

    cluster_primary_metrics(
        args.csv_path,
        k_values=args.k_values,
        normalization=args.normalization,
        metrics_to_exclude=args.exclude,
        random_state=args.random_state
    )


if __name__ == "__main__":
    main()
