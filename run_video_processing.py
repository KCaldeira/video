#!/usr/bin/env python3
"""
Wrapper script to run video processing pipeline.
This script runs both process_video.py and process_metrics.py with configurable parameters.

Usage:
    python run_video_processing.py <config_file.json>
    python run_video_processing.py --help

Examples:
    python run_video_processing.py default_config.json
    python run_video_processing.py my_custom_config.json
    python run_video_processing.py N29_3M2pM6dispA7_config.json

Note: Configuration is exclusively via JSON files. See default_config.json for template.
"""

import sys
import os
import json
import argparse

# Import the processing functions directly
from process_video import process_video_to_csv
from process_metrics import process_metrics_to_midi
from cluster_primary import cluster_primary_metrics

def run_process_video(subdir_name, beats_per_minute=64, **kwargs):
    """
    Run process_video_to_csv function directly with the specified parameters.
    """
    print(f"Running process_video_to_csv with subdir_name={subdir_name}, beats_per_minute={beats_per_minute}")
    
    try:
        # Call the function directly
        # Look for video file in data/input directory
        # Try .wmv first, then .mp4 if not found
        video_file = f"data/input/{subdir_name}.wmv"
        if not os.path.exists(video_file):
            video_file = f"data/input/{subdir_name}.mp4"
            if not os.path.exists(video_file):
                raise FileNotFoundError(f"Neither data/input/{subdir_name}.wmv nor data/input/{subdir_name}.mp4 found")
        print(f"Using video file: {video_file}")
        # Extract Farneback parameters and strip the prefix
        farneback_kwargs = {}
        for k, v in kwargs.items():
            if k.startswith("farneback_") and k != "farneback_preset":
                # Strip the "farneback_" prefix
                param_name = k.replace("farneback_", "")
                farneback_kwargs[param_name] = v
        
        process_video_to_csv(
            video_file,
            subdir_name,
            kwargs.get("frames_per_second", 30),
            kwargs.get("beats_per_midi_event", 1),
            kwargs.get("ticks_per_beat", 480),
            beats_per_minute,
            kwargs.get("downscale_large", 100),
            kwargs.get("downscale_medium", 10),
            kwargs.get("max_frames", None),
            kwargs.get("farneback_preset", "default"),
            **farneback_kwargs
        )
        print("process_video_to_csv completed successfully")
        return True
    except Exception as e:
        print(f"Error running process_video_to_csv: {e}")
        return False

def run_process_metrics(subdir_name, **kwargs):
    """
    Run process_metrics_to_midi function directly with the specified subdir_name and parameters.
    """
    print(f"Running process_metrics_to_midi with subdir_name={subdir_name}")
    
    try:
        # Create config dictionary with all parameters
        config = {
            'filter_periods': kwargs.get('filter_periods', [17, 65, 257]),
            'stretch_values': kwargs.get('stretch_values', [8]),
            'stretch_centers': kwargs.get('stretch_centers', [0.33, 0.67]),
            'cc_number': kwargs.get('cc_number', 1),
            'ticks_per_beat': kwargs.get('ticks_per_beat', 480),
            'beats_per_minute': kwargs.get('beats_per_minute', 100),
            'frames_per_second': kwargs.get('frames_per_second', 30),
            'beats_per_midi_event': kwargs.get('beats_per_midi_event', 1),
            'farneback_preset': kwargs.get('farneback_preset', 'default')
        }
        
        # Call the function directly with config
        process_metrics_to_midi(subdir_name, config)
        print("process_metrics_to_midi completed successfully")
        return True
    except Exception as e:
        print(f"Error running process_metrics_to_midi: {e}")
        return False

def load_config(config_file):
    """
    Load configuration from JSON file.
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Config file '{config_file}' not found!")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config file '{config_file}': {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description='Run video processing pipeline using JSON configuration',
        epilog='See default_config.json for a complete configuration template'
    )
    parser.add_argument('config',
                       help='Path to JSON configuration file')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    if config is None:
        return 1

    # Extract configuration values with defaults
    video_config = config.get('video', {})
    timing_config = config.get('timing', {})
    video_proc_config = config.get('video_processing', {})
    optical_flow_config = video_proc_config.get('optical_flow', {})
    metrics_config = config.get('metrics_processing', {})
    pipeline_config = config.get('pipeline_control', {})

    # Required configuration
    subdir_name = video_config.get('video_name')
    if not subdir_name or subdir_name == 'your_video_name_here':
        print("Error: 'video.video_name' must be set in configuration file")
        return 1

    # Timing parameters
    beats_per_minute = timing_config.get('beats_per_minute', 64)
    frames_per_second = timing_config.get('frames_per_second', 30)
    beats_per_midi_event = timing_config.get('beats_per_midi_event', 1)
    ticks_per_beat = timing_config.get('ticks_per_beat', 480)

    # Video processing parameters
    downscale_large = video_proc_config.get('downscale_large', 100)
    downscale_medium = video_proc_config.get('downscale_medium', 10)
    max_frames = video_config.get('max_frames', None)

    # Optical flow parameters
    farneback_preset = optical_flow_config.get('preset', 'default')
    farneback_pyr_scale = optical_flow_config.get('pyr_scale', 0.5)
    farneback_levels = optical_flow_config.get('levels', 3)
    farneback_winsize = optical_flow_config.get('winsize', 15)
    farneback_iterations = optical_flow_config.get('iterations', 3)
    farneback_poly_n = optical_flow_config.get('poly_n', 5)
    farneback_poly_sigma = optical_flow_config.get('poly_sigma', 1.2)

    # Metrics processing parameters
    filter_periods = metrics_config.get('filter_periods', [17, 65, 257])
    stretch_values = metrics_config.get('stretch_values', [8])
    stretch_centers = metrics_config.get('stretch_centers', [0.33, 0.67])
    cc_number = metrics_config.get('cc_number', 1)

    # Pipeline control
    process_video = pipeline_config.get('process_video', True)
    process_metrics = pipeline_config.get('process_metrics', True)
    process_clusters = pipeline_config.get('process_clusters', True)

    # Clustering parameters
    cluster_processing_config = config.get('cluster_processing', {})
    k_values = cluster_processing_config.get('k_values', [2, 3, 4, 5, 6, 8, 10, 12])
    clustering_normalization = cluster_processing_config.get('normalization', 'rank')
    metrics_to_exclude = cluster_processing_config.get('metrics_to_exclude', [])
    clustering_random_state = cluster_processing_config.get('random_state', 42)

    # Print configuration summary
    print(f"Starting video processing pipeline:")
    print(f"  Configuration file: {args.config}")
    print(f"  Subdir name: {subdir_name}")
    print(f"  Beats per minute: {beats_per_minute}")
    print(f"  Frames per second: {frames_per_second}")
    print(f"  Beats per MIDI event: {beats_per_midi_event}")
    print(f"  Ticks per beat: {ticks_per_beat}")
    print(f"  Downscale large: {downscale_large}")
    print(f"  Downscale medium: {downscale_medium}")
    print(f"  Filter periods: {filter_periods}")
    print(f"  Stretch values: {stretch_values}")
    print(f"  Stretch centers: {stretch_centers}")
    print(f"  CC number: {cc_number}")
    print(f"  Max frames: {max_frames if max_frames else 'All frames'}")
    print(f"  Optical flow preset: {farneback_preset}")
    print()

    # Check if video file exists in data/input directory (try configured extensions)
    file_extensions = video_config.get('file_extensions', ['.wmv', '.mp4'])
    video_file = None
    for ext in file_extensions:
        candidate = f"data/input/{subdir_name}{ext}"
        if os.path.exists(candidate):
            video_file = candidate
            print(f"Found video file: {video_file}")
            break

    if not video_file:
        print(f"Error: No video file found for '{subdir_name}' in data/input/ with extensions {file_extensions}")
        return 1

    success = True

    # Step 1: Run process_video.py
    if process_video:
        print("=" * 50)
        print("STEP 1: Running process_video.py")
        print("=" * 50)
        video_params = {
            'frames_per_second': frames_per_second,
            'beats_per_midi_event': beats_per_midi_event,
            'ticks_per_beat': ticks_per_beat,
            'downscale_large': downscale_large,
            'downscale_medium': downscale_medium,
            'max_frames': max_frames,
            'farneback_preset': farneback_preset,
            'farneback_pyr_scale': farneback_pyr_scale,
            'farneback_levels': farneback_levels,
            'farneback_winsize': farneback_winsize,
            'farneback_iterations': farneback_iterations,
            'farneback_poly_n': farneback_poly_n,
            'farneback_poly_sigma': farneback_poly_sigma
        }
        if not run_process_video(subdir_name, beats_per_minute, **video_params):
            success = False
            print("Failed to run process_video.py")
        else:
            print("process_video.py completed successfully")
        print()

    # Step 2: Run process_metrics.py
    if process_metrics and success:
        print("=" * 50)
        print("STEP 2: Running process_metrics.py")
        print("=" * 50)
        metrics_params = {
            'filter_periods': filter_periods,
            'stretch_values': stretch_values,
            'stretch_centers': stretch_centers,
            'cc_number': cc_number,
            'ticks_per_beat': ticks_per_beat,
            'beats_per_minute': beats_per_minute,
            'frames_per_second': frames_per_second,
            'beats_per_midi_event': beats_per_midi_event,
            'farneback_preset': farneback_preset
        }
        if not run_process_metrics(subdir_name, **metrics_params):
            success = False
            print("Failed to run process_metrics.py")
        else:
            print("process_metrics.py completed successfully")
        print()

    # Step 3: Run clustering analysis
    if process_clusters and success:
        print("=" * 50)
        print("STEP 3: Running cluster analysis")
        print("=" * 50)

        # Build path to CSV file
        csv_path = f"data/output/{subdir_name}_{farneback_preset}/{subdir_name}_{farneback_preset}_basic.csv"

        if not os.path.exists(csv_path):
            print(f"Warning: CSV file not found at {csv_path}, skipping clustering")
        else:
            cluster_primary_metrics(
                csv_path,
                k_values=k_values,
                normalization=clustering_normalization,
                metrics_to_exclude=metrics_to_exclude,
                random_state=clustering_random_state
            )
            print("Cluster analysis completed successfully")
        print()

    if success:
        print("=" * 50)
        print("Video processing pipeline completed successfully!")
        print(f"Output files are in: data/output/{subdir_name}_{farneback_preset}/")
        print("=" * 50)
        return 0
    else:
        print("=" * 50)
        print("Video processing pipeline failed!")
        print("=" * 50)
        return 1

if __name__ == "__main__":
    sys.exit(main())
