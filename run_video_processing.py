#!/usr/bin/env python3
"""
Wrapper script to run video processing pipeline.
This script runs both process_video.py and process_metrics.py with configurable parameters.

Usage:
    python run_video_processing.py <subdir_name> [beats_per_minute]
    python run_video_processing.py --config config.json
    python run_video_processing.py --help
    
Examples:
    python run_video_processing.py N17_Mz7fo6C2f
    python run_video_processing.py N17_Mz7fo6C2f 120
    python run_video_processing.py --config my_config.json
"""

import sys
import os
import subprocess
import json
import argparse

# Import the processing functions directly
from process_video import process_video_to_csv
from process_metrics import process_metrics_to_midi

def run_process_video(subdir_name, beats_per_minute=64, **kwargs):
    """
    Run process_video_to_csv function directly with the specified parameters.
    """
    print(f"Running process_video_to_csv with subdir_name={subdir_name}, beats_per_minute={beats_per_minute}")
    
    try:
        # Call the function directly
        video_file = f"{subdir_name}.wmv"
        process_video_to_csv(
            video_file=video_file,
            subdir_name=subdir_name,
            frames_per_second=kwargs.get("frames_per_second", 30),
            beats_per_midi_event=kwargs.get("beats_per_midi_event", 1),
            ticks_per_beat=kwargs.get("ticks_per_beat", 480),
            beats_per_minute=beats_per_minute,
            downscale_large=kwargs.get("downscale_large", 100),
            downscale_medium=kwargs.get("downscale_medium", 10)
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
            'beats_per_midi_event': kwargs.get('beats_per_midi_event', 1)
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
    parser = argparse.ArgumentParser(description='Run video processing pipeline')
    parser.add_argument('subdir_name', nargs='?', help='Name of the subdirectory/video file (without .wmv extension)')
    parser.add_argument('beats_per_minute', nargs='?', type=int, help='Beats per minute (default: 64)')
    parser.add_argument('--beats-per-minute', '-b', type=int, 
                       help='Beats per minute (alternative to positional argument)')
    parser.add_argument('--config', '-c', help='Path to config.json file')
    parser.add_argument('--skip-video', action='store_true',
                       help='Skip process_video.py and only run process_metrics.py')
    parser.add_argument('--skip-metrics', action='store_true',
                       help='Skip process_metrics.py and only run process_video.py')
    parser.add_argument('--frames-per-second', type=int, default=30,
                       help='Frames per second (default: 30)')
    parser.add_argument('--beats-per-midi-event', type=int, default=1,
                       help='Beats per MIDI event (default: 1)')
    parser.add_argument('--ticks-per-beat', type=int, default=480,
                       help='Ticks per beat (default: 480)')
    parser.add_argument('--downscale-large', type=int, default=100,
                       help='Large downscale factor (default: 100)')
    parser.add_argument('--downscale-medium', type=int, default=10,
                       help='Medium downscale factor (default: 10)')
    parser.add_argument('--filter-periods', nargs='+', type=int, default=[17, 65, 257],
                       help='Filter periods for smoothing (default: 17 65 257)')
    parser.add_argument('--stretch-values', nargs='+', type=int, default=[8],
                       help='Stretch values (default: 8)')
    parser.add_argument('--stretch-centers', nargs='+', type=float, default=[0.33, 0.67],
                       help='Stretch centers (default: 0.33 0.67)')
    parser.add_argument('--cc-number', type=int, default=1,
                       help='MIDI CC number (default: 1)')
    
    args = parser.parse_args()
    
    # Load config if specified
    if args.config:
        config = load_config(args.config)
        if config is None:
            return 1
        
        subdir_name = config.get('subdir_name')
        beats_per_minute = config.get('beats_per_minute', 64)
        frames_per_second = config.get('frames_per_second', 30)
        beats_per_midi_event = config.get('beats_per_midi_event', 1)
        ticks_per_beat = config.get('ticks_per_beat', 480)
        downscale_large = config.get('downscale_large', 100)
        downscale_medium = config.get('downscale_medium', 10)
        filter_periods = config.get('filter_periods', [17, 65, 257])
        stretch_values = config.get('stretch_values', [8])
        stretch_centers = config.get('stretch_centers', [0.33, 0.67])
        cc_number = config.get('cc_number', 1)
    else:
        # Use command line arguments
        if not args.subdir_name:
            print("Error: Either provide subdir_name as argument or use --config option")
            parser.print_help()
            return 1
        
        subdir_name = args.subdir_name
        
        # Handle beats_per_minute: positional argument takes precedence over named argument
        if args.beats_per_minute is not None:
            beats_per_minute = args.beats_per_minute
        else:
            beats_per_minute = 64  # default value
        
        frames_per_second = args.frames_per_second
        beats_per_midi_event = args.beats_per_midi_event
        ticks_per_beat = args.ticks_per_beat
        downscale_large = args.downscale_large
        downscale_medium = args.downscale_medium
        filter_periods = args.filter_periods
        stretch_values = args.stretch_values
        stretch_centers = args.stretch_centers
        cc_number = args.cc_number
    
    print(f"Starting video processing pipeline:")
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
    print(f"  Video file: {subdir_name}.wmv")
    print()
    
    # Check if video file exists
    video_file = f"{subdir_name}.wmv"
    if not os.path.exists(video_file):
        print(f"Error: Video file '{video_file}' not found!")
        return 1
    
    success = True
    
    # Step 1: Run process_video.py
    if not args.skip_video:
        print("=" * 50)
        print("STEP 1: Running process_video.py")
        print("=" * 50)
        video_params = {
            'frames_per_second': frames_per_second,
            'beats_per_midi_event': beats_per_midi_event,
            'ticks_per_beat': ticks_per_beat,
            'downscale_large': downscale_large,
            'downscale_medium': downscale_medium
        }
        if not run_process_video(subdir_name, beats_per_minute, **video_params):
            success = False
            print("Failed to run process_video.py")
        else:
            print("✓ process_video.py completed successfully")
        print()
    
    # Step 2: Run process_metrics.py
    if not args.skip_metrics and success:
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
            'beats_per_midi_event': beats_per_midi_event
        }
        if not run_process_metrics(subdir_name, **metrics_params):
            success = False
            print("Failed to run process_metrics.py")
        else:
            print("✓ process_metrics.py completed successfully")
        print()
    
    if success:
        print("=" * 50)
        print("✓ Video processing pipeline completed successfully!")
        print(f"Output files are in: ../video_midi/{subdir_name}/")
        print("=" * 50)
        return 0
    else:
        print("=" * 50)
        print("✗ Video processing pipeline failed!")
        print("=" * 50)
        return 1

if __name__ == "__main__":
    sys.exit(main())
