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

def run_process_video(subdir_name, beats_per_minute=64, **kwargs):
    """
    Run process_video.py with the specified parameters by temporarily modifying the script.
    """
    print(f"Running process_video.py with subdir_name={subdir_name}, beats_per_minute={beats_per_minute}")
    
    # Read the original process_video.py file
    with open('process_video.py', 'r') as f:
        content = f.read()
    
    # Create a temporary version with our parameters
    # Replace the hardcoded values at the bottom
    lines = content.split('\n')
    
    # Find and replace the video_file and subdir_name lines
    for i, line in enumerate(lines):
        if line.strip().startswith('video_file = '):
            lines[i] = f'video_file = "{subdir_name}.wmv"'
        elif line.strip().startswith('subdir_name = '):
            lines[i] = f'subdir_name = "{subdir_name}" # output prefix'
        elif line.strip().startswith('beats_per_minute='):
            lines[i] = f'                      beats_per_minute={beats_per_minute},  '
        elif line.strip().startswith('frames_per_second='):
            lines[i] = f'                      frames_per_second={kwargs.get("frames_per_second", 30)}, '
        elif line.strip().startswith('beats_per_midi_event='):
            lines[i] = f'                      beats_per_midi_event={kwargs.get("beats_per_midi_event", 1)},'
        elif line.strip().startswith('ticks_per_beat='):
            lines[i] = f'                      ticks_per_beat={kwargs.get("ticks_per_beat", 480)}, '
        elif line.strip().startswith('downscale_large='):
            lines[i] = f'                      downscale_large={kwargs.get("downscale_large", 100)}, # scale boundary means divide so 100x100 pixels in a cell (approximately square root of width and height of video)'
        elif line.strip().startswith('downscale_medium='):
            lines[i] = f'                      downscale_medium={kwargs.get("downscale_medium", 10)} ) # resolution reduction means divide so 10x10 pixels in a cell (approximately square root of the larger scale)'
    
    # Write the temporary file
    temp_content = '\n'.join(lines)
    with open('process_video_temp.py', 'w') as f:
        f.write(temp_content)
    
    try:
        # Run the temporary script with real-time output
        print("Starting process_video.py...")
        result = subprocess.run([sys.executable, 'process_video_temp.py'], 
                              check=True)
        print("process_video.py completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error running process_video.py: {e}")
        return False
    finally:
        # Clean up temporary file
        if os.path.exists('process_video_temp.py'):
            os.remove('process_video_temp.py')
    
    return True

def run_process_metrics(subdir_name):
    """
    Run process_metrics.py with the specified subdir_name.
    """
    print(f"Running process_metrics.py with subdir_name={subdir_name}")
    
    # Read the original process_metrics.py file
    with open('process_metrics.py', 'r') as f:
        content = f.read()
    
    # Create a temporary version with our subdir_name
    lines = content.split('\n')
    
    # Find and replace the prefix line
    for i, line in enumerate(lines):
        if line.strip().startswith('prefix = '):
            lines[i] = f'    prefix = "{subdir_name}"'
            break
    
    # Write the temporary file
    temp_content = '\n'.join(lines)
    with open('process_metrics_temp.py', 'w') as f:
        f.write(temp_content)
    
    try:
        # Run the temporary script with real-time output
        print("Starting process_metrics.py...")
        result = subprocess.run([sys.executable, 'process_metrics_temp.py'], 
                              check=True)
        print("process_metrics.py completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error running process_metrics.py: {e}")
        return False
    finally:
        # Clean up temporary file
        if os.path.exists('process_metrics_temp.py'):
            os.remove('process_metrics_temp.py')
    
    return True

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
    
    print(f"Starting video processing pipeline:")
    print(f"  Subdir name: {subdir_name}")
    print(f"  Beats per minute: {beats_per_minute}")
    print(f"  Frames per second: {frames_per_second}")
    print(f"  Beats per MIDI event: {beats_per_midi_event}")
    print(f"  Ticks per beat: {ticks_per_beat}")
    print(f"  Downscale large: {downscale_large}")
    print(f"  Downscale medium: {downscale_medium}")
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
        if not run_process_metrics(subdir_name):
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
