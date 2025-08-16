# Video to MIDI Processing Pipeline

Code to process video files to generate MIDI data that can be imported into a DAW (Digital Audio Workstation).

## Overview

This project processes video files frame by frame, extracts various visual metrics (color intensity, symmetry, texture, etc.), and converts them into MIDI control change messages. The output can be imported into any DAW for creating music that responds to visual content.

**Recent Updates:**
- ✅ Added wrapper script for easy parameter management
- ✅ Enhanced error dispersion metrics (ee0, es0, ed0 + ratio metrics)
- ✅ Automatic derived column generation (ee1r, ee2r, es1r, es2r)
- ✅ Real-time progress output during processing
- ✅ Cleaned up repository (removed old process.py and post_process.py)
- ✅ Comprehensive documentation and usage examples

## Quick Start

Instead of modifying code each time, use the wrapper script:

```bash
python run_video_processing.py <subdir_name> [beats_per_minute]
```

### Example
```bash
# Process video with default settings (64 BPM)
python run_video_processing.py N17_Mz7fo6C2f

# Process video with custom BPM
python run_video_processing.py N17_Mz7fo6C2f 96
```

## Usage Methods

### Method 1: Command Line Arguments (Simplest)

```bash
# Process video with default settings (64 BPM)
python run_video_processing.py N17_Mz7fo6C2f

# Process video with custom BPM
python run_video_processing.py N17_Mz7fo6C2f 96

# Process with all custom parameters
python run_video_processing.py N17_Mz7fo6C2f 96 --frames-per-second 30 --beats-per-midi-event 1
```

### Method 2: Configuration File

1. Edit `config.json` with your parameters:
```json
{
  "subdir_name": "N17_Mz7fo6C2f",
  "beats_per_minute": 64,
  "frames_per_second": 30,
  "beats_per_midi_event": 1,
  "ticks_per_beat": 480,
  "downscale_large": 100,
  "downscale_medium": 10
}
```

2. Run with config file:
```bash
python run_video_processing.py --config config.json
```

### Method 3: Partial Processing

```bash
# Only run process_video.py (skip metrics)
python run_video_processing.py N17_Mz7fo6C2f --skip-metrics

# Only run process_metrics.py (skip video processing)
python run_video_processing.py N17_Mz7fo6C2f --skip-video
```

## Processing Pipeline

The system consists of two main scripts:

1. **`process_video.py`** - Extracts frames from video and computes basic metrics
2. **`process_metrics.py`** - Processes the basic metrics, generates derived columns, and creates MIDI files

The wrapper script `run_video_processing.py` orchestrates both steps automatically with real-time progress feedback.

## Parameters

- **subdir_name**: Name of your video file without the `.wmv` extension
- **beats_per_minute**: Tempo for MIDI generation (default: 64)
- **frames_per_second**: Video frame rate (default: 30)
- **beats_per_midi_event**: Beats between each processed frame (default: 1)
- **ticks_per_beat**: MIDI resolution (default: 480)
- **downscale_large**: Large scale analysis factor (default: 100)
- **downscale_medium**: Medium scale analysis factor (default: 10)

## Output Files

The pipeline generates:

1. **`{subdir_name}_basic.csv`** - Raw metrics data from video processing
2. **`{subdir_name}_config.json`** - Processing configuration
3. **`../video_midi/{subdir_name}/`** - Directory containing:
   - Multiple MIDI files (one for each metric and processing method)
   - **`{subdir_name}_derived.xlsx`** - Processed metrics data (includes derived columns)
   - **`{subdir_name}_plots.pdf`** - Visualization plots of all metrics

## Metrics Computed

The system extracts various visual metrics from each frame:

### Color Channel Metrics (R, G, B, Gray, Saturation, Value)
- **avg**: Average intensity
- **var**: Variance (total information)
- **xps**: Transpose symmetry metric
- **rfl**: Reflection symmetry metric  
- **rad**: Radial symmetry metric
- **ee0/ee1/ee2**: Error dispersion metrics (total/large/small scale detail)
- **ee1r/ee2r**: Error dispersion ratios (ee1/ee0, ee2/ee0) - *automatically generated*
- **ed0/ed1/ed2**: Error distance metrics (total/large/small scale distance from center)
- **es0/es1/es2**: Error spatial variation metrics (total/large/small scale spatial variation)
- **es1r/es2r**: Error spatial variation ratios (es1/es0, es2/es0) - *automatically generated*
- **lmd/l10/l90**: Line symmetry metrics (median, 10th, 90th percentile)
- **dcd/dcl**: Dark/light count metrics

### Hue Metrics
- **H000-H360**: Presence of cardinal colors (0°, 60°, 120°, 180°, 240°, 300°)
- **Hmon**: Monochromaticity metric

### Derived Metrics (Auto-generated)
The system automatically creates ratio-based metrics:
- **ee1r**: Ratio of large-scale to total error dispersion
- **ee2r**: Ratio of small-scale to total error dispersion  
- **es1r**: Ratio of large-scale to total spatial variation
- **es2r**: Ratio of small-scale to total spatial variation

## Video File Requirements

- **Format**: `.wmv` files
- **Naming**: `{subdir_name}.wmv` (e.g., `N17_Mz7fo6C2f.wmv`)
- **Location**: Place video files in the same directory as the scripts

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Troubleshooting

- **Video file not found**: Make sure your `.wmv` file exists in the current directory
- **Permission errors**: Make sure the script is executable: `chmod +x run_video_processing.py`
- **Missing dependencies**: Install required packages: `pip install -r requirements.txt`

## Project Structure

### Core Files
- **`run_video_processing.py`** - Main wrapper script (use this!)
- **`process_video.py`** - Video frame extraction and basic metrics computation
- **`process_metrics.py`** - Advanced metrics processing, derived columns, and MIDI generation
- **`config.json`** - Configuration file template
- **`requirements.txt`** - Python dependencies

### Utility Files
- **`circle_test.py`** - Circle detection testing utility
- **`create_group_channel.py`** - Channel grouping utility
- **`kfs_to_csv.py`** - Data conversion utility

### Removed Files
- ~~`process.py`~~ - Old version (removed)
- ~~`post_process.py`~~ - Old version (removed)

## Recent Improvements

1. **Enhanced Error Dispersion Metrics**: Added ee0, es0, ed0 metrics for total information analysis
2. **Automatic Derived Columns**: System now generates ratio-based metrics (ee1r, ee2r, es1r, es2r)
3. **Improved User Experience**: Real-time progress output and flexible parameter management
4. **Code Cleanup**: Removed outdated files and consolidated functionality
5. **Better Documentation**: Comprehensive usage examples and troubleshooting guide

## License

This project is open source. See the repository for more details.
