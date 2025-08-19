# Video Processing Pipeline

This repository contains a comprehensive video processing pipeline that extracts visual metrics from video files and converts them into MIDI data for musical applications. The pipeline consists of three main scripts that work together to analyze video content and generate musical representations.

## Overview

The pipeline processes video files to extract various visual metrics (color, motion, symmetry, etc.) and transforms them into MIDI control change messages. This enables the creation of music that responds to visual content, making it useful for video scoring, interactive installations, and audiovisual art projects.

## Script Architecture

1. **`run_video_processing.py`** - Main entry point that orchestrates the entire pipeline
2. **`process_video.py`** - Module containing video analysis functions (called by run_video_processing.py)
3. **`process_metrics.py`** - Module containing metrics processing functions (called by run_video_processing.py)

---

## 1. run_video_processing.py

### Overview
The main entry point for the video processing pipeline. This script provides a convenient interface to run both `process_video.py` and `process_metrics.py` with configurable parameters.

### Command Line Arguments

#### Basic Usage
```bash
python run_video_processing.py <subdir_name> [beats_per_minute]
```

#### Advanced Options
```bash
python run_video_processing.py --config config.json
python run_video_processing.py --help
```

#### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `subdir_name` | string | required | Name of the video file (without .wmv extension) |
| `beats_per_minute` | int | 64 | Tempo in beats per minute |
| `--beats-per-minute, -b` | int | 64 | Alternative way to specify BPM |
| `--config, -c` | string | - | Path to JSON configuration file |
| `--skip-video` | flag | false | Skip process_video.py, only run process_metrics.py |
| `--skip-metrics` | flag | false | Skip process_metrics.py, only run process_video.py |
| `--frames-per-second` | int | 30 | Video frame rate |
| `--beats-per-midi-event` | int | 1 | Beats between each processed frame |
| `--ticks-per-beat` | int | 480 | MIDI ticks per beat |
| `--downscale-large` | int | 100 | Large scale analysis factor |
| `--downscale-medium` | int | 10 | Medium scale analysis factor |
| `--filter-periods` | int list | [17, 65, 257] | Filter periods for smoothing |
| `--stretch-values` | int list | [8] | Stretch values for transformation |
| `--stretch-centers` | float list | [0.33, 0.67] | Stretch centers for transformation |
| `--cc-number` | int | 1 | MIDI CC number |
| `--max-frames` | int | None | Maximum number of frames to process (default: process all frames) |

#### Configuration File Format
```json
{
  "subdir_name": "N17_Mz7fo6C2f",
  "beats_per_minute": 120,
  "frames_per_second": 30,
  "beats_per_midi_event": 1,
  "ticks_per_beat": 480,
  "downscale_large": 100,
  "downscale_medium": 10,
  "filter_periods": [17, 65, 257],
  "stretch_values": [8],
  "stretch_centers": [0.33, 0.67],
  "cc_number": 1,
  "max_frames": null
}
```

### Output Files Created

#### From process_video.py
- `{subdir_name}_basic.csv` - Raw metrics data for each frame
- `{subdir_name}_config.json` - Configuration parameters used

#### From process_metrics.py
- `../video_midi/{subdir_name}/` - Output directory containing:
  - `{subdir_name}_derived.xlsx` - All derived metrics in Excel format
  - `{subdir_name}_plots.pdf` - Visual plots of all metrics
  - Multiple MIDI files organized by:
    - Variable (R, G, B, Gray, H000, etc.)
    - Processing type (value vs rank)
    - Filter period (f001, f017, f065, f257)
    - Stretch parameters (s1-0.5, s8-0.33, etc.)

### Examples
```bash
# Basic usage
python run_video_processing.py N17_Mz7fo6C2f

# With custom tempo
python run_video_processing.py N17_Mz7fo6C2f 120

# Using configuration file
python run_video_processing.py --config my_config.json

# Skip video processing, only run metrics
python run_video_processing.py N17_Mz7fo6C2f --skip-video

# Process only first 1000 frames
python run_video_processing.py N17_Mz7fo6C2f --max-frames 1000
```

---

## 2. process_video.py

### Overview
Module containing functions to extract comprehensive visual metrics from video frames using computer vision techniques. This module is called by `run_video_processing.py` and is not designed for standalone execution.

### Key Features
- **Multi-scale analysis**: Analyzes video at different spatial resolutions
- **Color space support**: RGB, grayscale, HSV, and hue-specific metrics
- **Motion analysis**: ECC-based global transform analysis for zoom, rotation, and motion detection
- **Symmetry metrics**: Transpose, reflection, and radial symmetry analysis
- **Error dispersion**: Multi-scale information content analysis
- **Performance timing**: Detailed timing analysis for optimization

### Metrics Computed

#### Basic Intensity Metrics
- **`avg`** - Average intensity per color channel
- **`std`** - Standard deviation of intensity
- **`dcd`** - Dark count (pixels near minimum value)
- **`dcl`** - Light count (pixels near maximum value)

#### Symmetry Metrics
- **`xps`** - Transpose symmetry (flip around center point)
- **`rfl`** - Reflection symmetry (vertical and horizontal flips)
- **`rad`** - Radial symmetry (circular patterns)

#### Error Dispersion Metrics (Multi-scale Analysis)
- **`ee0`** - Total error dispersion (total detail content)
- **`ee1`** - Large-scale error dispersion (low-resolution detail)
- **`ee2`** - Small-scale error dispersion (high-resolution detail)
- **`ed0`** - Distance of total error center from image center
- **`ed1`** - Distance of large-scale error center from image center
- **`ed2`** - Distance of small-scale error center from image center
- **`es0`** - Spatial standard deviation of total error
- **`es1`** - Spatial standard deviation of large-scale error
- **`es2`** - Spatial standard deviation of small-scale error

#### Motion Metrics (Lucas-Kanade Optical Flow Analysis)
- **`czd`** - Zoom divergence (positive = zoom out, negative = zoom in)
- **`crc`** - Rotation curl (positive = counterclockwise, negative = clockwise)
- **`cmv`** - Motion variance (how uniform the motion is)

##### Motion Metrics Overview
These metrics are computed by analyzing the optical flow between consecutive video frames using Lucas-Kanade Farneback algorithm. This approach extracts zoom and rotation information from the flow field using a centered region analysis, focusing on center-anchored motion (zoom and rotation around center, no panning).

**`czd` - Zoom Divergence**
- **What it measures**: How much the image is zooming in or out
- **Mathematical basis**: Divergence of the flow field (∂u/∂x + ∂v/∂y) in centered region
- **Interpretation**: Positive values indicate zooming out (objects getting smaller), negative values indicate zooming in (objects getting larger), zero indicates no zoom effect

**`crc` - Rotation Curl**
- **What it measures**: How much the image is rotating around its center
- **Mathematical basis**: Curl of the flow field (∂v/∂x - ∂u/∂y) in centered region
- **Interpretation**: Positive values indicate counterclockwise rotation, negative values indicate clockwise rotation, zero indicates no rotation

**`cmv` - Motion Variance**
- **What it measures**: How uniform the motion is across the image
- **Mathematical basis**: Variance of flow components (var(flow_x) + var(flow_y))
- **Interpretation**: High values indicate non-uniform motion (complex scenes), low values indicate uniform motion (simple scenes)

##### Derived Motion Metrics
The system automatically computes additional derived metrics:
- **`crl`** - Positive rotation component (`max(crc, 0)`) - captures counterclockwise rotation
- **`crr`** - Negative rotation component (`max(-crc, 0)`) - captures clockwise rotation

##### Technical Implementation Details
- **Algorithm**: Uses Lucas-Kanade Farneback optical flow with optimized parameters
- **Analysis Region**: Computed on a centered region of the image (configurable via `center_region_ratio`)
- **Downscaling**: Centered region is downscaled for computational efficiency (factor of 2 by default)
- **Flow Calculation**: Full-resolution optical flow using Farneback algorithm with pyramid levels
- **Mathematical Framework**: Uses finite differences to compute divergence and curl of the flow field
- **Coordinate System**: Analyzes motion relative to image center for zoom and rotation detection

#### Color-Specific Metrics
- **`H000`** - Proximity to red hue (0°)
- **`H060`** - Proximity to yellow hue (60°)
- **`H120`** - Proximity to green hue (120°)
- **`H180`** - Proximity to cyan hue (180°)
- **`H240`** - Proximity to blue hue (240°)
- **`H300`** - Proximity to magenta hue (300°)
- **`Hmon`** - Monochromaticity (color uniformity)

### Color Channels Analyzed
- **R, G, B** - Red, Green, Blue channels
- **Gray** - Grayscale intensity
- **S, V** - Saturation and Value from HSV
- **H000-H300** - Hue-specific metrics (6 cardinal colors)
- **Hmon** - Monochromaticity metric

### Processing Parameters
- **`downscale_large`** - Large scale analysis factor (default: 100)
- **`downscale_medium`** - Medium scale analysis factor (default: 10)
- **`frames_per_second`** - Video frame rate (default: 30)
- **`beats_per_minute`** - Musical tempo (default: 64)
- **`beats_per_midi_event`** - Beats between processed frames (default: 1)
- **`max_frames`** - Maximum frames to process (default: None, processes all frames)

---

## 3. process_metrics.py

### Overview
Module containing functions to process basic metrics from `process_video.py`, compute derived metrics, apply various transformations, and generate MIDI files. This module is called by `run_video_processing.py` and is not designed for standalone execution.

### Derived Metrics Computation

#### Automatic Ratio Calculations
The script automatically computes ratio metrics for error dispersion data:
- **`ee1r`** - `ee1/ee0` (large-scale to total error ratio)
- **`ee2r`** - `ee2/ee0` (small-scale to total error ratio)
- **`es1r`** - `es1/es0` (large-scale to total spatial variation ratio)
- **`es2r`** - `es2/es0` (small-scale to total spatial variation ratio)

#### Processing Pipeline
The script applies multiple transformation stages to each metric in sequence:

1. **Rank/Value Processing** - Creates base entries
   - `_v` - Original values from CSV
   - `_r` - Percentile transformation (0-1 range)

2. **Scaling** - Normalizes data to 0-1 range
   - Applied to all data but doesn't change key names

3. **Filtering** - Triangular smoothing filters
   - `f001` - No filtering (original data, period 1)
   - `f017` - ~2 bars at 4/4 time (period 17)
   - `f065` - ~16 bars at 4/4 time (period 65)
   - `f257` - ~64 bars at 4/4 time (period 257)
   
   **Note**: All filter periods use consistent `_f{period:03d}` naming convention, including period 1 for unfiltered data.

4. **Stretching** - Non-linear transformation
   - Applies sigmoid-like stretching function
   - Parameters: `stretch_value` (1, 8) and `stretch_center` (0.33, 0.5, 0.67)
   - **Special case**: Always includes `stretch_value=1, stretch_center=0.5`

5. **Inversion** - Value inversion
   - `_o` - Original values (no inversion)
   - `_i` - Inverted versions (1 - original_value)
   - Useful for complementary musical patterns

**Key Structure**: The final key names reflect the processing order:
```
R_avg_v_f017_s1-0.5_o
│ │   │ │    │ │
│ │   │ │    │ └─ o/i (inversion)
│ │   │ │    └─── s1-0.5 (stretching)
│ │   │ └──────── f017 (filtering)
│ │   └────────── v (rank/value)
│ └────────────── avg (metric)
└──────────────── R (color channel)
```

### MIDI File Generation

#### File Organization
MIDI files are organized by several criteria:

1. **By Variable and Processing Type**:
   - `{variable}_{metric}_{rank/value}_{filter}_{stretch}_{inversion}.mid`
   - Example: `R_avg_v_f017_s1-0.5_o.mid`

2. **By Processing Method**:
   - `{base_suffix}.mid` (contains all variations of a base metric)

#### MIDI Parameters
- **Control Change Number**: Configurable (default: 1)
- **Channel**: 7 (default)
- **Value Range**: 0-127 (scaled from 0-1)
- **Timing**: Based on frame intervals and musical tempo

### Output Files

#### Data Files
- **`{prefix}_derived.xlsx`** - Complete dataset with all derived metrics
- **`{prefix}_plots.pdf`** - Visual plots of all metrics (30 per page)

#### MIDI Files
Multiple MIDI files are generated for different combinations:
- **Value vs Rank**: Direct values (`_v`) vs percentile rankings (`_r`)
- **Filter periods**: Different smoothing levels (`_f001`, `_f017`, `_f065`, `_f257`)
- **Stretch parameters**: Various non-linear transformations (`_s1-0.5`, `_s8-0.33`, etc.)
- **Inversion**: Original (`_o`) vs inverted (`_i`) values
- **Variables**: Each color channel and metric type

### Configuration
The script can read configuration from:
- **JSON config file**: `{prefix}_config.json`
- **Hardcoded defaults**: If no config file exists

**Automatic Detection**: The script automatically detects all available variables and metrics from the CSV file, so no explicit configuration of `vars` or `metric_names` is needed.

### Special Features

#### Automatic Special Case Inclusion
The script ensures that the combination `stretch_value=1, stretch_center=0.5` is always included in the processing, even if not in the original parameter lists.

#### Flexible Processing
- Can process any subset of variables and metrics
- Supports custom filter periods and stretch parameters
- Generates both individual MIDI files and combined files

---

## Usage Examples

### Complete Pipeline
```bash
# Process a video with default settings
python run_video_processing.py my_video

# Process with custom tempo
python run_video_processing.py my_video 120

# Use configuration file
python run_video_processing.py --config my_config.json
```

### Partial Processing
```bash
# Only run metrics processing (skip video analysis)
python run_video_processing.py my_video --skip-video

# Only run video processing (skip metrics)
python run_video_processing.py my_video --skip-metrics
```

### Advanced Configuration
```bash
# Custom parameters
python run_video_processing.py my_video 120 \
  --frames-per-second 24 \
  --beats-per-midi-event 2 \
  --ticks-per-beat 960 \
  --downscale-large 50 \
  --downscale-medium 5 \
  --filter-periods 17 65 \
  --stretch-values 4 8 \
  --stretch-centers 0.25 0.5 0.75 \
  --cc-number 7
```

## File Structure
```
project/
├── run_video_processing.py    # Main wrapper script
├── process_video.py           # Video analysis script
├── process_metrics.py         # Metrics processing script
├── {video_name}.wmv          # Input video file
├── {video_name}_basic.csv    # Basic metrics output
├── {video_name}_config.json  # Configuration file
└── ../video_midi/{video_name}/  # Output directory
    ├── {video_name}_derived.xlsx
    ├── {video_name}_plots.pdf
    └── *.mid                 # MIDI files
```

## Dependencies
- **OpenCV** (`cv2`) - Video processing and computer vision
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation and CSV handling
- **Mido** - MIDI file generation
- **Matplotlib** - Plotting and PDF generation
- **SciPy** - Statistical functions
- **JSON** - Configuration file handling

## Notes
- Input videos should be in WMV format
- The pipeline is optimized for musical applications with configurable tempo and timing
- All metrics are normalized to 0-1 range before MIDI conversion
- The special case `stretch_value=1, stretch_center=0.5` is always included for consistency
- Performance timing data is automatically generated and saved
