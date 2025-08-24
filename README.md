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
| `--farneback-preset` | string | default | Farneback optical flow preset (see Motion Analysis section) |
| `--farneback-pyr-scale` | float | 0.5 | Farneback pyramid scale |
| `--farneback-levels` | int | 3 | Farneback pyramid levels |
| `--farneback-winsize` | int | 15 | Farneback window size |
| `--farneback-iterations` | int | 3 | Farneback iterations |
| `--farneback-poly-n` | int | 5 | Farneback polynomial degree |
| `--farneback-poly-sigma` | float | 1.2 | Farneback Gaussian sigma |

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
  "max_frames": null,
  "farneback_preset": "default",
  "farneback_pyr_scale": 0.5,
  "farneback_levels": 3,
  "farneback_winsize": 15,
  "farneback_iterations": 3,
  "farneback_poly_n": 5,
  "farneback_poly_sigma": 1.2
}
```

### Output Files Created

#### From process_video.py
- `{subdir_name}_{farneback_preset}_basic.csv` - Raw metrics data for each frame
- `{subdir_name}_{farneback_preset}_config.json` - Configuration parameters used

#### From process_metrics.py
- `../video_midi/{subdir_name}_{farneback_preset}/` - Output directory containing:
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

# Use center-only motion preset (for rotation/zoom only)
python run_video_processing.py N17_Mz7fo6C2f --farneback-preset center_only

# Use small motion preset with custom window size
python run_video_processing.py N17_Mz7fo6C2f --farneback-preset small_motion --farneback-winsize 25
```

---

## 2. process_video.py

### Overview
Module containing functions to extract comprehensive visual metrics from video frames using computer vision techniques. This module is called by `run_video_processing.py` and is not designed for standalone execution.

### Key Features
- **Multi-scale analysis**: Analyzes video at different spatial resolutions
- **Color space support**: RGB, grayscale, HSV, and hue-specific metrics
- **Motion analysis**: Lucas-Kanade Farneback optical flow analysis for zoom, rotation, and motion detection
- **Symmetry metrics**: Transpose, reflection, and radial symmetry analysis (Gray channel only for performance)
- **Error dispersion**: Multi-scale information content analysis (Gray channel only for performance)
- **Gaussian mixture modeling**: Multi-Gaussian fit to gray-tone distribution with BIC model selection
- **Performance timing**: Detailed timing analysis for optimization
- **Performance optimization**: Computationally expensive metrics computed only for Gray channel to improve processing speed

### Metrics Computed

#### Basic Intensity Metrics
- **`avg`** - Average intensity per color channel
- **`std`** - Standard deviation of intensity per color channel

#### Symmetry Metrics (Gray channel only for performance)
- **`xps`** - Transpose symmetry (flip around center point) - computed only for Gray channel
- **`rfl`** - Reflection symmetry (vertical and horizontal flips) - computed only for Gray channel
- **`rad`** - Radial symmetry (circular patterns) - computed only for Gray channel

#### Error Dispersion Metrics (Gray channel only for performance)
- **`ee0`** - Total error dispersion (total detail content) - computed only for Gray channel
- **`ee1`** - Large-scale error dispersion (low-resolution detail) - computed only for Gray channel
- **`ee2`** - Small-scale error dispersion (high-resolution detail) - computed only for Gray channel
- **`ed0`** - Distance of total error center from image center - computed only for Gray channel
- **`ed1`** - Distance of large-scale error center from image center - computed only for Gray channel
- **`ed2`** - Distance of small-scale error center from image center - computed only for Gray channel
- **`es0`** - Spatial standard deviation of total error - computed only for Gray channel
- **`es1`** - Spatial standard deviation of large-scale error - computed only for Gray channel
- **`es2`** - Spatial standard deviation of small-scale error - computed only for Gray channel

#### Dark/Light Metrics (Gray channel only for performance)
- **`dcd`** - Dark count (pixels near minimum value) - computed only for Gray channel
- **`dcl`** - Light count (pixels near maximum value) - computed only for Gray channel

#### Gaussian Mixture Model Metrics
- **`gm_n`** - Number of Gaussian components in best model (0-6) - computed for all color channels
- **`gm_m1`** - Mean of first Gaussian component (highest weight) - computed for all color channels
- **`gm_m2`** - Mean of second Gaussian component (second highest weight) - computed for all color channels
- **`gm_m3`** - Mean of third Gaussian component (third highest weight) - computed for all color channels
- **`gm_m4`** - Mean of fourth Gaussian component (fourth highest weight) - computed for all color channels
- **`gm_m5`** - Mean of fifth Gaussian component (fifth highest weight) - computed for all color channels
- **`gm_m6`** - Mean of sixth Gaussian component (lowest weight) - computed for all color channels
- **`gm_s1`** - Standard deviation of first Gaussian component (highest weight) - computed for all color channels
- **`gm_s2`** - Standard deviation of second Gaussian component (second highest weight) - computed for all color channels
- **`gm_s3`** - Standard deviation of third Gaussian component (third highest weight) - computed for all color channels
- **`gm_s4`** - Standard deviation of fourth Gaussian component (fourth highest weight) - computed for all color channels
- **`gm_s5`** - Standard deviation of fifth Gaussian component (fifth highest weight) - computed for all color channels
- **`gm_s6`** - Standard deviation of sixth Gaussian component (lowest weight) - computed for all color channels
- **`gm_a1`** - Amplitude/weight of first Gaussian component (highest weight) - computed for all color channels
- **`gm_a2`** - Amplitude/weight of second Gaussian component (second highest weight) - computed for all color channels
- **`gm_a3`** - Amplitude/weight of third Gaussian component (third highest weight) - computed for all color channels
- **`gm_a4`** - Amplitude/weight of fourth Gaussian component (fourth highest weight) - computed for all color channels
- **`gm_a5`** - Amplitude/weight of fifth Gaussian component (fifth highest weight) - computed for all color channels
- **`gm_a6`** - Amplitude/weight of sixth Gaussian component (lowest weight) - computed for all color channels
- **`gm_bic`** - Bayesian Information Criterion value of best model - computed for all color channels

**Gaussian Mixture Model Overview**: These metrics analyze the distribution of gray-tone values in each frame using a multi-Gaussian mixture model with Bayesian Information Criterion (BIC) for model selection. The system fits models with 1-4 Gaussian components and selects the best model based on BIC scores.

**Parameters**:
- **`histogram_bin_count`** - Number of histogram bins (default: 256)
- **`downscale_factor`** - Image downscaling factor (default: 1 = full resolution)
- **`max_gaussians`** - Maximum number of Gaussian components to test (default: 6)
- **`penalty_factor`** - BIC penalty factor for model complexity (default: 8, higher values prefer fewer components)

**Interpretation**:
- **`gm_n`**: Number of distinct intensity regions in the image (1-6, or 0 for failure)
- **`gm_m1-gm_m6`**: Center intensity values of each region (sorted by weight, highest to lowest)
- **`gm_s1-gm_s6`**: Spread of each intensity region (sorted by weight, highest to lowest)
- **`gm_a1-gm_a6`**: Relative importance/area of each region (sorted by weight, highest to lowest)
- **`gm_bic`**: Model quality score (lower is better)

**Model Selection**: The system uses a modified Bayesian Information Criterion (BIC) with a penalty factor of 8 to strongly prefer simpler models (fewer Gaussian components). This helps avoid overfitting and produces more interpretable results.

#### Motion Metrics (Lucas-Kanade Optical Flow Analysis)
- **`czd`** - Zoom divergence (positive = zoom out, negative = zoom in) - computed only for Gray channel
- **`crc`** - Rotation curl (positive = counterclockwise, negative = clockwise) - computed only for Gray channel
- **`cmv`** - Motion variance (how uniform the motion is) - computed only for Gray channel

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

##### Motion Metrics Implementation
The motion metrics are computed using Lucas-Kanade Farneback optical flow algorithm with optimized parameters for different motion scenarios. The system provides several preset configurations to handle various types of video content.

##### Technical Implementation Details
- **Algorithm**: Uses Lucas-Kanade Farneback optical flow with optimized parameters
- **Analysis Region**: Computed on a centered region of the image (configurable via `center_region_ratio`)
- **Downscaling**: Centered region is downscaled for computational efficiency (factor of 2 by default)
- **Flow Calculation**: Full-resolution optical flow using Farneback algorithm with pyramid levels
- **Mathematical Framework**: Uses finite differences to compute divergence and curl of the flow field
- **Coordinate System**: Analyzes motion relative to image center for zoom and rotation detection
- **Performance**: Motion metrics computed only for Gray channel to improve processing speed

##### Farneback Parameter Presets
The system provides several preset configurations for different motion scenarios:

| Preset | Description | Best For | Parameters |
|--------|-------------|----------|------------|
| `default` | Optimized for noisy video content (recommended) | Low-quality video, compression artifacts | `pyr_scale=0.5, levels=3, winsize=25, iterations=4, poly_n=5, poly_sigma=1.4` |
| `balanced` | Standard parameters for general use | General video content | `pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2` |
| `small_motion` | Optimized for detecting very small motions | Near-still images, subtle movements | `pyr_scale=0.5, levels=4, winsize=21, iterations=5, poly_n=7, poly_sigma=1.5` |
| `large_motion` | Optimized for detecting large motions | Fast-moving content, action scenes | `pyr_scale=0.5, levels=2, winsize=11, iterations=2, poly_n=5, poly_sigma=1.1` |
| `noisy_scene` | Optimized for noisy video content | Low-quality video, compression artifacts | `pyr_scale=0.5, levels=3, winsize=25, iterations=4, poly_n=5, poly_sigma=1.4` |
| `smooth_scene` | Optimized for smooth, clean video content | High-quality video, minimal noise | `pyr_scale=0.5, levels=2, winsize=9, iterations=2, poly_n=5, poly_sigma=1.0` |
| `center_only` | Optimized for center-only rotation/zoom | Camera on tripod, no panning | `pyr_scale=0.5, levels=4, winsize=11, iterations=5, poly_n=7, poly_sigma=1.3` |

**Parameter Descriptions:**
- **`pyr_scale`**: Pyramid scale factor (0.5 = standard, 0.3 = better for small motions, 0.7 = better for large motions)
- **`levels`**: Number of pyramid levels (2-4, higher for smaller motions)
- **`winsize`**: Window size for flow calculation (9-25, larger for smaller motions)
- **`iterations`**: Number of iterations at each level (2-5, more for smaller motions)
- **`poly_n`**: Polynomial degree for expansion (5-7, higher for precision)
- **`poly_sigma`**: Gaussian sigma for polynomial expansion (1.0-1.5, higher for noise reduction)

#### Color-Specific Metrics
- **`H000`** - Proximity to red hue (0°)
- **`H060`** - Proximity to yellow hue (60°)
- **`H120`** - Proximity to green hue (120°)
- **`H180`** - Proximity to cyan hue (180°)
- **`H240`** - Proximity to blue hue (240°)
- **`H300`** - Proximity to magenta hue (300°)
- **`Hmon`** - Monochromaticity (color uniformity)

### Color Channels Analyzed
- **R, G, B** - Red, Green, Blue channels (basic intensity metrics and Gaussian mixture model metrics)
- **Gray** - Grayscale intensity (all metrics computed)
- **S, V** - Saturation and Value from HSV (basic intensity metrics and Gaussian mixture model metrics)
- **H000-H300** - Hue-specific metrics (6 cardinal colors, basic intensity metrics and Gaussian mixture model metrics)
- **Hmon** - Monochromaticity metric (basic intensity metrics and Gaussian mixture model metrics)

**Performance Optimization Note**: To improve processing speed, computationally expensive metrics (symmetry, error dispersion, dark/light, and motion metrics) are computed only for the Gray channel. Other color channels receive zero values for these metrics. This optimization can be easily reversed by removing the Gray channel conditions in the code.

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

**Performance Note**: Due to the optimizations in `process_video.py` where computationally expensive metrics are computed only for the Gray channel, many metrics will have zero values for non-Gray color channels. This is expected behavior and improves processing speed significantly.

### Derived Metrics Computation

#### Automatic Ratio Calculations
The script automatically computes ratio metrics for error dispersion data (Gray channel only):
- **`ee1r`** - `ee1/ee0` (large-scale to total error ratio)
- **`ee2r`** - `ee2/ee0` (small-scale to total error ratio)
- **`es1r`** - `es1/es0` (large-scale to total spatial variation ratio)
- **`es2r`** - `es2/es0` (small-scale to total spatial variation ratio)

#### Motion Metrics Derived Calculations
The script automatically computes derived motion metrics from the base Lucas-Kanade metrics:
- **`crl`** - Positive rotation component (`max(crc, 0)`) - captures counterclockwise rotation
- **`crr`** - Negative rotation component (`max(-crc, 0)`) - captures clockwise rotation
- **`cra`** - Absolute rotation magnitude (`abs(crc)`) - captures total rotation regardless of direction

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
├── {video_name}_{preset}_basic.csv    # Basic metrics output
├── {video_name}_{preset}_config.json  # Configuration file
└── ../video_midi/{video_name}_{preset}/  # Output directory
    ├── {video_name}_derived.xlsx
    ├── {video_name}_plots.pdf
    └── *.mid                 # MIDI files (original names, no preset suffix)
```

## Dependencies
- **OpenCV** (`cv2`) - Video processing and computer vision
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation and CSV handling
- **Mido** - MIDI file generation
- **Matplotlib** - Plotting and PDF generation
- **SciPy** - Statistical functions
- **scikit-learn** - Machine learning (Gaussian mixture models)
- **JSON** - Configuration file handling

## Notes
- Input videos should be in WMV format
- The pipeline is optimized for musical applications with configurable tempo and timing
- All metrics are normalized to 0-1 range before MIDI conversion
- The special case `stretch_value=1, stretch_center=0.5` is always included for consistency
- Performance timing data is automatically generated and saved
