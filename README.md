# Video Processing Pipeline

This repository contains a comprehensive video processing pipeline that extracts visual metrics from video files and converts them into MIDI data for musical applications.

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Create a configuration file**:
   ```bash
   cp default_config.json my_video.json
   ```

3. **Edit configuration** and set your video name:
   ```json
   "video": {
     "video_name": "my_video"
   }
   ```

4. **Place video file** in `data/input/`:
   ```bash
   # e.g., data/input/my_video.wmv or data/input/my_video.mp4
   ```

5. **Run the pipeline**:
   ```bash
   python run_video_processing.py my_video.json
   ```

6. **Find outputs** in `data/output/my_video_default/`

---

## Architecture Overview

The pipeline consists of four main Python scripts:

1. **`run_video_processing.py`** - Main orchestrator (entry point)
2. **`process_video.py`** - Video analysis module (extracts visual metrics)
3. **`process_metrics.py`** - Data processing module (transforms to MIDI)
4. **`cluster_primary.py`** - Clustering module (groups similar frames)

### Processing Flow

```
Video File (data/input/)
    ↓
Step 1: process_video.py → CSV + Config (data/output/)
    ↓
Step 2: process_metrics.py → MIDI + Excel + Plots (data/output/)
    ↓
Step 3: cluster_primary.py → Cluster assignments + Quality metrics (data/output/)
```

Each step can be enabled/disabled via configuration (`process_video`, `process_metrics`, `process_clusters`).

---

## Configuration

### JSON Configuration File

All pipeline parameters are configured via JSON files. The configuration file is the **only** required argument.

**Usage**:
```bash
python run_video_processing.py <config_file.json>
```

**Examples**:
```bash
python run_video_processing.py default_config.json
python run_video_processing.py my_video.json
python run_video_processing.py N29_3M2pM6dispA7_config.json
```

### Configuration Structure

```json
{
  "description": "Configuration description",
  "input": {
    "directory": "data/input",
    "note": "Video files are read from this directory"
  },
  "output": {
    "directory": "data/output",
    "note": "All output files are written to subdirectories here"
  },
  "video": {
    "video_name": "my_video",
    "file_extensions": [".wmv", ".mp4"],
    "max_frames": null
  },
  "timing": {
    "frames_per_second": 30,
    "beats_per_minute": 64,
    "beats_per_midi_event": 1,
    "ticks_per_beat": 480
  },
  "video_processing": {
    "downscale_large": 100,
    "downscale_medium": 10,
    "optical_flow": {
      "preset": "default",
      "pyr_scale": 0.5,
      "levels": 3,
      "winsize": 15,
      "iterations": 3,
      "poly_n": 5,
      "poly_sigma": 1.2
    }
  },
  "metrics_processing": {
    "filter_periods": [17, 65, 257],
    "stretch_values": [8],
    "stretch_centers": [0.33, 0.67],
    "cc_number": 1
  },
  "pipeline_control": {
    "skip_video": false,
    "skip_metrics": false
  }
}
```

### Configuration Parameters

#### Video Section (`video`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video_name` | string | **REQUIRED** | Video file name (without extension) |
| `file_extensions` | array | `[".wmv", ".mp4"]` | Video formats to search for |
| `max_frames` | int/null | `null` | Limit frames processed (useful for testing) |

**Example**:
```json
"video": {
  "video_name": "N17_Mz7fo6C2f",
  "file_extensions": [".wmv", ".mp4", ".avi"],
  "max_frames": 1000
}
```

#### Timing Section (`timing`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `frames_per_second` | int | `30` | Video frame rate |
| `beats_per_minute` | int | `64` | Tempo for MIDI output |
| `beats_per_midi_event` | int | `1` | MIDI event granularity |
| `ticks_per_beat` | int | `480` | MIDI resolution |

**Example**:
```json
"timing": {
  "frames_per_second": 30,
  "beats_per_minute": 120,
  "beats_per_midi_event": 1,
  "ticks_per_beat": 480
}
```

#### Video Processing Section (`video_processing`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `downscale_large` | int | `100` | Large downscale factor |
| `downscale_medium` | int | `10` | Medium downscale factor |

**Optical Flow Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `preset` | string | `"default"` | Preset configuration (see below) |
| `pyr_scale` | float | `0.5` | Pyramid scale factor (0-1) |
| `levels` | int | `3` | Number of pyramid levels |
| `winsize` | int | `15` | Averaging window size |
| `iterations` | int | `3` | Iterations at each level |
| `poly_n` | int | `5` | Polynomial expansion degree |
| `poly_sigma` | float | `1.2` | Gaussian standard deviation |

**Optical Flow Presets**:

| Preset | Best For | Description |
|--------|----------|-------------|
| `default` | Low-quality video | Optimized for noisy video, compression artifacts |
| `balanced` | General content | Standard parameters for general use |
| `small_motion` | Subtle movements | Detects very small motions |
| `large_motion` | Action scenes | Optimized for fast-moving content |
| `noisy_scene` | Compression artifacts | Better handling of noise |
| `smooth_scene` | High-quality video | Optimized for clean content |
| `center_only` | Tripod camera | Focus on rotation/zoom, no panning |

#### Metrics Processing Section (`metrics_processing`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filter_periods` | array | `[17, 65, 257]` | Smoothing filter window sizes |
| `stretch_values` | array | `[8]` | Non-linear stretch factors |
| `stretch_centers` | array | `[0.33, 0.67]` | Center points for stretching |
| `cc_number` | int | `1` | MIDI continuous controller number |

**Example - Multiple filter periods**:
```json
"metrics_processing": {
  "filter_periods": [17, 65, 257, 513],
  "stretch_values": [4, 8, 16],
  "stretch_centers": [0.25, 0.5, 0.75],
  "cc_number": 1
}
```

#### Pipeline Control Section (`pipeline_control`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `process_video` | bool | `true` | Run video analysis (extract primary metrics) |
| `process_metrics` | bool | `true` | Run metrics processing (MIDI generation) |
| `process_clusters` | bool | `true` | Run clustering analysis |

**Example - Reprocess metrics only**:
```json
"pipeline_control": {
  "process_video": false,
  "process_metrics": true,
  "process_clusters": true
}
```

**Example - Run clustering only**:
```json
"pipeline_control": {
  "process_video": false,
  "process_metrics": false,
  "process_clusters": true
}
```

### Common Configuration Use Cases

**Testing with limited frames**:
```json
"video": {
  "video_name": "test_video",
  "max_frames": 500
}
```

**High tempo music video**:
```json
"timing": {
  "beats_per_minute": 140
}
```

**Experimenting with filters**:
```json
"metrics_processing": {
  "filter_periods": [9, 17, 33, 65, 129, 257, 513]
}
```

**Regenerate MIDI from existing analysis**:
```json
"pipeline_control": {
  "process_video": false,
  "process_metrics": true,
  "process_clusters": true
}
```

---

## Video Analysis (process_video.py)

### Overview

Extracts comprehensive visual metrics from video frames using computer vision techniques. This module is called by `run_video_processing.py` and is not designed for standalone execution.

### Key Features

- **Multi-scale analysis**: Analyzes video at different spatial resolutions
- **Color space support**: RGB, grayscale, HSV, and hue-specific metrics
- **Motion analysis**: Farneback optical flow for zoom, rotation, and motion
- **Symmetry metrics**: Rotational and radial symmetry detection
- **Error dispersion**: Multi-scale information content analysis
- **Gaussian mixture modeling**: Multi-Gaussian fit with BIC model selection
- **Performance optimization**: Expensive metrics computed only for Gray channel

### Metrics Computed

#### Basic Intensity Metrics (All Channels)
- **`avg`** - Average intensity per color channel
- **`std`** - Standard deviation of intensity per color channel

#### Symmetry Metrics (Gray Channel Only)
- **`rsn`** - Rotational symmetry n-fold (detected order: 2, 3, 4, etc.)
- **`rss`** - Rotational symmetry strength (confidence ratio)
- **`rad`** - Radial symmetry (circular patterns)

#### Error Dispersion Metrics (Gray Channel Only)
- **`ee0`** - Total error dispersion (total detail content)
- **`ee1`** - Large-scale error dispersion (low-res detail)
- **`ee2`** - Small-scale error dispersion (high-res detail)
- **`ed0`** - Distance of total error center from image center
- **`ed1`** - Distance of large-scale error center
- **`ed2`** - Distance of small-scale error center
- **`es0`** - Spatial standard deviation of total error
- **`es1`** - Spatial standard deviation of large-scale error
- **`es2`** - Spatial standard deviation of small-scale error

#### Dark/Light Metrics (Gray Channel Only)
- **`dcd`** - Dark count (pixels near minimum value)
- **`dcl`** - Light count (pixels near maximum value)

#### Gaussian Mixture Model Metrics (All Channels)
- **`gm_n`** - Number of Gaussian components (0-6)
- **`gm_m1-gm_m6`** - Mean of each component (sorted by weight)
- **`gm_s1-gm_s6`** - Standard deviation of each component
- **`gm_a1-gm_a6`** - Amplitude/weight of each component
- **`gm_bic`** - Bayesian Information Criterion value

**Gaussian Mixture Model**: Analyzes gray-tone distribution using multi-Gaussian mixture with BIC model selection. Fits models with 1-6 components and selects the best based on BIC scores with penalty factor of 8 to prefer simpler models.

#### Motion Metrics (Gray Channel Only)

Computed using Farneback optical flow algorithm:

- **`czd`** - Zoom divergence (positive = zoom out, negative = zoom in)
- **`crc`** - Rotation curl (positive = CCW, negative = CW)
- **`cmv`** - Motion variance (uniformity of motion)

**Motion Analysis Details**:
- Uses divergence (∂u/∂x + ∂v/∂y) for zoom detection
- Uses curl (∂v/∂x - ∂u/∂y) for rotation detection
- Analyzes centered region for center-anchored motion
- Multiple presets available for different video types

#### Color-Specific Metrics (All Channels)
- **`H000`** - Proximity to red hue (0°)
- **`H060`** - Proximity to yellow hue (60°)
- **`H120`** - Proximity to green hue (120°)
- **`H180`** - Proximity to cyan hue (180°)
- **`H240`** - Proximity to blue hue (240°)
- **`H300`** - Proximity to magenta hue (300°)
- **`Hmon`** - Monochromaticity (color uniformity)

### Color Channels Analyzed

- **R, G, B** - Red, Green, Blue (basic + GMM metrics)
- **Gray** - Grayscale (ALL metrics computed here)
- **S, V** - Saturation, Value from HSV (basic + GMM metrics)
- **H000-H300** - Hue-specific metrics (basic + GMM metrics)
- **Hmon** - Monochromaticity (basic + GMM metrics)

**Performance Note**: Computationally expensive metrics (symmetry, error dispersion, dark/light, motion) are computed ONLY for Gray channel to improve processing speed. Other channels receive zero values for these metrics.

### Output Files

Created in `data/output/{video_name}_{preset}/`:
- `{video_name}_{preset}_basic.csv` - Raw metrics for each frame
- `{video_name}_{preset}_config.json` - Configuration used

---

## Metrics Processing (process_metrics.py)

### Overview

Processes basic metrics from `process_video.py`, computes derived metrics, applies transformations, and generates MIDI files. Called by `run_video_processing.py`.

### Derived Metrics

#### Automatic Ratio Calculations (Gray Channel Only)
- **`ee1r`** - `ee1/ee0` (large-scale to total error ratio)
- **`ee2r`** - `ee2/ee0` (small-scale to total error ratio)
- **`es1r`** - `es1/es0` (large-scale to total spatial variation)
- **`es2r`** - `es2/es0` (small-scale to total spatial variation)

#### Motion Derived Calculations
- **`crl`** - Positive rotation (`max(crc, 0)`) - counterclockwise
- **`crr`** - Negative rotation (`max(-crc, 0)`) - clockwise
- **`cra`** - Absolute rotation (`abs(crc)`) - total rotation magnitude

### Processing Pipeline

The script applies transformation stages in this exact order:

1. **Rank/Value Processing** - Creates base entries
   - `_v` - Original values from CSV
   - `_r` - Percentile transformation (0-1 range)

2. **Scaling** - Normalizes data to 0-1 range
   - Applied to all data (no key name change)

3. **Filtering** - Triangular smoothing filters ⚠️ **MUST BE LAST**
   - `_f001` - No filtering (original, period 1)
   - `_f017` - ~2 bars at 4/4 time (period 17)
   - `_f065` - ~16 bars at 4/4 time (period 65)
   - `_f257` - ~64 bars at 4/4 time (period 257)

4. **Stretching** - Non-linear transformation
   - Sigmoid-like stretching function
   - Parameters: `stretch_value` and `stretch_center`
   - Always includes `stretch_value=1, stretch_center=0.5`

5. **Inversion** - Value inversion
   - `_o` - Original values (no inversion)
   - `_i` - Inverted values (1 - original)

**⚠️ Critical**: Filtering MUST come last. Filtering raw data produces jagged curves. Filtering must be applied to processed (scaled) data to produce smooth curves.

### Key Naming Convention

Final key names reflect processing order:
```
R_avg_v_f017_s1-0.5_o
│ │   │ │    │      │
│ │   │ │    │      └─ Inversion (o=original, i=inverted)
│ │   │ │    └──────── Stretching (s{value}-{center})
│ │   │ └───────────── Filtering (f{period:03d})
│ │   └─────────────── Rank/Value (v=value, r=rank)
│ └─────────────────── Metric name
└───────────────────── Color channel
```

### MIDI File Generation

#### File Organization

MIDI files in `data/output/{video_name}_{preset}/`:
- Individual files: `{variable}_{metric}_{rank_type}_{filter}_{stretch}_{inversion}.mid`
- Example: `R_avg_v_f017_s8-0.33_o.mid`

#### MIDI Parameters
- **CC Number**: Configurable (default: 1)
- **Channel**: 7
- **Value Range**: 0-127 (scaled from 0-1)
- **Timing**: Based on frame intervals and tempo

### Output Files

Created in `data/output/{video_name}_{preset}/`:
- `{video_name}_derived.xlsx` - Complete dataset with all metrics
- `{video_name}_plots.pdf` - Visual plots (30 per page)
- Multiple `.mid` files for different combinations:
  - Value vs Rank (`_v`, `_r`)
  - Filter periods (`_f001`, `_f017`, `_f065`, `_f257`)
  - Stretch parameters (`_s1-0.5`, `_s8-0.33`, etc.)
  - Inversion (`_o`, `_i`)

---

## Frame Clustering (cluster_primary.py)

### Overview

Clusters video frames based on primary metrics using K-Means and Gaussian Mixture Models (GMM). This helps identify groups of similar frames for analysis and understanding video structure.

### Key Features

- **K-Means clustering** - Fast, deterministic baseline algorithm
- **GMM clustering** - Better for elliptical clusters, provides BIC/AIC model selection
- **Rank normalization** - Robust, scale-free (default)
- **Z-score option** - Available for alternative normalization
- **Quality metrics** - Silhouette, Calinski-Harabasz, Davies-Bouldin, BIC/AIC
- **Exemplar frames** - Identifies representative frames per cluster
- **Automatic cleaning** - Drops constant columns (std = 0)

### Clustering Methods

#### K-Means
- Uses Euclidean distance in normalized feature space
- Deterministic with fixed random_state
- Fast and scalable
- Good baseline for spherical clusters

#### Gaussian Mixture Model (GMM)
- Full covariance modeling
- Better for elliptical/complex cluster shapes
- Provides BIC/AIC for model selection
- Soft cluster assignments available

### Quality Metrics

**Silhouette Score** (range: -1 to 1, higher is better)
- Measures how similar frames are to their own cluster vs other clusters
- Values near 1: well-separated clusters
- Values near 0: overlapping clusters
- Values near -1: frames may be in wrong cluster

**Calinski-Harabasz Score** (higher is better)
- Ratio of between-cluster to within-cluster variance
- Higher values indicate better-defined clusters

**Davies-Bouldin Score** (lower is better)
- Average similarity between each cluster and its most similar cluster
- Lower values indicate better separation

**BIC/AIC** (GMM only, lower is better)
- Bayesian/Akaike Information Criterion
- Balances model fit with complexity
- Helps select optimal number of clusters

### Configuration

Add to JSON config file:

```json
"pipeline_control": {
  "process_video": true,
  "process_metrics": true,
  "process_clusters": true
},
"cluster_processing": {
  "k_values": [2, 3, 4, 5, 6, 8, 10, 12],
  "normalization": "rank",
  "metrics_to_exclude": [],
  "random_state": 42
}
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k_values` | array | `[2,3,4,5,6,8,10,12]` | Number of clusters to try |
| `normalization` | string | `"rank"` | `"rank"` or `"zscore"` |
| `metrics_to_exclude` | array | `[]` | Metrics to exclude from clustering |
| `random_state` | int | `42` | Random seed for reproducibility |

### Output Files

Created in `data/output/{video_name}_{preset}/`:

- **`*_clusters.csv`** - Cluster assignments for each frame
  - Columns: `kmeans_k2`, `kmeans_k3`, ..., `gmm_k2`, `gmm_k3`, ...
- **`*_cluster_scores.json`** - Quality metrics for each k and algorithm
- **`*_cluster_exemplars.json`** - Representative frames per cluster

### Standalone Usage

Can also be run independently:

```bash
# Basic usage
python cluster_primary.py data/output/N29_*/N29_*_basic.csv

# Custom k values
python cluster_primary.py data/output/N29_*/N29_*_basic.csv --k-values 2 3 4 5

# Z-score normalization
python cluster_primary.py data/output/N29_*/N29_*_basic.csv --normalization zscore

# Exclude specific metrics
python cluster_primary.py data/output/N29_*/N29_*_basic.csv --exclude czd crc cmv
```

### Interpreting Results

**Choosing K:**
1. Look at silhouette scores (higher is better)
2. Check BIC values for GMM (lower is better)
3. Examine cluster sizes for balance
4. Consider domain knowledge

**Example interpretation:**
- K=2: Often highest silhouette, divides video into major sections
- K=4: BIC minimum may suggest natural groupings
- Larger K: More granular scene divisions

**Using exemplars:**
- Representative frames closest to cluster centers
- Useful for understanding what each cluster represents
- Can be used for visualization or manual inspection

### Integration with Pipeline

Clustering runs as Step 3 after metrics processing:
1. Step 1: Video analysis → primary metrics CSV
2. Step 2: Metrics processing → derived metrics + MIDI
3. Step 3: Clustering → cluster assignments + quality metrics

Disable clustering with `"process_clusters": false` in config.

---

## Directory Structure

```
project/
├── run_video_processing.py           # Main entry point
├── process_video.py                  # Video analysis module
├── process_metrics.py                # Metrics processing module
├── cluster_primary.py                # Clustering module
├── default_config.json               # Configuration template
├── example_config.json               # Example configuration
├── requirements.txt                  # Python dependencies
├── CLAUDE.md                         # Instructions for Claude Code
├── README.md                         # This file
│
├── data/
│   ├── input/                        # Video files (gitignored)
│   │   ├── my_video.wmv
│   │   ├── my_video.mp4
│   │   └── ...
│   │
│   └── output/                       # All outputs (gitignored)
│       └── {video_name}_{preset}/
│           ├── {video_name}_{preset}_basic.csv
│           ├── {video_name}_{preset}_config.json
│           ├── {video_name}_{preset}_clusters.csv
│           ├── {video_name}_{preset}_cluster_scores.json
│           ├── {video_name}_{preset}_cluster_exemplars.json
│           ├── {video_name}_derived.xlsx
│           ├── {video_name}_plots.pdf
│           └── *.mid                 # MIDI files
```

---

## Usage Examples

### Basic Usage
```bash
# Create configuration
cp default_config.json my_video.json

# Edit configuration (set video_name)
# Place video in data/input/my_video.wmv

# Run pipeline
python run_video_processing.py my_video.json
```

### Testing with Limited Frames
```json
{
  "video": {
    "video_name": "test_video",
    "max_frames": 1000
  }
}
```

### Regenerate MIDI and Clusters (Skip Video)
```json
{
  "pipeline_control": {
    "process_video": false,
    "process_metrics": true,
    "process_clusters": true
  }
}
```

### Custom Filter Periods
```json
{
  "metrics_processing": {
    "filter_periods": [5, 17, 65, 257, 513],
    "stretch_values": [1, 2, 4, 8],
    "stretch_centers": [0.1, 0.33, 0.5, 0.67, 0.9]
  }
}
```

### Different Optical Flow Preset
```json
{
  "video_processing": {
    "optical_flow": {
      "preset": "center_only"
    }
  }
}
```

---

## Dependencies

Install via `pip install -r requirements.txt`:

- **OpenCV** (`cv2`) - Video processing and computer vision
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation and CSV handling
- **Mido** - MIDI file generation
- **Matplotlib** - Plotting and PDF generation
- **SciPy** - Statistical functions
- **scikit-learn** - Machine learning (Gaussian mixture models)

Python version: 3.10+

---

## Important Notes

### Performance Optimization
Computationally expensive metrics (symmetry, error dispersion, dark/light, motion) are computed **only for Gray channel** to improve processing speed. Other color channels receive zero values. This optimization can be reversed by removing Gray channel conditions in `process_video.py`.

### Processing Order
⚠️ **Critical**: The filtering step must come LAST in the processing pipeline. Applying filtering to raw data produces jagged curves. Filtering must be applied to processed (scaled) data to produce smooth curves. This is a design principle that should not be changed.

### Video Formats
The pipeline supports `.wmv` and `.mp4` formats by default. Additional formats can be added via the `file_extensions` configuration.

### MIDI Resolution
Standard MIDI resolution is 480 ticks per beat. Higher values provide more timing precision but larger files.

### Configuration Files
Keep multiple configuration files for different videos or experiments. Use descriptive names like `{video_name}_experiment_description.json`. Version control configurations alongside code.

---

## Troubleshooting

### Video file not found
- Ensure video is in `data/input/` directory
- Check `video_name` matches filename (without extension)
- Verify `file_extensions` includes your video format

### CSV file not found (metrics processing)
- Ensure video processing completed successfully
- Check `data/output/{video_name}_{preset}/` exists
- Verify CSV filename matches pattern

### High memory usage
- Use `max_frames` to limit processing
- Reduce `downscale_large` and `downscale_medium` values
- Process video in multiple chunks

### Slow processing
- Motion metrics are expensive; use `center_only` preset
- Reduce `max_frames` for testing
- Increase downscale factors
- Ensure using Gray channel optimization

---

## Configuration Templates

See `default_config.json` for a complete template with all parameters and defaults.

See `example_config.json` for a working example with a specific video.

---

## License

[Add your license information here]
