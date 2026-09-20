# Video Processing and Tempo Mapping Suite

This repository contains tools for video analysis, tempo mapping, and audio leveling, providing three complementary workflows:

1. **Visual Metrics Pipeline** (`run_video_processing.py`) - Extracts comprehensive visual metrics from videos and generates MIDI CC tracks
2. **Tempo Mapping** (`calculate_tempo_from_inverse.py`) - Generates variable tempo maps from zoom/speed data
3. **Audio Leveling** (`audio_level_curve.py` + `curve_to_dawproject.py`) - Computes a volume-riding curve from an audio file and delivers it to Cubase as a DAWproject automation envelope

---

## Table of Contents

- [Quick Start: Visual Metrics Pipeline](#quick-start-visual-metrics-pipeline)
- [Quick Start: Tempo Mapping](#quick-start-tempo-mapping)
- [Quick Start: Audio Leveling](#quick-start-audio-leveling)
- [Entry Points](#entry-points)
  - [Visual Metrics Pipeline](#1-run_video_processingpy---visual-metrics-pipeline)
  - [Tempo Mapping](#2-calculate_tempo_from_inversepy---variable-tempo-mapping)
  - [Audio Leveling](#3-audio_level_curvepy--curve_to_dawprojectpy---audio-leveling)
- [Architecture: Visual Metrics Pipeline](#architecture-visual-metrics-pipeline)
- [Configuration](#configuration)
- [Video Analysis](#video-analysis-process_videopy)
- [Metrics Processing](#metrics-processing-process_metricspy)
- [DAWproject Output](#dawproject-output-write_dawprojectpy)
- [Frame Clustering](#frame-clustering-cluster_primarypy)
- [Audio Leveling](#audio-leveling-audio_level_curvepy)
  - [Choosing the Mean Weighting Distance](#choosing-the-mean-weighting-distance)
  - [Choosing the Target](#choosing-the-target)
  - [Worked Example](#worked-example)
- [Future Integration](#future-integration-tempo-synchronized-visual-metrics)
- [Additional Utilities](#additional-utilities)

---

## Quick Start: Visual Metrics Pipeline

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Create a configuration file** in the `json/` folder (copy an existing
   config, or build one from the [Configuration Structure](#configuration-structure)
   below):
   ```bash
   cp json/some_existing_config.json json/my_video.json
   ```

3. **Edit configuration** — set your video name and the required
   `frame_interval` (see [Video Processing](#video-processing-section-video_processing)):
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
   python run_video_processing.py json/my_video.json
   ```

6. **Find outputs** in `data/output/my_video_default/`

---

## Quick Start: Tempo Mapping

1. **Prepare speed/zoom data**:
   - Create a Python file with speed values (e.g., `data/input/N32_speed.py`)
   - File should contain either `y_values = [...]` or `s = [...]`

2. **Edit parameters** in `calculate_tempo_from_inverse.py`:
   ```python
   # At bottom of file (around lines 875-910)
   y_values_path = "data/input/N32_speed.py"
   mean_tempo_bpm = 108.0

   # Choose mode:
   use_log_scale = False  # False=LINEAR (inverse), True=LOG (symmetric)
   scaling_param = 0.1    # LINEAR: 0.1=10% blend; LOG: 20.0=±20 BPM per octave
   ```

3. **Run tempo mapping**:
   ```bash
   python calculate_tempo_from_inverse.py
   ```

4. **Find outputs** in `data/output/`:
   - `beat_tempos_inverse_108bpm.csv` - Frame-level tempo data
   - `tempo_map_inverse_108bpm.mid` - MIDI with tempo changes and CC tracks

**Example Configurations**:
- **Constant tempo** (no zoom influence): `scaling_param = 0.0`
- **Linear 10% blend**: `use_log_scale = False, scaling_param = 0.1`
- **Log ±20 BPM per doubling**: `use_log_scale = True, scaling_param = 20.0`

---

## Quick Start: Audio Leveling

Compute a volume curve that evens out an audio file, then carry it into Cubase.

```bash
# 1. Analyze the audio and build the gain curve
python audio_level_curve.py "data/input/my_song.wav"

# 2. Package the curve and the audio into a DAWproject
python curve_to_dawproject.py \
    "data/output/my_song_cf25_gau15_gain_points.csv" \
    "data/input/my_song.wav"

# 3. In Cubase: File > Import > DAWproject
```

Step 1 also writes `my_song_cf25_gau15_leveled.wav`, a preview render, so you
can hear the result before opening Cubase.

The defaults need no knowledge of the file: the target is **derived from the
audio** so the volume is reduced 25% of the time. The two knobs worth reaching
for are `--target-fraction` (how often the level comes down) and
`--mean-distance` (how quickly the curve may move).

Both stages document themselves:

```bash
python audio_level_curve.py --help
python curve_to_dawproject.py --help
```

---

## Entry Points

### 1. `run_video_processing.py` - Visual Metrics Pipeline

Main pipeline for comprehensive video analysis with fixed-tempo MIDI generation.

**Usage**:
```bash
python run_video_processing.py <config_file.json>
```

**What it does**:
- Analyzes video frames for color, motion, symmetry, and other visual metrics
- Generates 100+ derived metrics with filtering and transformations
- Creates MIDI CC tracks for each metric (fixed tempo)
- Optionally performs clustering analysis on frames

**Key modules**:
- `process_video.py` - Extracts primary visual metrics
- `process_metrics.py` - Computes derived metrics and generates MIDI
- `process_clusters.py` - Groups similar frames using K-Means/GMM

### 2. `calculate_tempo_from_inverse.py` - Variable Tempo Mapping

Generates variable tempo maps where tempo is controlled by zoom/speed data. Supports two modes:

**Usage**:
```bash
# Edit the main section at bottom of file to configure paths and mode
python calculate_tempo_from_inverse.py
```

**What it does**:
- Reads zoom/speed values from Python files (e.g., `N32_speed.py`)
- Calculates frame-level tempo based on zoom rate (linear or log mode)
- Generates tempo map CSV with per-frame tempo data
- Creates MIDI file with:
  - Frame-by-frame tempo changes
  - Beat notes at variable intervals
  - Multiple CC1 tracks representing zoom depth (Normal, Inverted, Deep/Medium/Shallow variants)

**Two Modes Available**:

1. **LINEAR Mode** (`use_log_scale=False`, default):
   - Tempo inversely proportional to zoom rate
   - **Algorithm**: `tempo = mean_tempo + scaling_param × ((k / zoom_rate) - mean_tempo)`
   - **`scaling_param`**: Blend fraction (0-1), e.g., 0.1 = 10% zoom influence
   - **Behavior**: Doubling zoom rate → tempo halves; Halving → tempo doubles
   - **Use case**: Traditional inverse relationship

2. **LOG Mode** (`use_log_scale=True`):
   - Tempo varies symmetrically with log of zoom rate
   - **Algorithm**: `tempo = mean_tempo + tempo_change_per_octave × (log₂(zoom) - mean(log₂(zoom)))`
   - **`scaling_param`**: BPM change per octave, e.g., 20.0 = ±20 BPM per doubling/halving
   - **Behavior**: Doubling or halving zoom rate → ±same BPM change (symmetric)
   - **Use case**: Perceptually uniform zoom scaling
   - **Guarantee**: Average tempo exactly equals `mean_tempo_bpm` (no rescaling needed)

**Key Feature - Independent CC Tracks**:
- **IMPORTANT**: CC tracks represent zoom depth directly, NOT tempo
- This means you can set `scaling_param=0` for constant tempo while still getting zoom variation in CC tracks
- **LINEAR mode**: CC values based on `1/zoom_rate` (zoom depth)
- **LOG mode**: CC values based on `log₂(zoom_rate)` (log zoom depth)
- Tracks include: Normal, Inverted, Deep/Medium/Shallow zoom (based on percentiles)

**Example Use Case**:
```python
# Constant 108 BPM tempo, but CC tracks still show zoom variation
mean_tempo_bpm = 108.0
scaling_param = 0.0  # No tempo variation
use_log_scale = False  # or True, doesn't matter for constant tempo
```

---

### 3. `audio_level_curve.py` + `curve_to_dawproject.py` - Audio Leveling

**Usage**:
```bash
python audio_level_curve.py INPUT_AUDIO [options]
python curve_to_dawproject.py POINTS_CSV AUDIO_FILE [options]
```

**What it does**:
- Measures EBU R128 / ITU-R BS.1770 K-weighted loudness over short blocks
- Smooths that loudness with a kernel specified by its **mean weighting distance**
- Turns the smoothed loudness into a gain curve (a slow fader ride)
- Renders a preview and packages the curve into a `.dawproject` for Cubase

**Key modules**: `audio_smoothing.py` (kernels), `audio_level_curve.py` (analysis),
`curve_to_dawproject.py` (DAWproject writer)

---


## Architecture: Visual Metrics Pipeline

The main pipeline consists of four Python modules:

### Processing Flow

```
Video File (data/input/)
    ↓
Step 1: process_video.py → CSV + Config (data/output/)
    ↓
Step 2: process_metrics.py → MIDI + Excel + Plots (data/output/)
    ↓
Step 3: cluster_primary.py → Cluster assignments + Quality metrics (data/output/)
    ↓
Step 4: write_midi.py → MIDI CC files (data/output/)          [tempo-aware]
    ↓
Step 5: write_dawproject.py → .dawproject (data/output/)      [tempo-free]
```

Each step can be enabled/disabled via configuration (`process_video`, `process_metrics`, `process_clusters`, `write_midi`, `write_dawproject`).

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
    "ticks_per_beat": 480
  },
  "video_processing": {
    "downscale_large": 100,
    "downscale_medium": 10,
    "frame_interval": 28.125,
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
    "block_beats": [],
    "stretch_values": [8],
    "stretch_centers": [0.33, 0.67],
    "cc_number": 1
  },
  "cluster_processing": {
    "k_values": [2, 3, 4, 5, 6, 8, 10, 12],
    "normalization": "rank",
    "metrics_to_exclude": [],
    "random_state": 42,
    "boxcar_periods": null
  },
  "pipeline_control": {
    "process_video": true,
    "process_metrics": true,
    "process_clusters": true,
    "write_midi": true,
    "write_dawproject": false
  },
  "dawproject": {
    "columns": [],
    "interpolation": "linear"
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
| `beats_per_minute` | int/float/string | `64` | Tempo for MIDI output. A number = constant tempo; a string = path to a MIDI tempo file (variable tempo, e.g. produced by `calculate_tempo_from_inverse.py`) |
| `ticks_per_beat` | int | `480` | MIDI resolution |

> **Note:** `beats_per_midi_event` from older configs is no longer read. Analysis-frame sampling is now controlled by `video_processing.frame_interval` (see below). Tempo is applied only in the MIDI-writing stage, so video analysis is tempo-free.

**Example**:
```json
"timing": {
  "frames_per_second": 30,
  "beats_per_minute": 120,
  "ticks_per_beat": 480
}
```

#### Video Processing Section (`video_processing`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `downscale_large` | int | `100` | Large downscale factor |
| `downscale_medium` | int | `10` | Medium downscale factor |
| `frame_interval` | int/float | **REQUIRED** (when `process_video` runs) | Spacing, in video frames, between analysed frames (may be fractional). The analysis stage is tempo-free and samples every `frame_interval`-th frame. To get one analysis sample per beat, set it to `60 × frames_per_second / beats_per_minute` (e.g. `28.125` at 30 fps / 64 BPM, `16.667` at 30 fps / 108 BPM). Non-integer values are handled without drift — each target frame is recomputed as `round(k × frame_interval)`. |

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
| `filter_periods` | array | `[17, 65, 257]` | Triangular (weighted moving-average) smoothing window sizes. Produces smooth, continuous curves. Each period becomes a column suffixed `_f{period:03d}` (`_f001` = unfiltered pass-through). |
| `block_beats` | array | `[]` | **Block (boxcar) averaging** window sizes, in beats. Groups consecutive rows into blocks of N and replaces each block with its mean, producing a **stair-step (piecewise-constant)** curve — distinct from the triangular `filter_periods`. Each size becomes a column suffixed `_b{beats:03d}` (`_b001` = pass-through) and flows through the same downstream stages (stretch → invert → MIDI), yielding its own MIDI track. Units are beats **only when** one CSV row equals one beat, i.e. `frame_interval` = `60 × fps / bpm`. |
| `stretch_values` | array | `[8]` | Non-linear stretch factors |
| `stretch_centers` | array | `[0.33, 0.67]` | Center points for stretching |
| `cc_number` | int | `1` | MIDI continuous controller number |

**`filter_periods` vs `block_beats`** — the two smoothers are complementary:

| Option | Filter shape | Result | Column suffix |
|--------|--------------|--------|---------------|
| `filter_periods` | triangular (weighted) moving average | smooth, continuous curve | `_f{period:03d}` |
| `block_beats` | flat block mean over N beats | stair-step / piecewise-constant "hold for N beats" | `_b{beats:03d}` |

**Example - Filters plus block averaging (4 / 8 / 16 bars at 4 beats/bar)**:
```json
"metrics_processing": {
  "filter_periods": [65, 129],
  "block_beats": [16, 32, 64],
  "stretch_values": [4, 8, 16],
  "stretch_centers": [0.25, 0.5, 0.75],
  "cc_number": 1
}
```

#### Pipeline Control Section (`pipeline_control`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `process_video` | bool | `true` | Run video analysis (extract primary metrics) |
| `process_metrics` | bool | `true` | Run metrics processing (derive metrics, write values CSV) |
| `process_clusters` | bool | `true` | Run clustering analysis |
| `write_midi` | bool | `true` | Write MIDI files (the only tempo-aware stage; renders metrics and cluster CSVs to `.mid`) |
| `write_dawproject` | bool | `false` | Write a `.dawproject` of volume-automated group busses (tempo-free; see below) |

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

#### Cluster Processing Section (`cluster_processing`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k_values` | array | `[2, 3, 4, 5, 6, 8, 10, 12]` | Cluster counts (k) to try |
| `normalization` | string | `"rank"` | Feature normalization method |
| `metrics_to_exclude` | array | `[]` | Metric column names to drop before clustering |
| `random_state` | int | `42` | Random seed for reproducibility |
| `boxcar_periods` | array/null | `null` | **Iterative boxcar (majority-vote) smoothing** of cluster assignments — odd-integer widths, in **rows/frames**, applied repeatedly until convergence to remove cluster flickering. Distinct from `metrics_processing.block_beats`: it smooths integer cluster IDs, not metric values. Each period becomes a `_b`-suffixed cluster track. |

**Example - Smooth cluster assignments**:
```json
"cluster_processing": {
  "k_values": [4, 6, 8],
  "boxcar_periods": [15, 31]
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

#### Gaussian Mixture Model Metrics

GMM analyzes the channel's value distribution using a multi-Gaussian mixture with BIC model selection. Fits models with 1–6 components and picks the best by BIC (penalty factor 8 to prefer simpler models). Components are sorted by weight (component 1 is the largest).

**Stored on every channel (R, G, B, Gray, S, V):**
- **`gmn`** - Number of Gaussian components selected (1-6)
- **`gs1`** - Standard deviation of component 1 (the largest-weight component)

**Stored on Gray channel only** (full mixture):
- **`gm1`-`gm6`** - Mean of each component (sorted by weight; 0 if component absent)
- **`gs1`-`gs6`** - Standard deviation of each component
- **`ga1`-`ga6`** - Amplitude/weight of each component
- **`gmb`** - Bayesian Information Criterion of the selected fit

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

#### Hue-Distance Metrics

These are pseudo-channels keyed on distance from a target hue. They produce `_std` and `_int` columns directly (no `_avg`, no GMM, no symmetry etc.).

**Per-hue channels** — `H000` (red, 0°), `H060` (yellow), `H120` (green), `H180` (cyan), `H240` (blue), `H300` (magenta):
- **`_std`** - Negative RMS angular distance from the target hue across the frame. Always ≤ 0; closer to 0 means the frame's hue is closer to the target.
- **`_int`** - Post-pass derived value: `(180 - H{nnn}_std) × diff_monos`, where `diff_monos = 1 - Hmon_std / max(Hmon_std)` is normalized per video against the most-monochromatic frame. Computed after all frames are processed, so it depends on the whole video's `Hmon_std` distribution.

**Monochromaticity channel** — `Hmon`:
- **`_std`** - Negative circular standard deviation of hue weighted by saturation (larger = more monochromatic). No `_int` variant is emitted.

### Color Channels Analyzed

Each row of the basic CSV is `{channel}_{metric}` for every (channel, metric) pair listed below.

| Channel | `_avg` `_std` | Symmetry / error / dark-light / motion | Full GMM | `gmn` + `gs1` only | `_int` |
|---|---|---|---|---|---|
| R, G, B | ✓ | — | — | ✓ | — |
| Gray | ✓ | ✓ | ✓ (gm1-6, gs1-6, ga1-6, gmb) | — | — |
| S, V | ✓ | — | — | ✓ | — |
| H000–H300 | only `_std` | — | — | — | ✓ |
| Hmon | only `_std` | — | — | — | — |

**Performance Note**: Computationally expensive metrics (symmetry, error dispersion, dark/light, motion, full GMM) are computed ONLY for Gray channel to improve processing speed. Other channels skip them entirely (their columns are not emitted, rather than emitted as zeros).

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

## DAWproject Output (write_dawproject.py)

### Overview

Renders selected metric curves as **volume automation on group (buss) tracks**
in a single `.dawproject` file, as an alternative to the MIDI CC output.

Unlike `write_midi.py`, this stage is **tempo-free**. Automation times are
wall-clock seconds computed as `frame_count_list / frames_per_second`, so the
curves stay locked to the video and changing the project tempo in the DAW does
not move them.

### What Is Generated

One `.dawproject` at `data/output/{video_name}_{preset}/{name_prefix}.dawproject`,
a ZIP holding `project.xml` and `metadata.xml`. No audio is embedded, so the
file is small.

Inside:
- One **empty group buss per listed column**, named after the column, each
  routed to a master track. The busses carry automation only, with no clips;
  route your own sources into them in the DAW.
- A `Points` envelope per buss with `timeUnit="seconds"`, targeting that
  buss's `Volume` parameter, with one `RealPoint` per CSV row.
- A `Transport` tempo, which is cosmetic — nothing positional depends on it.

Group busses use `Channel role="submix"`. The `mixerRole` enumeration is
`regular | master | effect | submix | vca` — **there is no `group`**, and
Cubase silently imports an invalid role as an ordinary audio track rather than
reporting an error. Every generated file is therefore validated against the
vendored schema in `schema/dawproject/Project.xsd`, and a schema error aborts
the write.

### Cubase Limitation: Automation on a Group

**Cubase will not import automation onto a group (submix) channel.**
Confirmed on both 14.0.41 and 15.0.30. Decisively, Cubase 15 cannot reimport
its *own* exported group automation, so this is a Cubase limitation rather
than a defect in the generated XML.
Each metric therefore produces a *pair*: an audio track carrying the
automation, and a like-named group buss. Copy each lane from the audio track
to its buss by hand in Cubase.

This is a workaround for a DAW limitation, not the structure the format calls
for. It was established by testing eight structural variants:

| Variant | `contentType` | `role` | Structure | Result in Cubase |
|---|---|---|---|---|
| A | *(none)* | submix | Track-wrapped | no tracks at all |
| B | `automation` | submix | Track-wrapped | group busses, no automation |
| C | `audio` | submix | Track-wrapped | **audio tracks + group busses, automation on the audio tracks** |
| D | *(none)* | regular | Track-wrapped | no tracks at all |
| E, I, J, K | various | submix | bare `Channel` in `Structure` | busses appear, automation does not |

A second round tested whether *any* other automatable parameter would work,
since `Volume` might have been a special case:

| Variant | Automated parameter | Result |
|---|---|---|
| L | `Volume` on a `role="vca"` channel | no automation |
| M | `Pan` on a submix | no automation |
| N | `Send/Volume` on a submix | no automation |
| O | `Equalizer/OutputGain` on a submix | no automation |

Variant M is the informative one: `Pan` is not a volume parameter, and it
failed too. So Cubase refuses **all** automation on a submix or VCA channel,
not merely `Volume`. Channel-level automatable parameters are `Volume`, `Pan`,
`Mute`, `Send/Volume`, `Send/Pan`, `Send/Enable` and device parameters — that
is the whole list, and none of them attach. There is no remaining structural
workaround; the paired audio track is the only route until Cubase changes.

Track order is likewise outside our control: Cubase collects submix channels
into its Group folder, so emitting an audio track immediately followed by its
buss does not interleave them in the track list. Tested and ignored.

The round-trip test is the one to repeat after a Cubase update: put volume
automation on a group, export a `.dawproject`, reimport it. Until the
automation survives that, nothing this writer emits can work either.

Variants I/J/K matched the shape of Cubase's *own* DAWproject export, in which
a group buss is a bare `<Channel role="submix">` directly inside `<Structure>`,
never wrapped in a `<Track>`. Automation still did not attach. Cubase's
exporter is consistent with this: exporting a project whose group carries
volume automation yields a file with zero `Points` elements.

Two things follow. `contentType` is **required** — Cubase creates no track
without it. And schema validation cannot catch any of this, since every
variant above is schema-valid; only importing reveals the behaviour.

Revisit if a later Cubase version imports group automation; the writer would
then emit bare submix channels and drop the paired audio tracks.

### Track Time Base (Musical vs Linear)

Automation times are written in seconds, but a track's **time base is not part
of the DAWproject schema** — it is a DAW-side property applied when the track
is created. Cubase takes it from **Preferences > Editing > Default Track Time
Type**, so tracks will arrive in musical time base unless that is set to
**Linear** before importing.

This matters: on a musical-time-base track, the automation moves when the
project tempo changes, which defeats the purpose of exporting in seconds. Set
the preference to Linear before importing, or select the imported tracks
afterwards and switch their time base with the note/clock toggle in the
Inspector.

### Value Mapping

Metric values are 0-1 and are written as **linear gain directly**. The `Volume`
parameter is `unit="linear"` with `min=0` and `max=1.0`, so:

| Metric | Gain | dB |
|--------|------|-----|
| 0.0 | 0.0 | -inf |
| 0.5 | 0.5 | -6.0 |
| 1.0 | 1.0 | 0.0 (unity, fader top) |

### Configuration

The number of group busses is set **only** by `dawproject.columns`, so it is
independent of how many columns the values CSV holds:

```json
"dawproject": {
  "columns": [
    "Gray_avg_v_f065_s1-0.5_o",
    "Gray_std_v_f065_s1-0.5_o",
    "Gray_avg_v_f065_s1-0.5_i"
  ],
  "interpolation": "linear"
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `columns` | list | `[]` | Values-CSV column names to export, one group buss each. Required when `write_dawproject` is true |
| `interpolation` | string | `"linear"` | Interpolation written on each `RealPoint` |

#### Transport: Set Tempo and Time Signature to Match the Target Project

`Transport/Tempo` and `Transport/TimeSignature` are both optional in the
schema and both effectively mandatory in Cubase:

- **No `Tempo`:** the project fails to load at all, silently.
- **No `TimeSignature`:** the project loads, but Cubase imposes **4/4** on the
  target project rather than leaving its signature alone.

Both are therefore always written, and both are configurable, because
**importing into an existing project overwrites that project's initial tempo
marking and time signature** with these values. Set them to match the target
and the import leaves the tempo track as it was:

```json
"dawproject": {
  "tempo": 75,
  "time_signature": [1, 4]
}
```

The tempo is cosmetic for positioning — all times are in seconds — so a wrong
value on a variable-tempo video is harmless in that respect, but it will still
overwrite the initial marking.

This is the third case where **schema-valid is not the same as loadable**,
after `contentType` and `Tempo`. Schema validation catches malformed XML and
bad enumeration values, not Cubase's undocumented expectations. An optional
element in the schema is not safe to omit. After any structural change,
confirm with a real import.

#### No `timing` Section Needed

This stage reads **no** beats or ticks. `ticks_per_beat` is never read, and
`beats_per_minute` only supplies the `Transport` tempo, which is cosmetic but
must be present (see above); omitting it just means the 120 bpm default.

The frame rate comes from `{name_prefix}_config.json`, the config
`process_video` wrote beside the values CSV. That file is authoritative: it
records the rate actually used to produce those rows, so it cannot drift from
them. A missing file or missing key is an error, not a default.

A config that only runs this stage can therefore drop `timing` entirely. A
config that also runs `process_video` still needs `timing.frames_per_second`,
because that stage reads it and silently defaults to 30 otherwise — see
`json/N48_dawproject.json`.

The writer reads only `video.video_name`, the optical-flow preset, and
`dawproject.columns`. A config that runs nothing else needs nothing else.

A column name that is not in the CSV is an error naming the missing column; it
is never silently skipped.

### Standalone Usage

```bash
python write_dawproject.py json/N48_dawproject.json
```

`json/N48_dawproject.json` is a worked example. To re-render only the
`.dawproject` from an existing `_values.csv`, set `process_video` and
`process_metrics` to false; that takes a few seconds instead of reprocessing
the video.

**DAWproject is the default output.** `write_dawproject` defaults to `true`
and `write_midi` to `false`; MIDI is legacy and must be opted into.

Import in Cubase with **File > Import > DAWproject**.

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

## Audio Leveling (audio_level_curve.py)

### Overview

Produces a slowly-varying gain curve that evens out the loudness of an audio
file, then delivers it to Cubase as a DAWproject volume automation envelope.
Two stages, run separately:

1. `audio_level_curve.py` - audio in, gain curve + preview out
2. `curve_to_dawproject.py` - gain curve + audio in, `.dawproject` out

### Loudness Measurement

Loudness is EBU R128 / ITU-R BS.1770 K-weighted LUFS, measured over
overlapping blocks (`--block-length`, default 0.4 s) at a fixed spacing
(`--block-hop`, default 0.1 s). The K-weighting filter coefficients are derived
for the file's own sample rate, so nothing is resampled. Calibration check: a
-20 dBFS 997 Hz sine measures -20.00 LUFS.

### Smoothing: Mean Weighting Distance

Every smoothing method is specified by one method-independent quantity, the
**mean weighting distance**:

```
d = integral |t| * w(t) dt
```

for the normalized symmetric weight function `w`. For a boxcar this is a
quarter of the full width, so **a one-minute boxcar is `--mean-distance 15`**.
Because all methods are normalized the same way, the same `--mean-distance`
smooths by the same amount whichever method you pick — at `d` = 15 s all five
have a 10-90% step rise time of about 48 s.

| `--smoothing-method` | scale from `d` | notes |
|---|---|---|
| `boxcar` | half-width `h = 2d` | flat window; discretized by area overlap |
| `triangular` | half-width `h = 3d` | Bartlett, as in `process_metrics.py` |
| `gaussian` (default) | `sigma = d*sqrt(pi/2)` | no ringing, no hard edges |
| `tricube` | half-width `h = (22/7)d` | the Tukey tri-cube weight |
| `loess` | tricube span, `h = (22/7)d` | local **linear** fit; follows ramps into the edges |

The scale is solved numerically so the *discrete* kernel on the block grid has
exactly the requested mean weighting distance; the factors above are the
large-window limit.

### Choosing the Mean Weighting Distance

This is the main knob, and it changes the character of the result more than
anything else. Shorter distances track the material more closely, so the
smoothed curve swings further and the ride works harder.

| `--mean-distance` | Behaves like | Use when |
|---|---|---|
| 20-60 s | Riding a fader between sections | Balancing movements or scenes against each other |
| 8-15 s | A careful engineer riding phrases | General-purpose leveling; a good starting point |
| 3-8 s | Aggressive riding | Individual phrases stand out too much |
| < 3 s | Slow compression | Rarely what you want from an automation envelope |

Below roughly 3 s the tool stops behaving like volume automation and starts
behaving like a compressor with a very slow attack. If you find yourself
reaching for those values, a compressor is probably the better instrument.

**Worked comparison.** One source (679.8 s, integrated -26.05 LUFS), one
target (`--target-lufs -28`), four mean weighting distances. Everything except
the smoothing is held constant:

| `--mean-distance` | Gaussian sigma | Smoothed loudness range | Ride depth | Points | Preview peak |
|---|---|---|---|---|---|
| 2.5 s | 3.13 s | -34.6 to -19.5 LUFS | -8.45 dB | 241 | -9.46 dBFS |
| 5 s | 6.27 s | -33.5 to -20.7 LUFS | -7.31 dB | 116 | -8.53 dBFS |
| 10 s | 12.53 s | -32.1 to -23.0 LUFS | -5.01 dB | 53 | -7.43 dBFS |
| 15 s | 18.80 s | -31.1 to -23.9 LUFS | -4.06 dB | 37 | -6.69 dBFS |

Roughly, halving the mean weighting distance doubles the automation point
count and adds about 1.5 dB of ride depth. At 15 s the curve makes a handful
of slow 1-2 dB dips; at 2.5 s it makes about 25 distinct dips, several
reaching 6-8 dB. Both are legitimate, but they are different effects.

Because the tag encodes the setting, runs at different distances coexist in
`data/output/` and can be compared by ear:

```bash
python audio_level_curve.py song.wav --target-lufs -28 --mean-distance 15
python audio_level_curve.py song.wav --target-lufs -28 --mean-distance 5
# -> song_c28_gau15_leveled.wav  and  song_c28_gau5_leveled.wav
```

⚠️ **Match levels when auditioning.** Note the preview peaks in the table
above: a deeper ride pulls the loudest moments further down, so the previews
differ by nearly 3 dB end to end. Louder reliably sounds better in an A/B, so
comparing them as-is will favor the longest mean distance for the wrong
reason. Level-match in your player or DAW before judging.

### Choosing the Target

There are two ways to set the level the ride aims at. They are mutually
exclusive.

#### `--target-fraction` (default)

Sets the target **from the audio itself**, as the fraction of the time the
volume should be reduced. `--target-fraction 0.25` places the target where a
quarter of the material sits above it, so the volume comes down a quarter of
the time. The target is the `1 - fraction` quantile of the smoothed loudness,
computed over audible (ungated) blocks only — there is no audio in a gated
stretch to reduce.

This is the default (0.25) because it adapts to the material. The same setting
means the same thing on every file, whatever its absolute level.

```bash
python audio_level_curve.py song.wav --target-fraction 0.25
```

```
STEP 3: gain curve
  target derived from the audio: -26.58 LUFS (25% of the time above it)
  ride range -7.03 to +0.00 dB
  volume reduced 25.0% of the audible time
```

The achieved fraction is reported and matches the request to within 0.1%.

Not valid with `--mode boost-only`, which never reduces the volume.

#### `--target-lufs`

Sets the target as an absolute loudness. Use it when you want the same anchor
across several files, for instance when they must sit together in a mix.

The catch is that an absolute target may miss the material entirely. In
`cut-only` mode it is a ceiling, so if it sits above the file's whole smoothed
loudness range the curve does nothing and `STEP 3` reports a ride range of
`+0.00 to +0.00 dB`. Check the range first:

```bash
python audio_level_curve.py song.wav --no-preview
```

```
STEP 2: smoothing
  smoothed loudness range -31.1 to -23.9 LUFS
```

#### Why the fraction usually wins

Two recordings from the same session, same `--mean-distance 2.5`:

| File | Setting | Target | Ride depth | Time reduced |
|---|---|---|---|---|
| `-04` | `--target-lufs -28` | -28.00 LUFS | -8.45 dB | **36.6%** |
| `-06` | `--target-lufs -28` | -28.00 LUFS | -4.23 dB | **13.1%** |
| `-04` | `--target-fraction 0.25` | -26.58 LUFS | -7.03 dB | **25.0%** |
| `-06` | `--target-fraction 0.25` | -30.26 LUFS | -6.49 dB | **25.0%** |

The same absolute target treats the two files very differently, because they
sit at different absolute levels. The same fraction treats them consistently,
deriving a different target for each to do so.

### Headroom Report

Every run reports how much the result can be turned up before clipping:

```
STEP 5: headroom
  true peak with the curve applied: -8.33 dBFS
  headroom before 0 dBFS: 8.33 dB
  -> the whole file can be raised by up to 8.33 dB
```

By default this is an **inter-sample (true) peak**, measured by oversampling
4x per ITU-R BS.1770-4 Annex 2, so it accounts for the reconstructed waveform
overshooting the highest sample. `--peak-mode sample` reports only the highest
sample, which is faster and less conservative.

This is measurement only — the tool reports the number and does not act on it.
Applying the makeup gain is a separate step, consistent with this tool doing
volume riding and nothing else. The value is in the config JSON as
`headroom_db`.

Deeper rides leave more headroom, since they pull the loudest moments further
down. On one test file: 6.69 dB of headroom at `--mean-distance 15`, rising to
9.45 dB at 2.5 s.

### The Silence Gate

Blocks quieter than `--gate-lufs` (default -50) are excluded from the smoothing
and the remaining kernel weights are renormalized. A silent stretch therefore
inherits an interpolated gain from the music on either side, instead of
dragging the curve down and producing a spurious boost.

### Modes

`--mode` controls the direction of the ride. The algorithm is identical in all
three; only the clamp on the gain differs.

| Mode | Behavior |
|---|---|
| `cut-only` (default) | Gain clamped to <= 0 dB. Passages at or below `--target-lufs` are left **exactly** alone; only louder passages are pulled down. The target acts as a **ceiling**. |
| `both` | Symmetric leveling toward the target. |
| `boost-only` | Gain clamped to >= 0 dB. |

The clamp is applied after smoothing, so in `cut-only` the curve sits flat at
0 dB and dips through loud passages, with corners where it crosses the
threshold — the same shape a person riding a fader produces.

### What This Tool Does Not Do

It only rides the volume. It does **not** limit, normalize, or set the peak
level. Those are a separate concern and belong in a separate step run
afterwards. Keeping them apart avoids conflating two different operations on
the same curve.

### No Loss of Information

- Nothing is resampled; the output sample rate always equals the input's.
- The preview is always written as **32-bit float**, so applying gain to 24-bit
  source audio does not requantize it. Where the gain is unity the preview is
  bit-identical to the source.
- `curve_to_dawproject.py` copies the source audio into the archive **byte for
  byte** — never decoded, re-encoded, resampled, or requantized. Cubase
  receives the original file plus a fader move.

### Output Files

Written to `data/output/`, named `{input_stem}_{tag}_*`:

| File | Contents |
|---|---|
| `..._gain_curve.csv` | full resolution: `time_s,lufs_raw,lufs_smooth,gain_db,gain_linear,valid` |
| `..._gain_points.csv` | decimated automation points — **input to stage 2** |
| `..._leveled.wav` | preview render, 32-bit float at the source rate |
| `..._plot.png` | loudness before, gain curve, loudness after |
| `..._config.json` | every parameter used for the run |
| `....dawproject` | stage 2 output |

The tag encodes the settings that shape the curve, so different settings land
in different files and a repeat of the same settings overwrites cleanly. For
example `song_cf25_gau15_leveled.wav` is a cut-only ride reducing 25% of the
time, with gaussian smoothing at a 15 s mean weighting distance;
`song_c28_gau15_leveled.wav` is the same but anchored to an absolute -28 LUFS
ceiling. Use `--tag` to name
a run yourself.

### DAWproject Notes

[DAWproject](https://github.com/bitwig/dawproject) is a ZIP holding
`project.xml`, `metadata.xml`, and media. Cubase has supported it since
14.0.20; import with **File > Import > DAWproject**.

- **Everything is in seconds.** The schema makes `Transport`/`Tempo` optional
  and gives every timeline its own `timeUnit`, so no position in the generated
  file depends on tempo. A `Tempo` element is written anyway because Cubase
  always has a project tempo, but nothing positional uses it.
- **Automation values are linear gain**, matching the `unit="linear"` declared
  on the track's `Volume` parameter. `--volume-max-db` sets that parameter's
  ceiling and defaults to +12 dB, Cubase's own fader maximum. The writer fails
  loudly if the curve exceeds it.
- `--external-audio` references the audio by absolute path instead of
  embedding it, producing a tiny file — but the path must resolve on the
  importing machine.
- `--validate Project.xsd` validates the generated XML against the
  [official schema](https://raw.githubusercontent.com/bitwig/dawproject/main/Project.xsd)
  before writing. Requires `xmlschema`.

### Worked Example

A full pass over a 679.8 s, 48 kHz, 24-bit stereo recording.

**1. Run with the defaults.** The target is derived from the audio, so this
works without knowing anything about the file:

```bash
python audio_level_curve.py "data/input/N47 2026-09-17-04.wav" --no-preview
```

```
  Format:           48000 Hz, 2 ch, PCM_24, 679.8 s
  Target:           reduce the volume 25% of the time (derived from the audio)
STEP 1: measuring K-weighted loudness
  6798 blocks, loudness range -300.7 to -14.8 LUFS
  2 blocks (0.0%) below the gate
STEP 2: smoothing
  smoothed loudness range -31.1 to -23.9 LUFS
STEP 3: gain curve
  target derived from the audio: -28.61 LUFS (25% of the time above it)
  ride range -4.68 to +0.00 dB
  volume reduced 25.0% of the audible time
```

The two gated blocks are digital silence at the head of the recording.

**2. Adjust the fraction to taste and render a preview:**

```bash
python audio_level_curve.py "data/input/N47 2026-09-17-04.wav" --mean-distance 2.5
```

```
STEP 4: decimating to automation points
  6798 blocks -> 160 points (tolerance 0.05 dB)
STEP 5: headroom
  -> the whole file can be raised by up to 8.33 dB
STEP 6: writing outputs
  data/output/N47 2026-09-17-04_cf25_gau2.5_gain_curve.csv
  data/output/N47 2026-09-17-04_cf25_gau2.5_gain_points.csv
  data/output/N47 2026-09-17-04_cf25_gau2.5_plot.png
  data/output/N47 2026-09-17-04_cf25_gau2.5_leveled.wav  (peak -8.33 dBFS)
  data/output/N47 2026-09-17-04_cf25_gau2.5_config.json
```

**3. Check the plot and listen to the preview.** In `cut-only` mode the bottom
panel should show the smoothed trace clipped flat at the target where the
source exceeded it, and untouched below it.

**4. Package for Cubase:**

```bash
python curve_to_dawproject.py \
    "data/output/N47 2026-09-17-04_c28_gau15_gain_points.csv" \
    "data/input/N47 2026-09-17-04.wav"
```

```
  Gain range:       -4.06 to +0.00 dB
  Volume ceiling:   +12.00 dB (linear 3.981072)
  Audio placement:  embedded
Wrote data/output/N47 2026-09-17-04_c28_gau15.dawproject (195.8 MB)
```

The archive holds `project.xml` (~5 KB), `metadata.xml`, and the source audio
stored uncompressed and byte-identical to the input.

---

### Standalone Usage

```bash
# Reduce the volume 40% of the time instead of the default 25%
python audio_level_curve.py song.wav --target-fraction 0.40

# Anchor to an absolute level instead of deriving one
python audio_level_curve.py song.wav --target-lufs -28

# One-minute-boxcar-equivalent smoothing
python audio_level_curve.py song.wav --mean-distance 15 --smoothing-method boxcar

# Two-sided leveling
python audio_level_curve.py song.wav --mode both --target-lufs -20

# Tighter automation curve (more points)
python audio_level_curve.py song.wav --point-tolerance-db 0.01
```

### Tests

```bash
python test_audio_smoothing.py
```

Verifies the two claims the workflow rests on: that each kernel's mean
weighting distance is what was requested, and that the loudness measurement is
calibrated to the BS.1770 reference (a -20 dBFS 997 Hz sine reads -20 LUFS).
Also covers the silence gate and the loess edge behavior.

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
- **SoundFile** - Audio file reading and writing (audio leveling workflow)

Optional:

- **xmlschema** - Only needed for `curve_to_dawproject.py --validate`

Python version: 3.10+

---

## Important Notes

### Performance Optimization
Computationally expensive metrics (symmetry, error dispersion, dark/light, motion, full GMM) are computed **only for Gray channel** to improve processing speed. For other channels, those columns are not emitted at all (rather than emitted as zeros) — non-Gray channels store only `_avg`, `_std`, `gmn`, and `gs1`. This optimization can be reversed by removing the `if color_channel_name == "Gray"` guards in `process_video.py`.

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

## Additional Utilities

### Supporting Tools

- **`kfs_to_csv.py`** - Converts KFS binary files to CSV format
- **`find_extrema.py`** - Finds local maxima/minima in time series data
- **`create_group_channel.py`** - MIDI channel grouping utility
- **`speed_to_cc.py`** - Converts speed data into a fixed-tempo MIDI with CC1 tracks
- **`audio_smoothing.py`** - Smoothing kernels parameterized by mean weighting distance (library, used by `audio_level_curve.py`)
- **`audio_level_curve.py`** - Builds a loudness-leveling gain curve from an audio file
- **`curve_to_dawproject.py`** - Packages a gain curve and its audio into a `.dawproject` for Cubase
- **`test_audio_smoothing.py`** - Checks for the smoothing kernels and loudness calibration

### `speed_to_cc.py` - Speed Data to MIDI CC Tracks

Standalone CLI utility that reads speed/zoom data and generates a constant-tempo MIDI file with six CC1 tracks derived from the input. Unlike `calculate_tempo_from_inverse.py`, the tempo is fixed (no tempo map), and the CC values are emitted one per video frame at 30 fps.

**Usage**:
```bash
python speed_to_cc.py <input_file> <tempo_bpm> [output.mid]
```

**Arguments**:
- `input_file` - Path to a `.py`, `.csv`, or `.kfs` file containing speed values
  - `.py` files must define `y_values = [...]` or `s = [...]`
  - `.csv` files: any layout; all numeric cells are read in order
  - `.kfs` files: binary KFS point format (y values extracted)
- `tempo_bpm` - Tempo in beats per minute (fixed for the whole file)
- `output.mid` - Optional output path (default: `speed_cc.mid`)

**Output Tracks** (all CC1, control change per frame):

| Track | Description |
|-------|-------------|
| `CC1 Speed` | Speed scaled to 0-127 |
| `CC1 Inverse Speed` | `1/speed` scaled to 0-127 |
| `CC1 Speed Percentile` | Percentile rank of speed (0-127) |
| `CC1 Speed Inverted` | `127 - CC1 Speed` |
| `CC1 Inverse Speed Inverted` | `127 - CC1 Inverse Speed` |
| `CC1 Speed Percentile Inverted` | `127 - CC1 Speed Percentile` |

**Example**:
```bash
python speed_to_cc.py data/input/N32_speed.py 108 data/output/N32_speed_cc.mid
```

---

## Documentation Files

- **`README.md`** (this file) - Main documentation and usage guide
- **`CLAUDE.md`** - Coding guidelines and architecture principles for AI assistants
- **`default_config.json`** - Configuration template with all parameters
- **`example_config.json`** - Working example configuration

---

## License

[Add your license information here]
