# Video Processing and Tempo Mapping Suite

This repository contains tools for video analysis, tempo mapping, and audio leveling, providing three complementary workflows:

1. **Visual Metrics Pipeline** (`run_video_processing.py`) - Extracts comprehensive visual metrics from videos and generates MIDI CC tracks
2. **Tempo Mapping** (`calculate_tempo_from_inverse.py`) - Generates variable tempo maps from zoom/speed data
3. **Audio Leveling** (`audio_level_curve.py` + `curve_to_dawproject.py`) - Computes a volume-riding curve from an audio file and delivers it to Cubase as a DAWproject automation envelope
4. **Audio Synthesis** (`metrics_to_wav.py`) - Renders video metrics directly to a WAV file as harmonic tones whose timbre follows the video: a held drone, or one tone per note of a monophonic MIDI file

---

## Table of Contents

- [Quick Start: Visual Metrics Pipeline](#quick-start-visual-metrics-pipeline)
- [Quick Start: Tempo Mapping](#quick-start-tempo-mapping)
- [Quick Start: Audio Leveling](#quick-start-audio-leveling)
- [Quick Start: Audio Synthesis](#quick-start-audio-synthesis)
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
- [Audio Synthesis: Metric-Driven Drone](#audio-synthesis-metric-driven-drone-metrics_to_wavpy)
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
   `process_every_nth_frame` (see [Video Processing](#video-processing-section-video_processing)):
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

Turn a speed file into a MIDI tempo map in which faster camera motion means a
faster tempo.

```bash
python calculate_tempo_from_inverse.py data/input/N52_speed.py \
    -t 96 --log-scale -s 20 \
    -o data/output/N52_tempo.mid --csv data/output/N52_tempo.csv
```

- The input is a `.py` file holding `s = [...]` or `y_values = [...]`: one
  **inverse-speed** value per video frame (large = slow).
- `-t` sets the average tempo; `--log-scale -s 20` changes it by 20 BPM each
  time the speed doubles or halves.
- Without `-o` / `--csv`, the outputs are written **next to the input file**
  as `<input>_tempo.mid` and `<input>_tempo.csv` (e.g. in `data/input/`), so
  pass both to keep them in `data/output/`.
- Import the `.mid` into Cubase; its tempo events go to the Tempo Track.

```bash
python calculate_tempo_from_inverse.py --help
```

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

## Quick Start: Audio Synthesis

Render a video's metrics as a held harmonic drone, written straight to WAV.

```bash
# Needs the run's *_basic.csv (from process_video); the JSON names it
python metrics_to_wav.py json/N52_drone_006/N52_drone_BGR_r055_flat_006.json
# -> data/output/N52_..._default/N52_..._default_N52_drone_BGR_r055_flat_006.wav (+ _config.json, _plot.png)
```

See [Audio Synthesis: Metric-Driven Drone](#audio-synthesis-metric-driven-drone-metrics_to_wavpy).

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

Builds a MIDI tempo map, plus a per-frame CSV, in which tempo follows the
camera's speed.

**Usage**:
```bash
python calculate_tempo_from_inverse.py INPUT_FILE [-t MEAN_TEMPO] [-o OUTPUT]
       [--csv CSV] [-s SCALING_PARAM] [--log-scale] [--fps 30]
       [--division 480] [--cc-subsample 30]
```

| Argument | Default | Meaning |
|---|---|---|
| `input_file` | required | `.py` file with `s = [...]` or `y_values = [...]`, one inverse-speed value per frame (e.g. `data/input/N52_speed.py`). Only `.py` is accepted |
| `-t`, `--mean-tempo` | 64.0 | Average tempo in BPM |
| `-o`, `--output` | `<input>_tempo.mid` | MIDI output path. The default sits next to the input, not in `data/output/` |
| `--csv` | `<input>_tempo.csv` | CSV output path, same caveat |
| `-s`, `--scaling-param` | 1.0 | Strength of the effect; meaning depends on the mode (below) |
| `--log-scale` | off | Use LOG mode instead of LINEAR |
| `--fps` | 30.0 | Video frame rate; one input value per frame |
| `--division` | 480 | MIDI ticks per beat |
| `--cc-subsample` | 30 | Frames between points on the CC tracks |

**Two modes**. The input holds inverse speed `y`, so speed is `1/y`. In both
modes tempo **rises with speed**.

1. **LINEAR** (default): tempo is a blend of a constant and a value
   proportional to speed:
   `tempo = mean + s * (k / y - mean)`, with `k` chosen so the total number
   of beats matches `mean` over the whole video. `s` is the blend fraction:
   0 = constant tempo, 1 (default) = tempo fully proportional to speed (twice
   the speed, twice the tempo). Where a tempo would fall outside 3.58-300 BPM,
   the speed is averaged over a longer window until it fits.
2. **LOG** (`--log-scale`): tempo moves the same number of BPM for each
   doubling of speed:
   `tempo = mean + s * (log2(1/y) - mean(log2(1/y)))`, so `s` is BPM per
   octave of speed (e.g. `-s 20` = +/-20 BPM per doubling or halving). The
   average equals `mean` exactly; tempos are clipped to 3.58-300 BPM.

**Outputs**:
- **MIDI** (`-o`):
  - Track 1 "Tempo Map": a tempo change on every frame, 4/4, and a C3 note
    (MIDI 60) on every beat, so you can hear the beat.
  - Eight CC1 tracks that follow the speed itself, not the tempo: Normal,
    Inverted, Deep/Medium/Shallow Zoom and their inverses (percentile
    bands). Speed in LINEAR mode, log2 speed in LOG mode.
- **CSV** (`--csv`): one row per frame, with columns `frame`, `time` (m:ss.ss)
  and `tempo_bpm`.

Because the CC tracks follow speed rather than tempo, `-s 0` gives a constant
tempo while the CC tracks still carry the speed variation.

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
    "frames_per_second": 30
  },
  "video_processing": {
    "downscale_large": 100,
    "downscale_medium": 10,
    "process_every_nth_frame": 15,
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
    "color_channels": [],
    "metrics": [],
    "rank_types": [],
    "inversions": [],
    "filter_seconds": [32, 64],
    "block_seconds": [],
    "stretch_values": [8],
    "stretch_centers": [0.33, 0.67]
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
    "write_midi": false,
    "write_dawproject": true
  },
  "dawproject": {
    "tempo": 120,
    "time_signature": [4, 4],
    "max_columns": 200,
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
| `frames_per_second` | int | `30` | Video frame rate. **Required when `process_video` runs** — it silently defaults to 30 otherwise, and the value is recorded in the stage-1 config that later stages read |
| `beats_per_minute` | int/float/string | — | Legacy, MIDI output only. A number = constant tempo; a string = path to a MIDI tempo file. Not read by the DAWproject path |
| `ticks_per_beat` | int | `480` | Legacy, MIDI resolution only |

> **Note:** the whole pipeline is tempo-free apart from `write_midi.py`. A config
> that only produces a DAWproject needs nothing here but `frames_per_second`,
> and may omit `timing` entirely if it also skips `process_video`. Set the
> DAWproject's own `Transport` values under `dawproject` instead.

**Example**:
```json
"timing": {
  "frames_per_second": 30
}
```

#### Video Processing Section (`video_processing`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `downscale_large` | int | `100` | Large downscale factor |
| `downscale_medium` | int | `10` | Medium downscale factor |
| `process_every_nth_frame` | int | **REQUIRED** (when `process_video` runs) | Whole-number spacing, in video frames, between analysed frames. Seconds per analysis row = `process_every_nth_frame / frames_per_second` (15 / 30 = 0.5 s). **This fixes the time resolution permanently** — everything else can be re-derived from `basic.csv` in seconds, but changing this means re-decoding the video, so choose it before a long run. If the source has a repeating cycle, pick a divisor of its frame length so every cycle is sampled at the same phase (N48: 486-frame cycle, 486 = 2 × 3⁵, so 6 gives 81 rows per cycle with no drift). |

> `frame_interval` was the previous name and was allowed to be fractional, because
> it was derived from tempo to land samples on beats. It is now rejected with a
> message pointing at `process_every_nth_frame`.

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
**Selection — what the values CSV contains.** These four form an outer product
and are the only way to control how many tracks the DAWproject ends up with,
since every CSV column becomes one. An empty list means "all".

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `color_channels` | array | `[]` (all) | Channels to process: `R G B Gray S V H000 H060 H120 H180 H240 H300 Hmon` |
| `metrics` | array | `[]` (all) | Metrics to process. Combinations that do not exist are skipped, so non-Gray channels quietly contribute only `avg std gmn gs1` |
| `rank_types` | array | `[]` (both) | `v` = raw value, `r` = percentile rank |
| `inversions` | array | `[]` (both) | `o` = original, `i` = `1 - value` |

An unknown channel or metric name is an error listing what is available.

**Smoothing — both widths are in seconds.**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filter_seconds` | array | `[32, 64]` | Triangular (weighted moving-average) widths **in seconds**. Converted to the nearest odd number of analysis rows, so the realised width quantises to a multiple of the row spacing. Fractional values allowed. Column suffix `_f{width}s` (`_f032s`, `_f004.75s`) |
| `block_seconds` | array | `[]` | **Block (boxcar) averaging** widths in seconds, giving a **stair-step (piecewise-constant)** curve rather than a smooth one. Column suffix `_b{width}s` |
| `stretch_values` | array | `[8]` | Logistic stretch factors |
| `stretch_centers` | array | `[0.33, 0.67]` | Centre points for stretching |
| `cc_number` | int | `1` | Legacy, MIDI output only |

| Option | Filter shape | Result | Column suffix |
|--------|--------------|--------|---------------|
| `filter_seconds` | triangular (weighted) moving average | smooth, continuous curve | `_f{width}s` |
| `block_seconds` | flat block mean | stair-step / piecewise-constant hold | `_b{width}s` |

> Because widths are in seconds, the realised width depends on the sampling
> rate: at 0.5 s/row a requested 4.75 s becomes 5.50 s (+16%), while at
> 0.2 s/row it becomes 5.00 s (+5%). Choose `process_every_nth_frame` with the
> smoothing you want in mind.

> The identity stretch `_s1-0.5` is a **fallback, not an addition**: it is
> emitted only when `stretch_values` and `stretch_centers` are both empty, so
> asking for one stretch yields one column rather than two.

**Example — zoom direction, rank-normalised, one 4.75 s filter**:
```json
"metrics_processing": {
  "color_channels": ["Gray", "R", "G", "B"],
  "metrics": ["czi", "czo", "avg", "std"],
  "rank_types": ["r"],
  "inversions": ["o", "i"],
  "filter_seconds": [4.75],
  "block_seconds": [],
  "stretch_values": [8],
  "stretch_centers": [0.67]
}
```
That is 4 channels × 4 metrics (two of them Gray-only) × 1 rank × 1 filter ×
1 stretch × 2 inversions = **20 columns → 40 Cubase tracks**.

#### Pipeline Control Section (`pipeline_control`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `process_video` | bool | `true` | Run video analysis (extract primary metrics) |
| `process_metrics` | bool | `true` | Run metrics processing (derive metrics, write values CSV) |
| `process_clusters` | bool | `true` | Run clustering analysis |
| `write_midi` | bool | `true` | Write MIDI files (the only tempo-aware stage; renders metrics and cluster CSVs to `.mid`) |
| `write_dawproject` | bool | `false` | Write a `.dawproject` of volume-automated audio tracks (tempo-free; see below) |

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
| `boxcar_periods` | array/null | `null` | **Iterative boxcar (majority-vote) smoothing** of cluster assignments — odd-integer widths, in **rows**, applied repeatedly until convergence to remove cluster flickering. Distinct from `metrics_processing.block_seconds`: these are row counts, not seconds, and they smooth integer cluster IDs rather than metric values. Each period becomes a `_b`-suffixed cluster track. |

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

**Finer time resolution** (halves the row spacing; doubles the analysis time):
```json
"video_processing": {
  "process_every_nth_frame": 3
}
```

**Experimenting with filters** (widths in seconds):
```json
"metrics_processing": {
  "filter_seconds": [1, 2, 4, 8, 16, 32]
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

#### Chromaticity Channels

`Rn`, `Gn`, `Bn` are added by `process_metrics.py` from the frame means, so they
need no re-analysis of the video. Each has one metric, `_avg`:

- **`Rn_avg`** = `R_avg / (R_avg + G_avg + B_avg)`, and likewise `Gn_avg`, `Bn_avg`.

Each is that primary's share of the frame's light: unchanged by brightening
or darkening, neutral at 1/3, and the three always sum to 1. With the `i`
inversion each gives a colour-opposite pair (red/cyan, green/magenta,
blue/yellow). An all-black frame is given the neutral 1/3.

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

#### Motion Derived Calculations (Gray Channel Only)

Both `czd` (zoom divergence) and `crc` (rotation curl) are **signed**, so a
single track cannot drive one sound in each direction. Each is half-wave
rectified into a pair of speeds plus a magnitude. Each directional signal is
**zero whenever the camera is not moving that way**, so separate sounds can
follow each direction.

| Derived | From | Meaning |
|---------|------|---------|
| **`czi`** | `max(czd, 0)` | zoom-**in** speed |
| **`czo`** | `max(-czd, 0)` | zoom-**out** speed |
| **`cza`** | `abs(czd)` | zoom speed, either direction |
| **`crl`** | `max(-crc, 0)` | counter**clockwise** speed |
| **`crr`** | `max(crc, 0)` | **clockwise** speed |
| **`cra`** | `abs(crc)` | rotation speed, either direction |

> **Sign conventions, verified rather than assumed.** Optical flow runs
> `prev → curr`, so magnifying the image moves features outward and the field
> diverges: **positive `czd` is zoom IN**. And because the image y-axis points
> down, **positive `crc` is CLOCKWISE** on screen. Both were checked with
> synthetic warps (1.05× / 0.95× → `czd` +0.071 / −0.078; ±3° → `crc` −0.074 /
> +0.074), because the original source comments stated the opposite for both.
> Re-verify the same way before trusting either sign.

> **Rank normalisation and rectified metrics.** A rectified metric is zero for
> every frame the camera is not moving that way, and those zeros all tie.
> `percentile_data` therefore ranks ties by **lowest**, not average: with
> averaged ranks a large block of zeros lands mid-range (0.37 on a clip that
> zooms in a quarter of the time) and the inactive direction would never reach
> silence. For metrics with no repeated values the two conventions are
> identical.

### Processing Pipeline

The script applies transformation stages in this exact order:

1. **Rank/Value Processing** - Creates base entries
   - `_v` - Original values from CSV
   - `_r` - Percentile transformation (0-1 range)

2. **Scaling** - Normalizes data to 0-1 range
   - Applied to all data (no key name change)

3. **Filtering** - Triangular smoothing filters ⚠️ **MUST BE LAST**
   - Widths given in **seconds** via `filter_seconds`, converted to the
     nearest odd number of analysis rows
   - `_f001s` - 1 second
   - `_f004.75s` - 4.75 seconds (fractional widths allowed)
   - `_f032s` - 32 seconds
   - A width shorter than one row collapses to a no-op

4. **Stretching** - Non-linear transformation
   - Sigmoid-like stretching function
   - Parameters: `stretch_value` and `stretch_center`
   - `_s1-0.5` (identity) is a **fallback**, emitted only when no stretch is
     configured — so one configured stretch yields one column, not two

5. **Inversion** - Value inversion
   - `_o` - Original values (no inversion)
   - `_i` - Inverted values (1 - original)

**⚠️ Critical**: Filtering MUST come last. Filtering raw data produces jagged curves. Filtering must be applied to processed (scaled) data to produce smooth curves.

### Key Naming Convention

Final key names reflect processing order:
```
Gray_czi_r_f004.75s_s8-0.67_o
│    │   │ │        │       │
│    │   │ │        │       └─ Inversion (o=original, i=inverted)
│    │   │ │        └───────── Stretching (s{value}-{center})
│    │   │ └────────────────── Filtering, in SECONDS (f{width}s)
│    │   └──────────────────── Rank/Value (v=value, r=rank)
│    └──────────────────────── Metric name
└───────────────────────────── Color channel
```

The filter width is the number of **seconds** requested, not a row count, with
the integer part zero-padded to three digits so names still sort: `_f032s`,
`_f004.75s`. Block averages use `_b{width}s`. `width_tag()` in
`process_metrics.py` is the single source of that format.

### MIDI File Generation

#### File Organization

MIDI files in `data/output/{video_name}_{preset}/`:
- Individual files: `{variable}_{metric}_{rank_type}_{filter}_{stretch}_{inversion}.mid`
- Example: `R_avg_v_f032s_s8-0.33_o.mid`

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
  - Filter widths in seconds (`_f001s`, `_f032s`, `_f004.75s`)
  - Stretch parameters (`_s1-0.5`, `_s8-0.33`, etc.)
  - Inversion (`_o`, `_i`)

---

## DAWproject Output (write_dawproject.py)

### Overview

Renders metric curves as **volume automation on audio tracks**
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
- One **empty audio track per values-CSV column**, named after the column,
  each routed to the master. The tracks carry automation only, with no clips.
- A `Points` envelope per track with `timeUnit="seconds"`, targeting that
  track's `Volume` parameter, with one `RealPoint` per CSV row.
- A `Transport` tempo, which is cosmetic — nothing positional depends on it.

The tracks use `Channel role="regular"`. The `mixerRole` enumeration is
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
Each metric is therefore an audio track carrying the automation, routed
straight to the master. Earlier versions also emitted a like-named
`"<metric> BUS"` group for each track to feed, with the lane copied onto it by
hand; that was dropped as not worth doubling the track count.

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
workaround; an audio track is the only route until Cubase changes.

When busses were emitted, track order was likewise outside our control: Cubase
collects submix channels into its Group folder, so an audio track immediately
followed by its buss did not interleave them in the track list.

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

Every column of the values CSV becomes one track; there is no column list.
To get fewer tracks, narrow `metrics_processing` (`color_channels`, `metrics`,
`rank_types`, `inversions`, `filter_seconds`, `stretch_*`).

```json
"dawproject": {
  "tempo": 108,
  "time_signature": [4, 4],
  "max_columns": 200,
  "interpolation": "linear"
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_columns` | int | `200` | Refuse to write a file with more columns (tracks) than this |
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

### Custom Filter Widths
```json
{
  "metrics_processing": {
    "filter_seconds": [1, 2, 4, 8, 16],
    "stretch_values": [1, 2, 4, 8],
    "stretch_centers": [0.1, 0.33, 0.5, 0.67, 0.9]
  }
}
```

> Every combination multiplies the column count, and each column is one
> Cubase track. `dawproject.max_columns` (default 200) refuses a project that
> would be unopenable rather than emitting it.

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

## Audio Synthesis: Metric-Driven Drone (metrics_to_wav.py)

### Overview

A new direction, built up iteratively: instead of handing automation to a DAW,
render the video's metrics directly as audio, as harmonic tones whose
harmonic balance follows the picture. The notes come from one of two sources:

- **Drone** (`"root_hz"`): one note held for the length of the video.
- **MIDI** (`"midi_file"`): one tone per note of a monophonic MIDI file, with
  the note's pitch, start and duration (see [MIDI Notes](#midi-notes)).

Each note is a sum of **exact integer harmonics** of its frequency (no
tempering of the harmonics): with a root of A1 = 55 Hz, harmonic *n* sits at
*n* x 55 Hz. Each harmonic's loudness is driven by one metric expression. Both
sources go through the same per-note renderer (`render_note()`), so every
setting below applies to both.

No extra packages are needed: synthesis is NumPy, ranking is SciPy, and the
WAV is written by SoundFile, all already in `requirements.txt`.

### Signal

```
y(t) = sum_n  w_n * m_n(t) * sin(2 pi n f0 (t - t_start) + phi_n + drift_n(t))  /  sum_n w_n
```

- **`w_n`, background strength**: `n ** strength_exponent` times the
  row's `gain`. Exponent -1 gives 1/n (sawtooth-like balance); -2 is darker; 0
  makes all harmonics equal. The per-row `gain` shapes the spectrum by hand:
  raising the odd rows (1, 3, 5, 7) moves the timbre toward a hollow,
  clarinet/square-like sound, raising the even rows emphasizes the octaves.
- **`m_n(t)`, metric weight**: the harmonic's expression, rank-normalized (see
  below), linearly interpolated between CSV rows.
- **Which harmonics** (`cycle`): with `"cycle": true` the list must be
  n = 1…L and is cycled (below); every harmonic up to `max_hz` sounds. With
  `"cycle": false` exactly the listed `n` sound, in any increasing set, each
  with its own expression (see [Octave Thinning](#octave-thinning)).
- **Cycling** (`"cycle": true`): the listed expressions repeat up to `max_hz`. With L listed
  entries, harmonic *n* uses entry `((n - 1) mod L) + 1`, so harmonic L + 1
  follows the same metric as harmonic 1, L + 2 the same as 2, and so on. A
  short list therefore still fills the spectrum: 37 entries at 55 Hz with
  `max_hz` 20000 give 363 harmonics. The background strength `w_n` uses the
  actual *n*, so each cycle is quieter than the last.
- **Phase**: each harmonic has a fixed starting phase, and only its amplitude
  changes over time, so the waveform is continuous. With `"phases": "zero"`
  every harmonic starts at 0; because the frequencies are exact multiples,
  they then crest together once per period of the root and the upper
  harmonics fuse into a pulse train at the root frequency, heard as a
  **buzz**. With `"phases": "random"` (uniform on [0, 2 pi), reproducible via
  `random_seed`) the spectrum is unchanged but the pulse is broken up. On
  N52 this halved the 55 Hz pulsing of the harmonics above 300 Hz (envelope
  modulation depth 0.79 -> 0.41). Some pulsing remains whatever the phases,
  because any two adjacent harmonics beat at the root frequency. Each MIDI
  note gets a fresh draw.
- **Jitter** (`drift_n`): see [Pitch Jitter](#pitch-jitter).
- **Fades**: a raised-cosine fade of `fade_s` seconds at the start and end of
  every note, so each note starts and ends at exactly zero (random phases do
  not start at zero). A note shorter than `2 * fade_s` fades over half its
  length instead; the run reports how many.
- **Harmonic count**: every harmonic of the note up to `max_hz`, so higher
  notes have fewer harmonics (55 Hz: 363; 220 Hz: 90).
- **Scale**: dividing by the note's `sum w_n` is a fixed, data-independent
  gain that guarantees |y| <= 1 for one note. The level is therefore
  comparable between videos, but usually below full scale (N52 drones peak
  between about -8 and -2 dBFS). Overlapping MIDI notes add, so they can
  exceed 1; the float WAV keeps the value and the run reports the peak.

### Pitch Jitter

With exact harmonics the waveform repeats exactly once per period of f0, and
neighbouring harmonics beat at exactly f0. Random phases change the shape of
the repeating waveform but not the repetition, so a steady f0 buzz remains.
Jitter makes each harmonic drift independently around its exact frequency so
that the pattern never quite repeats:

- For each harmonic of each note, a random offset in cents is drawn from a
  normal distribution with standard deviation `jitter_cents` every `jitter_s`
  seconds, and linearly interpolated between those points.
- Harmonic *n* runs at `n * f0 * 2^(cents/1200)`. The frequency is integrated
  into phase, so the drift never causes a discontinuity or click.
- The drift is in **cents**, so it is proportionally the same for every
  harmonic but larger in Hz for upper harmonics: 5 cents is about 0.16 Hz at
  55 Hz but 3.2 Hz at harmonic 20. The pitch stays put while the upper
  harmonics, where the buzz is, slide in and out of alignment.
- The drift is **independent per harmonic**. A drift shared by all harmonics
  would only wobble the pitch; the waveform would still repeat.
- Slow drift only widens each harmonic by a few Hz; it adds no hiss. On N52
  with 5 cents every 0.3 s, energy more than 15 Hz from any harmonic is
  2e-6 of the total.
- `jitter_cents` 0 gives exact harmonics, and output identical to a render
  without the jitter feature. Jitter roughly doubles rendering time.

What it does to the buzz, measured on `N52_drone_B_r055_flat` (envelope of the
harmonics above 300 Hz, 2-second windows): the pulse **at exactly 55 Hz**
drops from 0.41 to 0.18, but the total pulse energy between 50 and 60 Hz is
unchanged (0.42 vs 0.41). Jitter does not remove the beating between
neighbouring harmonics, it makes it **irregular**: the pulse rate wanders
instead of locking to f0. Whether that reads as less buzzy is a listening
judgement.

### Octave Thinning

The buzz lives above about 500 Hz: with every harmonic present, neighbours
there are f0 apart, several share one critical band, and they beat at f0.
With `"cycle": false` the harmonic set can be thinned so that above some
point only **octave doublings of lower harmonics** remain, the way organ
ranks and Hammond drawbars build a tone. The gaps then double every octave
(110, 220, 440 Hz, ...), too fast to be heard as pulsing.

All N52 configs use this pattern: harmonics 1-8, then every doubling of 4,
5, 6 and 7 (n whose odd part is at most 7):

```
1 2 3 4 5 6 7 8 | 10 12 14 16 | 20 24 28 32 | 40 48 56 64 | 80 ... 1280
```

Above harmonic 8 each octave has the same four-note shape (root, major
third, fifth, harmonic seventh). The 37 rows run to n = 1280 so that all 37
expressions are kept; rows above `max_hz` are skipped, so at 55 Hz the top
sounding harmonic is 320 (29 harmonics), at 110 Hz 160, at 220 Hz 80.
Rendering is about ten times faster than with every harmonic.

Measured on N52 (envelope of everything above 500 Hz, four 2-second windows):

| Render | Pulse at 55 Hz | All pulsing, 30-90 Hz |
|---|---|---|
| all 363 harmonics | 0.35 | 0.37 |
| all 363 harmonics, 4 cents jitter | 0.06 | 0.29 |
| octave thinning (first 29 expressions) | 0.03 | 0.04 |

Jitter makes the beating irregular, while thinning removes it.

### Speed-Driven Pitch

WAV output is not limited to fixed MIDI pitches, so the pitch of every note
can glide with the camera. With

```json
"pitch_mod": {"speed_file": "data/input/N52_speed.py", "min_fraction": 0.8333333333333334}
```

the speed file is read with `read_speed_data()` (one value per video frame,
at `frames_per_second`). `*_speed.py` files hold **inverse** speed, so speed
is `1 / value`. Speed is rank-normalized to 0-1, and the fundamental becomes

```
f(t) = f0 * min_fraction ** (1 - rank(t))
```

so the fastest moment plays at `f0` (`root_hz`, or the MIDI note) and the
slowest at `f0 * min_fraction`. 5/6 is a just minor third (-315.6 cents),
spread evenly in cents. On N52 the fastest value is shared by 21% of frames;
that plateau sits exactly at `f0`. After the last frame the value is held.

The frequency is turned into phase by summing it sample by sample (a running
integral), and every harmonic n uses n times that phase, so all harmonics
glide together and the tone stays exactly harmonic. The rank between frames
is linearly interpolated, so the frequency is continuous and the phase is
smooth: no clicks, even where the speed jumps. Jitter still applies on top.
`max_hz` is never crossed, since the factor is at most 1. `"pitch_mod": null`
keeps the pitch fixed and reproduces earlier output exactly.

The `004`-`006` sets have it on with `min_fraction` 1.0 (no glide). The
plot gains a second panel showing the pitch in cents below the root.

### MIDI Notes

With `"midi_file"` in place of `"root_hz"`, each note in the file becomes one
tone:

- **Pitch**: equal temperament, A4 (MIDI 69) = 440 Hz. Pitch bend is ignored
  (the run reports how many pitch-bend messages it skipped).
- **Timing**: MIDI time 0 is video time 0. Note-on and note-off times are read
  in seconds with the file's tempo map applied (all tracks are merged), so
  tempo changes are honoured. The metric weights follow the **video's**
  timeline: a note at 100 s takes the harmonic balance at 100 s.
- **After the video**: notes past the last CSV row hold the last row's
  values. The WAV runs to `max(video length, last note-off)`.
- **Velocity** is ignored for now: every note has the same level.
- **Monophonic** input is expected, but overlaps are allowed: a legato
  overlap simply sums the two notes. The run reports the number of overlaps
  and the longest.
- **Errors**: a note that restarts before its note-off, a note never turned
  off, a note shorter than one sample, or a note above `max_hz`.

Between notes the output is exactly zero.


### Rank Normalization

Each expression is evaluated on every row of the CSV, then replaced by its
**rank**: the lowest value maps to 0, the highest to 1, and everything else is
spaced evenly by order (tied values share their average rank). Rank
normalization ignores the metric's units and outliers, so every harmonic uses
its full 0-1 range over the video. A constant expression cannot be ranked and
is an error.

### Metric Expressions

Expressions combine **raw** metric columns from the run's `*_basic.csv` using
only **multiplication, division, and parentheses**:

```
B_avg
B_avg / Gray_avg
(B_avg / Gray_avg) * Gray_cmv
```

Any other character (`+`, `-`, numbers with decimal points, functions) is
rejected, and so are unknown column names. A result containing NaN or
infinity, e.g. from dividing by zero, stops the run. The ranking happens
**after** the expression is computed, so `B_avg / Gray_avg` ranks the ratio,
not the two ranks.

The processed `*_values.csv` is not used: its columns are already ranked,
smoothed and stretched, and it lacks many raw metrics (e.g. `Gray_cmv`).

### Timing

The CSV's `frame_count_list` gives each row's frame; rows must be evenly
spaced. Row *k* sits at `frame_k / frames_per_second` seconds. The file lasts
`(last_frame + step) / frames_per_second`, so each row covers its whole
interval and the last row's value is held to the end (394 rows every 30 frames
at 30 fps = 394.000 s, the length of the N52 video). Between rows, each
harmonic's amplitude is interpolated linearly at the audio sample rate, so
there are no steps and no zipper noise.

There is **no smoothing** in this iteration: the amplitudes follow the raw
1-second rows, which can swing from 0 to 1 within a few seconds.

### Configuration

A JSON file, kept in `json/`. Every key is required, and exactly one of
`root_hz` (drone) or `midi_file` (MIDI notes) must be present.

```json
{
  "input_csv": "data/output/N52_Sin2B-A1DpiB256k-f-C2-AN245_default/N52_Sin2B-A1DpiB256k-f-C2-AN245_default_basic.csv",
  "frames_per_second": 30,
  "sample_rate": 48000,
  "root_hz": 55.0,
  "max_hz": 20000.0,
  "cycle": false,
  "strength_exponent": -2.0,
  "phases": "random",
  "random_seed": 1,
  "fade_s": 0.05,
  "jitter_cents": 0.0,
  "jitter_s": 0.3,
  "harmonics": [
    {"n": 1, "gain": 1.0, "expr": "B_avg"},
    {"n": 2, "gain": 1.0, "expr": "B_std"},
    {"n": 3, "gain": 1.0, "expr": "B_avg / Gray_avg"},
    {"n": 4, "gain": 1.0, "expr": "(B_avg / Gray_avg) * Gray_std"},
    {"n": 5, "gain": 1.0, "expr": "(B_avg / Gray_avg) * Gray_ed0"},
    {"n": 6, "gain": 1.0, "expr": "(B_avg / Gray_avg) * Gray_ed1"},
    {"n": 7, "gain": 1.0, "expr": "(B_avg / Gray_avg) * Gray_ed2"},
    {"n": 8, "gain": 1.0, "expr": "(B_avg / Gray_avg) * Gray_ee0"},
    {"n": 10, "gain": 1.0, "expr": "(B_avg / Gray_avg) * Gray_ee1"}
  ]
}
```

| Key | Meaning |
|---|---|
| `input_csv` | The run's raw metrics, `*_basic.csv` |
| `frames_per_second` | Video frame rate; the CSV holds frame numbers only |
| `sample_rate` | Output sample rate in Hz |
| `root_hz` | Drone: fundamental frequency (55 Hz = A1). Not with `midi_file` |
| `midi_file` | MIDI: path to a monophonic MIDI file. Not with `root_hz` |
| `max_hz` | Highest harmonic frequency; nothing above it sounds. Must be below Nyquist |
| `cycle` | `true`: list is n = 1…L, cycled up to `max_hz`. `false`: exactly the listed n |
| `strength_exponent` | `w_n = n ** exponent`; -1 by convention |
| `phases` | `"zero"` (all harmonics start at 0; buzzy) or `"random"` |
| `random_seed` | Seed for the random phases and jitter; same seed, same file |
| `fade_s` | Raised-cosine fade length at the start and end of every note, in seconds (e.g. 0.01 for MIDI, 0.05 for a drone) |
| `jitter_cents` | Standard deviation of each harmonic's random pitch drift, in cents; 0 = exact harmonics |
| `jitter_s` | Spacing in seconds of the drift's random points (how slowly it moves) |
| `pitch_mod` | `null`, or `{"speed_file", "min_fraction"}`: pitch glides with camera speed, fastest = f0, slowest = f0 * min_fraction |
| `harmonics` | List of `{n, gain, expr}` in increasing `n`: harmonic number, its gain multiplier, and its metric expression (one cycle when `cycle` is true) |

The N52 configs are sets in subfolders of `json/`, each a grid over color
channel, root frequency and row gains (`{R|G|B}` or a channel ordering,
`r{028|055|110|...}` for the root in Hz, and `flat` all 1.0, `odd4`/`even4`
4.0 alternating with 1.0 down the rows):

- **`json/N52_drone_006/N52_drone_{XYZ}_r{028|055|110}_{flat|odd4|even4}_006.json`**:
  the 14 harmonic numbers of `004` with all three color channels mixed in
  one tone (plus Gray-only rows n = 18, 27); `odd4`/`even4` by row position
  as in `004`. The six channel tags are the orderings of R, G and B,
  swapped throughout every row; the tag names the channel of the
  `C_avg / Gray_avg` rows n = 2, 3, 5. `BGR` is the user's template
  (originally `N52_drone_r055_flat_006.json`). 54 files.
- **`json/N52_drone_005/N52_drone_*_r{028|055|110}_*_005.json`**: same 14
  harmonic numbers, settings and row-position `odd4`/`even4` as `004`, with
  the expressions reassigned (e.g. n = 3 is `C_avg * C_std`, n = 4 `C_avg`,
  n = 32 `C_avg * (C_std / Gray_std)`). Roots 27.5, 55 and 110 Hz; 27
  configs, built from the template `N52_drone_B_r055_flat_005.json`.
- **`json/N52_drone_004/N52_drone_*_r{028|055|066|110|220}_*_004.json`**:
  14 rows, n = 2, 3, 4, 5, 6, 8, 10, 12, 16, 18, 20, 24, 27, 32 (top partial
  32 x root), each with its own channel expression; no pitch glide
  (`min_fraction` 1.0), 4 cents of jitter. Here `odd4` and `even4` go by
  **row position**, not by n: `odd4` puts 4.0 on rows 1, 3, 5, ... (n = 2, 4,
  6, 10, 16, 20, 27), `even4` on rows 2, 4, 6, ... (n = 3, 5, 8, 12, 18, 24,
  32). Roots 27.5, 55, 66, 110 and 220 Hz; 45 files.
Older sets (`000`-`003`, `h6`, `octave`) were deleted on 2026-10-25; they
are in git history up to commit `0299300` (`git show 0299300:json/N52_drone_h6/...`).
Their listening notes stay in `SYNTHESIS_NOTES.md`.

Rendering time grows with the number of harmonics: every harmonic of 55 Hz
(363) over 394 s at 48 kHz takes about two minutes; the 29-harmonic octave
pattern about ten seconds.

Variants are separate config files: the output is named after the config
file, so `json/N52_drone_006/N52_drone_BGR_r055_odd4_006.json` writes
`..._N52_drone_BGR_r055_odd4_006.wav` beside the `flat` render without
overwriting it.

### Output Files

Written next to the input CSV, named `{run}_{config_stem}`:

| File | Contents |
|---|---|
| `{run}_{config_stem}.wav` | Mono, 32-bit float, at `sample_rate` |
| `{run}_{config_stem}_config.json` | Copy of the config used |
| `{run}_{config_stem}_plot.png` | Amplitude `w_n * m_n` over time for the first cycle (the listed harmonics) |

To hear it in Cubase, copy the WAV to the Cubase hand-off folder for the
machine you are on (see `CLAUDE.md`, Environment) and import it as audio.

### Roadmap

Possible next iterations: velocity, smoothing the metric weights (rank first, then
filter, the same order as the main pipeline), stereo placement, tempered or
moving pitch, several voices, and a nonlinear map from weight to amplitude
(e.g. in dB).

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
- **SoundFile** - Audio file reading and writing (audio leveling and audio synthesis)

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
- **`speed_to_dawproject.py`** - Converts an inverse-speed file into `.dawproject` volume automation (six continuous tracks, or bar-quantized bins)
- **`speed_distribution.py`** - Plots and tabulates the distribution of speed in an inverse-speed file
- **`frame_similarity.py`** - Measures how much each video frame differs from the next, as CSV
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

### `speed_to_dawproject.py` - Inverse Speed Data to DAWproject Automation

The DAWproject counterpart of `speed_to_cc.py`: it reads the same input files and writes the same
six derived series, but as volume automation in a `.dawproject` for Cubase rather than as MIDI CC.

**The `*_speed.py` files hold inverse speed** (time per unit of motion: large when the camera is
slow). This script takes speed as `1/value` before deriving anything, so its track names are
correct. `speed_to_cc.py` does not invert, so its `Speed` and `Inverse Speed` tracks are the other
way round. Any value `<= 0` in an inverse-speed file is an error.

**Usage**:
```bash
python speed_to_dawproject.py <input_file> [-o OUTPUT] [--fps 30] [--tempo 120]
       [--time-signature 4/4] [--every-nth-frame 1] [--clip-fraction 0]
       [--interpolation linear]
       [--segment-bars N | --segment-beats N --bin-fractions F0 F1 ... [--lead-ms 10]]
```

**Arguments**:
- `input_file` - `.py`, `.csv`, or `.kfs` file, one value per frame (same formats as `speed_to_cc.py`)
- `-o`, `--output` - Output path (default: `data/output/<input stem>.dawproject`)
- `--fps` - Frame rate of the series; automation time is `frame_index / fps` seconds (default 30)
- `--tempo` - Transport tempo. Cosmetic, since all times are in seconds, but Cubase will not load a
  project without one, and importing overwrites the target project's initial tempo marking, so set
  it to match that project (default 120)
- `--time-signature` - Transport time signature; also overwrites the target's on import (default `4/4`)
- `--every-nth-frame` - Keep every Nth point to thin the automation. Scaling and percentiles are
  still computed from every frame. This decimates; it does not smooth (default 1)
- `--clip-fraction` - Fraction of each tail to clip before scaling each track to 0-1.
  `0.05` maps the 5th and 95th percentiles to 0 and 1 and clips everything beyond them, so a brief
  extreme (N50's slow-down over its last 12 s) cannot squash the rest of the curve into a sliver
  of the range. `0` uses the true min and max. Must be in `[0, 0.5)`. It applies to the percentile
  tracks too: percentile 0.05 maps to 0 and 0.95 to 1, with the tails clipped (default 0)
- `--interpolation` - Interpolation written on each point (default `linear`)

**Output Tracks** (each scaled independently to 0-1, written as linear gain: 0 = -inf dB, 1 = 0 dB):

| Track | Description |
|-------|-------------|
| `Speed` | Speed (`1/file value`) |
| `Inverse Speed` | Inverse speed (the file's values) |
| `Speed Percentile` | Percentile rank of speed |
| `Speed Inverted` | `1 - Speed` |
| `Inverse Speed Inverted` | `1 - Inverse Speed` |
| `Speed Percentile Inverted` | `1 - Speed Percentile` |

If the filename contains `rot`, the tracks are labelled `Rotation` and the values are used as they
are, not inverted: rotation is signed, so it has no meaningful reciprocal. Positive is
counter-clockwise.

As in `write_dawproject.py`, each track is an audio track carrying the automation, routed to the
master (Cubase will not import automation onto a group). Import with `File > Import > DAWproject` from Cubase 15 (see `CLAUDE.md`).

**Example**:
```bash
python speed_to_dawproject.py data/input/N50_speed.py --tempo 108 --time-signature 4/4 --clip-fraction 0.05
```

#### Binned mode: bar-quantized switches

`--segment-bars` (or `--segment-beats`) with `--bin-fractions` writes a separate file in which
volume changes only on bar lines. The video is cut into segments of that many bars at the
constant `--tempo` and `--time-signature`, with video time 0 on bar 1, beat 1. A trailing partial
segment is kept. Each segment's value is the **median of the per-frame percentile rank** of speed,
so the result does not change under any monotonic transform of the metric. The segments are
ranked and split so that each bin holds its given fraction of them. Bin 0 is the slowest.
Rounding is applied to the boundaries between bins, not to each bin's count, so a palindromic
split stays symmetric and the middle bins absorb the rounding (79 segments at `0.4 0.2 0.4` give
32 / 15 / 32). A boundary exactly on a half segment gives the extra segment to the more central
neighbouring bin.

Each bin is an empty audio track at unity gain (0 dB) during its segments and silent otherwise.
Cubase ignores `hold` interpolation on import and ramps across the whole segment, so each change
is written as two linear points one tick (1/480 quarter note, 1.3 ms at 96 bpm) apart (old
value, then new), which acts as a step. An inverted set ranks the segments the other way, so its
`inv B0` holds the fastest segments. Each track (trailing ` o`) has a complement (trailing ` i`),
which is at 0 dB exactly when the track is silent and vice versa. Track names carry
the segment length in bars (`4bt` if given in beats), the bin and its fraction as a whole percent.
`Sp`/`Ro` abbreviate the quantity:

| Track | On (0 dB) during |
|-------|------------------|
| `Sp 4 B0 50% o` | the slowest 50% of segments |
| `Sp 4 B0 50% i` | every segment except those |
| `Sp inv 4 B0 50% o` | the fastest 50% of segments |
| `Sp inv 4 B0 50% i` | every segment except those |

With K bins there are up to 4K tracks. Sets that would repeat other tracks are not written:

- **Two bins:** each ` i` track would equal the other bin's ` o` track, so the ` i` tracks are dropped.
- **Palindromic fractions** (the same reversed, e.g. `0.4 0.2 0.4`): the counts split
  symmetrically, so `inv B`*k* mirrors `B`*(K-1-k)* and the `inv` set is dropped. Ties between equal
  segment medians, or an even number of bins with an odd number of segments, can leave the mirror
  off by a segment.

So `0.4 0.2 0.4` gives 6 tracks, `0.75 0.25` gives 4 and `0.5 0.5` gives 2. Among the ` o` tracks of
one set, exactly one is on at any time.

**Track order:** all ` o` tracks first, then all ` i` tracks. Within each, the plain set comes
before `inv`, and then the tracks run in bin order: `B0 o`, `B1 o`, `B2 o`, `inv B0 o`, …, `B0 i`, …
The `--midi-in` output uses the same order, with each bin track's source tracks kept together.

- `--segment-bars N` / `--segment-beats N` - Segment length. A beat is a 1/denominator note and
  the tempo counts quarter notes. The project tempo must match, or the steps miss the bar lines
- `--bin-fractions F0 F1 ...` - Fraction of segments per bin, slowest first; each > 0, summing to 1
- `--lead-ms` - How far before each bar line a change sits, so Cubase's de-click ramp has finished
  when a note starting on the bar sounds. The first segment starts at 0 (default 10)

`--clip-fraction`, `--every-nth-frame` and `--interpolation` do not apply in this mode. Outputs
go to `data/output/<input stem>_<tag>.dawproject`, plus `<...>_segments.csv` with each segment's
start time, change time, frame count, median percentile, bin and inverted bin. The tag encodes the
settings, e.g. `s4bar_b50-25-25_lead10ms`.

```bash
python speed_to_dawproject.py N52_speed.py --tempo 96 --time-signature 4/4 \
    --segment-bars 4 --bin-fractions 0.5 0.25 0.25
```

#### Splitting MIDI notes by bin (`--midi-in`)

Switching volume cuts off reverb tails and long decays. To avoid that, the same bins can sort the
notes themselves, so a note either plays in full or is absent. Export the MIDI from Cubase
(*File > Export > MIDI File*, 480 ticks per quarter, with Export Inserts on), then add the
flags below.

Notes on the Cubase export:

- **Choosing tracks:** solo the tracks you want before exporting, and only those are written.
  Muted and disabled tracks are always left out.
- **Play Probability is applied at export.** A note set below 100% is randomly kept or dropped each
  time, so two exports can differ. Set it to 100% first, unless one random roll is what you want.
- **Inserts:** Export Inserts writes the effect of MIDI insert plug-ins into the notes, so turn
  those inserts off on the tracks you import into, or they act twice.
- **Note names:** Cubase calls MIDI note 60 C3, not C4.

The flags:

- `--midi-in FILE` - the exported MIDI file. Each track that has notes becomes one track per bin
  track, named `<track> <bin track>` (e.g. `Piano A melody Sp 4 B0 25% o`). Each holds only the
  notes that start in that bin track's on segments. The ` i` and `inv` tracks follow the same
  redundancy rules as the volume tracks.
- `--grid` - the grid the music was played to (default `1/16`). **Notes are never moved or
  shortened.** The grid only decides which segment a note belongs to: its start is snapped to the
  nearest grid step, so a note played up to half a step (60 ticks for 1/16) early counts as being
  on the bar line.

Notes starting after the video ends and zero-length notes are dropped, and tracks with no notes
are skipped. The file
must have one tempo and one time signature, equal to `--tempo` and `--time-signature`. The output
is `data/output/<midi stem>_<input stem>_<tag>.mid`; import it with *File > Import > MIDI File*.

```bash
python speed_to_dawproject.py /mnt/d/N52/N52_speed.py --tempo 96 --time-signature 4/4 \
    --segment-bars 4 --bin-fractions 0.25 0.5 0.25 --midi-in /mnt/d/N52/test.mid --grid 1/16
```

---

### `speed_distribution.py` - Distribution of Speed

Reads the same inverse-speed files as `speed_to_dawproject.py`, takes speed as `1/value`, and
shows how it is distributed, which helps when choosing `--bin-fractions`.

```bash
python speed_distribution.py <input_file> [-o OUTPUT_STEM] [--fps 30] [--thresholds S1 S2 ...]
```

- **`<stem>_distribution.csv`**: one row per frame, with `frame`, `seconds`, `inverse_speed`,
  `speed` and `percentile`. The percentile is the rank the binned mode uses, so ties are visible.
- **`<stem>_distribution.png` and `.pdf`**: four panels, showing speed over time, a histogram of
  speed, a histogram of log₂(speed), and the cumulative distribution.
- **Printed**: quantiles, the range, the share of frames at the maximum speed (on N52, 21% sit at a
  cap and tie in rank), and the fraction of time below and above each `--thresholds` speed.

Outputs go to `data/output/<input stem>_distribution.*`. Rotation files are refused, because
rotation is signed.

---

### `frame_similarity.py` - Frame-to-Frame Similarity CSV

Standalone CLI utility that decodes an entire video in one pass and writes one CSV row per frame
transition, describing how much frame `i` differs from frame `i+1`. Useful for locating still or
slow passages, finding held/duplicated frames, and producing a per-frame change signal that is far
cheaper than the optical-flow metrics in `process_video.py`.

For each consecutive pair, the signed difference `frame[i+1] - frame[i]` is taken per RGB channel,
and the mean and standard deviation of that difference are recorded. The headline metric,
`diff_std_sum`, is the sum of the three per-channel standard deviations.

**Usage**:
```bash
python frame_similarity.py <video> [-o OUTPUT] [--max-frames N]
```

**Arguments**:
- `video` - Any of: a path to a video file; a directory holding exactly one `.wmv`/`.mp4`; or a bare
  name resolved against `data/input/` in either of those forms. So `N48`, `data/input/N48`, and the
  full path to the file inside it are all equivalent.
- `-o`, `--output` - Output CSV path (default: `data/output/{video stem}_framesim.csv`)
- `--max-frames N` - Stop after N frames, for quick checks (default: the whole video)

**Output Columns**:

| Column | Description |
|--------|-------------|
| `frame` | Index of the first frame of the pair |
| `time_sec` | `frame / fps` |
| `diff_std_sum` | `R_std + G_std + B_std`, the headline similarity metric |
| `R_std`, `G_std`, `B_std` | Per-channel standard deviation of `frame[i+1] - frame[i]` |
| `R_mean`, `G_mean`, `B_mean` | Per-channel mean of `frame[i+1] - frame[i]` |
| `identical` | 1 if the two frames are bit-identical, else 0 |

A video of *n* frames produces *n-1* rows; the last frame has no successor.

**Performance**: frames are compared at native resolution with no downscaling, since downscaling
attenuates the metric and can hide small differences. Decoding is the bottleneck (a reader thread
gives no measurable gain), so expect roughly 85 frames per second - about 5.5 minutes for a
26,000-frame 1080p video.

**Example**:
```bash
python frame_similarity.py N48
python frame_similarity.py data/input/N48 --max-frames 400 -o data/output/N48_smoke.csv
```

---

## Documentation Files

- **`README.md`** (this file) - Main documentation and usage guide
- **`CLAUDE.md`** - Coding guidelines and architecture principles for AI assistants
- **`SYNTHESIS_NOTES.md`** - Listening notes, hypotheses and ideas for the audio synthesis (`metrics_to_wav.py`)
- **`default_config.json`** - Configuration template with all parameters
- **`example_config.json`** - Working example configuration

---

## License

[Add your license information here]
