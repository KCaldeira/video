# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Architecture

This is a video processing pipeline that extracts visual metrics from video files and converts them to MIDI data for musical applications. The codebase consists of three main Python scripts that work in sequence:

1. **`run_video_processing.py`** - Main orchestrator script with command-line interface
2. **`process_video.py`** - Video analysis module (extracts visual metrics from frames)
3. **`process_metrics.py`** - Data processing module (transforms metrics into MIDI files)

## Development Commands

### Running the Pipeline
```bash
# Basic usage
python run_video_processing.py <video_name>

# With custom tempo
python run_video_processing.py <video_name> <beats_per_minute>

# Using configuration file
python run_video_processing.py --config config.json

# Limit processing to first N frames (useful for testing)
python run_video_processing.py <video_name> --max-frames 1000

# Skip video processing, only run metrics
python run_video_processing.py <video_name> --skip-video

# Skip metrics processing, only run video analysis
python run_video_processing.py <video_name> --skip-metrics
```

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# The project uses Python 3.10+ with a virtual environment
source .venv/bin/activate
```

## Critical Processing Order

**WARNING**: The filtering step must come LAST in the processing pipeline. This is a recurring issue that has been fixed multiple times.

**Correct Processing Order**:
1. Create base entries (`_v`, `_r`)
2. Scale data to 0-1 range
3. **Apply filtering** (to scaled data) - MUST BE LAST
4. Apply stretching (to filtered data)
5. Apply inversion (to stretched data)

Applying filtering to raw data produces jagged, inconsistent curves. Filtering must be applied to processed (scaled) data to produce smooth curves.

## Code Philosophy

### CRITICAL: Fail Fast, Understand Everything
- **NEVER use try/except blocks** to hide errors or provide fallbacks
- **Let the code fail** when something goes wrong - this reveals the real problem
- **Scientific code principle**: Prefer failure over silent incorrectness
- This is NOT commercial production code - we want to see every failure

### Root Cause Analysis Required
- **DO NOT** apply band-aid patches or quick fixes
- Always understand the root cause before making changes
- Fix the fundamental issue, not just symptoms
- Prefer failing and understanding problems over hiding them

### Code Structure Principles
- **Single Source of Truth**: The `filter_periods` list controls which filters are applied
- **Simplified Logic**: Prefer linear processing over complex conditional logic
- **Consistent Naming**: Use `_f{period:03d}` format for all filter periods (including `_f001` for unfiltered)

## Key Data Flow

The processing uses separate dictionaries for each transformation stage:
- **`raw_entries`** - Initial base entries (`_v`, `_r`)
- **`scaled_entries`** - After scaling to 0-1 range
- **`filtered_entries`** - After filtering with `_f{period:03d}` suffixes
- **`stretched_entries`** - After stretching transformations
- **`final_entries`** - After inversion with `_o` and `_i` suffixes

## Performance Optimizations

Computationally expensive metrics (symmetry, error dispersion, dark/light, motion) are computed **only for Gray channel** to improve processing speed. Other color channels receive zero values for these metrics. This optimization can be reversed by removing Gray channel conditions if full color analysis is needed.

## Key Functions and Files

### process_video.py (Video Analysis)
- `process_video_to_csv()` - Main function (700+ lines - needs refactoring)
- `compute_basic_metrics()` - Basic visual metrics from frames
- `compute_change_metrics()` - Motion analysis using Lucas-Kanade optical flow

### process_metrics.py (Data Processing)
- `process_metrics_to_midi()` - Main function for CSV to MIDI processing
- `post_process()` - Applies filtering, ranking, stretching, inversion
- `add_derived_columns()` - Adds ratio and rotation metrics

### run_video_processing.py (Orchestration)
- `main()` - Command-line argument handling and pipeline orchestration
- `run_process_video()` - Calls video analysis with parameters
- `run_process_metrics()` - Calls metrics processing with parameters

## Output Structure

```
project/
├── {video_name}_{preset}_basic.csv      # Raw metrics from video analysis
├── {video_name}_{preset}_config.json    # Processing configuration
└── ../video_midi/{video_name}_{preset}/ # MIDI output directory
    ├── {video_name}_derived.xlsx
    ├── {video_name}_plots.pdf
    └── *.mid                            # Individual MIDI files
```

## Configurable Parameters

All processing parameters are configurable via JSON or command line:
- `filter_periods`: [17, 65, 257] - Smoothing filter periods
- `stretch_values`: [8] - Non-linear transformation values
- `stretch_centers`: [0.33, 0.67] - Stretch center points
- `max_frames`: null - Limit frames processed (useful for testing)
- `farneback_preset`: "default" - Optical flow algorithm preset

## Testing and Validation

- Test with different `filter_periods` combinations
- Verify smooth curves are produced by filtering
- Check that MIDI files match expected filter periods
- Use `max_frames` parameter for quick testing of changes