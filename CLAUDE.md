# CLAUDE.md

This file provides coding guidance to Claude Code when working with code in this repository.

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
- **Avoid if-then complexity**: Keep control flow simple and obvious
- **Consistent Naming**: Use `_f{period:03d}` format for all filter periods (including `_f001` for unfiltered)

## Critical Processing Order

**WARNING**: The filtering step must come LAST in the processing pipeline. This is a recurring issue that has been fixed multiple times.

**Correct Processing Order**:
1. Create base entries (`_v`, `_r`)
2. Scale data to 0-1 range
3. **Apply filtering** (to scaled data) - MUST BE LAST
4. Apply stretching (to filtered data)
5. Apply inversion (to stretched data)

Applying filtering to raw data produces jagged, inconsistent curves. Filtering must be applied to processed (scaled) data to produce smooth curves.

## Key Data Flow

The processing uses separate dictionaries for each transformation stage:
- **`raw_entries`** - Initial base entries (`_v`, `_r`)
- **`scaled_entries`** - After scaling to 0-1 range
- **`filtered_entries`** - After filtering with `_f{period:03d}` suffixes
- **`stretched_entries`** - After stretching transformations
- **`final_entries`** - After inversion with `_o` and `_i` suffixes

## Quick Reference

### Running the Pipeline
```bash
python run_video_processing.py config.json
```

### Pipeline Stages
1. **Video Analysis** (process_video.py) - Extract primary metrics
2. **Metrics Processing** (process_metrics.py) - Derive metrics, generate MIDI
3. **Clustering** (cluster_primary.py) - Group similar frames

### Directory Structure
- Videos: `data/input/`
- Outputs: `data/output/{video_name}_{preset}/`

### Configuration
All parameters are in JSON config files. See `default_config.json` for template.

Key config parameters:
- `video.video_name` - Video filename (without extension) - **REQUIRED**
- `timing.beats_per_minute` - Tempo (default: 64)
- `video_processing.optical_flow.preset` - Motion detection preset (default: "default")
- `metrics_processing.filter_periods` - [17, 65, 257] - Smoothing filters
- `cluster_processing.k_values` - [2, 3, 4, 5, 6, 8, 10, 12] - Cluster counts to try
- `pipeline_control.process_video` / `process_metrics` / `process_clusters` - Enable/disable stages

### Performance Optimizations

Computationally expensive metrics (symmetry, error dispersion, dark/light, motion) are computed **only for Gray channel** to improve processing speed. Other color channels receive zero values for these metrics. This optimization can be reversed by removing Gray channel conditions if full color analysis is needed.

## Documentation

For complete documentation, see `README.md`.
