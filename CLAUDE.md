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

## Output Location

**By default, all output files must be written to `./data/output/` or a subdirectory thereof.** This applies to generated data, MIDI files, plots, one-off scripts' outputs, and any other artifacts. Do not write outputs to the repository root or other locations unless the user explicitly requests a different path.

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

## Environment

- **DAW: Cubase 15.0.30** (Windows). Assume this version for any DAWproject or
  MIDI question; do not ask which DAW or version.
- Windows drives visible from WSL: `C:` and `D:` only, as `/mnt/c` and
  `/mnt/d`. `E:` is not mounted. To hand a file to Cubase, write it under
  `/mnt/d/`.

**Two Cubase versions are installed (14 and 15).** The Windows file
association for `.dawproject` and `.cpr` opens **14**, while the desktop
shortcut launches **15**. Always start Cubase from the shortcut and use
`File > Import > DAWproject` from inside it; never double-click the file, or
the test runs against the wrong version. To confirm which version is running,
export any `.dawproject` from it and read the `<Application version="...">`
line — that string is accurate and names the binary that wrote the file.

### Known Cubase DAWproject limitations

- **Automation will not import onto a group (`role="submix"`) channel.**
  Confirmed on both Cubase 14.0.41 and 15.0.30 across twelve structural
  variants, including Cubase's own export shape. Cubase 15 also fails to
  reimport its *own* exported group automation, so this is a Cubase
  limitation, not a problem with the generated XML. Settled approach in
  `write_dawproject.py`: `contentType="audio"` + `role="submix"`, giving an
  audio track carrying the automation plus a like-named dummy buss to copy it
  onto. Do not redesign this without first re-running the round-trip test.
- **`contentType` is required** on a `Track`, or Cubase creates no track.
- **`Transport/Tempo` is required**, or Cubase silently loads nothing.
- **`Transport/TimeSignature` is required**, or Cubase imposes 4/4 on the
  target project instead of leaving its signature alone.
- Importing into an existing project **overwrites its initial tempo marking
  and time signature**, so set `dawproject.tempo` and
  `dawproject.time_signature` to match that project (N48: 75 bpm, 1/4).
- **Track order is not controllable.** Cubase collects every submix channel
  into its own Group folder, so audio tracks and their busses cannot be
  interleaved no matter what order the `Structure` element lists them in.
  Tested with explicit interleaved Track/Channel pairs; ignored.
- `contentType`, `Tempo` and `TimeSignature` are all optional in the schema
  and all required in practice: **schema-valid is not the same as loadable**,
  and an optional element is not safe to omit. Confirm every structural
  change with a real import.
- **Track time base is not in the DAWproject schema.** Cubase applies
  *Preferences > Editing > Default Track Time Type* at import; set it to
  **Linear** or seconds-based automation will still move with tempo.

## Quick Reference

### Running the Pipeline
```bash
python run_video_processing.py config.json
```

### Pipeline Stages
1. **Video Analysis** (process_video.py) - Extract primary metrics
2. **Metrics Processing** (process_metrics.py) - Derive metrics, generate MIDI
3. **Clustering** (cluster_primary.py) - Group similar frames

### Audio Leveling Workflow (separate from the video pipeline)
```bash
python audio_level_curve.py INPUT_AUDIO [--target-lufs -28] [--mean-distance 15]
python curve_to_dawproject.py POINTS_CSV AUDIO_FILE
```
1. **audio_level_curve.py** - K-weighted (BS.1770) loudness -> smoothed -> gain curve
2. **curve_to_dawproject.py** - gain curve + audio -> `.dawproject` for Cubase

Uses flag-based CLIs (the `speed_to_cc.py` style), not JSON configs. Key
invariants for this workflow:
- **Smoothing is specified by mean weighting distance**, not window width, so
  all kernels are comparable. A one-minute boxcar is `--mean-distance 15`.
  `audio_smoothing.py` is the single source of truth for the conversions.
- **Volume riding only.** Limiting, normalizing, and peak-setting are a
  separate concern and deliberately excluded; do not add them here.
- **No loss of information.** Never resample, never reduce bit depth. The
  preview render is always 32-bit float at the source rate; the DAWproject
  embeds the source audio byte for byte.
- Outputs go directly in `data/output/` named `{input_stem}_{tag}_*`, where
  the tag encodes the curve-shaping settings (e.g. `c28_gau15`).

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

Computationally expensive metrics (symmetry, error dispersion, dark/light, motion, full GMM) are computed **only for Gray channel** to improve processing speed. For other channels, those columns are not emitted at all (rather than emitted as zeros) — non-Gray channels store only `_avg`, `_std`, `gmn`, and `gs1`. This optimization can be reversed by removing the `if color_channel_name == "Gray"` guards in `process_video.py` if full color analysis is needed.

## Documentation

For complete documentation, see `README.md`.
