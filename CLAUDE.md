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
- **Single Source of Truth**: The `filter_seconds` list controls which filters are applied
- **Simplified Logic**: Prefer linear processing over complex conditional logic
- **Avoid if-then complexity**: Keep control flow simple and obvious
- **Consistent Naming**: Smoothing widths are given in **seconds** and named in
  seconds: `_f{width}s`, where the integer part is zero-padded to three digits
  (`_f032s`, `_f004.75s`). `width_tag()` in `process_metrics.py` is the single
  source of that format. Block averages use `_b{width}s`.
- **Short names only for Cubase tracks**: Cubase shows only a few characters
  of a track name, so DAWproject track names are abbreviated (`Sp 4 B0 50% o`,
  `Sp`/`Ro` for speed/rotation, `inv`, `B0`). File names have no such limit:
  keep them long and descriptive (`N46_rot_...`, `_speed_`), and do not carry
  track-name abbreviations into them.

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
- **`filtered_entries`** - After filtering with `_f{width}s` suffixes
- **`stretched_entries`** - After stretching transformations
- **`final_entries`** - After inversion with `_o` and `_i` suffixes

## Environment

- **DAW: Cubase 15.0.30** (Windows). Assume this version for any DAWproject or
  MIDI question; do not ask which DAW or version.
- **Two machines: a laptop and a desktop (a tower).** The user tries to keep
  them aligned, but installed software and drive layouts can differ. Check
  which one you are on (`hostname`) before assuming a path or an installed
  program, and record facts here per machine.
  - **Laptop (`ProLaptopCompact`):** Windows drives visible from WSL are `C:` and `D:` only, as
    `/mnt/c` and `/mnt/d`; `E:` is not mounted. Each piece has a Cubase
    project folder at `/mnt/d/<piece>/` (e.g. `/mnt/d/N52/`), holding the
    `.cpr` files, the video and the `<piece>_speed.py` input. After
    generating a `.dawproject` for a piece, also copy it into that folder,
    so it is ready for `File > Import > DAWproject`.
  - **Desktop (`SmallPinkDot`, the tower):** Windows drives `C:` through `I:`
    are visible from WSL (`/mnt/c` … `/mnt/i`). Each piece has a Cubase
    project folder at `F:\Cubase Projects\<piece>\`, i.e.
    `/mnt/f/Cubase Projects/<piece>/` (e.g. `N52`); copy generated files
    there for Cubase. Windows refuses to overwrite a file Cubase has open
    ("Permission denied" from WSL); release it in Cubase or use a new name.

**Only Cubase 15 is installed on the laptop** (checked 2026-10-17). Cubase 14 was installed alongside it until
it was uninstalled in October 2026. While both were present, some early tests ran
on 14 by accident, because the Windows file association opened 14 and the
shortcut opened 15. Import with `File > Import > DAWproject` from inside
Cubase. To confirm which version wrote a file, export any `.dawproject` from
it and read the `<Application version="...">` line.

**The desktop** has `C:\Program Files\Steinberg\` folders for Cubase 10.5,
11, 12, 14 and 15, but only Cubase 15 in the Start menu (checked
2026-10-22); some of those folders may be leftovers. As on the laptop, start
Cubase 15 from its shortcut and import from inside it.

### Cubase MIDI conventions

- **Note names: Cubase calls MIDI note 60 C3**, not C4. Use Cubase's names
  when describing notes to the user.
- **MIDI export (`File > Export > MIDI File`) applies Play Probability.** A
  note with probability below 100% is randomly kept or dropped on each
  export, so two exports of the same track can differ. Standard MIDI files
  have no probability field, so the roll is baked in.
- **Muted and disabled tracks are left out of a MIDI export.** Tracks that
  appear in the file with no notes are genuinely empty tracks.

### Known Cubase DAWproject limitations

- **Automation will not import onto a group (`role="submix"`) channel.**
  Confirmed on both Cubase 14.0.41 and 15.0.30 across twelve structural
  variants, including Cubase's own export shape. Cubase 15 also fails to
  reimport its *own* exported group automation, so this is a Cubase
  limitation, not a problem with the generated XML. Settled approach in
  `write_dawproject.py`: each metric is an audio `Track` whose channel is
  `role="regular"`, carries the automation and routes to the master. The
  audio track carrying automation was verified importing on 15.0.30 when it
  fed a paired `"<metric> BUS"` submix channel; those busses were dropped on
  2026-10-04 as unhelpful, and the bus-free shape awaits a Cubase import. Do not redesign this without
  first re-running the round-trip test.
- **`contentType` is required** on a `Track`, or Cubase creates no track.
- **`Transport/Tempo` is required**, or Cubase silently loads nothing.
- **`Transport/TimeSignature` is required**, or Cubase imposes 4/4 on the
  target project instead of leaving its signature alone.
- Importing into an existing project **overwrites its initial tempo marking
  and time signature**, so set `dawproject.tempo` and
  `dawproject.time_signature` to match that project (N48: 75 bpm, 1/4).
- **Submix channels cannot be interleaved with tracks.** Cubase collects
  every submix channel into its own Group folder, whatever order the
  `Structure` element lists them in.
- `contentType`, `Tempo` and `TimeSignature` are all optional in the schema
  and all required in practice: **schema-valid is not the same as loadable**,
  and an optional element is not safe to omit. Confirm every structural
  change with a real import.
- **`interpolation="hold"` is ignored on import.** Cubase 15.0.30 ramps
  linearly between points regardless, so a step must be written as two
  linear points close together (old value, then new). `speed_to_dawproject.py`
  binned mode uses one tick (1/480 quarter).
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
4. **MIDI** (write_midi.py) - Legacy CC output, off by default
5. **DAWproject** (write_dawproject.py) - Volume automation; the default output

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

### Audio Synthesis (separate from the video pipeline)
```bash
python metrics_to_wav.py json/N52_drone_h6/N52_h6_G_r110_flat.json
```
Renders harmonic tones from a run's raw `*_basic.csv`, either one drone note
(`root_hz`) or one tone per note of a monophonic MIDI file (`midi_file`; MIDI
t=0 = video t=0); both go through `render_note()`. Harmonic n of the note has
amplitude `n**strength_exponent * gain * rank(expr)` (gain per row),
linearly interpolated between rows, divided by the sum of strengths. The listed
expressions cycle up to `max_hz` (harmonic L+1 reuses entry 1) when `cycle` is
true; with `cycle` false exactly the listed n sound (e.g. octave thinning). Random phases,
optional per-harmonic pitch jitter (`jitter_cents`), and a `fade_s` fade on
every note, optional speed-driven pitch glide (`pitch_mod`). Expressions
allow only `*`, `/` and parentheses. Output is a 32-bit float WAV next to the CSV.

### Directory Structure
- Videos: `data/input/`
- Outputs: `data/output/{video_name}_{preset}/`

### Configuration
All parameters are in JSON config files. See `default_config.json` for template.

Key config parameters:
- `video.video_name` - Video filename (without extension) - **REQUIRED**
- `timing.frames_per_second` - Required when `process_video` runs (it silently
  defaults to 30 otherwise)
- `video_processing.process_every_nth_frame` - Integer frame step; fixes the time
  resolution permanently, so choose it before a long run
- `video_processing.optical_flow.preset` - Motion detection preset (default: "default")
- `metrics_processing.filter_seconds` - Smoothing widths in seconds (fractional allowed)
- `metrics_processing.color_channels` / `metrics` / `rank_types` / `inversions` -
  Outer product controlling which columns the values CSV holds, and hence how
  many DAWproject tracks result
- `cluster_processing.k_values` - [2, 3, 4, 5, 6, 8, 10, 12] - Cluster counts to try
- `pipeline_control.process_video` / `process_metrics` / `process_clusters` /
  `write_dawproject` / `write_midi` - Enable/disable stages. DAWproject is the
  default output; MIDI is legacy and off by default

### Performance Optimizations

Computationally expensive metrics (symmetry, error dispersion, dark/light, motion, full GMM) are computed **only for Gray channel** to improve processing speed. For other channels, those columns are not emitted at all (rather than emitted as zeros) — non-Gray channels store only `_avg`, `_std`, `gmn`, and `gs1`. This optimization can be reversed by removing the `if color_channel_name == "Gray"` guards in `process_video.py` if full color analysis is needed.

## Documentation

For complete documentation, see `README.md`.
