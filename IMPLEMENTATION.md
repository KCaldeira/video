# Integration Proposal: Tempo Mapping + Visual Metrics Pipeline

## Executive Summary

This document proposes integrating two complementary workflows:

1. **Tempo Map Generation** (`calculate_tempo_from_inverse.py`) - Creates variable tempo maps from inverse zoom rate data
2. **Visual Metrics MIDI** (`run_video_processing.py`) - Generates MIDI tracks from comprehensive video analysis

The integrated system will produce **tempo-synchronized MIDI files** where:
- Tempo varies based on zoom depth analysis
- Visual metrics (color, motion, symmetry) are encoded as CC values
- All MIDI tracks share a common, dynamic tempo map
- Beat alignment is consistent across all outputs

---

## Current State Analysis

### Workflow 1: Tempo Map Generation (`calculate_tempo_from_inverse.py`)

**Input**:
- `y_values.py` file containing inverse zoom rates per frame
- Configuration: `mean_tempo_bpm`, `fps`, `division`

**Processing**:
- Calculates tempo inversely proportional to y_values: `tempo[i] = k / y_values[i]`
- Applies window extension to keep tempo in MIDI-valid range (3.58-300 BPM)
- Accumulates beats across frames using variable tempo
- Generates frame-level tempo changes

**Output**:
- CSV with frame-level tempo data
- MIDI file with:
  - Frame-by-frame tempo changes (very detailed)
  - Beat notes (quarter notes at variable tempo)
  - CC1 tracks (Normal, Inverted, Fast/Medium/Slow variants)

**Key Feature**: Tempo is **data-driven** and varies continuously based on zoom dynamics

---

### Workflow 2: Visual Metrics MIDI (`run_video_processing.py`)

**Input**:
- Video file (`.wmv`, `.mp4`)
- JSON configuration with timing, processing, and metrics parameters

**Processing**:
1. **Video Analysis** - Extract 100+ visual metrics per frame
2. **Metrics Processing** - Apply filtering, stretching, inversion transformations
3. **MIDI Generation** - Convert each metric to CC MIDI track
4. **Clustering** - Group similar frames (optional)

**Output**:
- CSV with primary metrics (`*_basic.csv`)
- Excel with all derived metrics (`*_derived.xlsx`)
- 100+ MIDI files (one per metric transformation)
- PDF plots, cluster assignments

**Key Feature**: Rich **visual analysis** with fixed tempo

---

## Integration Architecture

### Option A: Video-Driven Tempo Map (Recommended)

Generate tempo map **directly from video analysis**, eliminating the need for separate y_values files.

#### Data Flow

```
Video File
    ↓
┌─────────────────────────────────┐
│ Step 1: Video Analysis          │
│ (process_video.py)               │
│                                  │
│ Extract:                         │
│ - Visual metrics (colors, etc.)  │
│ - Motion metrics (czd, crc, cmv) │
│ - Zoom divergence (Gray_czd)     │
└─────────────────────────────────┘
    ↓
    ├─→ Primary Metrics CSV
    │
    ↓
┌─────────────────────────────────┐
│ Step 2: Tempo Map Generation    │
│ (NEW: tempo_from_video.py)       │
│                                  │
│ Use Gray_czd as inverse zoom     │
│ Calculate variable tempo         │
│ Generate beat timing             │
└─────────────────────────────────┘
    ↓
    ├─→ Tempo Map CSV
    ├─→ Tempo Map MIDI
    │
    ↓
┌─────────────────────────────────┐
│ Step 3: Metrics to MIDI          │
│ (MODIFIED: process_metrics.py)   │
│                                  │
│ Use variable tempo from Step 2   │
│ Generate CC tracks synchronized  │
│ with tempo map                   │
└─────────────────────────────────┘
    ↓
    ├─→ Visual Metrics MIDI (tempo-synced)
    ├─→ Derived Metrics Excel
    ├─→ Plots PDF
    │
    ↓
┌─────────────────────────────────┐
│ Step 4: Clustering (Optional)    │
│ (cluster_primary.py)             │
└─────────────────────────────────┘
    ↓
    └─→ Cluster Assignments
```

#### Key Changes

1. **New Module**: `tempo_from_video.py`
   - Reads `*_basic.csv` from video analysis
   - Uses `Gray_czd` (zoom divergence) as basis for tempo
   - Implements tempo calculation algorithms from `calculate_tempo_from_inverse.py`
   - Outputs tempo map CSV and MIDI

2. **Modified**: `process_metrics.py`
   - Accept optional tempo map input
   - When tempo map provided: use variable tempo for MIDI generation
   - When no tempo map: fallback to fixed `beats_per_minute` (backward compatible)
   - Synchronize all CC tracks with tempo changes

3. **Modified**: `run_video_processing.py`
   - Add Step 2.5: Generate tempo map from video metrics
   - Pass tempo map to metrics processing
   - Coordinate output files

---

### Option B: Separate Y-Values Tempo Map

Use existing `calculate_tempo_from_inverse.py` with external y_values files, coordinate with video processing.

#### Data Flow

```
Video File                    Y-Values File (separate)
    ↓                              ↓
    │                    ┌─────────────────────┐
    │                    │ Tempo Map Generation │
    │                    │ (calculate_tempo...py)│
    │                    └─────────────────────┘
    │                              ↓
    │                         Tempo Map CSV/MIDI
    ↓                              ↓
┌─────────────────┐               │
│ Video Analysis  │               │
│ (process_video) │               │
└─────────────────┘               │
    ↓                              │
Primary CSV                        │
    ↓                              │
    └──────────┬───────────────────┘
               ↓
    ┌─────────────────────┐
    │ Metrics Processing   │
    │ (process_metrics.py) │
    │ + Tempo Map Input    │
    └─────────────────────┘
               ↓
    Tempo-Synced MIDI Files
```

**Advantages**:
- Minimal code changes
- Allows external tempo control
- Separates tempo design from video analysis

**Disadvantages**:
- Requires separate y_values file preparation
- Two-step workflow (tempo map, then video)
- Potential frame count mismatch issues

---

## Recommended Implementation: Option A

### Rationale

1. **Single Source of Truth**: Video file is the only input needed
2. **Natural Integration**: Zoom data (`Gray_czd`) is already extracted during video analysis
3. **Consistency**: Frame counts guaranteed to match
4. **User Experience**: One-command workflow with JSON configuration
5. **Flexibility**: Can still use external y_values if needed (hybrid approach)

---

## Detailed Design: Option A

### 1. New Module: `tempo_from_video.py`

```python
def generate_tempo_map_from_csv(
    csv_path,
    output_dir,
    mean_tempo_bpm=64.0,
    fps=30.0,
    division=480,
    cc_subsample=30,
    zoom_metric="Gray_czd"
):
    """
    Generate tempo map from video analysis CSV.

    Parameters:
    - csv_path: Path to *_basic.csv from process_video.py
    - output_dir: Directory for tempo map outputs
    - mean_tempo_bpm: Target average tempo
    - fps: Frame rate (must match video processing)
    - division: MIDI ticks per beat
    - cc_subsample: CC track sub-sampling interval
    - zoom_metric: Column name for zoom data (default: "Gray_czd")

    Returns:
    - tempo_map_dict: Dictionary with tempo_bpm array and beat_frames
    - tempo_csv_path: Path to generated CSV
    - tempo_midi_path: Path to generated MIDI
    """
    # Read CSV
    # Extract zoom_metric column as y_values
    # Call compute_beat_tempos_from_inverse() logic
    # Return tempo map data structure
```

**Functions to Extract from `calculate_tempo_from_inverse.py`**:
- `calculate_tempo_with_window_extension()` - Core tempo calculation
- `calculate_midi_delta_ticks()` - MIDI timing conversion
- Core logic from `compute_beat_tempos_from_inverse()` - Tempo map building

---

### 2. Modified: `process_metrics.py`

**Current Behavior**:
- Uses fixed `beats_per_minute` for MIDI timing
- Calculates ticks per frame: `ticks_per_frame = (bpm/60) * (1/fps) * division`
- All frames have identical timing

**New Behavior**:
```python
def process_metrics_to_midi(subdir_name, config, tempo_map=None):
    """
    Process metrics and generate MIDI files.

    Parameters:
    - subdir_name: Video/project name
    - config: Configuration dictionary
    - tempo_map: Optional dict with 'tempo_bpm' and 'beat_frames' arrays
                 If None, use fixed tempo from config
    """
    if tempo_map is None:
        # Existing fixed tempo logic
        ticks_per_frame = calculate_fixed_ticks_per_frame(config)
    else:
        # NEW: Variable tempo logic
        ticks_per_frame_array = calculate_variable_ticks_per_frame(
            tempo_map['tempo_bpm'],
            config['frames_per_second'],
            config['ticks_per_beat']
        )
```

**MIDI Generation Changes**:
- Accept `ticks_per_frame_array` instead of scalar
- Accumulate delta ticks frame-by-frame: `delta_ticks[i] = ticks_per_frame_array[i]`
- Synchronize CC changes with tempo changes
- Optionally embed tempo changes in each MIDI file

---

### 3. Modified: `run_video_processing.py`

**New Pipeline Steps**:

```python
# After Step 1: Video Analysis
if process_video:
    run_process_video(...)  # Existing

# NEW Step 2: Generate Tempo Map (if enabled)
tempo_map = None
if pipeline_config.get('generate_tempo_map', False):
    from tempo_from_video import generate_tempo_map_from_csv

    csv_path = f"data/output/{subdir_name}_{preset}/{subdir_name}_{preset}_basic.csv"
    tempo_config = config.get('tempo_mapping', {})

    tempo_map, csv_path, midi_path = generate_tempo_map_from_csv(
        csv_path=csv_path,
        output_dir=f"data/output/{subdir_name}_{preset}",
        mean_tempo_bpm=tempo_config.get('mean_tempo_bpm', beats_per_minute),
        fps=frames_per_second,
        division=ticks_per_beat,
        cc_subsample=tempo_config.get('cc_subsample', 30),
        zoom_metric=tempo_config.get('zoom_metric', 'Gray_czd')
    )

# Modified Step 3: Metrics Processing (pass tempo_map)
if process_metrics:
    run_process_metrics(subdir_name, config, tempo_map=tempo_map)
```

---

### 4. Configuration Schema Updates

**New Section in JSON Config**:

```json
{
  "pipeline_control": {
    "process_video": true,
    "process_metrics": true,
    "process_clusters": true,
    "generate_tempo_map": true
  },

  "tempo_mapping": {
    "enabled": true,
    "mode": "video_based",
    "mean_tempo_bpm": 96.0,
    "zoom_metric": "Gray_czd",
    "cc_subsample": 30,
    "tempo_range": {
      "min_bpm": 3.58,
      "max_bpm": 300.0
    },
    "fallback_to_fixed": false
  }
}
```

**Configuration Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `false` | Enable variable tempo mapping |
| `mode` | string | `"video_based"` | `"video_based"` or `"external_file"` |
| `mean_tempo_bpm` | float | `96.0` | Target average tempo |
| `zoom_metric` | string | `"Gray_czd"` | CSV column to use as inverse zoom rate |
| `cc_subsample` | int | `30` | Sub-sampling interval for CC tracks in tempo MIDI |
| `tempo_range` | object | See above | Min/max BPM constraints |
| `fallback_to_fixed` | bool | `false` | Use fixed tempo if tempo map fails |

**External File Mode** (for Option B compatibility):

```json
{
  "tempo_mapping": {
    "enabled": true,
    "mode": "external_file",
    "y_values_path": "data/input/N30_T7a_speed.py",
    "mean_tempo_bpm": 96.0
  }
}
```

---

## Output File Organization

### Current Structure
```
data/output/{video_name}_{preset}/
├── {video_name}_{preset}_basic.csv
├── {video_name}_{preset}_config.json
├── {video_name}_derived.xlsx
├── {video_name}_plots.pdf
├── {video_name}_{preset}_clusters.csv
├── {video_name}_{preset}_cluster_scores.json
├── {video_name}_{preset}_cluster_exemplars.json
└── metrics_midi/
    ├── R_avg_v_f017_s1-0.5_o.mid
    ├── R_avg_v_f017_s1-0.5_i.mid
    └── ... (100+ files)
```

### Proposed Structure
```
data/output/{video_name}_{preset}/
├── {video_name}_{preset}_basic.csv
├── {video_name}_{preset}_config.json
├── {video_name}_derived.xlsx
├── {video_name}_plots.pdf
├── {video_name}_{preset}_clusters.csv
├── {video_name}_{preset}_cluster_scores.json
├── {video_name}_{preset}_cluster_exemplars.json
│
├── tempo_map/                               # NEW
│   ├── tempo_map_inverse_{bpm}bpm.csv       # Frame-level tempo data
│   ├── tempo_map_inverse_{bpm}bpm.mid       # Tempo changes + beats + CC tracks
│   └── tempo_map_config.json                # Tempo generation parameters
│
└── metrics_midi/
    ├── R_avg_v_f017_s1-0.5_o.mid           # Now tempo-synchronized
    ├── R_avg_v_f017_s1-0.5_i.mid
    └── ... (100+ files, all with variable tempo)
```

**Key Differences**:
- New `tempo_map/` subdirectory with tempo-specific outputs
- All MIDI files in `metrics_midi/` now contain tempo changes (optional)
- Tempo map configuration stored separately for reproducibility

---

## MIDI Synchronization Strategy

### Challenge

Tempo map MIDI has:
- Frame-by-frame tempo changes (potentially 1000s of events)
- Beat notes at variable intervals
- Sub-sampled CC tracks

Visual metrics MIDI has:
- One CC change per frame (or per beat)
- One MIDI file per metric transformation (100+ files)

### Solution: Two Approaches

#### Approach 1: Embed Tempo in Each MIDI File

**Advantages**:
- Self-contained files
- Import into DAW shows correct timing immediately
- Each file is standalone

**Disadvantages**:
- Large file sizes (tempo events duplicated 100+ times)
- Harder to edit tempo globally

**Implementation**:
```python
def write_metric_midi_with_tempo(metric_values, tempo_bpm_array, ...):
    """
    Write MIDI file with both tempo changes and CC events.

    Track 0: Tempo changes (frame-level)
    Track 1: CC values for this metric
    """
```

---

#### Approach 2: Separate Tempo Master + Metrics Slaves

**Advantages**:
- Smaller file sizes
- Single tempo map to edit
- Clear separation of concerns

**Disadvantages**:
- Requires DAW to sync multiple files
- Potential for tempo mismatch if files separated

**Implementation**:
```python
# tempo_map/tempo_map_master.mid
#   - Contains ALL tempo changes
#   - Contains beat markers (note-on/off)
#   - Contains tempo CC tracks

# metrics_midi/*.mid
#   - NO tempo changes
#   - Assumes external tempo from master
#   - Pure CC data at correct tick positions
```

**Recommended**: **Approach 1** for ease of use, with option to generate Approach 2 "slim" versions

---

## Implementation Phases

### Phase 1: Core Integration (Minimal Viable Product)

**Goal**: Basic tempo map generation from video

**Tasks**:
1. Create `tempo_from_video.py` module
   - Extract tempo calculation functions from `calculate_tempo_from_inverse.py`
   - Add CSV reading interface
   - Test with existing video outputs

2. Modify `process_metrics.py`
   - Add `tempo_map` parameter
   - Implement variable ticks-per-frame logic
   - Generate MIDI with tempo changes (Approach 1)

3. Modify `run_video_processing.py`
   - Add tempo map generation step
   - Pass tempo data to metrics processing
   - Update configuration handling

4. Update configuration schema
   - Add `tempo_mapping` section
   - Add `generate_tempo_map` flag

5. Test with existing video
   - Run full pipeline with tempo mapping enabled
   - Verify MIDI file synchronization
   - Validate tempo range and beat alignment

**Deliverables**:
- Working end-to-end pipeline with tempo mapping
- Updated documentation
- Example configuration file

---

### Phase 2: Enhanced Features

**Goal**: Flexibility, optimization, and alternative modes

**Tasks**:
1. External y_values support (Option B)
   - Add `mode: "external_file"` option
   - Load y_values from Python file
   - Coordinate frame counts

2. Tempo algorithm options
   - Support both `compute_beat_tempos_from_zoom()` and `compute_beat_tempos_from_inverse()`
   - Allow user selection via config
   - Document tradeoffs

3. Tempo visualization
   - Add tempo plot to PDF output
   - Show tempo changes over time
   - Annotate fast/medium/slow sections

4. MIDI export options
   - Implement Approach 2 (separate master/slaves)
   - Add configuration flag for export mode
   - Generate both versions optionally

5. Clustering integration
   - Cluster frames by tempo characteristics
   - Analyze tempo vs visual metric correlations
   - Generate tempo-aware cluster quality metrics

**Deliverables**:
- Multiple tempo modes
- Enhanced visualization
- Flexible MIDI output options

---

### Phase 3: Advanced Analysis

**Goal**: Deep integration and new insights

**Tasks**:
1. Multi-metric tempo synthesis
   - Combine multiple metrics for tempo (not just zoom)
   - Weight different visual features
   - Machine learning for optimal tempo mapping

2. Beat-aligned clustering
   - Cluster on beat-level features instead of frame-level
   - Analyze musical structure (verse/chorus detection)
   - Tempo-aware segmentation

3. Tempo-responsive filtering
   - Adaptive filter periods based on tempo
   - Faster tempo → shorter filters
   - Preserve musical time scale

4. Real-time preview
   - Generate tempo map preview before full processing
   - Interactive tempo curve editing
   - Quick iteration on tempo parameters

**Deliverables**:
- Advanced tempo synthesis
- Musical structure analysis
- Interactive tools

---

## Configuration Examples

### Example 1: Basic Video-Based Tempo Mapping

```json
{
  "description": "Standard video processing with automatic tempo mapping",
  "video": {
    "video_name": "N30_T7a"
  },
  "timing": {
    "frames_per_second": 30,
    "beats_per_minute": 96,
    "ticks_per_beat": 480
  },
  "pipeline_control": {
    "process_video": true,
    "process_metrics": true,
    "generate_tempo_map": true,
    "process_clusters": true
  },
  "tempo_mapping": {
    "enabled": true,
    "mode": "video_based",
    "mean_tempo_bpm": 96.0,
    "zoom_metric": "Gray_czd"
  }
}
```

---

### Example 2: External Y-Values Tempo Source

```json
{
  "description": "Use external y_values file for tempo mapping",
  "video": {
    "video_name": "N30_T7a"
  },
  "pipeline_control": {
    "process_video": true,
    "process_metrics": true,
    "generate_tempo_map": true
  },
  "tempo_mapping": {
    "enabled": true,
    "mode": "external_file",
    "y_values_path": "data/input/N30_T7a_speed.py",
    "mean_tempo_bpm": 96.0,
    "cc_subsample": 30
  }
}
```

---

### Example 3: Fixed Tempo (Backward Compatibility)

```json
{
  "description": "Traditional fixed-tempo processing (no tempo mapping)",
  "video": {
    "video_name": "N30_T7a"
  },
  "timing": {
    "beats_per_minute": 120
  },
  "pipeline_control": {
    "process_video": true,
    "process_metrics": true,
    "generate_tempo_map": false
  }
}
```

---

## Testing Strategy

### Unit Tests

1. **Tempo Calculation**
   - Test `calculate_tempo_with_window_extension()` with edge cases
   - Verify tempo stays in valid MIDI range
   - Test window extension logic

2. **MIDI Timing**
   - Verify ticks-per-frame calculation
   - Test tempo change event placement
   - Validate beat alignment

3. **CSV Integration**
   - Test reading video analysis CSV
   - Handle missing columns gracefully
   - Verify frame count consistency

### Integration Tests

1. **End-to-End Pipeline**
   - Run full pipeline with tempo mapping enabled
   - Verify all output files generated
   - Check MIDI file validity (import into DAW)

2. **Tempo Synchronization**
   - Compare beat positions across tempo map and metrics MIDI
   - Verify CC events align with tempo changes
   - Test with different FPS and BPM settings

3. **Backward Compatibility**
   - Run existing configurations without tempo mapping
   - Verify fixed-tempo MIDI generation unchanged
   - Test configuration migration

### Validation Tests

1. **MIDI Import Test**
   - Import generated MIDI into Ableton Live, Logic Pro, FL Studio
   - Verify tempo changes recognized
   - Check CC tracks playback correctly

2. **Tempo Range Test**
   - Test with extreme tempo variations (very slow/fast zoom)
   - Verify window extension prevents invalid tempos
   - Check for numerical stability

3. **Long Video Test**
   - Test with 10,000+ frame video
   - Check memory usage
   - Verify performance acceptable

---

## Migration Path

### For Existing Users

1. **No Breaking Changes**
   - Default: `generate_tempo_map: false` (fixed tempo)
   - Existing configurations work unchanged
   - Opt-in to tempo mapping

2. **Gradual Adoption**
   - Try tempo mapping on one video
   - Compare fixed vs variable tempo results
   - Adjust parameters incrementally

3. **Hybrid Workflows**
   - Generate tempo map separately, then reference it
   - Test tempo map with subset of metrics
   - Refine before full processing

---

## Performance Considerations

### Computational Cost

**Tempo Map Generation**: ~1-2 seconds per 1000 frames
- Reading CSV: negligible
- Tempo calculation: O(n) with potential window extensions
- MIDI generation: O(n)

**Impact on Pipeline**: Minimal (<5% overhead)

### Memory Usage

**Tempo Arrays**: ~8 KB per 1000 frames (float64 array)
- Negligible compared to video processing

**MIDI File Sizes**:
- Fixed tempo: ~10-50 KB per file
- With tempo changes: ~50-200 KB per file (4-10x larger)
- Total increase: ~5-20 MB for 100 files (acceptable)

### Optimization Opportunities

1. **Shared Tempo Track**
   - Generate tempo events once
   - Reference in each MIDI file
   - Reduces MIDI generation time

2. **Cached Tempo Map**
   - Save tempo map to disk
   - Reuse across multiple metric processing runs
   - Faster iteration on metrics without regenerating tempo

3. **Parallel MIDI Generation**
   - Generate metric MIDI files in parallel
   - Tempo map shared read-only
   - Leverage multi-core systems

---

## Open Questions and Decisions Needed

### 1. Default Tempo Source

**Question**: When tempo mapping is enabled, which metric should be the default tempo source?

**Options**:
- `Gray_czd` (zoom divergence) - matches conceptual "zoom depth"
- `Gray_cmv` (motion variance) - general activity level
- User-specified column
- Composite of multiple metrics

**Recommendation**: `Gray_czd` as default, with easy override in config

---

### 2. Tempo Map Always Embedded?

**Question**: Should all metrics MIDI files include tempo changes, or generate both versions?

**Options**:
- Always embed (simple, larger files)
- Never embed (small files, need master)
- Generate both versions (flexible, more files)
- Config option (user choice)

**Recommendation**: Config option with default = always embed

---

### 3. Beat Notes in Metrics MIDI?

**Question**: Should metrics MIDI files include beat notes, or just CC data?

**Options**:
- Include beats (helpful for alignment, more cluttered)
- Omit beats (cleaner, rely on tempo map file)
- Config option

**Recommendation**: Omit beats from metrics MIDI, beats only in tempo_map MIDI

---

### 4. Frame vs Beat Resolution

**Question**: Should tempo changes be frame-level or beat-level?

**Options**:
- Frame-level (high precision, many events)
- Beat-level (cleaner, less precise)
- Hybrid (tempo changes on beats, interpolated)

**Recommendation**: Frame-level for maximum fidelity, matches current `calculate_tempo_from_inverse.py` behavior

---

### 5. Tempo Map Caching

**Question**: Should tempo maps be cached/reused across runs?

**Options**:
- Always regenerate (simple, consistent)
- Cache and detect changes (faster iteration)
- Manual cache management (user control)

**Recommendation**: Start with always regenerate, add caching in Phase 2

---

## Success Criteria

### Minimum Viable Product (Phase 1)

- [ ] Generate tempo map from video analysis CSV
- [ ] Tempo map uses `Gray_czd` as inverse zoom rate
- [ ] All metrics MIDI files synchronized with tempo map
- [ ] Configuration via JSON (backward compatible)
- [ ] Output files organized in `tempo_map/` subdirectory
- [ ] MIDI files import correctly into DAW with tempo changes
- [ ] Documentation updated with examples
- [ ] One end-to-end test with real video

### Phase 2 Success

- [ ] External y_values file support working
- [ ] Tempo visualization in PDF output
- [ ] Both Approach 1 and Approach 2 MIDI generation
- [ ] Tempo-aware clustering integration
- [ ] Performance acceptable for 10,000+ frame videos

### Phase 3 Success

- [ ] Multi-metric tempo synthesis implemented
- [ ] Beat-aligned clustering working
- [ ] Tempo-responsive filtering available
- [ ] Interactive tempo preview tool

---

## Conclusion

This integration proposal provides a **comprehensive roadmap** for combining tempo mapping with visual metrics analysis. The recommended **Option A (Video-Driven Tempo Map)** offers:

✅ **Single-source workflow** - Video file as sole input
✅ **Natural integration** - Zoom data already extracted
✅ **Backward compatibility** - Existing configs unchanged
✅ **Flexibility** - Support for external tempo sources
✅ **Rich output** - Tempo-synchronized MIDI tracks

The **phased implementation** allows incremental development and testing, with Phase 1 delivering immediate value and later phases adding advanced features.

**Next Steps**:
1. Review and approve this proposal
2. Create detailed technical specifications for Phase 1
3. Implement `tempo_from_video.py` module
4. Modify `process_metrics.py` for tempo synchronization
5. Update pipeline orchestration in `run_video_processing.py`
6. Test with existing videos and iterate

**Questions?** Please review the "Open Questions and Decisions Needed" section and provide guidance on preferred approaches.
