# CONTINUATION.md

This file contains important information about the codebase that should be kept in mind when modifying code in this repository.

## Processing Pipeline Order

### **CRITICAL: Filtering Must Come Last**
The triangular filtering must be applied **as the final step** in the processing pipeline, not early in the sequence.

**Correct Order:**
1. Create base entries (`_v`, `_r`)
2. Scale data to 0-1 range
3. Apply stretching (if enabled)
4. Apply inversion (if enabled)
5. **Apply filtering as the final step** (to processed data)

**Why this matters:**
- Filtering applied to raw data produces jagged, inconsistent curves
- Filtering applied to processed data (after scaling/stretching) produces smooth curves
- The 65 and 257 point smoothing should produce progressively smoother curves
- This was a recurring bug that took multiple attempts to fix correctly

### **Processing Sequence Details**
```python
# 1. Base entries (unfiltered)
process_dict[key + "_v"] = csv[key]
process_dict[key + "_r"] = percentile_data(csv[key])

# 2. Scale to 0-1
process_dict[key] = scale_data(process_dict[key])

# 3. Stretch (if enabled)
# 4. Invert (if enabled)

# 5. Filter as FINAL step
new_key = key + f"_f{filter_period:03d}"
process_dict[new_key] = triangular_filter_odd(process_dict_copy[key], filter_period)
```

## Filter Periods Behavior

### **Controlled by filter_periods List**
- **`filter_periods = [17, 65, 257]`**: Only creates `_f017`, `_f065`, `_f257` entries
- **`filter_periods = [1]`**: Only creates `_f001` entries (unfiltered)
- **`filter_periods = [1, 17, 65, 257]`**: Creates `_f001`, `_f017`, `_f065`, `_f257` entries

### **Consistent Naming Convention**
All filter periods (including 1) use the `_f{period:03d}` naming convention:
- `_f001` = unfiltered data (period 1)
- `_f017`, `_f065`, `_f257` = filtered data with respective periods

### **No Conditional Filtering Logic**
- Filtering is always enabled (no "filter" in process_list check)
- To disable filtering, use `filter_periods = [1]`
- This simplifies the code and makes behavior predictable

## Problem-Solving Philosophy

### **Root Cause Analysis Required**
**DO NOT** apply band-aid patches or quick fixes. Always:

1. **Understand the root cause** of the problem
2. **Analyze why** the current approach isn't working
3. **Identify the fundamental issue** before making changes
4. **Fix the root cause**, not just the symptoms

### **Examples of Root Cause vs Band-Aid**
**❌ Band-Aid (Wrong):**
- Adding more conditions to hide unwanted output
- Changing variable names to avoid conflicts
- Adding workarounds without understanding the core issue

**✅ Root Cause Fix (Correct):**
- Understanding that filtering was applied to raw data instead of processed data
- Fixing the processing order to apply filtering at the end
- Ensuring the filter_periods list controls output exactly

## Code Structure Principles

### **Single Source of Truth**
- The `filter_periods` list should be the only control for which filters are applied
- No hidden logic that creates unexpected entries
- Output should match the filter_periods list exactly

### **Simplified Logic**
- Prefer simple, linear processing over complex conditional logic
- Each transformation should build on the previous one
- Avoid nested loops and complex state management

### **Consistent Naming**
- `_f001` = unfiltered data (period 1)
- `_f017`, `_f065`, `_f257` = filtered data with respective periods
- All filter periods use consistent `_f{period:03d}` naming

## Common Mistakes to Avoid

### **1. Applying Filters to Raw Data**
```python
# ❌ WRONG - applies filter to raw data
filtered_data = triangular_filter_odd(csv[key], filter_period)

# ✅ CORRECT - applies filter to processed data
filtered_data = triangular_filter_odd(process_dict_copy[key], filter_period)
```

### **2. Complex Conditional Logic**
```python
# ❌ WRONG - complex conditions
if "filter" in process_list:
    if 1 in filter_periods:
        # create f001
    else:
        # create basic versions
    # apply filters...

# ✅ CORRECT - simple loop
for filter_period in filter_periods:
    if filter_period == 1:
        # create unfiltered
    else:
        # create filtered
```

### **3. Ignoring Processing Order**
- Always consider the full pipeline when making changes
- Test that changes don't break the processing sequence
- Verify that filtering comes last

## Testing Guidelines

### **Visual Verification**
- Check that 257-point filtering produces smoother curves than 65-point
- Verify that filtering applied to processed data produces smooth curves
- Ensure no jagged or inconsistent output

### **Output Verification**
- Confirm that only specified filter periods appear in output
- Check that MIDI files and PDF plots match filter_periods list
- All entries should have consistent `_f{period:03d}` naming

## When Making Changes

### **Before Making Changes**
1. **Read this file** to understand the principles
2. **Understand the current processing order**
3. **Identify the root cause** of any issues
4. **Plan the fix** before implementing

### **After Making Changes**
1. **Verify the processing order** is maintained
2. **Test with different filter_periods** combinations
3. **Check that filtering is still applied last**
4. **Ensure no band-aid fixes** were introduced

## Key Functions to Understand

### **process_video.py**
- `process_video_to_csv()` - Main function that processes video and outputs CSV
- `compute_basic_metrics()` - Computes basic visual metrics from frames
- `compute_change_metrics()` - Computes motion-based metrics between frames

### **process_metrics.py**
- `process_metrics_to_midi()` - Main function that processes CSV and outputs MIDI
- `post_process()` - Applies transformations (filtering, ranking, stretching, inversion)
- `add_derived_columns()` - Adds ratio metrics and rotation metrics

### **run_video_processing.py**
- `run_process_video()` - Calls process_video_to_csv with parameters
- `run_process_metrics()` - Calls process_metrics_to_midi with parameters
- `main()` - Handles command-line arguments and orchestrates the pipeline

### **Core Processing Functions**
- `triangular_filter_odd(data, N)` - Applies triangular smoothing filter
- `scale_data(data)` - Scales data to 0-1 range
- `percentile_data(data)` - Converts values to percentiles (0-1 range)

### **Flexible Sorting System**
The sorting system uses a configurable field-based approach:

**Key Structure:**
```
R_avg_v_f017_s1-0.5_o
│ │   │ │    │ │
│ │   │ │    │ └─ o/i (field 6) - inversion
│ │   │ │    └─── s1-0.5 (field 5) - stretching
│ │   │ └──────── f017 (field 4) - filtering
│ │   └────────── v (field 3) - rank/value
│ └────────────── avg (field 2) - metric
└──────────────── R (field 1) - color channel
```

**Processing Order**: The key reflects the actual processing sequence:
1. Base data: `R_avg`
2. Rank/Value: `R_avg_v` or `R_avg_r`
3. Scaling: (applied, no key change)
4. Filtering: `R_avg_v_f017`
5. Stretching: `R_avg_v_f017_s1-0.5`
6. Inversion: `R_avg_v_f017_s1-0.5_o` or `R_avg_v_f017_s1-0.5_i`

**Configurable Sort Order:**
```python
SORT_ORDER = [
    'smoothing_period',  # f001, f017, f065, f257
    'rank_value',        # r or v
    'color_channel',     # R, G, B, Gray, H000, etc.
    'metric',           # avg, std, xps, etc.
    'stretching',       # s1-0.5, s8-0.33, etc.
    'inversion'         # o or i
]
```

**To Change Sort Order:**
Simply modify the `SORT_ORDER` list at the top of the sorting section. The system automatically applies the new order to both PDF plots and MIDI file organization.

### **Data Flow Architecture**
The processing pipeline uses separate dictionaries for each stage:
- **`raw_entries`** - Initial base entries (`_v`, `_r`)
- **`scaled_entries`** - After scaling to 0-1 range
- **`filtered_entries`** - After filtering with `_f{period:03d}` suffixes
- **`stretched_entries`** - After stretching transformations (applied to filtered data)
- **`final_entries`** - After inversion with `_o` and `_i` suffixes (applied to stretched data)
- **`process_dict`** - Final output added to `master_dict`

**Processing Order:**
1. Raw data → Rank/Value → Scale → Filter → Stretch → Invert → Output
2. Rank/Value processing creates `_v` and `_r` versions
3. Filtering comes BEFORE stretching to ensure smooth curves are stretched
4. Inversion comes last to create `_o` and `_i` versions

**Simplified Logic:**
- All transformations (ranking, filtering, stretching, inversion) are applied by default
- No conditional logic needed - the pipeline is always complete
- Cleaner, more predictable code flow

**Configurable Parameters:**
- All processing parameters are now configurable via JSON file or command line
- No hardcoded values in `process_metrics.py`
- Parameters include: `filter_periods`, `stretch_values`, `stretch_centers`, `cc_number`
- Default values are defined in `run_video_processing.py` and can be overridden
- **Automatic Detection**: Variables and metrics are automatically detected from CSV columns
- No need to configure `vars` or `metric_names` - the script processes everything available

**Benefits:**
- Clear data flow through each processing stage
- No risk of accidentally keeping unwanted data
- Self-documenting variable names
- Easy debugging and inspection of intermediate stages

## Remember
- **Filtering comes LAST** in the processing pipeline
- **Fix root causes**, not symptoms
- **Keep logic simple** and predictable
- **Test thoroughly** before considering changes complete

---

## **Future Code Cleanup Opportunities**

### **process_video.py - Major Cleanup Areas:**

#### **1. Massive Function Size**
- `process_video_to_csv()` is over 700 lines long - this is a huge red flag
- Should be broken down into smaller, focused functions like:
  - `extract_frames()`
  - `compute_basic_metrics()`
  - `compute_motion_metrics()`
  - `compute_symmetry_metrics()`
  - `compute_error_dispersion()`
  - `save_results()`

#### **2. Repetitive Code Patterns**
- Multiple nearly identical loops for different color channels (R, G, B, Gray, S, V)
- Could be abstracted into a single function that takes color channel as parameter
- Similar repetitive patterns for hue-specific metrics (H000, H060, etc.)

#### **3. Hardcoded Magic Numbers**
- `center_region_ratio = 0.5` - should be configurable
- `downscale_factor = 2` for optical flow - should be configurable
- Various array indices and thresholds scattered throughout

#### **4. Poor Error Handling**
- Limited try/catch blocks
- No validation of input parameters
- Silent failures in some areas

#### **5. Mixed Responsibilities**
- The main function does everything: file I/O, video processing, metrics computation, data saving
- Violates single responsibility principle

### **process_metrics.py - Major Cleanup Areas:**

#### **1. Complex Nested Logic**
- The `post_process()` function has deeply nested loops and conditionals
- The MIDI generation section is particularly convoluted with multiple nested loops
- Could be broken into: `create_processed_entries()`, `generate_midi_files()`, `create_plots()`

#### **2. Repetitive Data Structure Creation**
- Multiple similar dictionary creation patterns (`raw_entries`, `scaled_entries`, etc.)
- Could use a more functional approach with data transformation pipelines

#### **3. Hardcoded Processing Logic**
- The processing pipeline (scale → filter → stretch → invert) is hardcoded
- Could be made configurable with a processing pipeline definition

#### **4. Complex Sorting Logic**
- The flexible sorting system is clever but could be simplified
- The `parse_key_fields()` function has complex string parsing logic

#### **5. Mixed Data Processing and I/O**
- The same function handles data transformation, MIDI generation, and PDF creation
- Should be separated into distinct modules

### **Cross-File Issues:**

#### **1. Inconsistent Naming**
- Some functions use snake_case, others don't
- Variable names could be more descriptive (`csv` vs `dataframe`, `vars` vs `color_channels`)

#### **2. Configuration Management**
- Both files have their own config handling logic
- Could be centralized into a shared configuration module

#### **3. Error Handling**
- Limited error handling and logging throughout
- No graceful degradation for missing data or processing failures

#### **4. Performance Issues**
- Multiple passes over the same data
- Could benefit from vectorized operations in some areas

### **Architectural Improvements:**

#### **1. Class-Based Design**
- Could benefit from classes like `VideoProcessor`, `MetricsProcessor`, `MIDIGenerator`
- Would make the code more testable and maintainable

#### **2. Pipeline Pattern**
- The processing pipeline could be implemented as a chain of transformation objects
- Would make it easier to add/remove/reorder processing steps

#### **3. Configuration Objects**
- Replace dictionary-based config with proper configuration objects
- Would provide type safety and better IDE support

#### **4. Separation of Concerns**
- Data processing, file I/O, and visualization should be in separate modules
- Would make the code more testable and reusable

### **Priority Order for Cleanup:**
1. **High Priority**: Break down massive functions into smaller, focused functions
2. **High Priority**: Extract repetitive code patterns into reusable functions
3. **Medium Priority**: Improve error handling and validation
4. **Medium Priority**: Make hardcoded values configurable
5. **Low Priority**: Architectural improvements (classes, pipeline pattern)

**Note**: The code works, but it's definitely in the "technical debt" category - it would benefit significantly from a refactoring to make it more maintainable, testable, and extensible.
