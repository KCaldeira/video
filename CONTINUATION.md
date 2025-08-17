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

### **Data Flow Architecture**
The processing pipeline uses separate dictionaries for each stage:
- **`raw_entries`** - Initial base entries (`_v`, `_r`)
- **`scaled_entries`** - After scaling to 0-1 range
- **`filtered_entries`** - After filtering with `_f{period:03d}` suffixes
- **`stretched_entries`** - After stretching transformations (applied to filtered data)
- **`final_entries`** - After inversion (applied to stretched data)
- **`process_dict`** - Final output added to `master_dict`

**Processing Order:**
1. Raw data → Scale → Filter → Stretch → Invert → Output
2. Filtering comes BEFORE stretching to ensure smooth curves are stretched
3. Inversion comes last to create complementary patterns

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
