# Implementation Plan: Frame-Level Tempo from Inverse Zoom

## Overview
Create a new algorithm where tempo at each frame is inversely proportional to y_value, calibrated to produce a target number of beats across the video duration.

## Algorithm Design

### 1. Input Data
- **y_values**: Array of inverse zoom rates, one value per frame
- **target_beats**: Desired total number of beats across the video (from mean_tempo_bpm)
- **fps**: Video frame rate (default 30)
- **MIDI tempo range**: 3.58 - 300 BPM

### 2. Calculate Proportionality Constant

For each frame `i`, we want:
```
tempo_bpm[i] = k / y_value[i]
```

Where `k` is a constant we need to determine.

**Derivation:**
- Each frame has duration: `dt = 1/fps` seconds = `1/(fps*60)` minutes
- Beats in frame `i`: `beats[i] = tempo_bpm[i] * dt`
- Substituting tempo: `beats[i] = (k / y_value[i]) * (1/(fps*60))`
- Total beats: `sum(beats[i]) = k/(fps*60) * sum(1/y_value[i])`

To achieve `target_beats`:
```
k = (target_beats * fps * 60) / sum(1/y_value[i])
```

### 3. Calculate Per-Frame Tempo with Window Extension

Process frames sequentially, extending windows when needed to keep tempo in valid range:

```python
def calculate_tempo_with_window_extension(y_values, k, fps, min_tempo=3.58, max_tempo=300):
    """
    Calculate tempo for each frame, extending windows where needed to stay in range.

    Algorithm:
    - Start at frame i
    - Calculate tempo_i = k / y_values[i]
    - If in range [min_tempo, max_tempo]: use it, move to i+1
    - If out of range:
      - Try window [i, i+1]: avg_tempo = k * sum(1/y[i:i+2]) / 2
      - If in range: apply to both frames i and i+1, move to i+2
      - If not, try [i, i+2]: avg_tempo = k * sum(1/y[i:i+3]) / 3
      - If in range: apply to frames i, i+1, i+2, move to i+3
      - Continue extending until tempo is in range

    Division by zero handling: Treat as out-of-range (will trigger window extension)

    Returns:
    - tempo_bpm: Array of tempo values (one per frame), all in valid range
    """

    n_frames = len(y_values)
    tempo_bpm = np.zeros(n_frames)
    i = 0

    while i < n_frames:
        # Calculate single-frame tempo
        # Handle division by zero: if y_values[i] is 0 or very small, tempo will be inf or very large
        if y_values[i] == 0:
            single_tempo = float('inf')  # Out of range, will trigger windowing
        else:
            single_tempo = k / y_values[i]

        # Check if in range
        if min_tempo <= single_tempo <= max_tempo:
            # Use single-frame tempo
            tempo_bpm[i] = single_tempo
            i += 1
        else:
            # Out of range - extend window
            window_size = 1
            found_valid = False

            while i + window_size < n_frames and not found_valid:
                window_size += 1

                # Calculate average tempo over window [i, i+window_size)
                # avg_tempo = k * sum(1/y_values[i:i+window_size]) / window_size
                # Handle division by zero in window
                inverse_sum = 0.0
                for j in range(i, i + window_size):
                    if y_values[j] != 0:
                        inverse_sum += 1.0 / y_values[j]
                    else:
                        inverse_sum += float('inf')  # Will make avg_tempo inf

                avg_tempo = k * inverse_sum / window_size

                # Check if in range
                if min_tempo <= avg_tempo <= max_tempo:
                    # Apply this tempo to all frames in window
                    tempo_bpm[i:i+window_size] = avg_tempo
                    i += window_size
                    found_valid = True

            if not found_valid:
                # Reached end of array without finding valid tempo
                # Use remaining frames with clamped tempo
                remaining = n_frames - i
                inverse_sum = 0.0
                for j in range(i, n_frames):
                    if y_values[j] != 0:
                        inverse_sum += 1.0 / y_values[j]
                    # Skip zeros (they would contribute infinite tempo)

                if inverse_sum > 0:
                    avg_tempo = k * inverse_sum / remaining
                    clamped_tempo = np.clip(avg_tempo, min_tempo, max_tempo)
                else:
                    clamped_tempo = min_tempo  # Default to minimum

                tempo_bpm[i:] = clamped_tempo
                print(f"Warning: Frames {i}-{n_frames-1} required clamping to {clamped_tempo:.2f} BPM")
                break

    return tempo_bpm
```

### 4. Why This Algorithm Preserves Time

The key insight: **total beats over any window is the same whether we use individual frame tempos or averaged tempo.**

**Individual frame tempos:**
```
beats = sum(tempo[j] / (fps*60) for j in [i, i+n])
      = sum(k / y_values[j] / (fps*60) for j in [i, i+n])
      = k/(fps*60) * sum(1/y_values[j] for j in [i, i+n])
```

**Averaged tempo:**
```
avg_tempo = k * sum(1/y_values[j]) / n
beats = avg_tempo * n / (fps*60)
      = k * sum(1/y_values[j]) / n * n / (fps*60)
      = k/(fps*60) * sum(1/y_values[j])
```

**Same result!** Therefore, no drift between audio and video.

### 5. Accumulate Beats

Walk through frames, accumulating fractional beats:

```python
beat_accumulator = 0.0
beats_list = []

for i, frame in enumerate(frames):
    # Beats in this frame
    beats_this_frame = tempo_bpm[i] / (fps * 60)
    beat_accumulator += beats_this_frame

    # Emit beats when accumulator crosses integer thresholds
    while beat_accumulator >= 1.0:
        beats_list.append({
            'beat_index': len(beats_list) + 1,
            'time_sec': frame / fps,
            'frame': frame,
            'tempo_bpm': tempo_bpm[i]
        })
        beat_accumulator -= 1.0
```

### 6. Output Generation

**CSV Output:**
One row per beat with columns:
- `beat_index`: 1-indexed beat number
- `bar`: Bar number (beats // 4 + 1)
- `beat_in_bar`: Beat within bar (beats % 4 + 1)
- `time_sec`: Time in seconds
- `frame`: Frame number
- `tempo_bpm`: Tempo at this beat

**MIDI Output:**

The MIDI file contains multiple tracks:

1. **Track 1: Tempo Map with Notes**
   - Tempo changes at **every frame** (not just beats)
   - Notes on middle C (60) at each beat
   - Each tempo change occurs at the precise time of its frame

2. **Track 2: CC1 Normal (sub-sampled)**
   - CC values representing tempo (min→0, max→127)
   - Sub-sampled to reduce MIDI file size
   - Default: every 30 frames (adjustable parameter)

3. **Track 3: CC1 Inverted (sub-sampled)**
   - CC values representing inverted tempo (min→127, max→0)
   - Same sub-sampling as Track 2

#### MIDI Timing Calculation

For frame-level tempo changes, we need to calculate the delta-time in MIDI ticks between frames:

```python
def calculate_midi_delta_ticks(tempo_bpm_prev, fps, division):
    """
    Calculate MIDI ticks for one frame at given tempo.

    Parameters:
    - tempo_bpm_prev: Tempo in BPM during this frame
    - fps: Frames per second
    - division: MIDI ticks per beat

    Returns:
    - ticks: Number of MIDI ticks in one frame
    """
    # Time for one frame in seconds
    frame_duration_sec = 1.0 / fps

    # Beats in one frame at this tempo
    # tempo is in beats per minute, so beats per second = tempo / 60
    beats_per_frame = (tempo_bpm_prev / 60.0) * frame_duration_sec

    # Ticks for this frame
    ticks = int(round(beats_per_frame * division))

    return ticks
```

#### MIDI Generation Structure

```python
# Track 1: Tempo map with notes
track_tempo = mido.MidiTrack()
track_tempo.append(mido.MetaMessage('track_name', name='Tempo Map', time=0))

# First tempo at time 0
track_tempo.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo_bpm[0]), time=0))

# For each frame after the first
for i in range(1, len(tempo_bpm)):
    # Calculate delta-time from previous frame
    delta_ticks = calculate_midi_delta_ticks(tempo_bpm[i-1], fps, division)

    # Add tempo change
    track_tempo.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo_bpm[i]), time=delta_ticks))

# Add notes at beats (accumulated separately)
# Notes are added at their beat positions with appropriate delta-times

# Track 2 & 3: CC tracks (sub-sampled)
cc_subsample = 30  # Every 30 frames (adjustable parameter)
tempo_subsampled = tempo_bpm[::cc_subsample]
# Generate CC events at sub-sampled intervals
```

## Implementation Steps

1. **Copy existing file**:
   ```bash
   cp calculate_tempo_from_zoom.py calculate_tempo_from_inverse.py
   ```

2. **Implement helper functions**:
   - `calculate_tempo_with_window_extension()` - Main tempo calculation with range enforcement
   - `calculate_midi_delta_ticks()` - Convert frame duration to MIDI ticks

3. **Create new function** `compute_beat_tempos_from_inverse()` with parameters:
   - `input_path`: Path to y_values.py file
   - `mean_tempo_bpm`: Target average tempo (default 64.0)
   - `fps`: Frames per second (default 30.0)
   - `division`: MIDI ticks per beat (default 480)
   - `cc_subsample`: CC track sub-sampling interval in frames (default 30)
   - `csv_out_path`: CSV output path
   - `midi_out_path`: MIDI output path

   Function steps:
   - Load y_values from file
   - Calculate constant `k = (target_beats * fps * 60) / sum(1/y_values)`
   - Call `calculate_tempo_with_window_extension()` to get frame-level tempos
   - Accumulate beats across frames
   - Generate CSV output (beat-level data)
   - Generate MIDI output:
     - Track 1: Frame-level tempo changes + beat notes
     - Track 2: Sub-sampled CC1 normal
     - Track 3: Sub-sampled CC1 inverted

4. **Update main block** to call new function with appropriate parameters

5. **Test** with y_values.py:
   - Verify total beats ≈ target_beats
   - Verify all tempos in valid range (3.58-300 BPM)
   - Verify tempo inversely tracks y_values (small y → high tempo)
   - Verify no drift (total time matches video duration)
   - Verify MIDI file has frame-level tempo changes
   - Verify CC tracks are properly sub-sampled

## Expected Behavior

- **Small y_value** (deep zoom) → **high tempo** (fast beats)
- **Large y_value** (shallow zoom) → **low tempo** (slow beats)
- All tempo values within MIDI-valid range (3.58-300 BPM)
- Total beats matches target (within rounding)
- No drift between audio and video (time-conserving)
- Division by zero handled gracefully via window extension
