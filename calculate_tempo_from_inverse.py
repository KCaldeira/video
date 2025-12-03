import pandas as pd
import numpy as np
import mido
import struct

def read_kfs_to_dataframe(filepath):
    """
    Read a KFS binary file and return a DataFrame with x (frame) and y (speed) columns.

    KFS file format:
      - First 4 bytes: number of points (int32)
      - Then for each point: 2 floats (float32) = 8 bytes representing (x, y)
    """
    with open(filepath, 'rb') as f:
        # Read the number of points (4 bytes)
        nb_points_data = f.read(4)
        nb_points = struct.unpack('<i', nb_points_data)[0]

        # Read each point (2 floats = 8 bytes per point)
        points = []
        for _ in range(nb_points):
            point_data = f.read(8)
            x, y = struct.unpack('<ff', point_data)
            points.append((x, y))

        # Convert list of tuples to DataFrame
        df = pd.DataFrame(points, columns=['x', 'y'])
        return df

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

    Parameters:
    - y_values: Array of inverse zoom rates
    - k: Proportionality constant
    - fps: Frames per second
    - min_tempo, max_tempo: Valid MIDI tempo range

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


def compute_beat_tempos_from_zoom(
    input_path,
    mean_tempo_bpm=64.0,
    fps=30.0,
    division=480,
    csv_out_path="beat_tempos.csv",
    midi_out_path="tempo_map.mid",
    use_kfs=True,
    use_y_values=False,
):
    """
    Build a tempo map with beats evenly spaced in log(zoom depth).

    Algorithm:
      1. Load frame and speed/zoom data (from KFS, CSV, or y_values.py)
      2. Calculate target_beats from video duration and mean_tempo_bpm
      3. Compute cumulative zoom depth (integral of zoom over time)
      4. Create evenly spaced points in log(zoom depth)
      5. Interpolate to find time at each beat
      6. Calculate tempo from time differences between consecutive beats
      7. Generate MIDI file with tempo changes

    Inputs:
      - input_path: KFS file OR CSV file OR path to y_values.py
      - mean_tempo_bpm: desired average tempo in BPM (default 64.0)
      - fps: video frame rate (default 30.0)
      - division: MIDI ticks per beat (default 480)
      - use_kfs: if True, read KFS file; if False, read CSV file (ignored if use_y_values=True)
      - use_y_values: if True, read y_values.py file with inverse zoom rates

    Outputs:
      - CSV with beat-level tempo info
      - Standard MIDI file with tempo events at every beat
    """

    # --- Load data ---
    if use_y_values:
        # Read y_values.py file
        import sys
        import os
        # Add the directory containing y_values.py to the path
        sys.path.insert(0, os.path.dirname(input_path))
        # Import the module
        module_name = os.path.basename(input_path).replace('.py', '')
        y_values_module = __import__(module_name)
        y_values = np.array(y_values_module.y_values)

        # Calculate zoom as log2/y_values (since y_values contains inverse zoom rates)
        log_zoom = np.log(2) / y_values

        # Frames are sequential: 0, 1, 2, ..., len(y_values)-1
        frames = np.arange(len(y_values))

        print(f"Loaded y_values.py file: {len(frames)} frames")
    elif use_kfs:
        # Read KFS file
        df = read_kfs_to_dataframe(input_path)
        frames = df["x"].to_numpy()
        log_zoom = df["y"].to_numpy()
        print(f"Loaded KFS file: {len(frames)} keyframes")
    else:
        # Read CSV file
        df = pd.read_csv(input_path)
        frames = df["frame_count_list"].to_numpy()
        log_zoom = df["Gray_czd"].to_numpy()
        print(f"Loaded CSV file: {len(frames)} frames")

    print(f"Zoom stats: min={log_zoom.min():.4f}, max={log_zoom.max():.4f}, mean={log_zoom.mean():.4f}")

    # --- Convert frames to time ---
    time = frames / fps
    video_duration = time[-1]
    print(f"Total video duration: {video_duration:.2f} seconds")

    # --- Calculate target beats from mean tempo ---
    target_beats = int(video_duration * mean_tempo_bpm / 60.0)
    print(f"Target beats (from {mean_tempo_bpm:.1f} BPM): {target_beats}")

    # --- Compute cumulative zoom depth ---
    if use_y_values:
        # For y_values mode: cumulative sum of zoom (1/y_values), starting at 0.0
        cumulative_log_zoom_depth = np.concatenate([[0.0], np.cumsum(log_zoom)])
        # Need to adjust time and frames arrays to match (prepend 0.0 and 0)
        time = np.concatenate([[0.0], time])
        frames = np.concatenate([[0], frames])
    else:
        # For KFS/CSV mode: integral of zoom over time using trapezoidal integration
        cumulative_log_zoom_depth = np.zeros_like(time)
        for i in range(1, len(time)):
            dt = time[i] - time[i-1]
            avg_zoom = (log_zoom[i] + log_zoom[i-1]) / 2.0
            cumulative_log_zoom_depth[i] = cumulative_log_zoom_depth[i-1] + avg_zoom * dt

    print(f"Cumulative zoom depth range: {cumulative_log_zoom_depth.min():.4f} to {cumulative_log_zoom_depth.max():.4f}")

    # --- Create evenly spaced points in cumulative zoom depth ---
    zoom_depth_min = cumulative_log_zoom_depth.min()
    zoom_depth_max = cumulative_log_zoom_depth.max()
    log_zoom_depth_beats = np.linspace(zoom_depth_min, zoom_depth_max, target_beats)

    # --- Interpolate to find time at each beat ---
    beat_times_raw = np.interp(log_zoom_depth_beats, cumulative_log_zoom_depth, time)

    # --- Filter beats to ensure tempo is in valid MIDI range ---
    # MIDI tempo is stored as microseconds per beat (max 0xffffff = 16,777,215)
    # This gives minimum BPM of 60,000,000 / 16,777,215 ≈ 3.58
    MIN_TEMPO_BPM = 3.58
    MAX_TEMPO_BPM = 300.0

    filtered_beat_times = [beat_times_raw[0]]  # Always include first beat
    current_idx = 0

    while current_idx < len(beat_times_raw) - 1:
        # Try to find next beat that gives valid tempo
        found_valid = False
        for next_idx in range(current_idx + 1, len(beat_times_raw)):
            beat_duration = beat_times_raw[next_idx] - beat_times_raw[current_idx]
            tempo = 60.0 / beat_duration

            if MIN_TEMPO_BPM <= tempo <= MAX_TEMPO_BPM:
                # Found valid tempo, add this beat
                filtered_beat_times.append(beat_times_raw[next_idx])
                current_idx = next_idx
                found_valid = True
                break

        if not found_valid:
            # No valid tempo found in remaining beats, stop here
            print(f"Stopping at beat {len(filtered_beat_times)} - no valid tempo found for remaining beats")
            break

    # Convert to numpy array
    beat_times = np.array(filtered_beat_times)
    n_beats = len(beat_times)

    # --- Calculate tempo from filtered beat times ---
    tempo_bpm_midi = np.zeros(n_beats)

    for i in range(n_beats - 1):
        beat_duration = beat_times[i+1] - beat_times[i]
        tempo_bpm_midi[i] = 60.0 / beat_duration

    # Last beat uses same tempo as second-to-last
    if n_beats > 1:
        tempo_bpm_midi[-1] = tempo_bpm_midi[-2]
    else:
        tempo_bpm_midi[-1] = mean_tempo_bpm

    n_bars = n_beats / 4.0
    print(f"Total beats generated: {n_beats}")
    print(f"Total bars (at 4/4): {n_bars:.1f}")
    print(f"Tempo range: {tempo_bpm_midi.min():.1f} - {tempo_bpm_midi.max():.1f} BPM")
    print(f"Average tempo: {tempo_bpm_midi.mean():.1f} BPM")

    # --- Calculate frame numbers for beats ---
    beat_frames_array = np.interp(beat_times, time, frames).astype(int)

    # --- Build CSV ---
    bars_arr = (np.arange(n_beats) // 4) + 1
    beat_in_bar = (np.arange(n_beats) % 4) + 1

    beat_df = pd.DataFrame({
        "beat_index": np.arange(1, n_beats + 1),
        "bar": bars_arr, 
        
        "beat_in_bar": beat_in_bar,
        "time_sec": beat_times,
        "frame": beat_frames_array,
        "tempo_bpm": tempo_bpm_midi,
    })

    beat_df.to_csv(csv_out_path, index=False)

    # --- Build MIDI tempo map ---
    # Create MIDI file with tempo changes and notes at each beat
    midi_file = mido.MidiFile(ticks_per_beat=division)

    # Track 1: Tempo map with notes
    track = mido.MidiTrack()
    midi_file.tracks.append(track)

    # Add track name
    track.append(mido.MetaMessage('track_name', name='Tempo Map', time=0))

    # Add time signature (4/4) - required by most DAWs
    track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4,
                                 clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))

    # Add key signature (C major) - optional but recommended
    track.append(mido.MetaMessage('key_signature', key='C', time=0))

    # First tempo event at time 0
    # This tempo controls the duration until beat 1
    first_tempo = tempo_bpm_midi[0]
    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(first_tempo), time=0))

    # Add a note-on event for middle C at the start
    track.append(mido.Message('note_on', note=60, velocity=64, time=0))

    # Add a note-off event one quarter note later
    track.append(mido.Message('note_off', note=60, velocity=0, time=division))

    # Subsequent tempo events: one per beat, delta = one quarter note
    # Tempo at beat k controls duration from beat k to beat k+1
    for t in tempo_bpm_midi[1:]:
        # Tempo change at delta-time = 0 (simultaneous with note-off from previous beat)
        track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(t), time=0))

        # Note-on for middle C
        track.append(mido.Message('note_on', note=60, velocity=64, time=0))

        # Note-off one quarter note later
        track.append(mido.Message('note_off', note=60, velocity=0, time=division))

    # Track 2: CC 1 - Normal (min tempo → 0, max tempo → 127)
    cc_track_normal = mido.MidiTrack()
    midi_file.tracks.append(cc_track_normal)
    cc_track_normal.append(mido.MetaMessage('track_name', name='CC1 Normal', time=0))

    # Scale tempo to CC range 0-127
    tempo_min = tempo_bpm_midi.min()
    tempo_max = tempo_bpm_midi.max()
    tempo_range = tempo_max - tempo_min

    # First CC event
    cc_value = int(np.round((tempo_bpm_midi[0] - tempo_min) / tempo_range * 127))
    cc_value = np.clip(cc_value, 0, 127)
    cc_track_normal.append(mido.Message('control_change', control=1, value=cc_value, time=0))

    # Subsequent CC events at each beat
    for t in tempo_bpm_midi[1:]:
        cc_value = int(np.round((t - tempo_min) / tempo_range * 127))
        cc_value = np.clip(cc_value, 0, 127)
        cc_track_normal.append(mido.Message('control_change', control=1, value=cc_value, time=division))

    # Track 3: CC 1 - Inverted (min tempo → 127, max tempo → 0)
    cc_track_inverted = mido.MidiTrack()
    midi_file.tracks.append(cc_track_inverted)
    cc_track_inverted.append(mido.MetaMessage('track_name', name='CC1 Inverted', time=0))

    # First CC event (inverted)
    cc_value = int(np.round((tempo_max - tempo_bpm_midi[0]) / tempo_range * 127))
    cc_value = np.clip(cc_value, 0, 127)
    cc_track_inverted.append(mido.Message('control_change', control=1, value=cc_value, time=0))

    # Subsequent CC events at each beat (inverted)
    for t in tempo_bpm_midi[1:]:
        cc_value = int(np.round((tempo_max - t) / tempo_range * 127))
        cc_value = np.clip(cc_value, 0, 127)
        cc_track_inverted.append(mido.Message('control_change', control=1, value=cc_value, time=division))

    # Save MIDI file
    midi_file.save(midi_out_path)

    # Total video duration
    T_total = beat_times[-1]

    return beat_df, csv_out_path, midi_out_path, T_total


def compute_beat_tempos_from_inverse(
    input_path,
    mean_tempo_bpm=64.0,
    fps=30.0,
    division=480,
    cc_subsample=30,
    csv_out_path="beat_tempos_inverse.csv",
    midi_out_path="tempo_map_inverse.mid",
):
    """
    Build a tempo map where tempo is inversely proportional to y_values.

    Algorithm:
      1. Load y_values from file
      2. Calculate proportionality constant k to achieve target beats
      3. Calculate frame-level tempo = k / y_value[i]
      4. Apply window extension to keep all tempos in MIDI-valid range
      5. Accumulate beats across frames
      6. Generate CSV with beat-level data
      7. Generate MIDI file with:
         - Frame-level tempo changes
         - Beat notes
         - Sub-sampled CC tracks

    Parameters:
      - input_path: Path to y_values.py file
      - mean_tempo_bpm: Desired average tempo in BPM
      - fps: Video frame rate
      - division: MIDI ticks per beat
      - cc_subsample: CC track sub-sampling interval (frames)
      - csv_out_path: Output CSV path
      - midi_out_path: Output MIDI path

    Returns:
      - beat_df: DataFrame with beat-level data
      - csv_out_path: Path to output CSV
      - midi_out_path: Path to output MIDI
      - T_total: Total video duration
    """

    # --- Load y_values ---
    import sys
    import os

    # Load y_values by executing the file directly
    namespace = {}
    with open(input_path, 'r') as f:
        exec(f.read(), namespace)

    # Handle different variable names (y_values or s)
    if 'y_values' in namespace:
        y_values = np.array(namespace['y_values'])
    elif 's' in namespace:
        y_values = np.array(namespace['s'])
    else:
        raise ValueError(f"Could not find 'y_values' or 's' in {input_path}")

    print(f"Loaded y_values.py file: {len(y_values)} frames")
    print(f"Y-values stats: min={y_values.min():.4f}, max={y_values.max():.4f}, mean={y_values.mean():.4f}")

    # --- Calculate video duration ---
    n_frames = len(y_values)
    video_duration = n_frames / fps
    print(f"Total video duration: {video_duration:.2f} seconds")

    # --- Calculate target beats and proportionality constant k ---
    target_beats = int(video_duration * mean_tempo_bpm / 60.0)
    print(f"Target beats (from {mean_tempo_bpm:.1f} BPM): {target_beats}")

    # k = (target_beats * fps * 60) / sum(1/y_values)
    # Handle division by zero: skip zero values
    inverse_sum = 0.0
    for y in y_values:
        if y != 0:
            inverse_sum += 1.0 / y

    k = (target_beats * fps * 60) / inverse_sum
    print(f"Proportionality constant k: {k:.4f}")

    # --- Calculate frame-level tempo with window extension ---
    tempo_bpm = calculate_tempo_with_window_extension(y_values, k, fps)

    print(f"Tempo range: {tempo_bpm.min():.1f} - {tempo_bpm.max():.1f} BPM")
    print(f"Average tempo: {tempo_bpm.mean():.1f} BPM")

    # --- Accumulate beats to find beat positions ---
    beat_accumulator = 0.0
    beat_frames = []  # List of frame indices where beats occur

    for i in range(n_frames):
        # Beats in this frame
        beats_this_frame = tempo_bpm[i] / (fps * 60)
        beat_accumulator += beats_this_frame

        # Emit beats when accumulator crosses integer thresholds
        while beat_accumulator >= 1.0:
            beat_frames.append(i)
            beat_accumulator -= 1.0

    n_beats = len(beat_frames)
    print(f"Total beats generated: {n_beats}")
    print(f"Total bars (at 4/4): {n_beats / 4.0:.1f}")

    # --- Build CSV with frame-level data ---
    # Create DataFrame with one row per frame
    frame_data = []
    for i in range(n_frames):
        time_sec = i / fps
        minutes = int(time_sec // 60)
        seconds = time_sec % 60
        time_str = f"{minutes}:{seconds:05.2f}"

        frame_data.append({
            'frame': i,
            'time': time_str,
            'tempo_bpm': tempo_bpm[i]
        })

    frame_df = pd.DataFrame(frame_data)
    frame_df.to_csv(csv_out_path, index=False)
    print(f"CSV written to: {csv_out_path}")

    # --- Build MIDI file ---
    midi_file = mido.MidiFile(ticks_per_beat=division)

    # Track 1: Tempo map with notes at beats
    track_tempo = mido.MidiTrack()
    midi_file.tracks.append(track_tempo)
    track_tempo.append(mido.MetaMessage('track_name', name='Tempo Map', time=0))

    # Add time signature (4/4) - required by most DAWs
    track_tempo.append(mido.MetaMessage('time_signature', numerator=4, denominator=4,
                                       clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))

    # Add key signature (C major) - optional but recommended
    track_tempo.append(mido.MetaMessage('key_signature', key='C', time=0))

    # Build list of all events with absolute tick positions
    # This ensures proper synchronization between tempo changes and notes
    events = []  # List of (abs_tick, event_type, event_msg)

    # Add initial tempo change at tick 0
    events.append((0, 'tempo', mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo_bpm[0]), time=0)))

    # Calculate how many frames the CC tracks cover
    # CC tracks iterate through windows and sum ticks for each window
    num_windows = len(range(0, n_frames, cc_subsample))
    max_frame_covered = 0
    for i in range(1, num_windows):
        for j in range(cc_subsample):
            frame_idx = (i-1) * cc_subsample + j
            if frame_idx < n_frames:
                max_frame_covered = max(max_frame_covered, frame_idx)

    # Add tempo changes at EVERY frame up to max_frame_covered for detailed tempo map
    accumulated_ticks = 0
    for i in range(1, max_frame_covered + 1):
        # Calculate ticks from previous frame
        accumulated_ticks += calculate_midi_delta_ticks(tempo_bpm[i-1], fps, division)

        # Add tempo change at this frame
        events.append((accumulated_ticks, 'tempo',
            mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo_bpm[i]), time=0)))

    # Maximum track length is the accumulated ticks
    max_track_ticks = accumulated_ticks

    # Add beat notes at their actual beat positions
    for beat_frame in beat_frames:
        # Calculate ticks to this beat
        ticks_to_beat = 0
        for j in range(beat_frame):
            ticks_to_beat += calculate_midi_delta_ticks(tempo_bpm[j], fps, division)

        events.append((ticks_to_beat, 'note_on',
            mido.Message('note_on', note=60, velocity=64, time=0)))

        # Add note_off, clamping to track length if necessary
        note_off_tick = min(ticks_to_beat + division, max_track_ticks)
        events.append((note_off_tick, 'note_off',
            mido.Message('note_off', note=60, velocity=0, time=0)))

    # Sort events by absolute tick position
    # Priority: tempo changes first, then note_on, then note_off
    events.sort(key=lambda x: (x[0], 0 if x[1] == 'tempo' else 1 if x[1] == 'note_on' else 2))

    # Convert absolute ticks to delta ticks and add to track
    last_tick = 0
    for abs_tick, event_type, msg in events:
        delta = abs_tick - last_tick
        msg.time = delta
        track_tempo.append(msg)
        last_tick = abs_tick

    # Track 2: CC1 Normal (sub-sampled)
    cc_track_normal = mido.MidiTrack()
    midi_file.tracks.append(cc_track_normal)
    cc_track_normal.append(mido.MetaMessage('track_name', name='CC1 Normal', time=0))

    # Sub-sample tempo values
    tempo_subsampled_indices = range(0, n_frames, cc_subsample)
    tempo_subsampled = tempo_bpm[tempo_subsampled_indices]

    # Find min and max for scaling
    tempo_min = np.min(tempo_bpm)
    tempo_max = np.max(tempo_bpm)
    tempo_range = tempo_max - tempo_min

    if tempo_range > 0:
        cc_values_normal = ((tempo_subsampled - tempo_min) / tempo_range * 127).astype(int)
    else:
        cc_values_normal = np.full(len(tempo_subsampled), 64, dtype=int)
    cc_values_normal = np.clip(cc_values_normal, 0, 127)

    # Add first CC event
    cc_track_normal.append(mido.Message('control_change', control=1, value=cc_values_normal[0], time=0))

    # Add subsequent CC events
    for i in range(1, len(cc_values_normal)):
        # Calculate delta ticks for cc_subsample frames
        delta_ticks = 0
        for j in range(cc_subsample):
            frame_idx = (i-1) * cc_subsample + j
            if frame_idx < n_frames:
                delta_ticks += calculate_midi_delta_ticks(tempo_bpm[frame_idx], fps, division)

        cc_track_normal.append(mido.Message('control_change', control=1, value=cc_values_normal[i], time=delta_ticks))

    # Track 3: CC1 Inverted (sub-sampled)
    cc_track_inverted = mido.MidiTrack()
    midi_file.tracks.append(cc_track_inverted)
    cc_track_inverted.append(mido.MetaMessage('track_name', name='CC1 Inverted', time=0))

    if tempo_range > 0:
        cc_values_inverted = ((tempo_max - tempo_subsampled) / tempo_range * 127).astype(int)
    else:
        cc_values_inverted = np.full(len(tempo_subsampled), 64, dtype=int)
    cc_values_inverted = np.clip(cc_values_inverted, 0, 127)

    # Add first CC event
    cc_track_inverted.append(mido.Message('control_change', control=1, value=cc_values_inverted[0], time=0))

    # Add subsequent CC events
    for i in range(1, len(cc_values_inverted)):
        # Calculate delta ticks for cc_subsample frames
        delta_ticks = 0
        for j in range(cc_subsample):
            frame_idx = (i-1) * cc_subsample + j
            if frame_idx < n_frames:
                delta_ticks += calculate_midi_delta_ticks(tempo_bpm[frame_idx], fps, division)

        cc_track_inverted.append(mido.Message('control_change', control=1, value=cc_values_inverted[i], time=delta_ticks))

    # Calculate binary tempo threshold vectors for additional CC tracks
    # Create binary vectors based on tempo thresholds
    binary_fast = (tempo_bpm > 150).astype(float)     # Fast: tempo > 150
    binary_medium = ((tempo_bpm >= 100) & (tempo_bpm <= 150)).astype(float)  # Medium: 100 <= tempo <= 150
    binary_slow = (tempo_bpm < 100).astype(float)     # Slow: tempo < 100

    # Apply 60-frame (2 second) boxcar smoothing
    boxcar_window = 30
    from scipy.ndimage import uniform_filter1d

    # Smooth the binary vectors
    smooth_fast = uniform_filter1d(binary_fast, size=boxcar_window, mode='nearest')
    smooth_medium = uniform_filter1d(binary_medium, size=boxcar_window, mode='nearest')
    smooth_slow = uniform_filter1d(binary_slow, size=boxcar_window, mode='nearest')

    # Sub-sample to match CC track resolution
    smooth_fast_subsampled = smooth_fast[tempo_subsampled_indices]
    smooth_medium_subsampled = smooth_medium[tempo_subsampled_indices]
    smooth_slow_subsampled = smooth_slow[tempo_subsampled_indices]

    # Scale smoothed values to 0-127 range
    cc_fast = (smooth_fast_subsampled * 127).astype(int)
    cc_medium = (smooth_medium_subsampled * 127).astype(int)
    cc_slow = (smooth_slow_subsampled * 127).astype(int)

    cc_fast = np.clip(cc_fast, 0, 127)
    cc_medium = np.clip(cc_medium, 0, 127)
    cc_slow = np.clip(cc_slow, 0, 127)

    # Inverted versions
    cc_fast_inv = 127 - cc_fast
    cc_medium_inv = 127 - cc_medium
    cc_slow_inv = 127 - cc_slow

    # Track 4: CC1 Fast (tempo > 150 BPM, smoothed)
    cc_track_fast = mido.MidiTrack()
    midi_file.tracks.append(cc_track_fast)
    cc_track_fast.append(mido.MetaMessage('track_name', name='CC1 Fast', time=0))
    cc_track_fast.append(mido.Message('control_change', control=1, value=cc_fast[0], time=0))
    for i in range(1, len(cc_fast)):
        delta_ticks = 0
        for j in range(cc_subsample):
            frame_idx = (i-1) * cc_subsample + j
            if frame_idx < n_frames:
                delta_ticks += calculate_midi_delta_ticks(tempo_bpm[frame_idx], fps, division)
        cc_track_fast.append(mido.Message('control_change', control=1, value=cc_fast[i], time=delta_ticks))

    # Track 5: CC1 Medium (100 <= tempo <= 150 BPM, smoothed)
    cc_track_medium = mido.MidiTrack()
    midi_file.tracks.append(cc_track_medium)
    cc_track_medium.append(mido.MetaMessage('track_name', name='CC1 Medium', time=0))
    cc_track_medium.append(mido.Message('control_change', control=1, value=cc_medium[0], time=0))
    for i in range(1, len(cc_medium)):
        delta_ticks = 0
        for j in range(cc_subsample):
            frame_idx = (i-1) * cc_subsample + j
            if frame_idx < n_frames:
                delta_ticks += calculate_midi_delta_ticks(tempo_bpm[frame_idx], fps, division)
        cc_track_medium.append(mido.Message('control_change', control=1, value=cc_medium[i], time=delta_ticks))

    # Track 6: CC1 Slow (tempo < 100 BPM, smoothed)
    cc_track_slow = mido.MidiTrack()
    midi_file.tracks.append(cc_track_slow)
    cc_track_slow.append(mido.MetaMessage('track_name', name='CC1 Slow', time=0))
    cc_track_slow.append(mido.Message('control_change', control=1, value=cc_slow[0], time=0))
    for i in range(1, len(cc_slow)):
        delta_ticks = 0
        for j in range(cc_subsample):
            frame_idx = (i-1) * cc_subsample + j
            if frame_idx < n_frames:
                delta_ticks += calculate_midi_delta_ticks(tempo_bpm[frame_idx], fps, division)
        cc_track_slow.append(mido.Message('control_change', control=1, value=cc_slow[i], time=delta_ticks))

    # Track 7: CC1 Fast Inverted
    cc_track_fast_inv = mido.MidiTrack()
    midi_file.tracks.append(cc_track_fast_inv)
    cc_track_fast_inv.append(mido.MetaMessage('track_name', name='CC1 Fast-Inv', time=0))
    cc_track_fast_inv.append(mido.Message('control_change', control=1, value=cc_fast_inv[0], time=0))
    for i in range(1, len(cc_fast_inv)):
        delta_ticks = 0
        for j in range(cc_subsample):
            frame_idx = (i-1) * cc_subsample + j
            if frame_idx < n_frames:
                delta_ticks += calculate_midi_delta_ticks(tempo_bpm[frame_idx], fps, division)
        cc_track_fast_inv.append(mido.Message('control_change', control=1, value=cc_fast_inv[i], time=delta_ticks))

    # Track 8: CC1 Medium Inverted
    cc_track_medium_inv = mido.MidiTrack()
    midi_file.tracks.append(cc_track_medium_inv)
    cc_track_medium_inv.append(mido.MetaMessage('track_name', name='CC1 Medium-Inv', time=0))
    cc_track_medium_inv.append(mido.Message('control_change', control=1, value=cc_medium_inv[0], time=0))
    for i in range(1, len(cc_medium_inv)):
        delta_ticks = 0
        for j in range(cc_subsample):
            frame_idx = (i-1) * cc_subsample + j
            if frame_idx < n_frames:
                delta_ticks += calculate_midi_delta_ticks(tempo_bpm[frame_idx], fps, division)
        cc_track_medium_inv.append(mido.Message('control_change', control=1, value=cc_medium_inv[i], time=delta_ticks))

    # Track 9: CC1 Slow Inverted
    cc_track_slow_inv = mido.MidiTrack()
    midi_file.tracks.append(cc_track_slow_inv)
    cc_track_slow_inv.append(mido.MetaMessage('track_name', name='CC1 Slow-Inv', time=0))
    cc_track_slow_inv.append(mido.Message('control_change', control=1, value=cc_slow_inv[0], time=0))
    for i in range(1, len(cc_slow_inv)):
        delta_ticks = 0
        for j in range(cc_subsample):
            frame_idx = (i-1) * cc_subsample + j
            if frame_idx < n_frames:
                delta_ticks += calculate_midi_delta_ticks(tempo_bpm[frame_idx], fps, division)
        cc_track_slow_inv.append(mido.Message('control_change', control=1, value=cc_slow_inv[i], time=delta_ticks))

    # Save MIDI file
    midi_file.save(midi_out_path)
    print(f"MIDI written to: {midi_out_path}")

    T_total = video_duration

    return frame_df, csv_out_path, midi_out_path, T_total


if __name__ == "__main__":
    import os

    # Define paths and parameters
    input_dir = os.path.expanduser("~/video/data/input")
    output_dir = os.path.expanduser("~/video/data/output")

    # Use y_values.py file as input
    y_values_path = os.path.join(input_dir, "N30_T7a_speed.py")

    mean_tempo_bpm = 96.0

    csv_out_path = os.path.join(output_dir, f"beat_tempos_inverse_{mean_tempo_bpm:.0f}bpm.csv")
    midi_out_path = os.path.join(output_dir, f"tempo_map_inverse_{mean_tempo_bpm:.0f}bpm.mid")

    beat_df, csv_path_out, midi_path_out, T_total = compute_beat_tempos_from_inverse(
        input_path=y_values_path,
        mean_tempo_bpm=mean_tempo_bpm,
        fps=30.0,
        division=480,
        cc_subsample=30,
        csv_out_path=csv_out_path,
        midi_out_path=midi_out_path,
    )

    print(f"Total duration (s): {T_total:.2f}")
    print(f"CSV written to: {csv_path_out}")
    print(f"MIDI tempo map written to: {midi_path_out}")
