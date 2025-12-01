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
    csv_smooth_out_path=None,
    midi_smooth_out_path=None,
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

    # Add the directory containing y_values.py to the path
    sys.path.insert(0, os.path.dirname(input_path))
    # Import the module
    module_name = os.path.basename(input_path).replace('.py', '')
    y_values_module = __import__(module_name)
    y_values = np.array(y_values_module.y_values)

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

    # Add frame-level tempo changes
    track_tempo.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo_bpm[0]), time=0))

    beat_idx = 0
    pending_note_off_ticks = 0  # Track ticks owed from previous note-off

    for i in range(1, n_frames):
        # Calculate delta ticks from previous frame
        delta_ticks = calculate_midi_delta_ticks(tempo_bpm[i-1], fps, division)

        # Subtract any ticks we already advanced from previous note-off
        if pending_note_off_ticks > 0:
            if delta_ticks >= pending_note_off_ticks:
                delta_ticks -= pending_note_off_ticks
                pending_note_off_ticks = 0
            else:
                # Not enough ticks in this frame to account for debt
                pending_note_off_ticks -= delta_ticks
                delta_ticks = 0

        # Only add tempo change if we have positive delta_ticks or this is the last frame before a beat
        if delta_ticks > 0 or (beat_idx < len(beat_frames) and beat_frames[beat_idx] == i):
            track_tempo.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo_bpm[i]), time=delta_ticks))

            # Check if we should add a note at this frame (if beat occurs here)
            if beat_idx < len(beat_frames) and beat_frames[beat_idx] == i:
                # Add note-on and note-off
                track_tempo.append(mido.Message('note_on', note=60, velocity=64, time=0))
                track_tempo.append(mido.Message('note_off', note=60, velocity=0, time=division))
                beat_idx += 1
                # Track that we've advanced the timeline by division ticks
                pending_note_off_ticks = division

    # Track 2: CC1 Normal (sub-sampled)
    cc_track_normal = mido.MidiTrack()
    midi_file.tracks.append(cc_track_normal)
    cc_track_normal.append(mido.MetaMessage('track_name', name='CC1 Normal', time=0))

    # Sub-sample tempo values
    tempo_subsampled_indices = range(0, n_frames, cc_subsample)
    tempo_subsampled = tempo_bpm[tempo_subsampled_indices]

    # Scale to CC range 0-127
    tempo_min = tempo_bpm.min()
    tempo_max = tempo_bpm.max()
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

    # Calculate squared distance values for additional CC tracks
    target_tempo = mean_tempo_bpm

    # Distance squared to target, 2*target, and 0.5*target
    dist_sq_target = (tempo_subsampled - target_tempo) ** 2
    dist_sq_double = (tempo_subsampled - 2 * target_tempo) ** 2
    dist_sq_half = (tempo_subsampled - 0.5 * target_tempo) ** 2

    # Find min and max values for scaling
    min_dist_sq_target = dist_sq_target.min()
    max_dist_sq_target = dist_sq_target.max()
    min_dist_sq_double = dist_sq_double.min()
    max_dist_sq_double = dist_sq_double.max()
    min_dist_sq_half = dist_sq_half.min()
    max_dist_sq_half = dist_sq_half.max()

    # Scale to 0-127 using both min and max
    range_target = max_dist_sq_target - min_dist_sq_target
    range_double = max_dist_sq_double - min_dist_sq_double
    range_half = max_dist_sq_half - min_dist_sq_half

    if range_target > 0:
        cc_fast = 127- ((dist_sq_target - min_dist_sq_target) / range_target * 127).astype(int)
    else:
        cc_fast = np.zeros(len(tempo_subsampled), dtype=int)
    cc_fast = np.clip(cc_fast, 0, 127)

    if range_double > 0:
        cc_medium = 127 - ((dist_sq_double - min_dist_sq_double) / range_double * 127).astype(int)
    else:
        cc_medium = np.zeros(len(tempo_subsampled), dtype=int)
    cc_medium = np.clip(cc_medium, 0, 127)

    if range_half > 0:
        cc_slow = 127 - ((dist_sq_half - min_dist_sq_half) / range_half * 127).astype(int)
    else:
        cc_slow = np.zeros(len(tempo_subsampled), dtype=int)
    cc_slow = np.clip(cc_slow, 0, 127)

    # Inverted versions
    cc_fast_inv = 127 - cc_fast
    cc_medium_inv = 127 - cc_medium
    cc_slow_inv = 127 - cc_slow

    # Track 4: CC1 Fast (distance to target squared)
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

    # Track 5: CC1 Medium (distance to 2*target squared)
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

    # Track 6: CC1 Slow (distance to 0.5*target squared)
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

    # --- Generate smoothed/averaged tempo outputs ---
    if csv_smooth_out_path or midi_smooth_out_path:
        print("\n--- Generating smoothed tempo outputs ---")

        # Calculate averaged tempo for each window
        # Average frames [n, n+cc_subsample-1] and assign to all frames in that window
        tempo_averaged = np.zeros(n_frames)
        for i in range(0, n_frames, cc_subsample):
            window_end = min(i + cc_subsample, n_frames)
            avg_tempo = np.mean(tempo_bpm[i:window_end])
            tempo_averaged[i:window_end] = avg_tempo  # Assign to all frames in window

        print(f"Averaged tempo range: {tempo_averaged.min():.1f} - {tempo_averaged.max():.1f} BPM")
        print(f"Averaged tempo mean: {tempo_averaged.mean():.1f} BPM")

        # Accumulate beats with averaged tempo to find beat positions
        beat_accumulator_smooth = 0.0
        beat_frames_smooth = []

        for i in range(n_frames):
            # Beats in this frame with averaged tempo
            beats_this_frame = tempo_averaged[i] / (fps * 60)
            beat_accumulator_smooth += beats_this_frame

            # Emit beats when accumulator crosses integer thresholds
            while beat_accumulator_smooth >= 1.0:
                beat_frames_smooth.append(i)
                beat_accumulator_smooth -= 1.0

        n_beats_smooth = len(beat_frames_smooth)
        print(f"Total beats (smoothed): {n_beats_smooth}")
        print(f"Total bars (smoothed, at 4/4): {n_beats_smooth / 4.0:.1f}")

        # Generate smoothed CSV if requested - only cc_subsample frames
        if csv_smooth_out_path:
            frame_data_smooth = []
            for i in range(0, n_frames, cc_subsample):
                time_sec = i / fps
                minutes = int(time_sec // 60)
                seconds = time_sec % 60
                time_str = f"{minutes}:{seconds:05.2f}"

                frame_data_smooth.append({
                    'frame': i,
                    'time': time_str,
                    'tempo_bpm': tempo_averaged[i]
                })

            frame_df_smooth = pd.DataFrame(frame_data_smooth)
            frame_df_smooth.to_csv(csv_smooth_out_path, index=False)
            print(f"Smoothed CSV written to: {csv_smooth_out_path}")

        # Generate smoothed MIDI if requested
        if midi_smooth_out_path:
            midi_file_smooth = mido.MidiFile(ticks_per_beat=division)

            # Track 1: Tempo map with notes at beats (averaged tempo)
            # Only emit tempo changes at window boundaries (every cc_subsample frames)
            track_tempo_smooth = mido.MidiTrack()
            midi_file_smooth.tracks.append(track_tempo_smooth)
            track_tempo_smooth.append(mido.MetaMessage('track_name', name='Tempo Map Smoothed', time=0))

            # Add first tempo change at time 0
            track_tempo_smooth.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo_averaged[0]), time=0))

            # Build list of events with absolute tick positions
            events = []  # List of (abs_tick, event_type, event_msg)

            # Add tempo changes at window boundaries (every cc_subsample frames)
            accumulated_ticks = 0
            for idx, window_frame in enumerate(range(0, n_frames, cc_subsample)):
                if idx > 0:
                    # Calculate ticks to this window boundary
                    start_frame = (idx - 1) * cc_subsample
                    end_frame = min(idx * cc_subsample, n_frames)
                    for j in range(start_frame, end_frame):
                        accumulated_ticks += calculate_midi_delta_ticks(tempo_averaged[j], fps, division)

                events.append((accumulated_ticks, 'tempo',
                    mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo_averaged[window_frame]), time=0)))

            # Add beat notes at their actual beat positions
            for beat_frame in beat_frames_smooth:
                # Calculate ticks to this beat
                ticks_to_beat = 0
                for j in range(beat_frame):
                    ticks_to_beat += calculate_midi_delta_ticks(tempo_averaged[j], fps, division)

                events.append((ticks_to_beat, 'note_on',
                    mido.Message('note_on', note=60, velocity=64, time=0)))
                events.append((ticks_to_beat + division, 'note_off',
                    mido.Message('note_off', note=60, velocity=0, time=0)))

            # Sort events by absolute tick position
            events.sort(key=lambda x: (x[0], 0 if x[1] == 'tempo' else 1 if x[1] == 'note_on' else 2))

            # Convert absolute ticks to delta ticks and add to track
            last_tick = 0
            for abs_tick, event_type, msg in events:
                delta = abs_tick - last_tick
                msg.time = delta
                track_tempo_smooth.append(msg)
                last_tick = abs_tick

            # Track 2: CC1 Normal (using averaged values)
            cc_track_normal_smooth = mido.MidiTrack()
            midi_file_smooth.tracks.append(cc_track_normal_smooth)
            cc_track_normal_smooth.append(mido.MetaMessage('track_name', name='CC1 Normal', time=0))

            # Get tempo at each window
            tempo_windows = []
            for i in range(0, n_frames, cc_subsample):
                tempo_windows.append(tempo_averaged[i])

            # Scale to CC range 0-127
            tempo_win_min = min(tempo_windows)
            tempo_win_max = max(tempo_windows)
            tempo_win_range = tempo_win_max - tempo_win_min

            if tempo_win_range > 0:
                cc_values_smooth = [int(((t - tempo_win_min) / tempo_win_range * 127)) for t in tempo_windows]
            else:
                cc_values_smooth = [64] * len(tempo_windows)
            cc_values_smooth = [np.clip(v, 0, 127) for v in cc_values_smooth]

            # Add first CC event
            cc_track_normal_smooth.append(mido.Message('control_change', control=1, value=cc_values_smooth[0], time=0))

            # Add subsequent CC events
            for idx in range(1, len(cc_values_smooth)):
                # Calculate delta ticks for cc_subsample frames
                delta_ticks = 0
                start_frame = (idx - 1) * cc_subsample
                end_frame = min(idx * cc_subsample, n_frames)
                for j in range(start_frame, end_frame):
                    delta_ticks += calculate_midi_delta_ticks(tempo_averaged[j], fps, division)

                cc_track_normal_smooth.append(mido.Message('control_change', control=1, value=cc_values_smooth[idx], time=delta_ticks))

            # Track 3: CC1 Inverted (using averaged values)
            cc_track_inverted_smooth = mido.MidiTrack()
            midi_file_smooth.tracks.append(cc_track_inverted_smooth)
            cc_track_inverted_smooth.append(mido.MetaMessage('track_name', name='CC1 Inverted', time=0))

            if tempo_win_range > 0:
                cc_values_inv_smooth = [int(((tempo_win_max - t) / tempo_win_range * 127)) for t in tempo_windows]
            else:
                cc_values_inv_smooth = [64] * len(tempo_windows)
            cc_values_inv_smooth = [np.clip(v, 0, 127) for v in cc_values_inv_smooth]

            # Add first CC event
            cc_track_inverted_smooth.append(mido.Message('control_change', control=1, value=cc_values_inv_smooth[0], time=0))

            # Add subsequent CC events
            for idx in range(1, len(cc_values_inv_smooth)):
                # Calculate delta ticks for cc_subsample frames
                delta_ticks = 0
                start_frame = (idx - 1) * cc_subsample
                end_frame = min(idx * cc_subsample, n_frames)
                for j in range(start_frame, end_frame):
                    delta_ticks += calculate_midi_delta_ticks(tempo_averaged[j], fps, division)

                cc_track_inverted_smooth.append(mido.Message('control_change', control=1, value=cc_values_inv_smooth[idx], time=delta_ticks))

            # Calculate squared distance values for additional CC tracks (smoothed)
            tempo_windows_arr = np.array(tempo_windows)

            # Distance squared to target, 2*target, and 0.5*target
            dist_sq_target_smooth = (tempo_windows_arr - target_tempo) ** 2
            dist_sq_double_smooth = (tempo_windows_arr - 2 * target_tempo) ** 2
            dist_sq_half_smooth = (tempo_windows_arr - 0.5 * target_tempo) ** 2

            # Find min and max values for scaling
            min_dist_sq_target_s = dist_sq_target_smooth.min()
            max_dist_sq_target_s = dist_sq_target_smooth.max()
            min_dist_sq_double_s = dist_sq_double_smooth.min()
            max_dist_sq_double_s = dist_sq_double_smooth.max()
            min_dist_sq_half_s = dist_sq_half_smooth.min()
            max_dist_sq_half_s = dist_sq_half_smooth.max()

            # Scale to 0-127 using both min and max
            range_target_s = max_dist_sq_target_s - min_dist_sq_target_s
            range_double_s = max_dist_sq_double_s - min_dist_sq_double_s
            range_half_s = max_dist_sq_half_s - min_dist_sq_half_s

            if range_target_s > 0:
                cc_fast_s = ((dist_sq_target_smooth - min_dist_sq_target_s) / range_target_s * 127).astype(int)
            else:
                cc_fast_s = np.zeros(len(tempo_windows), dtype=int)
            cc_fast_s = np.clip(cc_fast_s, 0, 127)

            if range_double_s > 0:
                cc_medium_s = ((dist_sq_double_smooth - min_dist_sq_double_s) / range_double_s * 127).astype(int)
            else:
                cc_medium_s = np.zeros(len(tempo_windows), dtype=int)
            cc_medium_s = np.clip(cc_medium_s, 0, 127)

            if range_half_s > 0:
                cc_slow_s = ((dist_sq_half_smooth - min_dist_sq_half_s) / range_half_s * 127).astype(int)
            else:
                cc_slow_s = np.zeros(len(tempo_windows), dtype=int)
            cc_slow_s = np.clip(cc_slow_s, 0, 127)

            # Inverted versions
            cc_fast_inv_s = 127 - cc_fast_s
            cc_medium_inv_s = 127 - cc_medium_s
            cc_slow_inv_s = 127 - cc_slow_s

            # Track 4: CC1 Fast (distance to target squared)
            cc_track_fast_s = mido.MidiTrack()
            midi_file_smooth.tracks.append(cc_track_fast_s)
            cc_track_fast_s.append(mido.MetaMessage('track_name', name='CC1 Fast', time=0))
            cc_track_fast_s.append(mido.Message('control_change', control=1, value=cc_fast_s[0], time=0))
            for idx in range(1, len(cc_fast_s)):
                delta_ticks = 0
                start_frame = (idx - 1) * cc_subsample
                end_frame = min(idx * cc_subsample, n_frames)
                for j in range(start_frame, end_frame):
                    delta_ticks += calculate_midi_delta_ticks(tempo_averaged[j], fps, division)
                cc_track_fast_s.append(mido.Message('control_change', control=1, value=cc_fast_s[idx], time=delta_ticks))

            # Track 5: CC1 Medium (distance to 2*target squared)
            cc_track_medium_s = mido.MidiTrack()
            midi_file_smooth.tracks.append(cc_track_medium_s)
            cc_track_medium_s.append(mido.MetaMessage('track_name', name='CC1 Medium', time=0))
            cc_track_medium_s.append(mido.Message('control_change', control=1, value=cc_medium_s[0], time=0))
            for idx in range(1, len(cc_medium_s)):
                delta_ticks = 0
                start_frame = (idx - 1) * cc_subsample
                end_frame = min(idx * cc_subsample, n_frames)
                for j in range(start_frame, end_frame):
                    delta_ticks += calculate_midi_delta_ticks(tempo_averaged[j], fps, division)
                cc_track_medium_s.append(mido.Message('control_change', control=1, value=cc_medium_s[idx], time=delta_ticks))

            # Track 6: CC1 Slow (distance to 0.5*target squared)
            cc_track_slow_s = mido.MidiTrack()
            midi_file_smooth.tracks.append(cc_track_slow_s)
            cc_track_slow_s.append(mido.MetaMessage('track_name', name='CC1 Slow', time=0))
            cc_track_slow_s.append(mido.Message('control_change', control=1, value=cc_slow_s[0], time=0))
            for idx in range(1, len(cc_slow_s)):
                delta_ticks = 0
                start_frame = (idx - 1) * cc_subsample
                end_frame = min(idx * cc_subsample, n_frames)
                for j in range(start_frame, end_frame):
                    delta_ticks += calculate_midi_delta_ticks(tempo_averaged[j], fps, division)
                cc_track_slow_s.append(mido.Message('control_change', control=1, value=cc_slow_s[idx], time=delta_ticks))

            # Track 7: CC1 Fast Inverted
            cc_track_fast_inv_s = mido.MidiTrack()
            midi_file_smooth.tracks.append(cc_track_fast_inv_s)
            cc_track_fast_inv_s.append(mido.MetaMessage('track_name', name='CC1 Fast-Inv', time=0))
            cc_track_fast_inv_s.append(mido.Message('control_change', control=1, value=cc_fast_inv_s[0], time=0))
            for idx in range(1, len(cc_fast_inv_s)):
                delta_ticks = 0
                start_frame = (idx - 1) * cc_subsample
                end_frame = min(idx * cc_subsample, n_frames)
                for j in range(start_frame, end_frame):
                    delta_ticks += calculate_midi_delta_ticks(tempo_averaged[j], fps, division)
                cc_track_fast_inv_s.append(mido.Message('control_change', control=1, value=cc_fast_inv_s[idx], time=delta_ticks))

            # Track 8: CC1 Medium Inverted
            cc_track_medium_inv_s = mido.MidiTrack()
            midi_file_smooth.tracks.append(cc_track_medium_inv_s)
            cc_track_medium_inv_s.append(mido.MetaMessage('track_name', name='CC1 Medium-Inv', time=0))
            cc_track_medium_inv_s.append(mido.Message('control_change', control=1, value=cc_medium_inv_s[0], time=0))
            for idx in range(1, len(cc_medium_inv_s)):
                delta_ticks = 0
                start_frame = (idx - 1) * cc_subsample
                end_frame = min(idx * cc_subsample, n_frames)
                for j in range(start_frame, end_frame):
                    delta_ticks += calculate_midi_delta_ticks(tempo_averaged[j], fps, division)
                cc_track_medium_inv_s.append(mido.Message('control_change', control=1, value=cc_medium_inv_s[idx], time=delta_ticks))

            # Track 9: CC1 Slow Inverted
            cc_track_slow_inv_s = mido.MidiTrack()
            midi_file_smooth.tracks.append(cc_track_slow_inv_s)
            cc_track_slow_inv_s.append(mido.MetaMessage('track_name', name='CC1 Slow-Inv', time=0))
            cc_track_slow_inv_s.append(mido.Message('control_change', control=1, value=cc_slow_inv_s[0], time=0))
            for idx in range(1, len(cc_slow_inv_s)):
                delta_ticks = 0
                start_frame = (idx - 1) * cc_subsample
                end_frame = min(idx * cc_subsample, n_frames)
                for j in range(start_frame, end_frame):
                    delta_ticks += calculate_midi_delta_ticks(tempo_averaged[j], fps, division)
                cc_track_slow_inv_s.append(mido.Message('control_change', control=1, value=cc_slow_inv_s[idx], time=delta_ticks))

            # Save smoothed MIDI file
            midi_file_smooth.save(midi_smooth_out_path)
            print(f"Smoothed MIDI written to: {midi_smooth_out_path}")

    # Total video duration
    T_total = video_duration

    return frame_df, csv_out_path, midi_out_path, T_total


if __name__ == "__main__":
    import os

    # Define paths and parameters
    input_dir = os.path.expanduser("~/video/data/input")
    output_dir = os.path.expanduser("~/video/data/output")

    # Use y_values.py file as input
    y_values_path = os.path.join(input_dir, "y_values.py")

    mean_tempo_bpm = 96.0

    csv_out_path = os.path.join(output_dir, f"beat_tempos_inverse_{mean_tempo_bpm:.0f}bpm.csv")
    midi_out_path = os.path.join(output_dir, f"tempo_map_inverse_{mean_tempo_bpm:.0f}bpm.mid")
    csv_smooth_out_path = os.path.join(output_dir, f"beat_tempos_inverse_smooth_{mean_tempo_bpm:.0f}bpm.csv")
    midi_smooth_out_path = os.path.join(output_dir, f"tempo_map_inverse_smooth_{mean_tempo_bpm:.0f}bpm.mid")

    beat_df, csv_path_out, midi_path_out, T_total = compute_beat_tempos_from_inverse(
        input_path=y_values_path,
        mean_tempo_bpm=mean_tempo_bpm,
        fps=30.0,
        division=480,
        cc_subsample=30,
        csv_out_path=csv_out_path,
        midi_out_path=midi_out_path,
        csv_smooth_out_path=csv_smooth_out_path,
        midi_smooth_out_path=midi_smooth_out_path,
    )

    print(f"Total duration (s): {T_total:.2f}")
    print(f"CSV written to: {csv_path_out}")
    print(f"MIDI tempo map written to: {midi_path_out}")
