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

        # Calculate zoom as 1/y_values (since y_values contains inverse zoom rates)
        zoom = 1.0 / y_values

        # Frames are sequential: 0, 1, 2, ..., len(y_values)-1
        frames = np.arange(len(y_values))

        print(f"Loaded y_values.py file: {len(frames)} frames")
    elif use_kfs:
        # Read KFS file
        df = read_kfs_to_dataframe(input_path)
        frames = df["x"].to_numpy()
        zoom = df["y"].to_numpy()
        print(f"Loaded KFS file: {len(frames)} keyframes")
    else:
        # Read CSV file
        df = pd.read_csv(input_path)
        frames = df["frame_count_list"].to_numpy()
        zoom = df["Gray_czd"].to_numpy()
        print(f"Loaded CSV file: {len(frames)} frames")

    print(f"Zoom stats: min={zoom.min():.4f}, max={zoom.max():.4f}, mean={zoom.mean():.4f}")

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
        cumulative_zoom_depth = np.concatenate([[0.0], np.cumsum(zoom)])
        # Need to adjust time and frames arrays to match (prepend 0.0 and 0)
        time = np.concatenate([[0.0], time])
        frames = np.concatenate([[0], frames])
    else:
        # For KFS/CSV mode: integral of zoom over time using trapezoidal integration
        cumulative_zoom_depth = np.zeros_like(time)
        for i in range(1, len(time)):
            dt = time[i] - time[i-1]
            avg_zoom = (zoom[i] + zoom[i-1]) / 2.0
            cumulative_zoom_depth[i] = cumulative_zoom_depth[i-1] + avg_zoom * dt

    print(f"Cumulative zoom depth range: {cumulative_zoom_depth.min():.4f} to {cumulative_zoom_depth.max():.4f}")

    # --- Create evenly spaced points in cumulative zoom depth ---
    zoom_depth_min = cumulative_zoom_depth.min()
    zoom_depth_max = cumulative_zoom_depth.max()
    zoom_depth_beats = np.linspace(zoom_depth_min, zoom_depth_max, target_beats)

    # --- Interpolate to find time at each beat ---
    beat_times_raw = np.interp(zoom_depth_beats, cumulative_zoom_depth, time)

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


if __name__ == "__main__":
    import os

    # Define paths and parameters
    input_dir = os.path.expanduser("~/video/data/input")
    output_dir = os.path.expanduser("~/video/data/output")

    # Use y_values.py file as input
    y_values_path = os.path.join(input_dir, "y_values.py")

    mean_tempo_bpm = 96.0

    csv_out_path = os.path.join(output_dir, f"beat_tempos_yvalues_{mean_tempo_bpm:.0f}bpm.csv")
    midi_out_path = os.path.join(output_dir, f"tempo_map_yvalues_{mean_tempo_bpm:.0f}bpm.mid")

    beat_df, csv_path_out, midi_path_out, T_total = compute_beat_tempos_from_zoom(
        input_path=y_values_path,
        mean_tempo_bpm=mean_tempo_bpm,
        fps=30.0,
        division=480,
        csv_out_path=csv_out_path,
        midi_out_path=midi_out_path,
        use_y_values=True,
    )

    print(f"Total duration (s): {T_total:.2f}")
    print(f"CSV written to: {csv_path_out}")
    print(f"MIDI tempo map written to: {midi_path_out}")
