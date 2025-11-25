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
    target_beats=500,
    fps=30.0,
    division=480,
    csv_out_path="beat_tempos.csv",
    midi_out_path="tempo_map.mid",
    use_kfs=True,
):
    """
    Build a tempo map with beats evenly spaced in log(zoom depth).

    Algorithm:
      1. Load frame and speed/zoom data (from KFS or CSV)
      2. Compute cumulative zoom depth (integral of zoom over time)
      3. Create evenly spaced points in log(zoom depth)
      4. Interpolate to find time at each beat
      5. Calculate tempo from time differences between consecutive beats
      6. Generate MIDI file with tempo changes

    Inputs:
      - input_path: KFS file with speed data OR CSV with frame_count_list and Gray_czd columns
      - target_beats: desired number of beats (default 500)
      - fps: video frame rate (default 30.0)
      - division: MIDI ticks per beat (default 480)
      - use_kfs: if True, read KFS file; if False, read CSV file

    Outputs:
      - CSV with beat-level tempo info
      - Standard MIDI file with tempo events at every beat
    """

    # --- Load data ---
    if use_kfs:
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

    # --- Compute cumulative zoom depth (integral of zoom over time) ---
    # Use trapezoidal integration
    cumulative_zoom_depth = np.zeros_like(time)
    for i in range(1, len(time)):
        dt = time[i] - time[i-1]
        avg_zoom = (zoom[i] + zoom[i-1]) / 2.0
        cumulative_zoom_depth[i] = cumulative_zoom_depth[i-1] + avg_zoom * dt

    # Handle edge case: if cumulative zoom depth starts at 0, add small offset
    if cumulative_zoom_depth[0] == 0:
        cumulative_zoom_depth = cumulative_zoom_depth + 1e-10

    print(f"Cumulative zoom depth range: {cumulative_zoom_depth.min():.4f} to {cumulative_zoom_depth.max():.4f}")

    # --- Take log of zoom depth ---
    log_zoom_depth = np.log(cumulative_zoom_depth)
    print(f"Log zoom depth range: {log_zoom_depth.min():.4f} to {log_zoom_depth.max():.4f}")

    # --- Create evenly spaced points in log(zoom depth) ---
    log_zoom_min = log_zoom_depth.min()
    log_zoom_max = log_zoom_depth.max()
    log_zoom_beats = np.linspace(log_zoom_min, log_zoom_max, target_beats)

    # --- Interpolate to find time at each beat ---
    beat_times = np.interp(log_zoom_beats, log_zoom_depth, time)

    # --- Calculate tempo from time differences ---
    # tempo_at_beat_N = 60 / (time[N+1] - time[N])
    # For the last beat, use the tempo from the previous beat
    n_beats = len(beat_times)
    tempo_bpm_midi = np.zeros(n_beats)

    for i in range(n_beats - 1):
        beat_duration = beat_times[i+1] - beat_times[i]
        tempo_bpm_midi[i] = 60.0 / beat_duration

    # Last beat uses same tempo as second-to-last
    tempo_bpm_midi[-1] = tempo_bpm_midi[-2]

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
    output_dir = os.path.expanduser("~/video/data/output/N30_T7a_default")

    # Use KFS file as input
    kfs_path = os.path.join(input_dir, "N30_T7a_speed.kfs")

    target_beats = 500

    csv_out_path = os.path.join(output_dir, f"beat_tempos_kfs_{target_beats}.csv")
    midi_out_path = os.path.join(output_dir, f"tempo_map_kfs_{target_beats}.mid")

    beat_df, csv_path_out, midi_path_out, T_total = compute_beat_tempos_from_zoom(
        input_path=kfs_path,
        target_beats=target_beats,
        fps=30.0,
        division=480,
        csv_out_path=csv_out_path,
        midi_out_path=midi_out_path,
        use_kfs=True,
    )

    print(f"Total duration (s): {T_total:.2f}")
    print(f"CSV written to: {csv_path_out}")
    print(f"MIDI tempo map written to: {midi_path_out}")
