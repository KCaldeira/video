import numpy as np
import pandas as pd
import mido


def compute_beat_tempos_from_zoom_simple(
    y_values_path="./data/input/y_values.py",
    target_mean_bpm=96.0,
    fps=30.0,
    division=480,
    csv_out_path="beat_tempos_simple.csv",
    midi_out_path="tempo_map_simple.mid",
):
    """
    Build a tempo map where each beat represents equal amount of zooming.

    Algorithm:
      1. Load y_values (inverse of zoom rate) from Python file
      2. Calculate zoom_rate = 1/y for each frame
      3. Distribute beats so each beat has equal "zoom amount"
      4. Calculate actual tempo from beat timing
      5. Generate MIDI file with tempo changes, notes, and CC tracks

    Inputs:
      - y_values_path: Path to Python file containing y_values list
      - target_mean_bpm: Desired mean tempo across entire video (default 96.0)
      - fps: Video frame rate (default 30.0)
      - division: MIDI ticks per beat (default 480)
      - csv_out_path: Output path for CSV file
      - midi_out_path: Output path for MIDI file

    Outputs:
      - CSV with beat-level tempo info
      - Standard MIDI file with tempo events, notes, and CC tracks
    """

    # --- Load y_values ---
    print(f"Loading y_values from: {y_values_path}")
    namespace = {}
    with open(y_values_path, 'r') as f:
        exec(f.read(), namespace)
    y_values = np.array(namespace['y_values'])

    n_frames = len(y_values)
    video_duration_sec = n_frames / fps
    video_duration_min = video_duration_sec / 60.0

    print(f"Loaded {n_frames} frames")
    print(f"Video duration: {video_duration_sec:.2f} seconds ({video_duration_min:.2f} minutes)")

    # --- Calculate zoom rates ---
    # zoom_rate = 1/y (since y is inverse of zoom rate)
    zoom_rates = 1.0 / y_values

    print(f"Zoom rate stats: min={zoom_rates.min():.4f}, max={zoom_rates.max():.4f}, mean={zoom_rates.mean():.4f}")

    # --- Calculate total zoom and target beats ---
    total_zoom = np.sum(zoom_rates)
    target_beats = int(np.round(target_mean_bpm * video_duration_min))
    zoom_per_beat = total_zoom / target_beats

    print(f"Total zoom amount: {total_zoom:.2f}")
    print(f"Target beats: {target_beats}")
    print(f"Zoom per beat: {zoom_per_beat:.4f}")

    # --- Walk through frames and place beats ---
    beat_times = []
    beat_frames = []

    # First beat at time 0
    beat_times.append(0.0)
    beat_frames.append(0.0)

    zoom_accumulator = 0.0
    dt = 1.0 / fps  # Time per frame in seconds

    for frame_num in range(n_frames):
        zoom_this_frame = zoom_rates[frame_num]

        # Track position within frame as we process this frame's zoom
        remaining_zoom = zoom_this_frame

        while remaining_zoom > 0:
            # How much zoom do we need to reach the next beat?
            zoom_needed = zoom_per_beat - zoom_accumulator

            if remaining_zoom >= zoom_needed:
                # We cross a beat threshold in this frame
                # Calculate fractional position within frame where beat occurs
                zoom_used_so_far = zoom_this_frame - remaining_zoom
                fraction_of_frame = (zoom_used_so_far + zoom_needed) / zoom_this_frame if zoom_this_frame > 0 else 0.0

                # Beat time is at start of frame plus fractional position
                beat_time = frame_num * dt + fraction_of_frame * dt
                beat_frame = frame_num + fraction_of_frame

                beat_times.append(beat_time)
                beat_frames.append(beat_frame)

                # Update accumulators
                remaining_zoom -= zoom_needed
                zoom_accumulator = 0.0  # Reset after placing a beat
            else:
                # Not enough zoom left to reach next beat
                zoom_accumulator += remaining_zoom
                remaining_zoom = 0.0

    # Convert to numpy arrays
    beat_times = np.array(beat_times)
    beat_frames = np.array(beat_frames)
    n_beats = len(beat_times)

    print(f"Total beats generated: {n_beats}")
    print(f"Actual mean BPM: {(n_beats / video_duration_min):.2f}")

    # --- Calculate tempo at each beat ---
    # Tempo = 60 / (time between beats)
    tempo_bpm = np.zeros(n_beats)

    # For all beats except the last one, calculate from time to next beat
    for i in range(n_beats - 1):
        time_to_next_beat = beat_times[i + 1] - beat_times[i]
        tempo_bpm[i] = 60.0 / time_to_next_beat

    # Last beat uses same tempo as second-to-last
    tempo_bpm[-1] = tempo_bpm[-2] if n_beats > 1 else target_mean_bpm

    print(f"Tempo range: {tempo_bpm.min():.1f} - {tempo_bpm.max():.1f} BPM")
    print(f"Mean tempo: {tempo_bpm.mean():.1f} BPM")

    # --- Build CSV ---
    bars_arr = (np.arange(n_beats) // 4) + 1
    beat_in_bar = (np.arange(n_beats) % 4) + 1

    beat_df = pd.DataFrame({
        "beat_index": np.arange(1, n_beats + 1),
        "bar": bars_arr,
        "beat_in_bar": beat_in_bar,
        "time_sec": beat_times,
        "frame": beat_frames,
        "tempo_bpm": tempo_bpm,
    })

    beat_df.to_csv(csv_out_path, index=False)
    print(f"CSV written to: {csv_out_path}")

    # --- Build MIDI tempo map ---
    midi_file = mido.MidiFile(ticks_per_beat=division)

    # Track 1: Tempo map with notes
    track = mido.MidiTrack()
    midi_file.tracks.append(track)

    # Add track name
    track.append(mido.MetaMessage('track_name', name='Tempo Map', time=0))

    # First tempo event at time 0
    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo_bpm[0]), time=0))

    # Add a note-on event for middle C at the start
    track.append(mido.Message('note_on', note=60, velocity=64, time=0))

    # Add a note-off event one quarter note later
    track.append(mido.Message('note_off', note=60, velocity=0, time=division))

    # Subsequent tempo events: one per beat, delta = one quarter note
    for t in tempo_bpm[1:]:
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
    tempo_min = tempo_bpm.min()
    tempo_max = tempo_bpm.max()
    tempo_range = tempo_max - tempo_min

    # First CC event
    if tempo_range > 0:
        cc_value = int(np.round((tempo_bpm[0] - tempo_min) / tempo_range * 127))
    else:
        cc_value = 64  # Middle value if no range
    cc_value = np.clip(cc_value, 0, 127)
    cc_track_normal.append(mido.Message('control_change', control=1, value=cc_value, time=0))

    # Subsequent CC events at each beat
    for t in tempo_bpm[1:]:
        if tempo_range > 0:
            cc_value = int(np.round((t - tempo_min) / tempo_range * 127))
        else:
            cc_value = 64
        cc_value = np.clip(cc_value, 0, 127)
        cc_track_normal.append(mido.Message('control_change', control=1, value=cc_value, time=division))

    # Track 3: CC 1 - Inverted (min tempo → 127, max tempo → 0)
    cc_track_inverted = mido.MidiTrack()
    midi_file.tracks.append(cc_track_inverted)
    cc_track_inverted.append(mido.MetaMessage('track_name', name='CC1 Inverted', time=0))

    # First CC event (inverted)
    if tempo_range > 0:
        cc_value = int(np.round((tempo_max - tempo_bpm[0]) / tempo_range * 127))
    else:
        cc_value = 64
    cc_value = np.clip(cc_value, 0, 127)
    cc_track_inverted.append(mido.Message('control_change', control=1, value=cc_value, time=0))

    # Subsequent CC events at each beat (inverted)
    for t in tempo_bpm[1:]:
        if tempo_range > 0:
            cc_value = int(np.round((tempo_max - t) / tempo_range * 127))
        else:
            cc_value = 64
        cc_value = np.clip(cc_value, 0, 127)
        cc_track_inverted.append(mido.Message('control_change', control=1, value=cc_value, time=division))

    # Track 4: CC 2 - Distance from target tempo (127 at target, 0 at max deviation)
    cc_track_distance = mido.MidiTrack()
    midi_file.tracks.append(cc_track_distance)
    cc_track_distance.append(mido.MetaMessage('track_name', name='CC2 Distance from Target', time=0))

    # Calculate distance from target for each tempo
    # For tempo < target: value = 127 * tempo / target
    # For tempo >= target: value = 127 * target / tempo
    def tempo_distance_to_cc(tempo, target):
        if tempo < target:
            # Slower than target
            ratio = tempo / target
        else:
            # Faster than or equal to target
            ratio = target / tempo
        cc_val = int(np.round(127 * ratio))
        return np.clip(cc_val, 0, 127)

    # First CC event
    cc_value = tempo_distance_to_cc(tempo_bpm[0], target_mean_bpm)
    cc_track_distance.append(mido.Message('control_change', control=2, value=cc_value, time=0))

    # Subsequent CC events at each beat
    for t in tempo_bpm[1:]:
        cc_value = tempo_distance_to_cc(t, target_mean_bpm)
        cc_track_distance.append(mido.Message('control_change', control=2, value=cc_value, time=division))

    # Save MIDI file
    midi_file.save(midi_out_path)
    print(f"MIDI file written to: {midi_out_path}")

    return beat_df, csv_out_path, midi_out_path


if __name__ == "__main__":
    import os

    # Define paths
    y_values_path = "./data/input/y_values.py"
    output_dir = "./data/output"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    target_mean_bpm = 96.0

    csv_out_path = os.path.join(output_dir, f"beat_tempos_simple_{target_mean_bpm:.0f}.csv")
    midi_out_path = os.path.join(output_dir, f"tempo_map_simple_{target_mean_bpm:.0f}.mid")

    beat_df, csv_path, midi_path = compute_beat_tempos_from_zoom_simple(
        y_values_path=y_values_path,
        target_mean_bpm=target_mean_bpm,
        fps=30.0,
        division=480,
        csv_out_path=csv_out_path,
        midi_out_path=midi_out_path,
    )

    print("\nFirst 10 beats:")
    print(beat_df.head(10))
    print("\nLast 10 beats:")
    print(beat_df.tail(10))
