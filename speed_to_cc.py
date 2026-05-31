import numpy as np
import mido
import struct
import sys
import os


def read_speed_data(filepath):
    """
    Read speed data from a .py, .csv, or .kfs file.
    Returns a 1D numpy array of speed values.
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.kfs':
        with open(filepath, 'rb') as f:
            nb_points_data = f.read(4)
            nb_points = struct.unpack('<i', nb_points_data)[0]
            points = []
            for _ in range(nb_points):
                point_data = f.read(8)
                x, y = struct.unpack('<ff', point_data)
                points.append(y)
        return np.array(points)

    elif ext == '.csv':
        import csv
        values = []
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                for val in row:
                    val = val.strip()
                    if val:
                        values.append(float(val))
        return np.array(values)

    elif ext == '.py':
        namespace = {}
        with open(filepath, 'r') as f:
            exec(f.read(), namespace)
        if 'y_values' in namespace:
            return np.array(namespace['y_values'])
        elif 's' in namespace:
            return np.array(namespace['s'])
        raise ValueError(f"Could not find 'y_values' or 's' in {filepath}")

    else:
        raise ValueError(f"Unsupported file extension: {ext} (expected .py, .csv, or .kfs)")


def scale_to_cc(values):
    """Scale an array to 0-127 integer range."""
    vmin = np.min(values)
    vmax = np.max(values)
    vrange = vmax - vmin
    if vrange > 0:
        scaled = ((values - vmin) / vrange * 127).astype(int)
    else:
        scaled = np.full(len(values), 64, dtype=int)
    return np.clip(scaled, 0, 127)


def make_cc_track(name, cc_values, ticks_per_frame, tempo_us=None):
    """Create a MIDI track with CC1 messages, one per frame.

    ticks_per_frame is a float; per-event tick positions are accumulated in
    floating point and rounded only when computing the integer delta for each
    MIDI message, so cumulative timing does not drift.
    """
    track = mido.MidiTrack()
    track.append(mido.MetaMessage('track_name', name=name, time=0))
    if tempo_us is not None:
        track.append(mido.MetaMessage('set_tempo', tempo=tempo_us, time=0))
    track.append(mido.Message('control_change', control=1, value=int(cc_values[0]), time=0))
    prev_tick = 0
    for i in range(1, len(cc_values)):
        abs_tick = int(round(i * ticks_per_frame))
        delta = abs_tick - prev_tick
        track.append(mido.Message('control_change', control=1, value=int(cc_values[i]), time=delta))
        prev_tick = abs_tick
    return track


def speed_to_cc_midi(input_path, tempo_bpm, output_path):
    fps = 30.0
    division = 480

    speed = read_speed_data(input_path)
    n_frames = len(speed)
    print(f"Loaded {n_frames} frames from {input_path}")
    print(f"Speed stats: min={speed.min():.4f}, max={speed.max():.4f}, mean={speed.mean():.4f}")

    ticks_per_frame = (tempo_bpm / 60.0) * (1.0 / fps) * division
    tempo_us = mido.bpm2tempo(tempo_bpm)

    # Track 1: proportional to speed
    cc_speed = scale_to_cc(speed)

    # Track 2: proportional to inverse of speed
    inverse_speed = 1.0 / (speed + 1e-10)
    cc_inverse = scale_to_cc(inverse_speed)

    # Track 3: percentile rank of speed
    from scipy.stats import rankdata
    ranks = rankdata(speed, method='average')
    percentile_rank = (ranks - 1) / (n_frames - 1)  # 0 to 1
    cc_percentile = scale_to_cc(percentile_rank)

    # Tracks 4, 5, 6: inverted versions
    cc_speed_inv = 127 - cc_speed
    cc_inverse_inv = 127 - cc_inverse
    cc_percentile_inv = 127 - cc_percentile

    # Build MIDI file. set_tempo lives on the first track so any DAW reading
    # the file knows the intended tempo without guessing.
    midi_file = mido.MidiFile(ticks_per_beat=division)

    midi_file.tracks.append(make_cc_track('CC1 Speed', cc_speed, ticks_per_frame, tempo_us=tempo_us))
    midi_file.tracks.append(make_cc_track('CC1 Inverse Speed', cc_inverse, ticks_per_frame))
    midi_file.tracks.append(make_cc_track('CC1 Speed Percentile', cc_percentile, ticks_per_frame))
    midi_file.tracks.append(make_cc_track('CC1 Speed Inverted', cc_speed_inv, ticks_per_frame))
    midi_file.tracks.append(make_cc_track('CC1 Inverse Speed Inverted', cc_inverse_inv, ticks_per_frame))
    midi_file.tracks.append(make_cc_track('CC1 Speed Percentile Inverted', cc_percentile_inv, ticks_per_frame))

    midi_file.save(output_path)
    print(f"MIDI written to {output_path} ({len(midi_file.tracks)} tracks, {n_frames} frames, "
          f"tempo {tempo_bpm} BPM, ticks/frame≈{ticks_per_frame:.3f})")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <input_file> <tempo_bpm> [output.mid]")
        print(f"  input_file: .py, .csv, or .kfs file with speed data")
        print(f"  tempo_bpm:  tempo in beats per minute")
        print(f"  output.mid: optional output path (default: speed_cc.mid)")
        sys.exit(1)

    input_file = sys.argv[1]
    tempo_bpm = float(sys.argv[2])
    output_file = sys.argv[3] if len(sys.argv) > 3 else "speed_cc.mid"

    speed_to_cc_midi(input_file, tempo_bpm, output_file)
