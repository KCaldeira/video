import argparse
import numpy as np
import mido
import struct
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
        elif 'r' in namespace:
            return np.array(namespace['r'])
        raise ValueError(f"Could not find 'y_values', 's', or 'r' in {filepath}")

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


def make_cc_track(name, cc_values, ticks_per_frame, tempo_us=None, note=None):
    """Create a MIDI track with CC1 messages, one per frame.

    ticks_per_frame is a float; per-event tick positions are accumulated in
    floating point and rounded only when computing the integer delta for each
    MIDI message, so cumulative timing does not drift.
    """
    track = mido.MidiTrack()
    track.append(mido.MetaMessage('track_name', name=name, time=0))
    if note is not None:
        track.append(mido.MetaMessage('text', text=note, time=0))
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

    # Label tracks by the source quantity: rotation files ("*_rot.*") vs speed.
    is_rotation = 'rot' in os.path.basename(input_path).lower()
    quantity = 'Rotation' if is_rotation else 'Speed'
    note = 'Convention: positive is counter-clockwise' if is_rotation else None

    speed = read_speed_data(input_path)
    n_frames = len(speed)
    print(f"Loaded {n_frames} frames from {input_path}")
    print(f"{quantity} stats: min={speed.min():.4f}, max={speed.max():.4f}, mean={speed.mean():.4f}")
    if note is not None:
        print(f"Note: {note}")

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

    midi_file.tracks.append(make_cc_track(f'CC1 {quantity}', cc_speed, ticks_per_frame, tempo_us=tempo_us, note=note))
    midi_file.tracks.append(make_cc_track(f'CC1 Inverse {quantity}', cc_inverse, ticks_per_frame))
    midi_file.tracks.append(make_cc_track(f'CC1 {quantity} Percentile', cc_percentile, ticks_per_frame))
    midi_file.tracks.append(make_cc_track(f'CC1 {quantity} Inverted', cc_speed_inv, ticks_per_frame))
    midi_file.tracks.append(make_cc_track(f'CC1 Inverse {quantity} Inverted', cc_inverse_inv, ticks_per_frame))
    midi_file.tracks.append(make_cc_track(f'CC1 {quantity} Percentile Inverted', cc_percentile_inv, ticks_per_frame))

    midi_file.save(output_path)
    print(f"MIDI written to {output_path} ({len(midi_file.tracks)} tracks, {n_frames} frames, "
          f"tempo {tempo_bpm} BPM, ticks/frame≈{ticks_per_frame:.3f})")


DESCRIPTION = """\
Convert a per-frame speed (or rotation) time series into a multi-track MIDI
file of CC1 (modulation) automation, one CC event per video frame at 30 fps.

Six tracks are written, all carrying CC1 on channel 1:

  1. CC1 <Quantity>                     value scaled linearly to 0-127
  2. CC1 Inverse <Quantity>             1/value, scaled to 0-127
  3. CC1 <Quantity> Percentile          percentile rank of value, 0-127
  4. CC1 <Quantity> Inverted            127 - track 1
  5. CC1 Inverse <Quantity> Inverted    127 - track 2
  6. CC1 <Quantity> Percentile Inverted 127 - track 3

Each track is scaled independently so it spans the full 0-127 range.

If the input filename contains "rot", the tracks are labelled "Rotation"
instead of "Speed" and a text note recording the sign convention
(positive is counter-clockwise) is embedded in the first track.

A set_tempo meta message is written to the first track, and event times are
accumulated in floating point and rounded per event, so a long file does not
accumulate timing drift.
"""

EPILOG = """\
Input formats (chosen by file extension):

  .kfs  Blender/keyframe binary: int32 point count, then that many
        (float32 x, float32 y) pairs; the y values are used.
  .csv  Any layout - every non-empty cell in the file is read as one value,
        in row-major order.
  .py   Executed as Python; the first of the variables y_values, s, or r
        that is defined is used as the value list.

Examples:

  python speed_to_cc.py data/input/N45_speed.csv 64
  python speed_to_cc.py data/input/N45_rot.kfs 96 data/output/N45_rot_cc.mid
"""


def main():
    parser = argparse.ArgumentParser(
        prog="speed_to_cc.py",
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_file",
        help="speed/rotation data file (.py, .csv, or .kfs) with one value per frame",
    )
    parser.add_argument(
        "tempo_bpm",
        type=float,
        help="tempo in beats per minute; sets set_tempo and the frames-to-ticks scaling",
    )
    parser.add_argument(
        "output_file",
        nargs="?",
        default="speed_cc.mid",
        help="output MIDI path (default: %(default)s)",
    )
    args = parser.parse_args()

    speed_to_cc_midi(args.input_file, args.tempo_bpm, args.output_file)


if __name__ == "__main__":
    main()
