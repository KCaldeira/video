import argparse
import math
import os

import mido
import pandas as pd

# --- the 16.2 s cycle -------------------------------------------------------
# Each cycle is one 1/4 bar at 75 BPM followed by seven 4/4 bars at 1200/11 BPM.
# Both tempos are exact integers in microseconds per quarter note.
SHORT_US = 800_000            # 75 BPM
LONG_US = 550_000             # 1200/11 = 109.0909... BPM
SHORT_BEATS = 1
LONG_BARS = 7
LONG_BEATS_PER_BAR = 4
LONG_BEATS = LONG_BARS * LONG_BEATS_PER_BAR
BEATS_PER_CYCLE = SHORT_BEATS + LONG_BEATS

FRAMES_PER_SECOND = 30

# 1056 ticks per beat is the smallest value that makes a frame an exact whole
# number of ticks in BOTH tempo sections, so frame positions carry no rounding
# error at all: 1056/24 = 44 ticks per frame at 75 BPM (24 frames per beat),
# and 1056/16.5 = 64 ticks per frame at 1200/11 BPM (16.5 frames per beat).
TICKS_PER_BEAT = 1056
SHORT_TICKS_PER_FRAME = 44
LONG_TICKS_PER_FRAME = 64

SHORT_FRAMES = SHORT_BEATS * SHORT_US * FRAMES_PER_SECOND // 1_000_000      # 24
LONG_FRAMES = LONG_BEATS * LONG_US * FRAMES_PER_SECOND // 1_000_000         # 462
FRAMES_PER_CYCLE = SHORT_FRAMES + LONG_FRAMES                               # 486
TICKS_PER_CYCLE = BEATS_PER_CYCLE * TICKS_PER_BEAT                          # 30624

CC_MAX = 127
RUNNING_SUM_SOURCES = ["R_mean", "G_mean", "B_mean"]
EXCLUDED_COLUMNS = ("frame", "time_sec", "identical")


def frame_to_tick(frame):
    """Absolute tick of a video frame, exactly, with no accumulated drift."""
    cycle, offset = divmod(frame, FRAMES_PER_CYCLE)
    if offset < SHORT_FRAMES:
        within = offset * SHORT_TICKS_PER_FRAME
    else:
        within = (SHORT_FRAMES * SHORT_TICKS_PER_FRAME
                  + (offset - SHORT_FRAMES) * LONG_TICKS_PER_FRAME)
    return cycle * TICKS_PER_CYCLE + within


def build_conductor_track(cycles):
    """Track 0: the repeating tempo and time-signature map."""
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name="Tempo/Sig Map", time=0))

    events = []
    for cycle in range(cycles):
        tick = cycle * TICKS_PER_CYCLE
        events.append((tick, 0, mido.MetaMessage(
            "time_signature", numerator=SHORT_BEATS, denominator=4)))
        events.append((tick, 1, mido.MetaMessage("set_tempo", tempo=SHORT_US)))

        tick += SHORT_BEATS * TICKS_PER_BEAT
        events.append((tick, 0, mido.MetaMessage(
            "time_signature", numerator=LONG_BEATS_PER_BAR, denominator=4)))
        events.append((tick, 1, mido.MetaMessage("set_tempo", tempo=LONG_US)))

    events.sort(key=lambda event: (event[0], event[1]))

    previous = 0
    for abs_tick, _order, message in events:
        message.time = abs_tick - previous
        previous = abs_tick
        track.append(message)

    total = cycles * TICKS_PER_CYCLE
    track.append(mido.MetaMessage("end_of_track", time=total - previous))
    return track


def scale_to_cc(values):
    """Scale a series to integer 0..127, min to 0 and max to 127."""
    low = values.min()
    high = values.max()
    return ((values - low) / (high - low) * CC_MAX).round().astype(int)


def build_cc_track(name, cc_values, ticks, cc_number, channel):
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name=name, time=0))
    previous = 0
    for tick, value in zip(ticks, cc_values):
        track.append(mido.Message(
            "control_change", channel=channel, control=cc_number,
            value=int(value), time=tick - previous))
        previous = tick
    track.append(mido.MetaMessage("end_of_track", time=0))
    return track


def csv_to_cc_midi(csv_path, output_path, cc_number=1, channel=0):
    """
    Turn every data column of a frame-similarity CSV into a pair of MIDI CC
    tracks -- one scaled 0..127, one inverted -- in a single file whose track 0
    carries the 16.2 s tempo map.
    """
    data = pd.read_csv(csv_path)
    print(f"Read {csv_path}: {len(data)} rows, {len(data) / FRAMES_PER_SECOND:.2f} s at "
          f"{FRAMES_PER_SECOND} fps")

    for column in RUNNING_SUM_SOURCES:
        data[column + "_cumsum"] = data[column].cumsum()

    columns = [c for c in data.columns if c not in EXCLUDED_COLUMNS]
    print(f"Data columns ({len(columns)}): {', '.join(columns)}")

    ticks = [frame_to_tick(f) for f in data["frame"]]
    cycles = math.ceil((data["frame"].iloc[-1] + 1) / FRAMES_PER_CYCLE)

    midi = mido.MidiFile(type=1, ticks_per_beat=TICKS_PER_BEAT)
    midi.tracks.append(build_conductor_track(cycles))

    print()
    print(f"{'track':<34}{'source min':>12}{'source max':>12}   CC range")
    for column in columns:
        scaled = scale_to_cc(data[column])
        for name, series in ((column, scaled), (column + " Inverted", CC_MAX - scaled)):
            label = f"CC{cc_number} {name}"
            midi.tracks.append(build_cc_track(label, series, ticks, cc_number, channel))
            print(f"{label:<34}{data[column].min():>12.4f}{data[column].max():>12.4f}"
                  f"   {series.min()}-{series.max()}")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    midi.save(output_path)

    size_mb = os.path.getsize(output_path) / 1e6
    print()
    print(f"Saved: {output_path}  ({size_mb:.2f} MB)")
    print(f"  MIDI type 1, {TICKS_PER_BEAT} ticks per beat")
    print(f"  {len(midi.tracks)} tracks = 1 conductor + {len(columns)} columns x 2")
    print(f"  {len(data)} CC events per track, {len(data) * len(columns) * 2} total")
    cycle_seconds = (SHORT_BEATS * SHORT_US + LONG_BEATS * LONG_US) / 1e6
    print(f"  tempo map: {cycles} cycles of "
          f"[1 bar 1/4 @ 75 BPM + {LONG_BARS} bars 4/4 @ 1200/11 BPM]"
          f" = {cycle_seconds:g} s each, {cycles * cycle_seconds:.1f} s total")
    print(f"  playback length: {midi.length:.3f} s "
          f"(CSV spans {len(data) / FRAMES_PER_SECOND:.3f} s)")

    return midi


DESCRIPTION = """\
Turn every data column of a frame-similarity CSV into MIDI CC automation.

Running-sum columns are added for R_mean, G_mean and B_mean, then each data
column (everything except frame, time_sec and identical) becomes two tracks:
one scaled linearly so the minimum is 0 and the maximum is 127, and one
inverted as 127 minus that. One CC event is written per video frame.

Track 0 is a conductor track carrying the 16.2 s tempo cycle -- one 1/4 bar at
75 BPM followed by seven 4/4 bars at 1200/11 BPM -- so the file is
self-contained and the CC events land at the right wall-clock times without a
separate tempo import.

Ticks per beat is 1056, the smallest value that makes one video frame an exact
whole number of ticks in both tempo sections (44 and 64), so frame positions
carry no rounding error at all.
"""

EPILOG = """\
Examples:

  python csv_to_cc_midi.py data/output/N48_SSSj\\(10\\)a\\(5\\)a6_framesim.csv
  python csv_to_cc_midi.py data/output/N48_framesim.csv --cc 11 -o data/output/N48_cc11.mid
"""


def main():
    parser = argparse.ArgumentParser(
        prog="csv_to_cc_midi.py",
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("csv_path", help="frame-similarity CSV from frame_similarity.py")
    parser.add_argument("-o", "--output", default=None,
                        help="output MIDI path (default: alongside the CSV, _cc.mid)")
    parser.add_argument("--cc", type=int, default=1,
                        help="MIDI controller number (default: %(default)s)")
    parser.add_argument("--channel", type=int, default=0,
                        help="MIDI channel, 0-based (default: %(default)s)")
    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        stem = os.path.splitext(args.csv_path)[0]
        output_path = stem + "_cc.mid"

    csv_to_cc_midi(args.csv_path, output_path, args.cc, args.channel)


if __name__ == "__main__":
    main()
