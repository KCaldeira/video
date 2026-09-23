#!/usr/bin/env python3
"""
Build a MIDI tempo + time-signature conductor track from a repeating bar
pattern, for import into Cubase.

The tempo is constant; only the time signature changes. A pattern is a list
of bar lengths in beats, e.g. the N49 zoom cycle at 120 bpm:

    translate  3 s   =  6 beats  =  3/4 3/4
    zoom in    7 s   = 14 beats  =  1/4 3/4 3/4 3/4 3/4 1/4
    zoom out   4 s   =  8 beats  =  1/4 3/4 3/4 1/4
                       --------
                       28 beats  = 12 bars = 14 s

giving --pattern 3,3,1,3,3,3,3,1,1,3,3,1. The one-beat bars mark the start
and end of each zoom phase, so bar lines land on the visual phase boundaries.

The pattern repeats enough times to cover --seconds (or the duration of a
video given with --video). Cubase imports set_tempo into its Tempo Track and
time_signature into its Signature Track.

Usage:
    python create_cycle_timesig_track.py --pattern 3,3,1,3,3,3,3,1,1,3,3,1 \\
        --tempo 120 --video "data/input/N49_BSlgnw9a.wmv"
"""

import argparse
import os
import sys

import mido

TICKS_PER_BEAT = 480


def video_duration_seconds(path):
    """Duration of a video file, in seconds."""
    import cv2
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise FileNotFoundError(f"could not open {path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    capture.release()
    if not fps:
        raise ValueError(f"{path} reports no frame rate")
    return frames / fps


def build_track(pattern, tempo_bpm, cycles, denominator):
    """One conductor track: the bar pattern repeated `cycles` times."""
    tempo_us = round(60_000_000 / tempo_bpm)

    midi = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("track_name", name="Tempo/Sig Map", time=0))

    # Collect absolute-tick events first, then convert to deltas. The sort key
    # puts the time signature before the tempo when both fall on one tick.
    events = []
    tick = 0
    for cycle in range(cycles):
        for beats in pattern:
            events.append((tick, 0, mido.MetaMessage(
                "time_signature", numerator=beats, denominator=denominator)))
            if tick == 0:
                # The tempo never changes, so it is stated once at the start.
                events.append((tick, 1, mido.MetaMessage(
                    "set_tempo", tempo=tempo_us)))
            tick += beats * TICKS_PER_BEAT

    events.sort(key=lambda e: (e[0], e[1]))
    previous = 0
    for abs_tick, _, message in events:
        track.append(message.copy(time=abs_tick - previous))
        previous = abs_tick

    track.append(mido.MetaMessage("end_of_track", time=0))
    return midi, tick


def main():
    parser = argparse.ArgumentParser(
        prog="create_cycle_timesig_track.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--pattern", required=True,
        help="comma-separated bar lengths in beats, one cycle "
             "(e.g. 3,3,1,3,3,3,3,1,1,3,3,1)")
    parser.add_argument(
        "--tempo", type=float, required=True, help="tempo in BPM (constant)")
    parser.add_argument(
        "--denominator", type=int, default=4,
        help="time-signature denominator; the pattern gives the numerators "
             "(default: %(default)s)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seconds", type=float,
                       help="cover at least this many seconds")
    group.add_argument("--video",
                       help="cover at least this video's duration")
    parser.add_argument(
        "-o", "--output",
        help="output .mid path (default: data/output/<name>_tempo_timesig.mid)")
    args = parser.parse_args()

    pattern = [int(x) for x in args.pattern.split(",")]
    if any(b < 1 for b in pattern):
        raise ValueError(f"every bar must be at least 1 beat, got {pattern}")

    if args.video:
        target = video_duration_seconds(args.video)
        source = os.path.splitext(os.path.basename(args.video))[0]
        print(f"{args.video}: {target:.3f} s")
    else:
        target = args.seconds
        source = "cycle"

    beats_per_cycle = sum(pattern)
    seconds_per_beat = 60.0 / args.tempo
    cycle_seconds = beats_per_cycle * seconds_per_beat

    # Round up, so the track is at least as long as the target.
    cycles = int(target / cycle_seconds) + 1

    midi, total_ticks = build_track(
        pattern, args.tempo, cycles, args.denominator)
    covered = total_ticks / TICKS_PER_BEAT * seconds_per_beat

    output_path = args.output or os.path.join(
        "data", "output", f"{source}_tempo_timesig.mid")
    midi.save(output_path)

    sig = " ".join(f"{b}/{args.denominator}" for b in pattern)
    print(f"pattern: {sig}")
    print(f"  {len(pattern)} bars, {beats_per_cycle} beats, "
          f"{cycle_seconds:.4f} s per cycle at {args.tempo:g} bpm")
    print(f"  {cycles} cycles -> {cycles * len(pattern)} bars, "
          f"{cycles * beats_per_cycle} beats, {covered:.3f} s")
    print(f"  covers {target:.3f} s with {covered - target:.3f} s to spare")
    print(f"Wrote {output_path}")
    print("Import in Cubase with File > Import > MIDI File "
          "(enable 'Import Tempo Track').")
    return 0


if __name__ == "__main__":
    sys.exit(main())
