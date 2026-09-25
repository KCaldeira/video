import argparse
import math
import os
from fractions import Fraction

import mido

TICKS_PER_BEAT = 480

# MIDI stores tempo as a 24-bit microseconds-per-quarter-note value.
MAX_TEMPO_US = 0xFFFFFF


def parse_bpm(text):
    """Parse a tempo given as a decimal or as an exact fraction like 1200/11."""
    return Fraction(text)


def tempo_us(bpm):
    """Microseconds per quarter note for a tempo in beats per minute."""
    exact = Fraction(60_000_000) / bpm
    value = round(exact)
    if value > MAX_TEMPO_US:
        raise ValueError(f"{float(bpm):g} BPM needs {value} us/quarter, above the MIDI limit")
    return value, exact


def build_tempo_track(
    output_path,
    long_bars,
    long_beats_per_bar,
    long_bpm,
    short_beats,
    short_bpm,
    total_minutes,
    expected_cycle_seconds=None,
    short_first=False,
):
    """
    Build a conductor track that alternates a block of long_bars bars with a
    single short bar of short_beats beats, repeating to cover total_minutes.
    """
    long_us, long_exact = tempo_us(long_bpm)
    short_us, short_exact = tempo_us(short_bpm)

    long_beats = long_bars * long_beats_per_bar
    beats_per_cycle = long_beats + short_beats

    # Cycle length from the integer tempos actually written to the file, not
    # from the requested BPM, so the reported figure is what a DAW will play.
    cycle_us = long_beats * long_us + short_beats * short_us
    cycle_seconds = cycle_us / 1e6

    total_seconds = total_minutes * 60
    cycles = math.ceil(total_seconds / cycle_seconds)

    midi = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("track_name", name="Tempo/Sig Map", time=0))

    # (abs_tick, order, message); order breaks ties so time_signature precedes
    # set_tempo at the same tick.
    if short_first:
        segments = [(short_beats, short_beats, short_us), (long_beats, long_beats_per_bar, long_us)]
    else:
        segments = [(long_beats, long_beats_per_bar, long_us), (short_beats, short_beats, short_us)]

    events = []
    for cycle in range(cycles):
        tick = cycle * beats_per_cycle * TICKS_PER_BEAT
        for beats, numerator, tempo in segments:
            events.append((tick, 0, mido.MetaMessage(
                "time_signature", numerator=numerator, denominator=4)))
            events.append((tick, 1, mido.MetaMessage("set_tempo", tempo=tempo)))
            tick += beats * TICKS_PER_BEAT

    events.sort(key=lambda event: (event[0], event[1]))

    previous_tick = 0
    for abs_tick, _order, message in events:
        message.time = abs_tick - previous_tick
        previous_tick = abs_tick
        track.append(message)

    total_ticks = cycles * beats_per_cycle * TICKS_PER_BEAT
    track.append(mido.MetaMessage("end_of_track", time=total_ticks - previous_tick))

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    midi.save(output_path)

    print(f"Saved: {output_path}")
    print(f"Ticks per beat: {TICKS_PER_BEAT}")
    print(f"Cycle order: {'short bar first' if short_first else 'long block first'}")
    print()
    print(f"Long block  ({long_bars} bars of {long_beats_per_bar}/4, {long_beats} beats):")
    print(f"    tempo    = {float(long_bpm):.10g} BPM = {long_us} us/quarter"
          f"{'' if long_exact == long_us else f' (requested {float(long_exact):.3f}, rounded)'}")
    print(f"    duration = {long_beats * long_us / 1e6:.6f} s")
    print()
    print(f"Short bar   ({short_beats}/4, {short_beats} beat{'s' if short_beats != 1 else ''}):")
    print(f"    tempo    = {float(short_bpm):.10g} BPM = {short_us} us/quarter"
          f"{'' if short_exact == short_us else f' (requested {float(short_exact):.3f}, rounded)'}")
    print(f"    duration = {short_beats * short_us / 1e6:.6f} s")
    print()
    print(f"Cycle: {beats_per_cycle} beats = {cycle_seconds:.6f} s"
          f" = {cycle_seconds * 30:.3f} frames at 30 fps")
    if expected_cycle_seconds is not None:
        difference = cycle_seconds - expected_cycle_seconds
        verdict = "MATCH" if abs(difference) < 1e-9 else f"MISMATCH by {difference:+.6f} s"
        print(f"    expected {expected_cycle_seconds} s -> {verdict}")
    print(f"Cycles: {cycles}  ->  total {cycles * cycle_seconds:.3f} s"
          f" = {cycles * cycle_seconds / 60:.3f} min (requested at least {total_minutes} min)")
    print(f"Total beats: {cycles * beats_per_cycle}   Total ticks: {total_ticks}")

    return cycle_seconds, cycles


DESCRIPTION = """\
Build a MIDI conductor track that alternates a block of bars at one tempo with
a single short bar at another, repeating the pattern to cover a requested
duration.

The track carries set_tempo and time_signature meta events, which Cubase
imports into its Tempo Track and Signature Track respectively.

Tempos may be given as exact fractions (1200/11) so that a repeating decimal
is not lost to rounding; the cycle length is reported from the integer
microsecond tempo actually written to the file, so the printed figure is what
a DAW will play.
"""

EPILOG = """\
Examples:

  # 6 bars of 4/4 at 1200/11 BPM, then one 1/4 bar at 20 BPM: a 16.2 s cycle
  python make_tempo_track.py --long-bpm 1200/11 --short-bpm 20 \\
      --minutes 16 --expect-cycle 16.2

  # the same pattern with a 75 BPM short bar (a 14.0 s cycle)
  python make_tempo_track.py --long-bpm 1200/11 --short-bpm 75 --minutes 16
"""


def main():
    parser = argparse.ArgumentParser(
        prog="make_tempo_track.py",
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--long-bars", type=int, default=6,
                        help="bars in the long block (default: %(default)s)")
    parser.add_argument("--long-beats-per-bar", type=int, default=4,
                        help="beats per bar in the long block (default: %(default)s)")
    parser.add_argument("--long-bpm", type=parse_bpm, default=Fraction(1200, 11),
                        help="tempo of the long block, decimal or fraction (default: 1200/11)")
    parser.add_argument("--short-beats", type=int, default=1,
                        help="beats in the short bar (default: %(default)s)")
    parser.add_argument("--short-bpm", type=parse_bpm, default=Fraction(20),
                        help="tempo of the short bar (default: %(default)s)")
    parser.add_argument("--minutes", type=float, default=16.0,
                        help="cover at least this many minutes (default: %(default)s)")
    parser.add_argument("--short-first", action="store_true",
                        help="put the short bar at the start of each cycle instead of the end")
    parser.add_argument("--expect-cycle", type=float, default=None,
                        help="assert the cycle is this many seconds and report match/mismatch")
    parser.add_argument("-o", "--output", default=None,
                        help="output MIDI path (default: data/output/tempo_track_{cycle}s.mid)")
    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        long_beats = args.long_bars * args.long_beats_per_bar
        cycle = (long_beats * tempo_us(args.long_bpm)[0]
                 + args.short_beats * tempo_us(args.short_bpm)[0]) / 1e6
        output_path = os.path.join("data", "output", f"tempo_track_{cycle:g}s.mid")

    build_tempo_track(
        output_path,
        args.long_bars,
        args.long_beats_per_bar,
        args.long_bpm,
        args.short_beats,
        args.short_bpm,
        args.minutes,
        args.expect_cycle,
        args.short_first,
    )


if __name__ == "__main__":
    main()
