#!/usr/bin/env python3
"""
Convert a per-frame speed (or rotation) time series into a .dawproject.

The DAWproject counterpart of speed_to_cc.py: same input files and the same
six derived series, written as volume automation instead of MIDI CC.

The "speed" files hold INVERSE speed (time per unit of motion: large when the
camera is slow), so speed is taken as 1/value before anything is derived.
speed_to_cc.py does not do this, so its "Speed" and "Inverse Speed" tracks
are the other way round.

Six tracks are written, each scaled independently to span 0-1:

  1. <Quantity>                     speed
  2. Inverse <Quantity>             1/speed, i.e. the file's values
  3. <Quantity> Percentile          percentile rank of speed
  4. <Quantity> Inverted            1 - track 1
  5. Inverse <Quantity> Inverted    1 - track 2
  6. <Quantity> Percentile Inverted 1 - track 3

With --clip-fraction F, tracks 1-3 (and so 4-6) are each scaled so that
the F and 1-F quantiles span 0-1, with anything beyond clipped to the bounds.
A brief extreme -- such as a slow-down at the very end -- then no longer
squashes the rest of the curve into a sliver of the range. On the
percentile track this maps percentile F to 0 and 1-F to 1.

If the input filename contains "rot" the tracks are labelled "Rotation"
instead of "Speed", and the values are used as they are, not inverted:
rotation is signed, so it has no meaningful reciprocal. The sign convention
there is that positive is counter-clockwise.

Automation times are wall-clock seconds, `frame_index / fps`, so the curves
line up with the video and with the output of write_dawproject.py. Nothing
positional depends on tempo.

Values are written as linear gain: 0 is -infinity dB and 1.0 is 0 dB.

Binned mode
-----------
With --segment-bars (or --segment-beats) and --bin-fractions, the script
writes a separate, bar-quantized file instead of the six continuous tracks.
The video is cut into segments of that many bars at the constant --tempo and
--time-signature, with video time 0 on bar 1, beat 1. A trailing partial
segment is kept. Each segment's value is the median, over its frames, of the
per-frame percentile rank of speed, so the result does not change under any
monotonic transform of the metric. Segments are ranked by that value and
split into bins holding the given fractions of the segments: with
--bin-fractions 0.5 0.2 0.3, the slowest 50% go to bin 0, the next 20% to
bin 1 and the fastest 30% to bin 2.

Each bin is an empty audio track at unity gain (1.0) during its segments and
0 otherwise. Cubase ignores hold interpolation on import, so each change is
a pair of linear points one tick (1/480 quarter note) apart, which is a step
in practice. An
inverted set ("inv") ranks the segments the other way, so its bin 0 holds the
fastest segments. Each track (trailing " o") has a complement (trailing
" i"), which is at unity exactly when the track is silent. All " o" tracks
come first, then all " i" tracks; within each, the plain set comes before
inv, then bin number: 4 x bins
tracks in all, less any redundant set: with two bins a complement is just
the other bin, so the " i" tracks are not written; with palindromic fractions
(e.g. 0.4 0.2 0.4) the inv set mirrors the plain one, so it is not written.
Names are short: "Sp 4 B0 50% o" is speed, 4-bar segments,
bin 0 holding 50% of them; then "Sp 4 B0 50% i", ..., "Sp inv 4 B0 50% o".
Rotation is "Ro", a segment in beats reads "4bt", and the percent is rounded
to an integer. Every change after the first sits
--lead-ms before its bar line, so Cubase's de-click ramp has finished when a
note starting on the bar sounds. The first segment starts at time 0.

A segments CSV beside the .dawproject records each segment's times, median
percentile and bins.

With --midi-in, a MIDI file exported from Cubase is split the same way: each
track with notes becomes one track per bin track, named "<track> <bin track>",
holding only the notes that start in that bin track's on segments. Notes are
never moved or shortened, so they ring to their natural end. A note's segment
is decided by its start snapped to the nearest --grid step, so a note played
up to half a step early still counts as on the bar line. Notes starting after
the video ends, and zero-length notes, are dropped. The file must have the same single tempo and time
signature as --tempo and --time-signature.

Cubase will not accept automation on a group channel, so -- exactly as in
write_dawproject.py -- each series is an audio track carrying the
automation, routed to the master.

Usage:
    python speed_to_dawproject.py data/input/N50_speed.py --tempo 108
    python speed_to_dawproject.py data/input/N50_speed.py --tempo 108 \
        --segment-bars 4 --bin-fractions 0.5 0.2 0.3
    python speed_to_dawproject.py /mnt/d/N52/N52_speed.py --tempo 96 \
        --segment-bars 4 --bin-fractions 0.25 0.5 0.25 \
        --midi-in /mnt/d/N52/test.mid --grid 1/16
"""

import argparse
import os
import sys

import mido
import numpy as np
import pandas as pd

from process_metrics import percentile_data
from speed_to_cc import read_speed_data
from write_dawproject import (
    DEFAULT_MAX_COLUMNS,
    build_project_xml,
    render_archive,
    serialize,
    validate_project_xml,
)

DEFAULT_FPS = 30.0
DEFAULT_CLIP_FRACTION = 0.0
DEFAULT_EVERY_NTH_FRAME = 1
DEFAULT_INTERPOLATION = "linear"

# Binned tracks switch between these two linear gains: unity (0 dB) and silence.
BIN_ON = 1.0
BIN_OFF = 0.0

# Cubase 15.0.30 ignores interpolation="hold" on import and ramps linearly
# from one point to the next, across the whole segment. So each change is
# written as two linear points one tick apart: the old value, then the new
# one. Ticks are Cubase's display resolution, 480 per quarter note, so the
# step is 60 / tempo / 480 seconds (1.3 ms at 96 bpm): a step for musical
# purposes.
TICKS_PER_QUARTER = 480

# Short quantity names for the binned track names.
QUANTITY_ABBREVIATION = {"Speed": "Sp", "Rotation": "Ro"}


def scale_unit(values, clip_fraction=0.0):
    """Scale an array to span 0-1, clipping the extreme tails first.

    The `clip_fraction` and `1 - clip_fraction` quantiles map to 0 and 1, and
    values beyond them are clipped to those bounds. 0 uses the true min and
    max, which matches scale_to_cc without the 0-100 step.
    """
    low = float(np.quantile(values, clip_fraction))
    high = float(np.quantile(values, 1.0 - clip_fraction))
    if high == low:
        raise ValueError(
            f"series is constant between the {clip_fraction} and "
            f"{1.0 - clip_fraction} quantiles, so it cannot be scaled to 0-1")
    return (np.clip(values, low, high) - low) / (high - low)


def build_series(speed, quantity, clip_fraction):
    """The six derived series, in the same order speed_to_cc.py writes them.

    Clipping applies to all three base series, so on the percentile series
    percentile `clip_fraction` maps to 0 and `1 - clip_fraction` to 1.
    """
    value = scale_unit(speed, clip_fraction)
    inverse = scale_unit(1.0 / (speed + 1e-10), clip_fraction)
    percentile = scale_unit(percentile_data(speed), clip_fraction)

    return {
        f"{quantity}": value,
        f"Inverse {quantity}": inverse,
        f"{quantity} Percentile": percentile,
        f"{quantity} Inverted": 1.0 - value,
        f"Inverse {quantity} Inverted": 1.0 - inverse,
        f"{quantity} Percentile Inverted": 1.0 - percentile,
    }


def segment_seconds(segment_bars, segment_beats, tempo, numerator, denominator):
    """Length of one segment in seconds at a constant tempo.

    A beat is a 1/denominator note and a bar is `numerator` beats; the tempo
    counts quarter notes per minute.
    """
    beats = segment_beats if segment_bars is None else segment_bars * numerator
    return beats * (4.0 / denominator) * 60.0 / tempo


def bin_counts(fractions, n_segments):
    """Segments per bin, from the bin boundaries rounded to whole segments.

    Boundary k sits at the cumulative fraction of bins 0..k times
    n_segments, rounded to the nearest segment. Rounding the boundaries
    rather than each bin's count keeps palindromic fractions symmetric
    (79 segments at 0.4 0.2 0.4 give 32 15 32), so the rounding is absorbed
    by the middle bins. A boundary exactly on a half segment gives the extra
    segment to whichever neighbouring bin is more central; when both are
    equally central (an even number of bins, an odd number of segments,
    symmetric fractions) it goes to the lower bin.
    """
    cumulative = np.cumsum(fractions)
    position = cumulative[:-1] * n_segments
    whole = np.floor(position)
    remainder = position - whole

    # Distance of each bin's midpoint, in cumulative fraction, from the centre.
    starts = np.concatenate([[0.0], cumulative[:-1]])
    centrality = np.abs((starts + cumulative) / 2 - 0.5)
    lower_more_central = centrality[:-1] <= centrality[1:]

    # Rounding a boundary up gives the half segment to the bin below it.
    on_half = np.isclose(remainder, 0.5, rtol=0.0, atol=1e-9)
    round_up = np.where(on_half, lower_more_central, remainder > 0.5)
    boundaries = (whole + round_up).astype(int)

    counts = np.diff(np.concatenate([[0], boundaries, [n_segments]]))
    empty = np.flatnonzero(counts == 0)
    if empty.size:
        raise ValueError(
            f"bin(s) {list(empty)} receive no segments: {n_segments} segments "
            f"cannot be split as {list(fractions)}; use larger fractions or "
            f"shorter segments")
    return counts


def assign_bins(order, counts):
    """Bin of each segment, given segment indices ordered from bin 0 upward."""
    bins = np.empty(len(order), dtype=int)
    bins[order] = np.repeat(np.arange(len(counts)), counts)
    return bins


def build_binned_series(speed, fps, segment_length, fractions):
    """Assign each segment to a bin, normally and inverted.

    Returns a DataFrame with one row per segment: start time, frame count,
    median frame percentile, bin and inverted bin.
    """
    percentile = percentile_data(speed)
    segment = np.floor(np.arange(len(speed)) / fps / segment_length).astype(int)
    frames_per_segment = np.bincount(segment)
    if np.any(frames_per_segment == 0):
        raise ValueError(
            f"segments of {segment_length:.6f} s are shorter than a frame at "
            f"{fps:g} fps, so some hold no frames")

    median = np.array([np.median(percentile[segment == s])
                       for s in range(len(frames_per_segment))])
    counts = bin_counts(fractions, len(median))

    # Stable sorts break ties in segment order, in both directions.
    ascending = np.argsort(median, kind="stable")
    descending = np.argsort(-median, kind="stable")

    return pd.DataFrame({
        "segment": np.arange(len(median)),
        "start_seconds": np.arange(len(median)) * segment_length,
        "frames": frames_per_segment,
        "median_percentile": median,
        "bin": assign_bins(ascending, counts),
        "inverted_bin": assign_bins(descending, counts),
    })


def binned_tag(segment_bars, segment_beats, fractions, lead_ms):
    """Filename tag encoding the binning settings, e.g. s4bar_b50-20-30_lead10ms."""
    size = (f"s{segment_bars:g}bar" if segment_bars is not None
            else f"s{segment_beats:g}beat")
    shares = "-".join(f"{100 * f:g}" for f in fractions)
    return f"{size}_b{shares}_lead{lead_ms:g}ms"


def read_notes(track):
    """Pair a track's note-ons and note-offs into (start, end, channel, note, velocity).

    Times are absolute ticks. A note-on with velocity 0 is a note-off, and
    repeated notes on the same key and channel close first-in, first-out.
    Any other channel message (controllers, pitch bend, program change) is
    returned separately with its absolute tick, to be copied unchanged.
    """
    notes, other, sounding = [], [], {}
    tick = 0
    for message in track:
        tick += message.time
        if message.type == "note_on" and message.velocity > 0:
            sounding.setdefault((message.channel, message.note), []).append(
                (tick, message.velocity))
        elif message.type in ("note_on", "note_off"):
            start, velocity = sounding[(message.channel, message.note)].pop(0)
            notes.append((start, tick, message.channel, message.note, velocity))
        elif not message.is_meta:
            other.append((tick, message))
    hanging = [key for key, starts in sounding.items() if starts]
    if hanging:
        raise ValueError(f"notes with no note-off (channel, key): {hanging}")
    return notes, other


def build_track(name, notes, other):
    """A MIDI track holding the given notes and other channel messages."""
    # Order at equal ticks: note-offs first, so a repeated key re-strikes
    # cleanly; then other messages; then note-ons. Zero-length notes were
    # dropped by the caller, so no note's off can precede its own on.
    events = [(tick, 1, message) for tick, message in other]
    for start, end, channel, note, velocity in notes:
        events.append((start, 2, mido.Message(
            "note_on", channel=channel, note=note, velocity=velocity)))
        events.append((end, 0, mido.Message(
            "note_off", channel=channel, note=note, velocity=0)))
    events.sort(key=lambda e: (e[0], e[1]))

    track = mido.MidiTrack([mido.MetaMessage("track_name", name=name, time=0)])
    previous = 0
    for tick, _, message in events:
        track.append(message.copy(time=tick - previous))
        previous = tick
    return track


def write_binned_midi(args, values, numerator, denominator, segment_quarters,
                      duration, output_path):
    """Split each track of --midi-in into one track per bin track.

    A note goes on the bin tracks that are on in its segment and is never
    moved or shortened, so it rings to its natural end. Its segment is
    decided by its start snapped to the nearest --grid step, so a note
    played up to half a step early or late counts as being on the bar line.
    Notes starting after the video ends are dropped. Tracks with no notes
    (disabled or automation-only tracks in Cubase) are skipped.
    """
    source = mido.MidiFile(args.midi_in)
    if source.type != 1:
        raise ValueError(f"{args.midi_in} is MIDI type {source.type}; expected 1")
    quarter = source.ticks_per_beat
    conductor = source.tracks[0]

    # The bins are laid out at --tempo and --time-signature; a file at any
    # other tempo, or with a tempo or signature change, would not line up.
    tempos = [m for m in conductor if m.type == "set_tempo"]
    signatures = [m for m in conductor if m.type == "time_signature"]
    if len(tempos) != 1 or not np.isclose(mido.tempo2bpm(tempos[0].tempo), args.tempo):
        raise ValueError(
            f"{args.midi_in} must hold one tempo equal to --tempo {args.tempo:g}; "
            f"found {[round(mido.tempo2bpm(m.tempo), 3) for m in tempos]}")
    if (len(signatures) != 1 or (signatures[0].numerator, signatures[0].denominator)
            != (numerator, denominator)):
        raise ValueError(
            f"{args.midi_in} must hold one time signature equal to "
            f"--time-signature {numerator}/{denominator}; found "
            f"{[(m.numerator, m.denominator) for m in signatures]}")

    grid_numerator, grid_denominator = (int(x) for x in args.grid.split("/"))
    grid = quarter * 4 * grid_numerator / grid_denominator
    segment_ticks = quarter * segment_quarters
    seconds_per_tick = 60.0 / args.tempo / quarter

    output = mido.MidiFile(type=1, ticks_per_beat=quarter)
    output.tracks.append(conductor)
    print(f"MIDI: {args.midi_in}, grid {args.grid} = {grid:g} ticks, "
          f"segment = {segment_ticks:g} ticks")
    sources = []
    for track in source.tracks[1:]:
        notes, other = read_notes(track)
        # A zero-length note (on and off on the same tick) makes no sound.
        total = len(notes)
        zero_length = sum(1 for n in notes if n[1] == n[0])
        notes = [n for n in notes if n[1] > n[0]]
        if not notes:
            continue
        name = track.name
        starts = np.array([n[0] for n in notes])
        snapped = np.floor(starts / grid + 0.5) * grid
        segment = np.floor(snapped / segment_ticks).astype(int)
        in_video = starts * seconds_per_tick < duration
        print(f"  {name}: {total} notes, "
              f"{int(np.sum(~in_video))} after the video and {zero_length} "
              f"zero-length dropped, "
              f"{int(np.sum(snapped != starts))} off the grid")
        sources.append((name, notes, other, segment, in_video))

    # Tracks sharing a partition stay together: bin tracks in the order of
    # values' columns, and within each every source track in file order.
    for column in values.columns:
        for name, notes, other, segment, in_video in sources:
            on = np.zeros(len(notes), dtype=bool)
            on[in_video] = values[column].to_numpy()[segment[in_video]] == BIN_ON
            output.tracks.append(build_track(
                f"{name} {column}", [n for n, keep in zip(notes, on) if keep],
                other))

    output.save(output_path)
    print(f"Wrote {output_path} ({len(output.tracks) - 1} tracks)")


def write_binned(args, speed, quantity, numerator, denominator):
    """Write the bar-quantized binned .dawproject and its segments CSV."""
    fractions = args.bin_fractions
    if any(f <= 0 for f in fractions):
        raise ValueError(f"--bin-fractions must all be > 0, got {fractions}")
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(
            f"--bin-fractions must sum to 1, got {fractions} "
            f"(sum {sum(fractions):g})")
    if args.clip_fraction != DEFAULT_CLIP_FRACTION:
        raise ValueError("--clip-fraction does not apply in binned mode")
    if args.every_nth_frame != DEFAULT_EVERY_NTH_FRAME:
        raise ValueError("--every-nth-frame does not apply in binned mode")
    if args.interpolation != DEFAULT_INTERPOLATION:
        raise ValueError("--interpolation does not apply in binned mode; "
                         "binned changes are always written as 1-tick steps")
    size = args.segment_bars if args.segment_bars is not None else args.segment_beats
    if size <= 0:
        raise ValueError(f"segment length must be > 0, got {size}")

    segment_length = segment_seconds(
        args.segment_bars, args.segment_beats, args.tempo, numerator, denominator)
    lead = args.lead_ms / 1000.0
    step = 60.0 / args.tempo / TICKS_PER_QUARTER
    # Each step starts one tick before its change time, and the first
    # step must still come after time 0.
    if not 0.0 <= lead < segment_length - step:
        raise ValueError(
            f"--lead-ms must be in [0, segment length - 1 tick = "
            f"{1000 * (segment_length - step):g} ms), "
            f"got {args.lead_ms:g}")

    segments = build_binned_series(speed, args.fps, segment_length, fractions)

    # The first value starts at 0; every later change leads its bar line.
    change = segments["start_seconds"] - lead
    change.iloc[0] = 0.0
    segments.insert(2, "change_seconds", change)

    # Each bin track (" o") is followed by its complement (" i"), which is
    # silent exactly when the bin is on. Names are kept short for Cubase's
    # track list: "Sp 4 B0 50% o" is speed, 4-bar segments, bin 0 holding
    # 50% of them. A segment given in beats reads "4bt" instead of "4".
    short = QUANTITY_ABBREVIATION[quantity]
    segment_label = (f"{args.segment_bars:g}" if args.segment_bars is not None
                     else f"{args.segment_beats:g}bt")
    # Redundant sets are not written. With two bins, a bin's complement is
    # exactly the other bin of the same set, so the " i" tracks would only
    # duplicate " o" tracks. With palindromic fractions (e.g. 0.4 0.2 0.4),
    # bin_counts splits symmetrically, so inv bin k mirrors bin K-1-k and the
    # inv set is dropped. Ties between equal segment medians, or an even
    # number of bins with an odd number of segments, can leave the mirror
    # off by a segment.
    write_complements = len(fractions) > 2
    palindromic = np.allclose(fractions, fractions[::-1], rtol=0.0, atol=1e-6)
    sets = [(f"{short} {segment_label}", "bin")]
    if not palindromic:
        sets.append((f"{short} inv {segment_label}", "inverted_bin"))
    # Track order, used by both the .dawproject and the --midi-in output:
    # every " o" track before every " i" track, then the plain set before
    # inv, then bin number.
    originals, complements = {}, {}
    for label, bin_column in sets:
        for k, fraction in enumerate(fractions):
            on = segments[bin_column].to_numpy() == k
            name = f"{label} B{k} {round(100 * fraction)}%"
            originals[f"{name} o"] = np.where(on, BIN_ON, BIN_OFF)
            if write_complements:
                complements[f"{name} i"] = np.where(on, BIN_OFF, BIN_ON)
    values = pd.DataFrame(originals | complements)
    columns = list(values.columns)
    if len(columns) > DEFAULT_MAX_COLUMNS:
        raise ValueError(f"{len(columns)} columns exceeds {DEFAULT_MAX_COLUMNS}")

    # Point list: segment 0's value at time 0, then for each later segment
    # the previous segment's value one tick before its change time and
    # its own value at the change time.
    later = segments["change_seconds"].to_numpy()[1:]
    point_times = np.concatenate(
        [[0.0], np.column_stack([later - step, later]).ravel()])
    n_segments = len(segments)
    point_segment = np.concatenate(
        [[0], np.column_stack([np.arange(n_segments - 1),
                               np.arange(1, n_segments)]).ravel()])
    point_values = values.iloc[point_segment].reset_index(drop=True)

    tag = binned_tag(args.segment_bars, args.segment_beats, fractions, args.lead_ms)
    stem = os.path.splitext(os.path.basename(args.input_file))[0]
    output_path = args.output or os.path.join(
        "data", "output", f"{stem}_{tag}.dawproject")
    segments_csv = os.path.splitext(output_path)[0] + "_segments.csv"

    project_xml = serialize(build_project_xml(
        point_times, columns, point_values, args.tempo,
        [numerator, denominator], "linear"))
    validate_project_xml(project_xml)
    render_archive(
        output_path, project_xml,
        os.path.splitext(os.path.basename(output_path))[0],
        f"Binned {quantity.lower()} automation from "
        f"{os.path.basename(args.input_file)}")
    segments.to_csv(segments_csv, index=False)
    print(f"Wrote {segments_csv}")
    if args.midi_in is not None:
        midi_stem = os.path.splitext(os.path.basename(args.midi_in))[0]
        write_binned_midi(
            args, values, numerator, denominator,
            segment_length * args.tempo / 60.0, len(speed) / args.fps,
            os.path.join(os.path.dirname(output_path),
                         f"{midi_stem}_{stem}_{tag}.mid"))

    print(f"  Segment: {size:g} {'bars' if args.segment_bars is not None else 'beats'}"
          f" = {segment_length:.6f} s at {args.tempo:g} bpm, "
          f"{numerator}/{denominator}; lead {args.lead_ms:g} ms")
    print(f"  {len(segments)} segments "
          f"(last holds {segments['frames'].iloc[-1]} frames)")
    counts = segments["bin"].value_counts().sort_index()
    for k in range(len(fractions)):
        print(f"  Bin {k}: {counts[k]} segments "
              f"({counts[k] / len(segments):.3f}, asked {fractions[k]:g})")
    print(f"  {len(columns)} audio tracks, {len(point_times)} points each "
          f"(1-tick steps of {1000 * step:.3f} ms)")
    print("Import in Cubase with File > Import > DAWproject.")
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="speed_to_dawproject.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_file",
        help="speed/rotation data file (.py, .csv, or .kfs), one value per frame")
    parser.add_argument(
        "-o", "--output",
        help="output .dawproject path "
             "(default: data/output/<input stem>.dawproject)")
    parser.add_argument(
        "--fps", type=float, default=DEFAULT_FPS,
        help="frame rate the series was sampled at, used to turn frame "
             "indices into seconds (default: %(default)s)")
    parser.add_argument(
        "--tempo", type=float, default=120.0,
        help="Transport tempo written into the file. Cosmetic -- all times "
             "are seconds -- but Cubase will not load a project without one, "
             "and importing overwrites the target project's initial tempo "
             "marking, so set it to match (default: %(default)s)")
    parser.add_argument(
        "--time-signature", default="4/4",
        help="Transport time signature, as numerator/denominator. Also "
             "overwrites the target project's initial signature on import "
             "(default: %(default)s)")
    parser.add_argument(
        "--every-nth-frame", type=int, default=DEFAULT_EVERY_NTH_FRAME,
        help="keep only every Nth frame, to thin the automation. 1 keeps the "
             "full per-frame resolution (default: %(default)s)")
    parser.add_argument(
        "--clip-fraction", type=float, default=DEFAULT_CLIP_FRACTION,
        help="fraction of each tail to clip before scaling each track to "
             "0-1: 0.05 maps the 5th and 95th percentiles "
             "to 0 and 1 and clips everything beyond them, so a brief "
             "extreme cannot squash the rest of the curve. 0 uses the true "
             "min and max. On the percentile tracks, percentile 0.05 "
             "maps to 0 and 0.95 to 1 (default: %(default)s)")
    parser.add_argument(
        "--interpolation", default=DEFAULT_INTERPOLATION,
        help="interpolation written on each point (default: %(default)s)")
    segment = parser.add_mutually_exclusive_group()
    segment.add_argument(
        "--segment-bars", type=float,
        help="binned mode: segment length in bars at the constant --tempo "
             "and --time-signature")
    segment.add_argument(
        "--segment-beats", type=float,
        help="binned mode: segment length in beats (1/denominator notes)")
    parser.add_argument(
        "--bin-fractions", type=float, nargs="+",
        help="binned mode: fraction of segments in each bin, slowest first; "
             "must sum to 1, e.g. 0.5 0.2 0.3")
    parser.add_argument(
        "--lead-ms", type=float, default=10.0,
        help="binned mode: how far before each bar line a change sits, in "
             "milliseconds; the first segment starts at 0 "
             "(default: %(default)s)")
    parser.add_argument(
        "--midi-in",
        help="binned mode: a MIDI file exported from Cubase. Each track is "
             "split into one track per bin track, holding only the notes that "
             "start in that track's on segments")
    parser.add_argument(
        "--grid", default="1/16",
        help="with --midi-in: the note grid the music was played to. Notes "
             "are not moved; a note starting within half a grid step of a "
             "bar line counts as being on it (default: %(default)s)")
    args = parser.parse_args()

    binned = args.segment_bars is not None or args.segment_beats is not None
    if binned != (args.bin_fractions is not None):
        raise ValueError(
            "--bin-fractions and --segment-bars/--segment-beats go together")
    if args.midi_in is not None and not binned:
        raise ValueError("--midi-in needs the binned mode")

    numerator, denominator = (int(x) for x in args.time_signature.split("/"))

    is_rotation = "rot" in os.path.basename(args.input_file).lower()
    quantity = "Rotation" if is_rotation else "Speed"

    data = read_speed_data(args.input_file)
    print(f"Loaded {len(data)} frames from {args.input_file}")
    print(f"File stats: min={data.min():.4f}, max={data.max():.4f}, "
          f"mean={data.mean():.4f}")
    if is_rotation:
        print("Note: convention is that positive is counter-clockwise")
        speed = data
    else:
        # The file holds inverse speed; a zero or negative value has no
        # meaningful reciprocal, so refuse rather than write infinities.
        if np.any(data <= 0):
            raise ValueError(
                f"inverse-speed file has {int(np.sum(data <= 0))} values <= 0; "
                "cannot take 1/value")
        speed = 1.0 / data
        print(f"Treating file values as inverse speed; speed = 1/value "
              f"(min={speed.min():.6f}, max={speed.max():.6f})")

    if binned:
        return write_binned(args, speed, quantity, numerator, denominator)

    if not 0.0 <= args.clip_fraction < 0.5:
        raise ValueError(
            f"--clip-fraction must be in [0, 0.5), got {args.clip_fraction}")
    if args.every_nth_frame < 1:
        raise ValueError(
            f"--every-nth-frame must be at least 1, got {args.every_nth_frame}")

    # Derive from the full series, then thin, so that the percentile ranks and
    # the 0-1 scaling reflect every frame rather than only the kept ones.
    series = build_series(speed, quantity, args.clip_fraction)
    frames = np.arange(len(speed))[::args.every_nth_frame]
    values = pd.DataFrame({k: v[::args.every_nth_frame] for k, v in series.items()})
    times = frames / args.fps

    columns = list(values.columns)
    if len(columns) > DEFAULT_MAX_COLUMNS:
        raise ValueError(f"{len(columns)} columns exceeds {DEFAULT_MAX_COLUMNS}")

    output_path = args.output or os.path.join(
        "data", "output",
        os.path.splitext(os.path.basename(args.input_file))[0] + ".dawproject")

    project_xml = serialize(build_project_xml(
        times, columns, values, args.tempo, [numerator, denominator],
        args.interpolation))
    validate_project_xml(project_xml)
    render_archive(
        output_path, project_xml,
        os.path.splitext(os.path.basename(output_path))[0],
        f"{quantity} automation from {os.path.basename(args.input_file)}")

    print(f"  {len(columns)} audio tracks, {len(times)} points each")
    print(f"  Time range: 0.000 to {times[-1]:.3f} seconds")
    print(f"  Clip fraction: {args.clip_fraction:g} of each tail "
          f"(all tracks)")
    print(f"  Transport: {args.tempo:g} bpm, time signature "
          f"{numerator}/{denominator}")
    print("Import in Cubase with File > Import > DAWproject.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
