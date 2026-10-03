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

Cubase will not accept automation on a group channel, so -- exactly as in
write_dawproject.py -- each series emits an audio track carrying the
automation plus a like-named "<track> BUS" it feeds. Copy the lane onto the
buss by hand in Cubase.

Usage:
    python speed_to_dawproject.py data/input/N50_speed.py --tempo 108
"""

import argparse
import os
import sys

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
        "--every-nth-frame", type=int, default=1,
        help="keep only every Nth frame, to thin the automation. 1 keeps the "
             "full per-frame resolution (default: %(default)s)")
    parser.add_argument(
        "--clip-fraction", type=float, default=0.0,
        help="fraction of each tail to clip before scaling each track to "
             "0-1: 0.05 maps the 5th and 95th percentiles "
             "to 0 and 1 and clips everything beyond them, so a brief "
             "extreme cannot squash the rest of the curve. 0 uses the true "
             "min and max. On the percentile tracks, percentile 0.05 "
             "maps to 0 and 0.95 to 1 (default: %(default)s)")
    parser.add_argument(
        "--interpolation", default="linear",
        help="interpolation written on each point (default: %(default)s)")
    args = parser.parse_args()

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

    print(f"  {len(columns)} audio tracks (automation) + {len(columns)} group "
          f"busses, {len(times)} points each")
    print(f"  Each audio track is routed into its own \"<track> BUS\".")
    print("  Cubase will not import automation onto a group: copy each lane")
    print("  from the audio track to its buss by hand.")
    print(f"  Time range: 0.000 to {times[-1]:.3f} seconds")
    print(f"  Clip fraction: {args.clip_fraction:g} of each tail "
          f"(all tracks)")
    print(f"  Transport: {args.tempo:g} bpm, time signature "
          f"{numerator}/{denominator}")
    print("Import in Cubase with File > Import > DAWproject.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
