#!/usr/bin/env python3
"""
Convert a per-frame speed (or rotation) time series into a .dawproject.

The DAWproject counterpart of speed_to_cc.py: same input files and the same
six derived series, written as volume automation instead of MIDI CC.

Six tracks are written, each scaled independently to span 0-1:

  1. <Quantity>                     value
  2. Inverse <Quantity>             1/value
  3. <Quantity> Percentile          percentile rank of value
  4. <Quantity> Inverted            1 - track 1
  5. Inverse <Quantity> Inverted    1 - track 2
  6. <Quantity> Percentile Inverted 1 - track 3

If the input filename contains "rot" the tracks are labelled "Rotation"
instead of "Speed"; the sign convention there is that positive is
counter-clockwise.

Automation times are wall-clock seconds, `frame_index / fps`, so the curves
line up with the video and with the output of write_dawproject.py. Nothing
positional depends on tempo.

Values are written as linear gain: 0 is -infinity dB and 1.0 is 0 dB.

Cubase will not accept automation on a group channel, so -- exactly as in
write_dawproject.py -- each series emits an audio track carrying the
automation plus a like-named "<track> BUS" it feeds. Copy the lane onto the
buss by hand in Cubase.

Usage:
    python speed_to_dawproject.py data/input/N51_speed.py --tempo 108
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


def scale_unit(values):
    """Scale an array to span 0-1. Matches scale_to_cc but without the 0-100 step."""
    low, high = float(np.min(values)), float(np.max(values))
    if high == low:
        raise ValueError("series is constant, so it cannot be scaled to 0-1")
    return (values - low) / (high - low)


def build_series(speed, quantity):
    """The six derived series, in the same order speed_to_cc.py writes them."""
    value = scale_unit(speed)
    inverse = scale_unit(1.0 / (speed + 1e-10))
    percentile = scale_unit(percentile_data(speed))

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
        "--interpolation", default="linear",
        help="interpolation written on each point (default: %(default)s)")
    args = parser.parse_args()

    numerator, denominator = (int(x) for x in args.time_signature.split("/"))

    is_rotation = "rot" in os.path.basename(args.input_file).lower()
    quantity = "Rotation" if is_rotation else "Speed"

    speed = read_speed_data(args.input_file)
    print(f"Loaded {len(speed)} frames from {args.input_file}")
    print(f"{quantity} stats: min={speed.min():.4f}, max={speed.max():.4f}, "
          f"mean={speed.mean():.4f}")
    if is_rotation:
        print("Note: convention is that positive is counter-clockwise")

    if args.every_nth_frame < 1:
        raise ValueError(
            f"--every-nth-frame must be at least 1, got {args.every_nth_frame}")

    # Derive from the full series, then thin, so that the percentile ranks and
    # the 0-1 scaling reflect every frame rather than only the kept ones.
    series = build_series(speed, quantity)
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
    print(f"  Transport: {args.tempo:g} bpm, time signature "
          f"{numerator}/{denominator}")
    print("Import in Cubase with File > Import > DAWproject.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
