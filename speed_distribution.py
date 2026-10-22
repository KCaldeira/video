#!/usr/bin/env python3
"""
Show the distribution of speed in an inverse-speed file, as CSV and a plot.

Reads the same input files as speed_to_dawproject.py. Those files hold
INVERSE speed (time per unit of motion: large when the camera is slow), so
speed is taken as 1/value, exactly as there.

Writes, in data/output/ by default:

  <stem>_distribution.csv      one row per frame: frame, seconds,
                               inverse_speed, speed, percentile
  <stem>_distribution.png/.pdf four panels: speed over time, histogram of
                               speed, histogram of log2(speed), and the
                               cumulative distribution

and prints quantiles and the fraction of time below and above any given
speeds. The percentile column is the same rank speed_to_dawproject.py bins
on, so the CSV shows directly which frames tie.

Rotation files are refused: rotation is signed, so it has no reciprocal and
no logarithm.

Usage:
    python speed_distribution.py /mnt/d/N52/N52_speed.py
    python speed_distribution.py /mnt/d/N52/N52_speed.py --thresholds 0.015 0.025
"""

import argparse
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

from process_metrics import percentile_data  # noqa: E402
from speed_to_cc import read_speed_data  # noqa: E402

DEFAULT_FPS = 30.0
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]

# Single-series chart: one blue, with text and grid recessive.
SERIES = "#2a78d6"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def plot_distribution(seconds, speed, title, output_stem):
    """Write the four-panel figure as PNG and PDF."""
    plt.rcParams.update({
        "font.size": 10,
        "axes.edgecolor": TEXT_SECONDARY, "axes.labelcolor": TEXT_SECONDARY,
        "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    })
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    marks = np.quantile(speed, [0.05, 0.5, 0.95])

    ax = axes[0, 0]
    ax.plot(seconds, speed, color=SERIES, lw=1.2)
    ax.set(title="Speed over time", xlabel="seconds",
           ylabel="speed (1 / file value)")

    ax = axes[0, 1]
    ax.hist(speed, bins=80, color=SERIES, edgecolor=SURFACE, linewidth=0.5)
    for value, label in zip(marks, ["5%", "median", "95%"]):
        ax.axvline(value, color=TEXT_SECONDARY, lw=0.8, ls="--")
        ax.text(value, ax.get_ylim()[1] * 0.97, f" {label}",
                color=TEXT_SECONDARY, va="top", fontsize=8)
    ax.set(title="Histogram of speed", xlabel="speed", ylabel="frames")

    ax = axes[1, 0]
    ax.hist(np.log2(speed), bins=80, color=SERIES, edgecolor=SURFACE,
            linewidth=0.5)
    ax.set(title="Histogram of log2(speed)  (1 unit = doubling)",
           xlabel="log2(speed)", ylabel="frames")

    ax = axes[1, 1]
    ax.plot(np.sort(speed), np.linspace(0, 1, len(speed)), color=SERIES, lw=2)
    ax.set(title="Cumulative distribution", xlabel="speed",
           ylabel="fraction of frames below")

    fig.suptitle(title, color=TEXT_PRIMARY, fontsize=11)
    for extension in ("png", "pdf"):
        fig.savefig(f"{output_stem}.{extension}", dpi=130)
        print(f"Wrote {output_stem}.{extension}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        prog="speed_distribution.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_file",
        help="inverse-speed data file (.py, .csv, or .kfs), one value per frame")
    parser.add_argument(
        "-o", "--output-stem",
        help="output path without extension "
             "(default: data/output/<input stem>_distribution)")
    parser.add_argument(
        "--fps", type=float, default=DEFAULT_FPS,
        help="frame rate the series was sampled at (default: %(default)s)")
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=[],
        help="speeds at which to report the fraction of time below and above")
    args = parser.parse_args()

    stem = os.path.splitext(os.path.basename(args.input_file))[0]
    if "rot" in stem.lower():
        raise ValueError(
            f"{args.input_file} looks like a rotation file; rotation is "
            f"signed, so it has no speed reciprocal or logarithm")

    inverse = read_speed_data(args.input_file)
    if np.any(inverse <= 0):
        raise ValueError(
            f"inverse-speed file has {int(np.sum(inverse <= 0))} values <= 0; "
            "cannot take 1/value")
    speed = 1.0 / inverse
    seconds = np.arange(len(speed)) / args.fps

    output_stem = args.output_stem or os.path.join(
        "data", "output", f"{stem}_distribution")
    pd.DataFrame({
        "frame": np.arange(len(speed)),
        "seconds": seconds,
        "inverse_speed": inverse,
        "speed": speed,
        "percentile": percentile_data(speed),
    }).to_csv(f"{output_stem}.csv", index=False)
    print(f"Wrote {output_stem}.csv")

    quantiles = np.quantile(speed, QUANTILES)
    quantile_text = " / ".join(f"{q:.4f}" for q in quantiles)
    plot_distribution(
        seconds, speed,
        f"{stem}: {len(speed)} frames, {seconds[-1]:.0f} s at {args.fps:g} fps"
        f"   |   speed 5/25/50/75/95%: {quantile_text}",
        output_stem)

    duration = len(speed) / args.fps
    print(f"  {len(speed)} frames, {duration:.1f} s; "
          f"{len(np.unique(inverse))} distinct values")
    print(f"  Speed quantiles 5/25/50/75/95%: {quantile_text}")
    print(f"  Range: {speed.min():.6f} to {speed.max():.6f} "
          f"(ratio {speed.max() / speed.min():.1f}; "
          f"95%/5% ratio {quantiles[4] / quantiles[0]:.2f})")
    at_max = speed == speed.max()
    print(f"  At the maximum speed: {at_max.mean():.1%} of frames "
          f"({at_max.sum() / args.fps:.0f} s), all tied at percentile "
          f"{percentile_data(speed)[at_max][0]:.4f}")
    for threshold in args.thresholds:
        below = speed < threshold
        above = speed > threshold
        print(f"  Speed {threshold:g}: below {below.mean():.1%} "
              f"({below.sum() / args.fps:.0f} s), above {above.mean():.1%} "
              f"({above.sum() / args.fps:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
