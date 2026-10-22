#!/usr/bin/env python3
"""
Render a harmonic drone whose timbre follows video metrics, as a WAV file.

One note is held for the length of the video. It is the sum of exact integer
harmonics of a root frequency (no tempering). Each harmonic's amplitude is a
fixed background strength times a time-varying weight taken from a metric
expression evaluated on the raw metrics CSV (`*_basic.csv`):

    y(t) = sum_n  w_n * m_n(t) * sin(2 pi n f0 t)  /  sum_n w_n

    w_n    = n ** strength_exponent * (odd_gain if n is odd else even_gain)
    m_n(t) = rank of the expression's value, mapped so the lowest row is 0
             and the highest is 1, linearly interpolated between rows

Every sine starts at phase 0 at t = 0 and only its amplitude changes, so the
waveform is continuous. Dividing by sum(w_n) bounds |y| by 1 regardless of
the data.

Usage:
    python metrics_to_wav.py CONFIG.json
"""

import json
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.stats import rankdata

OUTPUT_SUBTYPE = "FLOAT"
FRAME_COLUMN = "frame_count_list"
# Expressions may use column names, multiplication, division and parentheses.
EXPR_PATTERN = re.compile(r"^[A-Za-z0-9_\s*/()]+$")


def evaluate_expression(df, expr):
    """Evaluate a `*`/`/` expression over the raw metric columns."""
    if not EXPR_PATTERN.match(expr):
        raise ValueError(f"Expression {expr!r} may only use column names, '*', '/' and parentheses")
    values = df.eval(expr).to_numpy(dtype="float64")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Expression {expr!r} produced non-finite values")
    return values


def rank_normalize(values, expr):
    """Map values to their rank, lowest -> 0, highest -> 1 (ties share the average rank)."""
    ranks = rankdata(values)
    span = ranks.max() - ranks.min()
    if span == 0:
        raise ValueError(f"Expression {expr!r} is constant; cannot rank-normalize")
    return (ranks - ranks.min()) / span


def harmonic_strength(n, config):
    parity_gain = config["odd_gain"] if n % 2 == 1 else config["even_gain"]
    return n ** config["strength_exponent"] * parity_gain


def write_plot(out_path, control_times, harmonics, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 7))
    cmap = plt.get_cmap("viridis")
    for i, h in enumerate(harmonics):
        ax.plot(control_times, h["strength"] * h["weight"],
                color=cmap(i / max(len(harmonics) - 1, 1)), linewidth=0.9,
                label=f"{h['n']}: {h['expr']}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude  w_n * m_n  (before / sum w_n)")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    config_path = sys.argv[1]
    with open(config_path) as f:
        config = json.load(f)

    df = pd.read_csv(config["input_csv"])
    frames = df[FRAME_COLUMN].to_numpy()
    steps = np.unique(np.diff(frames))
    if len(steps) != 1:
        raise ValueError(f"{FRAME_COLUMN} is not uniformly spaced: steps {steps}")
    step = int(steps[0])

    fps = config["frames_per_second"]
    sample_rate = config["sample_rate"]
    root_hz = config["root_hz"]
    control_times = frames / fps
    duration = (frames[-1] + step) / fps
    n_samples = int(round(duration * sample_rate))
    top_hz = root_hz * max(h["n"] for h in config["harmonics"])
    if top_hz >= sample_rate / 2:
        raise ValueError(f"Highest harmonic {top_hz} Hz is at or above Nyquist ({sample_rate / 2} Hz)")

    harmonics = []
    for entry in config["harmonics"]:
        values = evaluate_expression(df, entry["expr"])
        harmonics.append({
            "n": entry["n"],
            "expr": entry["expr"],
            "strength": harmonic_strength(entry["n"], config),
            "weight": rank_normalize(values, entry["expr"]),
        })
    total_strength = sum(h["strength"] for h in harmonics)

    print(f"Input:    {config['input_csv']}")
    print(f"Rows:     {len(df)} every {step} frames at {fps} fps -> {duration:.3f} s")
    print(f"Root:     {root_hz} Hz, {len(harmonics)} harmonics (top {top_hz} Hz)")
    print(f"Strength: exponent {config['strength_exponent']}, "
          f"odd {config['odd_gain']}, even {config['even_gain']}, sum {total_strength:.4f}")

    t = np.arange(n_samples) / sample_rate
    signal = np.zeros(n_samples)
    for h in harmonics:
        envelope = np.interp(t, control_times, h["weight"])
        signal += h["strength"] * envelope * np.sin(2 * np.pi * h["n"] * root_hz * t)
    signal /= total_strength

    csv_stem = os.path.splitext(os.path.basename(config["input_csv"]))[0]
    if not csv_stem.endswith("_basic"):
        raise ValueError(f"Expected a *_basic.csv input, got {config['input_csv']}")
    config_stem = os.path.splitext(os.path.basename(config_path))[0]
    out_base = os.path.join(os.path.dirname(config["input_csv"]),
                            f"{csv_stem[:-len('_basic')]}_{config_stem}")

    wav_path = f"{out_base}.wav"
    sf.write(wav_path, signal, sample_rate, subtype=OUTPUT_SUBTYPE)
    shutil.copyfile(config_path, f"{out_base}_config.json")
    write_plot(f"{out_base}_plot.png", control_times, harmonics,
               f"{os.path.basename(out_base)}  (root {root_hz} Hz)")

    peak = float(np.max(np.abs(signal)))
    print(f"Peak:     {peak:.4f} ({20 * np.log10(peak):.2f} dBFS)")
    print(f"Wrote:    {wav_path}")
    print(f"          {out_base}_config.json")
    print(f"          {out_base}_plot.png")


if __name__ == "__main__":
    main()
