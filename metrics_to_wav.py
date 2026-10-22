#!/usr/bin/env python3
"""
Render harmonic tones whose timbre follows video metrics, as a WAV file.

The notes come from one of two sources, chosen by which key the config has:
  "root_hz"    one note at root_hz held for the length of the video (a drone)
  "midi_file"  one tone per MIDI note, at the note's equal-tempered pitch
               (A4 = 440 Hz), start time and duration. Time 0 of the MIDI file
               is time 0 of the video; tempo changes are honoured. Velocity is
               ignored. Overlapping notes simply sum.

Each note is the sum of exact integer harmonics of its frequency f0 (no
tempering of the harmonics). Each harmonic's amplitude is a fixed background
strength times a time-varying weight taken from a metric expression evaluated
on the raw metrics CSV (`*_basic.csv`):

    y(t) = sum_n  w_n * m_n(t) * sin(2 pi n f0 (t - t_start) + phi_n)  /  sum_n w_n

    w_n    = n ** strength_exponent * gain_n   (gain_n from the harmonic's row)
    m_n(t) = rank of the expression's value, mapped so the lowest row is 0
             and the highest is 1, linearly interpolated between rows

Which harmonics sound is set by `cycle`:
  true   the list must be n = 1..L, and it is cycled up to `max_hz`: harmonic
         n uses entry ((n - 1) mod L) + 1, so harmonic L + 1 follows the same
         metric as harmonic 1. Every harmonic up to `max_hz` sounds.
  false  exactly the listed n sound (any increasing set, e.g. 1..8 then only
         octave doublings 10, 12, 14, 16, 20, ...), each with its own
         expression; listed harmonics above `max_hz` are skipped.
Higher notes therefore have fewer harmonics. The weights m_n(t) follow the video's
timeline; after the last CSV row they hold the last row's values.

Each harmonic has a fixed starting phase phi_n: 0 for every harmonic
(`"phases": "zero"`), or drawn uniformly from [0, 2 pi) with `random_seed`
(`"phases": "random"`, a fresh draw for each note). Zero phases align every harmonic once per period of the
root, which the ear hears as a buzz at the root frequency; random phases keep
the same spectrum without the pulse. Only amplitudes change over time, so the
waveform is continuous. A raised-cosine fade of `fade_s` seconds at each end
of every note removes the click at note-on and note-off; a note shorter than
2 * fade_s fades over half its length instead.

Optional pitch jitter breaks up the buzz that remains with random phases (any
two neighbouring harmonics beat at f0, and with exact harmonics that beat is
perfectly regular). Each harmonic of each note drifts independently: a random
offset in cents, drawn with standard deviation `jitter_cents`, is placed every
`jitter_s` seconds and linearly interpolated, and the resulting frequency
n f0 2^(cents/1200) is integrated into phase so there are no discontinuities.
Because the drift is in cents, upper harmonics move more Hz than the
fundamental, so the pitch stays put while the upper harmonics slide in and out
of alignment. `jitter_cents` 0 gives exact harmonics.

Dividing by each note's sum(w_n) bounds |y| by 1 for a single note regardless
of the data; overlapping notes can sum above 1.

Usage:
    python metrics_to_wav.py CONFIG.json
"""

import json
import os
import re
import shutil
import sys

import mido
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


def harmonic_strength(n, gain, config):
    return n ** config["strength_exponent"] * gain


def midi_note_hz(note):
    return 440.0 * 2.0 ** ((note - 69) / 12)


def read_midi_notes(path):
    """
    Return [(start_s, end_s, hz)] for every note in the file, sorted by start,
    and the number of pitch-bend messages (which are ignored). Iterating a
    MidiFile merges the tracks and gives message times in seconds with the
    tempo map applied.
    """
    notes = []
    sounding = {}
    pitch_bends = 0
    now = 0.0
    for msg in mido.MidiFile(path):
        now += msg.time
        pitch_bends += msg.type == "pitchwheel"
        is_on = msg.type == "note_on" and msg.velocity > 0
        is_off = msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0)
        if is_on:
            key = (msg.channel, msg.note)
            if key in sounding:
                raise ValueError(f"Note {msg.note} on channel {msg.channel} restarted at {now:.3f} s "
                                 f"before its note-off")
            sounding[key] = now
        if is_off:
            start = sounding.pop((msg.channel, msg.note))
            notes.append((start, now, midi_note_hz(msg.note)))
    if sounding:
        raise ValueError(f"Notes never turned off: {sorted(sounding)}")
    return sorted(notes), pitch_bends


def jitter_phase(n, f0, tau, config, rng):
    """
    Extra phase (radians) of harmonic n from an independent random drift in
    cents: N(0, jitter_cents) knots every jitter_s seconds, linearly
    interpolated, converted to a frequency offset and integrated.
    """
    jitter_s = config["jitter_s"]
    knot_tau = np.arange(int(np.ceil(tau[-1] / jitter_s)) + 2) * jitter_s
    cents = rng.normal(0.0, config["jitter_cents"], len(knot_tau))
    offset_hz = n * f0 * (2.0 ** (np.interp(tau, knot_tau, cents) / 1200) - 1)
    return 2 * np.pi * np.cumsum(offset_hz) / config["sample_rate"]


def render_note(start_s, end_s, f0, entries, control_times, config, rng):
    """
    Render one note as an array of round(end_s * sr) - round(start_s * sr)
    samples. Returns (samples, fade_was_shortened).
    """
    sample_rate = config["sample_rate"]
    i0 = int(round(start_s * sample_rate))
    i1 = int(round(end_s * sample_rate))
    if i1 <= i0:
        raise ValueError(f"Note at {start_s:.4f} s ({f0:.2f} Hz) is shorter than one sample")
    n_max = int(config["max_hz"] // f0)
    if n_max < 1:
        raise ValueError(f"Note at {start_s:.4f} s ({f0:.2f} Hz) is above max_hz {config['max_hz']}")

    phase_draws = {
        "zero": lambda: np.zeros(n_max + 1),
        "random": lambda: rng.uniform(0, 2 * np.pi, n_max + 1),
    }
    # phases[n] is harmonic n's starting phase; index 0 is unused.
    phases = phase_draws[config["phases"]]()

    member_rules = {
        True: lambda entry: range(entry["n"], n_max + 1, len(entries)),
        False: lambda entry: [n for n in (entry["n"],) if n <= n_max],
    }
    members = [list(member_rules[config["cycle"]](entry)) for entry in entries]
    sounding = sorted((n, entry["gain"]) for entry, group in zip(entries, members) for n in group)
    if not sounding:
        raise ValueError(f"Note at {start_s:.4f} s ({f0:.2f} Hz) has no listed harmonic below max_hz")
    total_strength = sum(harmonic_strength(n, gain, config) for n, gain in sounding)

    t = (i0 + np.arange(i1 - i0)) / sample_rate  # video time, for the weights
    tau = t - i0 / sample_rate                   # note time, for the sines
    jittered = config["jitter_cents"] > 0
    samples = np.zeros(i1 - i0)
    for entry, group in zip(entries, members):
        carrier = np.zeros(i1 - i0)
        for n in group:
            drift = jitter_phase(n, f0, tau, config, rng) if jittered else 0.0
            carrier += harmonic_strength(n, entry["gain"], config) * np.sin(2 * np.pi * n * f0 * tau + phases[n] + drift)
        samples += np.interp(t, control_times, entry["weight"]) * carrier
    samples /= total_strength

    n_fade_wanted = int(round(config["fade_s"] * sample_rate))
    n_fade = min(n_fade_wanted, (i1 - i0) // 2)
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(n_fade) / n_fade)
    samples[:n_fade] *= ramp
    samples[len(samples) - n_fade:] *= ramp[::-1]
    return samples, n_fade < n_fade_wanted


def write_plot(out_path, control_times, entries, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 7))
    cmap = plt.get_cmap("viridis")
    for i, h in enumerate(entries):
        ax.plot(control_times, h["strength"] * h["weight"],
                color=cmap(i / max(len(entries) - 1, 1)), linewidth=0.9,
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
    control_times = frames / fps
    video_duration = (frames[-1] + step) / fps
    if config["max_hz"] >= sample_rate / 2:
        raise ValueError(f"max_hz {config['max_hz']} is at or above Nyquist ({sample_rate / 2} Hz)")

    sources = [key for key in ("root_hz", "midi_file") if key in config]
    if len(sources) != 1:
        raise ValueError(f"Config needs exactly one of 'root_hz' or 'midi_file'; has {sources}")
    note_sources = {
        "root_hz": lambda: ([(0.0, video_duration, config["root_hz"])], 0),
        "midi_file": lambda: read_midi_notes(config["midi_file"]),
    }
    notes, pitch_bends = note_sources[sources[0]]()
    if not notes:
        raise ValueError("No notes to render")

    listed_n = [entry["n"] for entry in config["harmonics"]]
    if listed_n[0] < 1 or any(b <= a for a, b in zip(listed_n, listed_n[1:])):
        raise ValueError(f"Harmonic numbers must be positive and strictly increasing; got {listed_n}")
    if config["cycle"] and listed_n != list(range(1, len(listed_n) + 1)):
        raise ValueError(f"With cycle true, harmonics must be listed as n = 1, 2, ... in order; got {listed_n}")
    entries = []
    for entry in config["harmonics"]:
        values = evaluate_expression(df, entry["expr"])
        entries.append({
            "n": entry["n"],
            "expr": entry["expr"],
            "gain": entry["gain"],
            "strength": harmonic_strength(entry["n"], entry["gain"], config),
            "weight": rank_normalize(values, entry["expr"]),
        })

    duration = max(video_duration, max(end for _, end, _ in notes))
    overlaps = [prev[1] - nxt[0] for prev, nxt in zip(notes, notes[1:]) if nxt[0] < prev[1]]
    freqs = [hz for _, _, hz in notes]

    print(f"Input:    {config['input_csv']}")
    print(f"Rows:     {len(df)} every {step} frames at {fps} fps -> video {video_duration:.3f} s")
    print(f"Notes:    {sources[0]}: {len(notes)} notes, {min(freqs):.2f}-{max(freqs):.2f} Hz, "
          f"output {duration:.3f} s")
    if sources[0] == "midi_file":
        print(f"          {config['midi_file']}")
        print(f"          {len(overlaps)} overlaps (longest {max(overlaps, default=0) * 1000:.1f} ms), "
              f"{pitch_bends} pitch-bend messages ignored")
    print(f"Harmonics: up to {config['max_hz']} Hz, {len(entries)} listed, cycle {config['cycle']}")
    print(f"Phases:   {config['phases']} (seed {config['random_seed']}), fade {config['fade_s']} s")
    print(f"Jitter:   {config['jitter_cents']} cents every {config['jitter_s']} s")
    print(f"Strength: exponent {config['strength_exponent']}, per-row gains "
          f"{sorted(set(entry['gain'] for entry in entries))}")

    signal = np.zeros(int(round(duration * sample_rate)))
    rng = np.random.default_rng(config["random_seed"])
    shortened = 0
    for start, end, hz in notes:
        samples, was_shortened = render_note(start, end, hz, entries, control_times, config, rng)
        i0 = int(round(start * sample_rate))
        signal[i0:i0 + len(samples)] += samples
        shortened += was_shortened
    print(f"Fades:    {shortened} notes shorter than 2 x fade_s faded over half their length")

    csv_stem = os.path.splitext(os.path.basename(config["input_csv"]))[0]
    if not csv_stem.endswith("_basic"):
        raise ValueError(f"Expected a *_basic.csv input, got {config['input_csv']}")
    config_stem = os.path.splitext(os.path.basename(config_path))[0]
    out_base = os.path.join(os.path.dirname(config["input_csv"]),
                            f"{csv_stem[:-len('_basic')]}_{config_stem}")

    wav_path = f"{out_base}.wav"
    sf.write(wav_path, signal, sample_rate, subtype=OUTPUT_SUBTYPE)
    shutil.copyfile(config_path, f"{out_base}_config.json")
    write_plot(f"{out_base}_plot.png", control_times, entries, os.path.basename(out_base))

    peak = float(np.max(np.abs(signal)))
    print(f"Peak:     {peak:.4f} ({20 * np.log10(peak):.2f} dBFS)")
    print(f"Wrote:    {wav_path}")
    print(f"          {out_base}_config.json")
    print(f"          {out_base}_plot.png")


if __name__ == "__main__":
    main()
