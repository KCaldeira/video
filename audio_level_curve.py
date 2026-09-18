#!/usr/bin/env python3
"""
Compute a slowly-varying gain curve that levels the loudness of an audio file.

Stage 1 of the audio auto-leveling workflow. Measures EBU R128 / ITU-R BS.1770
K-weighted loudness over time, smooths it with a kernel of a specified mean
weighting distance, and turns the result into a gain curve that can be applied
as a volume fader. Stage 2 (curve_to_dawproject.py) carries the curve into
Cubase.

Usage:
    python audio_level_curve.py INPUT_AUDIO [options]

Examples:
    # Default: pull down anything above -18 LUFS, leave quieter passages alone
    python audio_level_curve.py "data/input/N47 2026-09-17-04.wav"

    # Gentler ride, one-minute-boxcar-equivalent smoothing
    python audio_level_curve.py song.wav --mean-distance 15 --smoothing-method boxcar

    # Two-sided leveling toward a target
    python audio_level_curve.py song.wav --mode both --target-lufs -20
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, sosfilt, sosfilt_zi

import audio_smoothing

DESCRIPTION = """\
Compute a gain curve that levels the loudness of an audio file.

Loudness is measured as EBU R128 / ITU-R BS.1770 K-weighted LUFS over short
overlapping blocks, then smoothed with a kernel specified by its MEAN WEIGHTING
DISTANCE -- the average distance of the kernel's weight from its center. For a
boxcar this is a quarter of the full width, so a one-minute boxcar is
--mean-distance 15. Every method is normalized the same way, so the same
--mean-distance smooths by the same amount whichever --smoothing-method you pick.

The gain is then target minus smoothed loudness, clamped according to --mode.
The default mode is cut-only: passages quieter than the target are left exactly
alone and only louder passages are pulled down, so the target acts as a ceiling.

By default the target is derived from the audio rather than given as a number.
--target-fraction says how much of the time the volume should be reduced, and
the target is placed at the matching quantile of the smoothed loudness. The
default of 0.25 reduces the volume a quarter of the time. Because it adapts to
the material, the same setting means the same thing on every file. Use
--target-lufs instead to anchor to an absolute level across several files.

This tool only rides the volume. It does not limit, normalize, or otherwise set
the peak level -- those are a separate concern and belong in a separate step,
applied after this one.
"""

EPILOG = """\
outputs (written to data/output/, named {input}_{tag}_*):
  ..._gain_curve.csv     full-resolution analysis, one row per block
  ..._gain_points.csv    decimated automation points -- input to stage 2
  ..._leveled.wav        preview render, 32-bit float at the source rate
  ..._plot.png           loudness before, loudness after, and the gain curve
  ..._config.json        every parameter used for this run

The tag encodes the settings that shape the curve -- mode, target, kernel and
mean distance -- so "song_cf25_gau15_leveled.wav" is a cut-only ride reducing
the volume 25% of the time, gaussian smoothing with a 15 s mean weighting
distance. An absolute target appears instead as "song_c28_gau15_*". Use --tag
to name a run yourself.

The preview is rendered by interpolating the DECIMATED points exactly the way a
DAW interpolates an automation envelope, so what you hear is what Cubase will
do -- not a higher-resolution approximation of it.

Nothing is resampled and no bit depth is reduced anywhere in this pipeline. The
preview is always written as 32-bit float at the source sample rate, so
applying gain neither requantizes the audio nor changes its rate.

This tool does not limit or normalize. If the ride leaves the result too hot or
too quiet, that is a job for a separate limiting or normalizing step run
afterwards.

Every run also reports how much headroom is left before 0 dBFS, using an
inter-sample (true) peak by default. That is a report only; nothing acts on it.

examples:
  # defaults: reduce the volume 25% of the time, 15 s mean weighting distance
  python audio_level_curve.py "data/input/N47 2026-09-17-04.wav"

  # ride harder, and let the curve move faster
  python audio_level_curve.py song.wav --target-fraction 0.4 --mean-distance 5

  # anchor to an absolute level instead of deriving one from the audio
  python audio_level_curve.py song.mp3 --target-lufs -16 --mean-distance 30

  python audio_level_curve.py song.wav --mode both --smoothing-method loess
"""

# ITU-R BS.1770-4 K-weighting, stage 1: high-frequency shelf.
SHELF_F0 = 1681.974450955533
SHELF_G_DB = 3.999843853973347
SHELF_Q = 0.7071752369554196
SHELF_VB_EXPONENT = 0.4996667741545416

# ITU-R BS.1770-4 K-weighting, stage 2: high-pass.
HIGHPASS_F0 = 38.13547087602444
HIGHPASS_Q = 0.5003270373238773

# Offset in the BS.1770 loudness equation.
LOUDNESS_OFFSET_DB = -0.691

# Per-channel weights for the loudness sum. Left and right are unity; this
# workflow does not handle surround layouts.
CHANNEL_WEIGHT = 1.0

READ_BLOCK_FRAMES = 1 << 18

# Default fraction of the audible time spent under volume reduction.
DEFAULT_TARGET_FRACTION = 0.25

# Where the loudest point of the finished curve is placed, in dBFS.
DEFAULT_PEAK_DBFS = -0.1

# Largest boost allowed by default. Boosting lifts the noise floor with the
# signal, so this stays finite. Cutting does not, so it is unlimited by
# default -- see DEFAULT_MAX_CUT_DB.
DEFAULT_MAX_BOOST_DB = 12.0

# No limit on how far the ride may pull the level down.
DEFAULT_MAX_CUT_DB = float("inf")

# Cubase's fader maximum. Gains above this cannot be represented on a Cubase
# volume fader, so the run warns when the curve exceeds it.
CUBASE_FADER_MAX_DB = 12.0

# The preview is always written as 32-bit float at the source sample rate, so
# applying gain never requantizes the audio or changes its rate.
PREVIEW_SUBTYPE = "FLOAT"

# Oversampling factor for the true-peak report. ITU-R BS.1770-4 Annex 2 calls
# for at least 4x at 48 kHz. Analysis only; never applied to written audio.
TRUE_PEAK_OVERSAMPLE = 4

# Source frames of context carried between blocks so the oversampling filter
# does not see an artificial edge. Far longer than the filter itself needs.
TRUE_PEAK_OVERLAP = 1024


def k_weighting_sos(sample_rate):
    """Second-order sections for the BS.1770 K-weighting filter."""
    k = np.tan(np.pi * SHELF_F0 / sample_rate)
    vh = 10.0 ** (SHELF_G_DB / 20.0)
    vb = vh**SHELF_VB_EXPONENT
    denom = 1.0 + k / SHELF_Q + k * k
    shelf = [
        (vh + vb * k / SHELF_Q + k * k) / denom,
        2.0 * (k * k - vh) / denom,
        (vh - vb * k / SHELF_Q + k * k) / denom,
        1.0,
        2.0 * (k * k - 1.0) / denom,
        (1.0 - k / SHELF_Q + k * k) / denom,
    ]

    k = np.tan(np.pi * HIGHPASS_F0 / sample_rate)
    denom = 1.0 + k / HIGHPASS_Q + k * k
    highpass = [
        1.0,
        -2.0,
        1.0,
        1.0,
        2.0 * (k * k - 1.0) / denom,
        (1.0 - k / HIGHPASS_Q + k * k) / denom,
    ]

    return np.array([shelf, highpass], dtype=float)


def measure_block_power(path, hop_frames):
    """
    Stream the file and accumulate, per hop-sized bin, the K-weighted mean
    square summed over channels.

    Returns (weighted_mean_square, info).
    """
    info = sf.info(path)
    sos = k_weighting_sos(info.samplerate)
    n_bins = int(np.ceil(info.frames / hop_frames))

    power_sum = np.zeros(n_bins)
    counts = np.zeros(n_bins)

    handle = sf.SoundFile(path)
    n_channels = handle.channels
    # One filter state per channel, initialized to the steady-state response of
    # the first sample so the filter does not ring in at t=0.
    state = None
    offset = 0

    for block in handle.blocks(blocksize=READ_BLOCK_FRAMES, dtype="float64", always_2d=True):
        if state is None:
            zi = sosfilt_zi(sos)
            state = zi[:, :, None] * block[0][None, None, :]
        filtered, state = sosfilt(sos, block, axis=0, zi=state)

        square = CHANNEL_WEIGHT * np.sum(filtered * filtered, axis=1)
        bins = (np.arange(len(block)) + offset) // hop_frames

        power_sum += np.bincount(bins, weights=square, minlength=n_bins)[:n_bins]
        counts += np.bincount(bins, minlength=n_bins)[:n_bins]
        offset += len(block)

    handle.close()

    if np.any(counts == 0):
        raise ValueError("internal error: empty loudness bin")
    return power_sum / counts, info


def momentary_loudness(mean_square, blocks_per_window):
    """
    Centered moving average of the per-bin mean square over the analysis
    window, converted to LUFS. Partial windows at the edges are normalized by
    the number of bins actually covered, so the series is not biased toward
    silence at the start and end.
    """
    kernel = np.ones(blocks_per_window)
    padded = np.concatenate(
        [np.zeros(blocks_per_window), mean_square, np.zeros(blocks_per_window)]
    )
    ones = np.concatenate(
        [np.zeros(blocks_per_window), np.ones(len(mean_square)), np.zeros(blocks_per_window)]
    )
    start = blocks_per_window + (blocks_per_window - 1) // 2 - (blocks_per_window - 1)
    total = np.convolve(padded, kernel, mode="full")[start : start + len(mean_square)]
    count = np.convolve(ones, kernel, mode="full")[start : start + len(mean_square)]

    mean = np.where(count > 0, total / np.maximum(count, 1.0), 0.0)
    with np.errstate(divide="ignore"):
        return LOUDNESS_OFFSET_DB + 10.0 * np.log10(np.maximum(mean, 1e-30))


def douglas_peucker(times, values, tolerance):
    """Indices of the points that approximate the curve within tolerance."""
    keep = np.zeros(len(times), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(times) - 1)]

    while stack:
        lo, hi = stack.pop()
        if hi <= lo + 1:
            continue
        span = times[hi] - times[lo]
        if span <= 0:
            continue
        slope = (values[hi] - values[lo]) / span
        segment = slice(lo + 1, hi)
        deviation = np.abs(
            values[segment] - (values[lo] + slope * (times[segment] - times[lo]))
        )
        worst = int(np.argmax(deviation))
        if deviation[worst] > tolerance:
            split = lo + 1 + worst
            keep[split] = True
            stack.append((lo, split))
            stack.append((split, hi))

    return np.flatnonzero(keep)


def measure_leveled_peak(path, point_times, point_gain_linear, mode="true"):
    """
    Peak of the audio with the gain curve applied, using the same point
    interpolation as the render.

    mode="true" reports the inter-sample (true) peak: the signal is oversampled
    by TRUE_PEAK_OVERSAMPLE so that peaks of the reconstructed waveform falling
    between samples are counted. This is the loudest point a converter will
    actually produce, and it can sit meaningfully above the highest sample.
    mode="sample" reports only the highest sample, which is cheaper.

    This is measurement only. Nothing here changes the audio -- the result is
    reported so the caller knows how much headroom is left, and the
    oversampled signal is discarded.
    """
    if mode not in ("true", "sample"):
        raise ValueError(f"unknown peak mode: {mode!r}")

    handle = sf.SoundFile(path)
    tail = np.zeros((TRUE_PEAK_OVERLAP, handle.channels))
    offset = 0
    peak = 0.0

    for block in handle.blocks(blocksize=READ_BLOCK_FRAMES, dtype="float64", always_2d=True):
        if mode == "sample":
            times = (np.arange(len(block)) + offset) / handle.samplerate
            gain = np.interp(times, point_times, point_gain_linear)
            peak = max(peak, float(np.max(np.abs(block * gain[:, None]))))
        else:
            chunk = np.concatenate([tail, block])
            times = (np.arange(len(chunk)) + offset - len(tail)) / handle.samplerate
            gain = np.interp(times, point_times, point_gain_linear)
            upsampled = resample_poly(chunk * gain[:, None], TRUE_PEAK_OVERSAMPLE, 1, axis=0)
            interior = upsampled[len(tail) * TRUE_PEAK_OVERSAMPLE :]
            peak = max(peak, float(np.max(np.abs(interior))))
            tail = block[-TRUE_PEAK_OVERLAP:]
        offset += len(block)

    handle.close()
    return peak


def render_preview(path, out_path, point_times, point_gain_linear):
    """
    Apply the gain curve and write the result. The gain is interpolated
    linearly between the decimated points, matching how a DAW draws an
    automation envelope, so the preview reflects what the DAWproject will do.
    """
    source = sf.SoundFile(path)
    target = sf.SoundFile(
        out_path,
        mode="w",
        samplerate=source.samplerate,
        channels=source.channels,
        subtype=PREVIEW_SUBTYPE,
    )

    offset = 0
    peak = 0.0
    for block in source.blocks(blocksize=READ_BLOCK_FRAMES, dtype="float64", always_2d=True):
        times = (np.arange(len(block)) + offset) / source.samplerate
        gain = np.interp(times, point_times, point_gain_linear)
        scaled = block * gain[:, None]
        peak = max(peak, float(np.max(np.abs(scaled))))
        target.write(scaled)
        offset += len(block)

    source.close()
    target.close()
    return peak


def write_plot(out_path, times, lufs_raw, lufs_smooth, gain_db, target_lufs,
               gate_lufs, valid, title, makeup_db=0.0):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    minutes = times / 60.0
    figure, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    # Scale the loudness axes to the audible material. A single block of
    # digital silence measures around -300 LUFS and would otherwise compress
    # everything of interest into a sliver.
    audible = lufs_raw[valid]
    low = min(np.percentile(audible, 0.5), lufs_smooth.min(), target_lufs) - 4.0
    high = max(np.percentile(audible, 99.9), lufs_smooth.max(), target_lufs) + 4.0

    axes[0].plot(minutes, lufs_raw, lw=0.5, color="0.7", label="measured (momentary)")
    axes[0].plot(minutes, lufs_smooth, lw=1.8, color="C0", label="smoothed")
    axes[0].axhline(target_lufs, color="C3", ls="--", lw=1.2, label=f"target {target_lufs:g} LUFS")
    if gate_lufs > low:
        axes[0].axhline(gate_lufs, color="0.5", ls=":", lw=1.0, label=f"gate {gate_lufs:g} LUFS")
    if not valid.all():
        axes[0].fill_between(
            minutes, low, high, where=~valid, color="C1", alpha=0.35, label="gated"
        )
    axes[0].set_ylim(low, high)
    axes[0].set_ylabel("loudness (LUFS)")
    axes[0].legend(loc="lower right", fontsize=8, ncol=2)
    axes[0].set_title(title)

    axes[1].plot(minutes, gain_db, lw=1.5, color="C2")
    axes[1].axhline(0.0, color="0.4", lw=0.8)
    axes[1].set_ylabel("gain (dB)")

    axes[2].plot(minutes, lufs_raw + gain_db, lw=0.5, color="0.7", label="measured + gain")
    axes[2].plot(minutes, lufs_smooth + gain_db, lw=1.8, color="C0", label="smoothed + gain")
    # The gain includes any constant makeup, so the result sits makeup_db
    # above the target the ride aimed at. Shift both the reference line and
    # the axis by the same amount, keeping the vertical scale identical to
    # the panel above so the two can be read against each other.
    axes[2].axhline(target_lufs + makeup_db, color="C3", ls="--", lw=1.2,
                    label=f"target {target_lufs + makeup_db:.1f} LUFS")
    axes[2].set_ylim(low + makeup_db, high + makeup_db)
    axes[2].set_ylabel("resulting loudness (LUFS)")
    axes[2].set_xlabel("time (minutes)")
    axes[2].legend(loc="lower right", fontsize=8)

    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(out_path, dpi=120)
    plt.close(figure)


MODE_CODE = {"cut-only": "c", "both": "b", "boost-only": "o"}


def build_tag(mode, target_lufs, target_fraction, method, mean_distance_s,
              peak_dbfs, makeup):
    """
    Short suffix encoding the settings that change the curve, so two runs with
    different settings cannot overwrite each other and a repeat of the same
    settings lands on the same files.
    """
    if target_fraction is not None:
        target_part = f"f{target_fraction * 100:g}"
    else:
        target_part = f"{abs(target_lufs):g}"
    peak_part = {
        "none": "pkoff",
        "curve": f"pk{abs(peak_dbfs):g}",
        "clip-gain": f"cg{abs(peak_dbfs):g}",
    }[makeup]
    return f"{MODE_CODE[mode]}{target_part}_{method[:3]}{mean_distance_s:g}_{peak_part}"


def main():
    parser = argparse.ArgumentParser(
        prog="audio_level_curve.py",
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_file",
        help="input audio file (.wav, .flac, .mp3, or anything libsndfile reads)",
    )
    parser.add_argument(
        "--mode",
        choices=("cut-only", "both", "boost-only"),
        default="cut-only",
        help="cut-only leaves quiet passages alone and only pulls down loud ones; "
        "both levels in each direction; boost-only only lifts quiet passages "
        "(default: %(default)s)",
    )
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "--target-fraction",
        type=float,
        default=None,
        help="set the target from the audio itself, as the fraction of the "
        "time the volume should be reduced: 0.25 puts the target where a "
        "quarter of the material sits above it (default: 0.25)",
    )
    target_group.add_argument(
        "--target-lufs",
        type=float,
        default=None,
        help="set the target as an absolute loudness instead; a ceiling in "
        "cut-only mode. Use when you want the same anchor across files",
    )
    parser.add_argument(
        "--mean-distance",
        type=float,
        default=15.0,
        help="mean weighting distance of the smoothing kernel, in seconds; "
        "equals a quarter of a boxcar's full width (default: %(default)s)",
    )
    parser.add_argument(
        "--smoothing-method",
        choices=audio_smoothing.METHODS,
        default="gaussian",
        help="smoothing kernel shape (default: %(default)s)",
    )
    parser.add_argument(
        "--max-boost-db",
        type=float,
        default=None,
        help="largest gain increase allowed; forced to 0 by --mode cut-only "
        "(default: 12 in --mode both and boost-only)",
    )
    parser.add_argument(
        "--max-cut-db",
        type=float,
        default=None,
        help="largest gain reduction allowed, as a positive number; forced to 0 "
        "by --mode boost-only (default: 12 in --mode both and cut-only)",
    )
    parser.add_argument(
        "--gate-lufs",
        type=float,
        default=-50.0,
        help="blocks quieter than this are excluded from the smoothing, so "
        "silence neither gets boosted nor drags the curve down (default: %(default)s)",
    )
    parser.add_argument(
        "--block-hop",
        type=float,
        default=0.1,
        help="spacing between loudness measurements, in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--block-length",
        type=float,
        default=0.4,
        help="loudness analysis window, in seconds; must be a whole multiple of "
        "--block-hop (default: %(default)s)",
    )
    parser.add_argument(
        "--point-tolerance-db",
        type=float,
        default=0.05,
        help="maximum error allowed when decimating the curve to automation "
        "points (default: %(default)s)",
    )
    parser.add_argument(
        "--peak-dbfs",
        type=float,
        default=DEFAULT_PEAK_DBFS,
        help="place the loudest point of the finished curve here, by adding a "
        "constant to the whole curve; this shifts the level without changing "
        "the shape of the ride (default: %(default)s)",
    )
    parser.add_argument(
        "--makeup",
        choices=("clip-gain", "curve", "none"),
        default="clip-gain",
        help="where the constant makeup gain goes. clip-gain keeps it out of "
        "the automation curve and reports it for you to dial in by hand, so "
        "the envelope stays within a fader's range; curve folds it into the "
        "envelope, which can exceed what a DAW fader allows; none applies no "
        "makeup at all (default: %(default)s)",
    )
    parser.add_argument(
        "--peak-mode",
        choices=("true", "sample"),
        default="true",
        help="how the headroom report measures the peak; true accounts for "
        "inter-sample peaks of the reconstructed waveform, sample looks only "
        "at the highest sample and is faster (default: %(default)s)",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="skip the preview render (analysis and curves are still written)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("data", "output"),
        help="output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="suffix distinguishing this run's output files; the default "
        "encodes the settings that change the curve, so different settings "
        "land in different files and a repeat of the same settings overwrites "
        "cleanly (default: e.g. cf25_gau15)",
    )

    args = parser.parse_args()

    # Resolve the gain limits from the mode, refusing silent overrides.
    if args.mode == "cut-only":
        if args.max_boost_db is not None and args.max_boost_db != 0.0:
            parser.error("--mode cut-only implies --max-boost-db 0; remove one of them")
        max_boost_db = 0.0
        max_cut_db = DEFAULT_MAX_CUT_DB if args.max_cut_db is None else args.max_cut_db
    elif args.mode == "boost-only":
        if args.max_cut_db is not None and args.max_cut_db != 0.0:
            parser.error("--mode boost-only implies --max-cut-db 0; remove one of them")
        max_cut_db = 0.0
        max_boost_db = DEFAULT_MAX_BOOST_DB if args.max_boost_db is None else args.max_boost_db
    else:
        max_boost_db = DEFAULT_MAX_BOOST_DB if args.max_boost_db is None else args.max_boost_db
        max_cut_db = DEFAULT_MAX_CUT_DB if args.max_cut_db is None else args.max_cut_db

    # Exactly one way of setting the target; fraction is the default because
    # it adapts to the material, where an absolute value may miss it entirely.
    target_fraction = args.target_fraction
    if target_fraction is None and args.target_lufs is None:
        target_fraction = DEFAULT_TARGET_FRACTION
    if target_fraction is not None:
        if not 0.0 < target_fraction < 1.0:
            parser.error(
                f"--target-fraction is a fraction of the time, strictly between "
                f"0 and 1, got {target_fraction}"
            )
        if args.mode == "boost-only":
            parser.error(
                "--mode boost-only never reduces the volume, so "
                "--target-fraction has no meaning; use --target-lufs"
            )

    if max_cut_db < 0.0:
        parser.error(f"--max-cut-db is a positive magnitude, got {max_cut_db}")
    if max_boost_db < 0.0:
        parser.error(f"--max-boost-db is a positive magnitude, got {max_boost_db}")

    if not os.path.exists(args.input_file):
        parser.error(f"input file not found: {args.input_file}")

    stem = os.path.splitext(os.path.basename(args.input_file))[0]
    tag = args.tag or build_tag(
        args.mode, args.target_lufs, target_fraction, args.smoothing_method,
        args.mean_distance, args.peak_dbfs, args.makeup,
    )
    name_prefix = f"{stem}_{tag}"
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    probe = sf.info(args.input_file)
    hop_frames = int(round(args.block_hop * probe.samplerate))
    if hop_frames < 1:
        parser.error(f"--block-hop {args.block_hop} is shorter than one sample")
    ratio = args.block_length / args.block_hop
    blocks_per_window = int(round(ratio))
    if abs(ratio - blocks_per_window) > 1e-9 or blocks_per_window < 1:
        parser.error(
            f"--block-length ({args.block_length}) must be a whole multiple of "
            f"--block-hop ({args.block_hop})"
        )

    print("=" * 60)
    print("AUDIO LEVEL CURVE")
    print("=" * 60)
    print(f"  Input:            {args.input_file}")
    print(f"  Format:           {probe.samplerate} Hz, {probe.channels} ch, "
          f"{probe.subtype}, {probe.frames / probe.samplerate:.1f} s")
    print(f"  Mode:             {args.mode}")
    if target_fraction is not None:
        print(f"  Target:           reduce the volume {target_fraction * 100:g}% "
              f"of the time (derived from the audio)")
    else:
        print(f"  Target:           {args.target_lufs:g} LUFS"
              f"{' (ceiling)' if args.mode == 'cut-only' else ''}")
    cut_label = "no limit" if np.isinf(max_cut_db) else f"-{max_cut_db:g} dB"
    print(f"  Gain limits:      +{max_boost_db:g} dB boost / {cut_label} cut")
    print(f"  Gate:             {args.gate_lufs:g} LUFS")
    peak_label = {
        "none": "no makeup applied (headroom reported only)",
        "curve": f"{args.peak_dbfs:+g} dBFS, makeup folded into the curve",
        "clip-gain": f"{args.peak_dbfs:+g} dBFS, makeup reported for clip gain",
    }[args.makeup]
    print(f"  Peak:             {peak_label} ({args.peak_mode} peak)")
    print(f"  Smoothing:        {audio_smoothing.describe(args.smoothing_method, args.mean_distance, args.block_hop)}")
    print(f"  Output:           {output_dir}/{name_prefix}_*")
    print()

    print("STEP 1: measuring K-weighted loudness")
    mean_square, info = measure_block_power(args.input_file, hop_frames)
    times = (np.arange(len(mean_square)) + 0.5) * hop_frames / info.samplerate
    lufs_raw = momentary_loudness(mean_square, blocks_per_window)
    print(f"  {len(lufs_raw)} blocks, loudness range "
          f"{lufs_raw.min():.1f} to {lufs_raw.max():.1f} LUFS")

    valid = lufs_raw > args.gate_lufs
    gated = int((~valid).sum())
    print(f"  {gated} blocks ({100.0 * gated / len(valid):.1f}%) below the gate")
    if not valid.any():
        raise ValueError(
            f"every block is below the gate of {args.gate_lufs} LUFS; "
            f"the loudest block is {lufs_raw.max():.1f} LUFS"
        )

    print("\nSTEP 2: smoothing")
    lufs_smooth = audio_smoothing.smooth(
        lufs_raw, valid, args.block_hop, args.smoothing_method, args.mean_distance
    )
    print(f"  smoothed loudness range {lufs_smooth.min():.1f} to {lufs_smooth.max():.1f} LUFS")

    print("\nSTEP 3: gain curve")
    if target_fraction is not None:
        # Reduction happens wherever the smoothed loudness exceeds the target,
        # so placing the target at this quantile makes that true for exactly
        # the requested fraction of the audible material. Gated blocks are
        # excluded: there is no audio there to reduce.
        target_lufs = float(np.quantile(lufs_smooth[valid], 1.0 - target_fraction))
        print(f"  target derived from the audio: {target_lufs:.2f} LUFS "
              f"({target_fraction * 100:g}% of the time above it)")
    else:
        target_lufs = args.target_lufs

    ride_db = np.clip(target_lufs - lufs_smooth, -max_cut_db, max_boost_db)
    reduced = ride_db < 0.0
    achieved = float(reduced[valid].mean())
    print(f"  ride range {ride_db.min():+.2f} to {ride_db.max():+.2f} dB")
    print(f"  volume reduced {achieved * 100:.1f}% of the audible time")

    print("\nSTEP 4: decimating to automation points")
    keep = douglas_peucker(times, ride_db, args.point_tolerance_db)
    point_times = times[keep]
    point_ride_db = ride_db[keep]
    # Hold the first and last values out to the ends of the file so the
    # envelope covers the whole timeline.
    duration = info.frames / info.samplerate
    point_times = np.concatenate([[0.0], point_times, [duration]])
    point_ride_db = np.concatenate([[point_ride_db[0]], point_ride_db, [point_ride_db[-1]]])
    print(f"  {len(lufs_raw)} blocks -> {len(point_times)} points "
          f"(tolerance {args.point_tolerance_db:g} dB)")

    print("\nSTEP 5: peak")
    # Measured from the decimated points, so this is the peak of exactly what
    # the DAW envelope will produce, not of a higher-resolution curve.
    ride_peak = measure_leveled_peak(
        args.input_file, point_times, 10.0 ** (point_ride_db / 20.0), args.peak_mode
    )
    ride_peak_dbfs = 20.0 * np.log10(max(ride_peak, 1e-30))
    headroom_db = -ride_peak_dbfs
    print(f"  {args.peak_mode} peak after the ride: {ride_peak_dbfs:+.2f} dBFS")
    print(f"  headroom before 0 dBFS: {headroom_db:.2f} dB")

    # A single constant, so it shifts the level without changing the shape of
    # the ride. Where it gets applied is what --makeup selects.
    makeup_db = 0.0 if args.makeup == "none" else args.peak_dbfs - ride_peak_dbfs

    # The curve written to CSV (and on to the DAWproject) carries the makeup
    # only in "curve" mode. The preview always renders the finished result, so
    # what you hear is the end state including a clip-gain move you make later.
    curve_makeup_db = makeup_db if args.makeup == "curve" else 0.0
    gain_db = ride_db + curve_makeup_db
    point_gain_db = point_ride_db + curve_makeup_db
    point_gain_linear = 10.0 ** (point_gain_db / 20.0)
    preview_gain_linear = 10.0 ** ((point_ride_db + makeup_db) / 20.0)
    peak_dbfs = ride_peak_dbfs + makeup_db

    if args.makeup == "clip-gain":
        print()
        print(f"  ==> APPLY {makeup_db:+.2f} dB OF CLIP GAIN IN CUBASE <==")
        print(f"      the automation curve holds the ride only "
              f"({gain_db.min():+.2f} to {gain_db.max():+.2f} dB), which fits any fader;")
        print(f"      with that clip gain the peak lands at {peak_dbfs:+.2f} dBFS")
        print()
    elif args.makeup == "curve":
        print(f"  makeup folded into the curve: {makeup_db:+.2f} dB")
        print(f"  final gain range {gain_db.min():+.2f} to {gain_db.max():+.2f} dB")
        print(f"  final {args.peak_mode} peak: {peak_dbfs:+.2f} dBFS")
        if gain_db.max() > CUBASE_FADER_MAX_DB:
            print(f"  NOTE: the curve peaks at {gain_db.max():+.2f} dB, above Cubase's "
                  f"+{CUBASE_FADER_MAX_DB:g} dB fader maximum; --makeup clip-gain avoids this")
    else:
        print(f"  no makeup applied; curve range {gain_db.min():+.2f} to "
              f"{gain_db.max():+.2f} dB")

    print("\nSTEP 6: writing outputs")
    gain_linear = 10.0 ** (gain_db / 20.0)
    curve_path = os.path.join(output_dir, f"{name_prefix}_gain_curve.csv")
    np.savetxt(
        curve_path,
        np.column_stack([times, lufs_raw, lufs_smooth, gain_db, gain_linear, valid.astype(int)]),
        delimiter=",",
        header="time_s,lufs_raw,lufs_smooth,gain_db,gain_linear,valid",
        comments="",
        fmt=["%.6f", "%.4f", "%.4f", "%.5f", "%.8f", "%d"],
    )
    print(f"  {curve_path}")

    points_path = os.path.join(output_dir, f"{name_prefix}_gain_points.csv")
    np.savetxt(
        points_path,
        np.column_stack([point_times, point_gain_db, point_gain_linear]),
        delimiter=",",
        header="time_s,gain_db,gain_linear",
        comments="",
        fmt=["%.6f", "%.5f", "%.8f"],
    )
    print(f"  {points_path}")

    plot_path = os.path.join(output_dir, f"{name_prefix}_plot.png")
    write_plot(
        plot_path, times, lufs_raw, lufs_smooth, ride_db + makeup_db, target_lufs,
        args.gate_lufs, valid,
        f"{stem}  |  {args.mode}, target {target_lufs:.1f} LUFS, "
        f"{args.smoothing_method} d={args.mean_distance:g}s"
        + (f", makeup {makeup_db:+.2f} dB" if makeup_db else ""),
        makeup_db,
    )
    print(f"  {plot_path}")

    preview_path = None
    if not args.no_preview:
        preview_path = os.path.join(output_dir, f"{name_prefix}_leveled.wav")
        rendered_peak = render_preview(
            args.input_file, preview_path, point_times, preview_gain_linear
        )
        print(f"  {preview_path}  (peak {20.0 * np.log10(max(rendered_peak, 1e-30)):+.2f} dBFS)")

    config_path = os.path.join(output_dir, f"{name_prefix}_config.json")
    with open(config_path, "w") as handle:
        json.dump(
            {
                "input_file": args.input_file,
                "tag": tag,
                "source": {
                    "samplerate": info.samplerate,
                    "channels": info.channels,
                    "subtype": info.subtype,
                    "frames": info.frames,
                    "duration_s": info.frames / info.samplerate,
                },
                "mode": args.mode,
                "target_lufs": float(target_lufs),
                "target_fraction": target_fraction,
                "target_from_audio": target_fraction is not None,
                "reduced_fraction_achieved": achieved,
                "mean_distance_s": args.mean_distance,
                "smoothing_method": args.smoothing_method,
                "max_boost_db": max_boost_db,
                "max_cut_db": None if np.isinf(max_cut_db) else max_cut_db,
                "gate_lufs": args.gate_lufs,
                "block_hop_s": args.block_hop,
                "block_length_s": args.block_length,
                "point_tolerance_db": args.point_tolerance_db,
                "peak_mode": args.peak_mode,
                "peak_dbfs_target": args.peak_dbfs,
                "makeup_mode": args.makeup,
                "clip_gain_to_apply_db": (
                    float(makeup_db) if args.makeup == "clip-gain" else 0.0
                ),
                "ride_peak_dbfs": float(ride_peak_dbfs),
                "headroom_db": float(headroom_db),
                "makeup_db": float(makeup_db),
                "leveled_peak_dbfs": float(peak_dbfs),
                "ride_db_min": float(ride_db.min()),
                "ride_db_max": float(ride_db.max()),
                "preview_subtype": PREVIEW_SUBTYPE,
                "n_blocks": int(len(lufs_raw)),
                "n_points": int(len(point_times)),
                "n_gated_blocks": gated,
                "gain_db_min": float(gain_db.min()),
                "gain_db_max": float(gain_db.max()),
                "outputs": {
                    "gain_curve_csv": curve_path,
                    "gain_points_csv": points_path,
                    "plot_png": plot_path,
                    "preview_wav": preview_path,
                },
            },
            handle,
            indent=2,
        )
    print(f"  {config_path}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
