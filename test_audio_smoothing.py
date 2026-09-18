#!/usr/bin/env python3
"""
Checks for audio_smoothing.py and the loudness measurement in
audio_level_curve.py.

These verify the two claims the audio leveling workflow rests on: that a
kernel's mean weighting distance is what the user asked for, and that the
loudness measurement is calibrated to the BS.1770 reference.

Run with:  python test_audio_smoothing.py
"""

import os
import sys
import tempfile

import numpy as np
import soundfile as sf

import audio_level_curve
import audio_smoothing

DT = 0.1


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}{'  ' + detail if detail else ''}")
    if not condition:
        check.failures += 1


check.failures = 0


def test_mean_distance_is_honored():
    print("\nmean weighting distance matches the request")
    for method in audio_smoothing.METHODS:
        worst = 0.0
        for requested in (1.0, 5.0, 15.0, 60.0):
            scale = audio_smoothing.solve_scale(method, requested, DT)
            actual = audio_smoothing.discrete_mean_distance(method, scale, DT)
            worst = max(worst, abs(actual - requested) / requested)
        check(f"{method:11s}", worst < 1e-6, f"worst relative error {worst:.2e}")


def test_boxcar_matches_the_documented_example():
    print("\na one-minute boxcar is a mean weighting distance of 15 s")
    scale = audio_smoothing.solve_scale("boxcar", 15.0, DT)
    _, weights = audio_smoothing.kernel("boxcar", scale)
    span = (len(weights) - 1) * DT
    check("full support is 60 s", abs(span - 60.0) < 0.15, f"got {span:.2f} s")


def test_constant_input_is_preserved():
    print("\na constant signal is returned unchanged")
    values = np.full(2000, -13.7)
    valid = np.ones(2000, dtype=bool)
    for method in audio_smoothing.METHODS:
        result = audio_smoothing.smooth(values, valid, DT, method, 15.0)
        deviation = np.abs(result + 13.7).max()
        check(f"{method:11s}", deviation < 1e-9, f"max deviation {deviation:.2e}")


def test_loess_follows_a_ramp_into_the_edges():
    print("\nloess extrapolates at the edges where symmetric kernels flatten")
    n = 2000
    values = 0.01 * np.arange(n) * DT
    valid = np.ones(n, dtype=bool)
    symmetric = audio_smoothing.smooth(values, valid, DT, "gaussian", 15.0)
    local_linear = audio_smoothing.smooth(values, valid, DT, "loess", 15.0)
    check(
        "gaussian lags at the edge",
        abs(symmetric[0] - values[0]) > 0.1,
        f"error {abs(symmetric[0] - values[0]):.4f}",
    )
    check(
        "loess does not",
        abs(local_linear[0] - values[0]) < 1e-6,
        f"error {abs(local_linear[0] - values[0]):.2e}",
    )


def test_gate_interpolates_across_silence():
    print("\ngated silence interpolates instead of dragging the curve down")
    n = 3000
    values = np.full(n, -20.0)
    values[1000:1500] = -80.0
    valid = values > -50.0

    masked = audio_smoothing.smooth(values, valid, DT, "gaussian", 15.0)
    unmasked = audio_smoothing.smooth(values, np.ones(n, dtype=bool), DT, "gaussian", 15.0)
    check(
        "masked smoothing holds the surrounding level",
        np.allclose(masked, -20.0, atol=1e-9),
        f"min {masked.min():.4f}",
    )
    check(
        "unmasked would dip (the behavior being avoided)",
        unmasked.min() < -30.0,
        f"min {unmasked.min():.2f}",
    )

    values = np.concatenate([np.full(1000, -30.0), np.full(500, -80.0), np.full(1000, -10.0)])
    valid = values > -50.0
    across = audio_smoothing.smooth(values, valid, DT, "gaussian", 15.0)
    check(
        "an asymmetric gap interpolates between the two levels",
        -30.0 < across[1250] < -10.0,
        f"gap centre {across[1250]:.2f}",
    )


def test_all_masked_fails_loudly():
    print("\nno valid samples is an error, not a silent result")
    raised = False
    try:
        audio_smoothing.smooth(np.zeros(10), np.zeros(10, dtype=bool), DT)
    except ValueError:
        raised = True
    check("raises ValueError", raised)


def test_loudness_calibration():
    print("\nloudness is calibrated to the BS.1770 reference")
    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "sine.wav")
    for rate in (44100, 48000, 96000):
        samples = np.arange(rate * 5)
        tone = 10.0 ** (-20.0 / 20.0) * np.sin(2 * np.pi * 997.0 * samples / rate)
        sf.write(path, np.column_stack([tone, tone]), rate, subtype="FLOAT")
        mean_square, _ = audio_level_curve.measure_block_power(path, int(0.1 * rate))
        loudness = audio_level_curve.momentary_loudness(mean_square, 4)
        interior = loudness[len(loudness) // 4 : 3 * len(loudness) // 4].mean()
        # Two identical channels sum to +3.01 dB, a sine's RMS is -3.01 dB of
        # its amplitude, so the two cancel and the result is the amplitude.
        check(f"{rate} Hz, -20 dBFS 997 Hz sine", abs(interior + 20.0) < 0.05,
              f"measured {interior:+.4f} LUFS")
    os.remove(path)
    os.rmdir(directory)


def main():
    test_mean_distance_is_honored()
    test_boxcar_matches_the_documented_example()
    test_constant_input_is_preserved()
    test_loess_follows_a_ramp_into_the_edges()
    test_gate_interpolates_across_silence()
    test_all_masked_fails_loudly()
    test_loudness_calibration()

    print()
    if check.failures:
        print(f"{check.failures} check(s) FAILED")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
