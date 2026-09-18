"""
Smoothing kernels parameterized by mean weighting distance.

Every kernel here is specified by a single, method-independent quantity: the
mean weighting distance

    d = integral |t| * w(t) dt

of the normalized symmetric weight function w. This makes methods directly
comparable -- a boxcar and a Gaussian with the same mean weighting distance
smooth by the same amount, even though their shapes and supports differ.

For a boxcar of full width W the mean weighting distance is W/4, so a
one-minute boxcar corresponds to d = 15 seconds.

Continuous scale factors (half-width h, or standard deviation sigma):

    boxcar       h = 2 d
    triangular   h = 3 d
    gaussian     sigma = d * sqrt(pi/2)  ~= 1.2533 d
    tricube      h = (22/7) d            ~= 3.1429 d
    loess        tricube span, h = (22/7) d

The scale actually used is solved numerically so that the *discrete* kernel on
the sample grid has exactly the requested mean weighting distance. The
continuous factors above are the large-window limit of that solve.

Smoothing is mask-aware: samples marked invalid are dropped from the weighted
average and the remaining weights are renormalized. A gap in the valid data
therefore interpolates across from the valid samples on either side instead of
pulling the result toward zero. Edges are handled by the same renormalization.
"""

import numpy as np
from scipy.ndimage import correlate1d
from scipy.optimize import brentq

METHODS = ("boxcar", "triangular", "gaussian", "tricube", "loess")

# Large-window limit of scale / mean_distance, for reference and testing.
CONTINUOUS_SCALE_FACTOR = {
    "boxcar": 2.0,
    "triangular": 3.0,
    "gaussian": np.sqrt(np.pi / 2.0),
    "tricube": 22.0 / 7.0,
    "loess": 22.0 / 7.0,
}

# Number of standard deviations at which the Gaussian is truncated. Generous,
# so the truncated tail weight stays negligible and the kernel varies smoothly
# with the scale parameter.
GAUSSIAN_TRUNCATE = 5.0


def _shape(method, u):
    """Unnormalized weight for offsets u, in units of the scale parameter."""
    if method == "triangular":
        return np.clip(1.0 - np.abs(u), 0.0, None)
    if method == "gaussian":
        return np.exp(-0.5 * u * u)
    if method in ("tricube", "loess"):
        return np.clip(1.0 - np.abs(u) ** 3, 0.0, None) ** 3
    raise ValueError(f"unknown smoothing method: {method!r} (expected one of {METHODS})")


def kernel(method, scale_samples):
    """
    Build a normalized, symmetric discrete kernel.

    scale_samples is the half-width in samples for the compact kernels, or the
    standard deviation in samples for the Gaussian. Returns (offsets, weights)
    where offsets runs from -K to K.
    """
    if scale_samples <= 0.0:
        raise ValueError(f"scale must be positive, got {scale_samples}")

    if method == "gaussian":
        half = int(np.ceil(GAUSSIAN_TRUNCATE * scale_samples))
        offsets = np.arange(-half, half + 1, dtype=float)
        weights = _shape(method, offsets / scale_samples)
    elif method == "boxcar":
        # Area overlap between each sample's cell [k-1/2, k+1/2] and the window
        # [-h, h]. This is the exact discretization of a rectangular window and,
        # unlike a hard cutoff, it varies continuously with the half-width -- so
        # the solve below can hit any requested mean weighting distance.
        half = int(np.ceil(scale_samples))
        offsets = np.arange(-half, half + 1, dtype=float)
        weights = np.clip(
            np.minimum(offsets + 0.5, scale_samples)
            - np.maximum(offsets - 0.5, -scale_samples),
            0.0,
            None,
        )
    else:
        # These shapes fall to zero at |u| = 1, so including the outermost
        # partially covered sample keeps the kernel continuous in the scale.
        half = int(np.ceil(scale_samples))
        offsets = np.arange(-half, half + 1, dtype=float)
        weights = _shape(method, offsets / scale_samples)

    total = weights.sum()
    if total <= 0.0:
        raise ValueError(f"degenerate kernel for method {method!r} at scale {scale_samples}")
    return offsets, weights / total


def discrete_mean_distance(method, scale_samples, sample_interval_s):
    """Mean weighting distance, in seconds, of the discrete kernel."""
    offsets, weights = kernel(method, scale_samples)
    return float(np.abs(offsets) @ weights) * sample_interval_s


def solve_scale(method, mean_distance_s, sample_interval_s):
    """
    Find the scale (in samples) whose discrete kernel has exactly the requested
    mean weighting distance.
    """
    if method not in METHODS:
        raise ValueError(f"unknown smoothing method: {method!r} (expected one of {METHODS})")
    if mean_distance_s <= 0.0:
        raise ValueError(f"mean distance must be positive, got {mean_distance_s}")

    guess = CONTINUOUS_SCALE_FACTOR[method] * mean_distance_s / sample_interval_s

    def excess(scale):
        return discrete_mean_distance(method, scale, sample_interval_s) - mean_distance_s

    # The discrete mean distance increases monotonically with scale, so bracket
    # outward from the continuous guess.
    lo = hi = guess
    for _ in range(60):
        if excess(hi) >= 0.0:
            break
        hi *= 1.5
    else:
        raise ValueError(f"could not bracket scale for {method!r} at d={mean_distance_s}")

    for _ in range(60):
        if excess(lo) <= 0.0:
            break
        lo /= 1.5
    else:
        raise ValueError(
            f"mean distance {mean_distance_s} s is too small to resolve on a "
            f"{sample_interval_s} s grid for method {method!r}"
        )

    if excess(lo) == 0.0:
        return lo
    return brentq(excess, lo, hi, xtol=1e-10, rtol=1e-12)


def _masked_moments(values, valid, weights, powers):
    """
    Weighted moments sum_k w_k * valid[i+k] * offset_k**p * values[i+k].

    Returns one array per requested (power, use_values) pair. Offsets are
    handled by correlate1d, which computes sum_j a[i + j - K] * weights[j].
    """
    out = []
    half = (len(weights) - 1) // 2
    offsets = np.arange(-half, half + 1, dtype=float)
    for power, use_values in powers:
        signal = valid * values if use_values else valid
        kern = weights * offsets**power
        out.append(correlate1d(signal, kern, mode="constant", cval=0.0))
    return out


def smooth(values, valid, sample_interval_s, method="gaussian", mean_distance_s=15.0):
    """
    Mask-aware smoothing with the requested mean weighting distance.

    values             1-D array to smooth (uniformly sampled)
    valid              boolean array, same length; False samples are excluded
    sample_interval_s  spacing between samples, in seconds
    method             one of METHODS
    mean_distance_s    mean weighting distance, in seconds

    Samples with no valid data anywhere in their window take the nearest valid
    smoothed value. Returns a float array the same length as values.
    """
    values = np.asarray(values, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if values.shape != valid.shape:
        raise ValueError(f"values {values.shape} and valid {valid.shape} must have the same shape")
    if values.ndim != 1:
        raise ValueError(f"values must be 1-D, got shape {values.shape}")
    if not valid.any():
        raise ValueError("no valid samples to smooth: every sample is masked out")

    scale = solve_scale(method, mean_distance_s, sample_interval_s)
    _, weights = kernel(method, scale)
    mask = valid.astype(float)

    if method == "loess":
        # Local linear fit: solve the weighted 2x2 normal equations at every
        # sample and evaluate the fit at the sample itself.
        s0, s1, s2, t0, t1 = _masked_moments(
            values, mask, weights, [(0, False), (1, False), (2, False), (0, True), (1, True)]
        )
        det = s0 * s2 - s1 * s1
        # Where the design is degenerate (fewer than two distinct valid
        # offsets) fall back to the weighted mean.
        good = np.abs(det) > 1e-12 * np.maximum(s0 * s2, 1e-30)
        result = np.full(values.shape, np.nan)
        result[good] = (s2[good] * t0[good] - s1[good] * t1[good]) / det[good]
        plain = s0 > 0.0
        fallback = plain & ~good
        result[fallback] = t0[fallback] / s0[fallback]
        covered = plain
    else:
        numerator, denominator = _masked_moments(
            values, mask, weights, [(0, True), (0, False)]
        )
        covered = denominator > 0.0
        result = np.full(values.shape, np.nan)
        result[covered] = numerator[covered] / denominator[covered]

    if not covered.all():
        # Hold the nearest computed value across windows with no valid data.
        index = np.arange(len(values))
        result = np.interp(index, index[covered], result[covered])

    return result


def describe(method, mean_distance_s, sample_interval_s):
    """Human-readable summary of the kernel actually used, for logging."""
    scale = solve_scale(method, mean_distance_s, sample_interval_s)
    scale_s = scale * sample_interval_s
    _, weights = kernel(method, scale)
    span_s = (len(weights) - 1) * sample_interval_s
    label = "sigma" if method == "gaussian" else "half-width"
    return (
        f"{method}: mean weighting distance {mean_distance_s:g} s, "
        f"{label} {scale_s:.3f} s, full support {span_s:.3f} s ({len(weights)} samples)"
    )
