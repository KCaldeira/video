import argparse
import glob
import json
import os
import subprocess
import time

import cv2
import numpy as np
import pandas as pd

VIDEO_EXTENSIONS = (".wmv", ".mp4")

# Bytes per pixel of the raw rgb24 stream ffmpeg writes to the pipe.
BYTES_PER_PIXEL = 3

# How often to print a progress line, in frames.
PROGRESS_INTERVAL = 1000


def resolve_video_path(video):
    """
    Resolve a video argument to a single video file path.

    Accepts a path to a video file, a directory holding exactly one video, or
    a bare name that is interpreted relative to data/input/ in either form.
    """
    candidates = [video, os.path.join("data", "input", video)]

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
        if os.path.isdir(candidate):
            matches = sorted(
                path
                for ext in VIDEO_EXTENSIONS
                for path in glob.glob(os.path.join(candidate, "*" + ext))
            )
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise ValueError(
                    f"Directory {candidate} holds {len(matches)} videos; "
                    f"name one explicitly: {matches}"
                )
            raise FileNotFoundError(
                f"Directory {candidate} holds no {' or '.join(VIDEO_EXTENSIONS)} file"
            )

    raise FileNotFoundError(
        f"No video found for '{video}' - tried {candidates[0]} and {candidates[1]}, "
        f"as both a file and a directory"
    )


def probe_video(video_path):
    """Return (width, height, fps) for the first video stream of video_path."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate",
            "-of", "json",
            video_path,
        ],
        stdout=subprocess.PIPE,
        check=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    numerator, denominator = stream["avg_frame_rate"].split("/")
    return stream["width"], stream["height"], float(numerator) / float(denominator)


def frame_similarity_to_csv(video_path, output_path, max_frames=None):
    """
    Compute a frame-to-frame similarity metric for every frame of a video and
    write it to CSV.

    Row i describes the transition from frame i to frame i+1, so a video of n
    frames produces n-1 rows; the final frame has no successor.
    """
    width, height, fps = probe_video(video_path)
    frame_bytes = width * height * BYTES_PER_PIXEL
    print(f"Reading {video_path}: {width}x{height} at {fps:g} fps")

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", video_path,
        "-an", "-sn",
    ]
    # Let ffmpeg stop on its own rather than closing the pipe under it, which
    # would make it report a broken pipe as an error.
    if max_frames is not None:
        command += ["-frames:v", str(max_frames)]
    command += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]

    decoder = subprocess.Popen(command, stdout=subprocess.PIPE)

    frames = []
    means = []
    stds = []
    identicals = []

    previous = None
    frame_index = 0
    reached_end = False
    start_time = time.perf_counter()

    while max_frames is None or frame_index < max_frames:
        buffer = decoder.stdout.read(frame_bytes)
        if not buffer:
            reached_end = True
            break
        if len(buffer) != frame_bytes:
            raise ValueError(
                f"Truncated frame {frame_index}: got {len(buffer)} of {frame_bytes} bytes"
            )
        current = np.frombuffer(buffer, np.uint8).reshape(height, width, BYTES_PER_PIXEL)

        if previous is not None:
            # Signed difference in int16 so that darkening and brightening do
            # not both fold into the same positive value.
            difference = cv2.subtract(current, previous, dtype=cv2.CV_16S)
            mean, std = cv2.meanStdDev(difference)
            frames.append(frame_index - 1)
            means.append(mean.ravel())
            stds.append(std.ravel())
            identicals.append(int(not difference.any()))

        previous = current
        frame_index += 1

        if frame_index % PROGRESS_INTERVAL == 0:
            elapsed = time.perf_counter() - start_time
            print(f"  {frame_index} frames, {frame_index / elapsed:.0f} fps", flush=True)

    decoder.stdout.close()
    decoder.wait()
    if reached_end and decoder.returncode != 0:
        raise RuntimeError(f"ffmpeg exited with code {decoder.returncode}")

    elapsed = time.perf_counter() - start_time
    print(f"Decoded {frame_index} frames in {elapsed:.1f} s ({frame_index / elapsed:.0f} fps)")

    means = np.array(means)
    stds = np.array(stds)
    frames = np.array(frames)

    data = pd.DataFrame({
        "frame": frames,
        "time_sec": frames / fps,
        "diff_std_sum": stds.sum(axis=1),
        "R_std": stds[:, 0],
        "G_std": stds[:, 1],
        "B_std": stds[:, 2],
        "R_mean": means[:, 0],
        "G_mean": means[:, 1],
        "B_mean": means[:, 2],
        "identical": identicals,
    })

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    data.to_csv(output_path, index=False, float_format="%.6g")
    print(f"Wrote {len(data)} rows to {output_path}")

    metric = data["diff_std_sum"]
    print(f"diff_std_sum: min {metric.min():.4f}  median {metric.median():.4f}  max {metric.max():.4f}")
    print(f"Exact-identical consecutive frame pairs: {data['identical'].sum()} of {len(data)}")
    print("10 most similar transitions (frame, diff_std_sum):")
    for frame, value in data.nsmallest(10, "diff_std_sum")[["frame", "diff_std_sum"]].values:
        print(f"  frame {int(frame):6d} -> {int(frame) + 1:6d}   {value:.4f}")

    return data


DESCRIPTION = """\
Measure how much each video frame differs from the one after it, and write the
result to CSV - one row per frame transition.

The whole video is decoded in a single pass. For each pair of consecutive
frames the signed difference frame[i+1] - frame[i] is taken per RGB channel,
and the mean and standard deviation of that difference are recorded. The
headline metric, diff_std_sum, is the sum of the three per-channel standard
deviations: near zero when consecutive frames are nearly the same, large when
the image changes a lot.

An 'identical' flag marks pairs of frames that are bit-identical at full
resolution, which is useful for finding held or duplicated frames.

Frames are compared at native resolution with no downscaling, since
downscaling attenuates the metric and can hide small differences. Decoding is
the bottleneck; expect roughly 85 frames per second, so about five and a half
minutes for a 26,000-frame 1080p video.
"""

EPILOG = """\
The video argument may be any of:

  a path to a video file
  a directory holding exactly one .wmv or .mp4
  a bare name, resolved against data/input/ as either of the above

Output columns:

  frame         index of the first frame of the pair
  time_sec      frame / fps
  diff_std_sum  R_std + G_std + B_std, the headline similarity metric
  R/G/B_std     per-channel standard deviation of frame[i+1] - frame[i]
  R/G/B_mean    per-channel mean of frame[i+1] - frame[i]
  identical     1 if the two frames are bit-identical, else 0

A video of n frames produces n-1 rows; the last frame has no successor.

Examples:

  python frame_similarity.py N48
  python frame_similarity.py data/input/N48 --max-frames 400
  python frame_similarity.py data/input/N40_M2ntg5f.wmv -o data/output/N40_sim.csv
"""


def main():
    parser = argparse.ArgumentParser(
        prog="frame_similarity.py",
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "video",
        help="video file, directory holding one video, or name under data/input/",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="output CSV path (default: data/output/{video stem}_framesim.csv)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="stop after this many frames, for quick checks (default: whole video)",
    )
    args = parser.parse_args()

    video_path = resolve_video_path(args.video)
    output_path = args.output
    if output_path is None:
        stem = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join("data", "output", f"{stem}_framesim.csv")

    frame_similarity_to_csv(video_path, output_path, args.max_frames)


if __name__ == "__main__":
    main()
