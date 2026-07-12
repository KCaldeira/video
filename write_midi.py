"""MIDI rendering stage.

This is the only stage that knows about tempo.  It reads the frame-indexed CSVs
produced by the metrics and clustering stages, builds a frame-to-tick mapping
from the tempo specification (constant BPM or a MIDI tempo file), and writes the
`.mid` files.  Because it consumes only CSVs, MIDI can be re-rendered at a new
tempo without recomputing metrics or clusters:

    python write_midi.py json/N42_post.json
"""

import json
import os
import sys

import mido
import numpy as np
import pandas as pd

from tempo_map import build_frame_tick_map

# Continuous metrics are scaled to this range (0..CC_VALUE_SCALE) before rounding
# to an integer MIDI CC value; matches the original process_metrics behaviour.
CC_VALUE_SCALE = 104
MIDI_CHANNEL = 7


def _append_cc_track(midi_file, track_name, values_0_127, tick_list, control,
                     channel=MIDI_CHANNEL, tempo_events=None):
    """Append one control_change track, merging tempo_events onto it when given.

    values_0_127 and tick_list are parallel per-frame sequences.  tempo_events is
    a list of (abs_tick, MetaMessage); when provided (first track of a file) the
    events are interleaved with the CC events by absolute tick, with tempo/time
    signature ordered ahead of a CC event at the same tick.
    """
    track = mido.MidiTrack()
    midi_file.tracks.append(track)
    track.append(mido.MetaMessage('track_name', name=track_name, time=0))

    events = []  # (abs_tick, order, msg)
    if tempo_events:
        for abs_tick, msg in tempo_events:
            events.append((abs_tick, 0, msg.copy(time=0)))
    for i, value in enumerate(values_0_127):
        events.append((tick_list[i], 1,
                       mido.Message('control_change', control=control,
                                    value=int(value), channel=channel, time=0)))

    events.sort(key=lambda e: (e[0], e[1]))
    prev_tick = 0
    for abs_tick, _order, msg in events:
        msg.time = abs_tick - prev_tick
        prev_tick = abs_tick
        track.append(msg)


def render_metrics_midi(values_csv, output_dir, frame_tick_map, cc_number,
                        filter_periods, block_beats, stretch_values, stretch_centers):
    """Render the metrics MIDI files from the frame-indexed values CSV.

    Reproduces the two original groupings: (1) one file per
    var/rank/averaging/stretch, and (2) one file per postprocessing base_suffix.
    Tracks within a file are ordered by descending mean; the first track carries
    the tempo map.
    """
    df = pd.read_csv(values_csv)
    frame_list = df['frame_count_list'].tolist()
    value_cols = [c for c in df.columns if c != 'frame_count_list']
    master_dict = {c: df[c].to_numpy() for c in value_cols}

    tick_list = frame_tick_map.ticks_for_frames(frame_list)
    tempo_events = frame_tick_map.tempo_events()
    ticks_per_beat = frame_tick_map.ticks_per_beat

    variables = sorted(set(c.split('_')[0] for c in value_cols))
    averaging_suffixes = [f"f{period:03d}" for period in filter_periods] + \
                         [f"b{beats:03d}" for beats in block_beats]

    midi_output_dir = os.path.join(output_dir, "metrics_midi")
    os.makedirs(midi_output_dir, exist_ok=True)

    def midi_values(key):
        return np.round(CC_VALUE_SCALE * master_dict[key]).astype(int).tolist()

    # --- Grouping 1: one file per var / rank / averaging / stretch ---
    print("Writing out midi files by var, rank/value, and averaging period")
    for var in variables:
        for averaging in averaging_suffixes:
            for rank_type in ["v", "r"]:
                stretch_values_to_process = list(stretch_values)
                if 1 not in stretch_values_to_process:
                    stretch_values_to_process.append(1)

                stretch_centers_to_process = list(stretch_centers)
                if 0.5 not in stretch_centers_to_process:
                    stretch_centers_to_process.append(0.5)

                for stretch_value in stretch_values_to_process:
                    for stretch_center in stretch_centers_to_process:
                        midi_file = mido.MidiFile(ticks_per_beat=ticks_per_beat)

                        matching_keys = []
                        for inversion_type in ["o", "i"]:
                            for key in master_dict:
                                if "_" + averaging + "_" not in key:
                                    continue
                                if not key.startswith(f"{var}_"):
                                    continue
                                if f"_{rank_type}_{averaging}_s{stretch_value}-{stretch_center}" not in key:
                                    continue
                                if not key.endswith(f"_{inversion_type}"):
                                    continue
                                matching_keys.append(key)

                        matching_keys.sort(key=lambda k: np.mean(master_dict[k]), reverse=True)

                        for ix, key in enumerate(matching_keys):
                            _append_cc_track(
                                midi_file, key, midi_values(key), tick_list,
                                control=cc_number,
                                tempo_events=tempo_events if ix == 0 else None)

                        if midi_file.tracks:
                            file_name = f"{midi_output_dir}/{var}_{rank_type}_{averaging}_s{stretch_value}-{stretch_center}.mid"
                            midi_file.save(file_name)

    # --- Grouping 2: one file per postprocessing base_suffix ---
    print("Writing out midi files by postprocessing methods")
    base_suffix_set = set()
    for key in master_dict:
        suffix = "_".join(key.split("_")[1:])
        base_suffix_set.add(suffix.split("_s")[0] if "_s" in suffix else suffix)

    for base_suffix in base_suffix_set:
        suffix_key_list = []
        for key in master_dict:
            key_suffix = "_".join(key.split("_")[1:])
            key_base_suffix = key_suffix.split("_s")[0] if "_s" in key_suffix else key_suffix
            if key_base_suffix == base_suffix:
                suffix_key_list.append(key)

        suffix_key_list.sort(key=lambda k: np.mean(master_dict[k]), reverse=True)

        midi_file = mido.MidiFile(ticks_per_beat=ticks_per_beat)
        for ix, key in enumerate(suffix_key_list):
            _append_cc_track(
                midi_file, key, midi_values(key), tick_list,
                control=cc_number,
                tempo_events=tempo_events if ix == 0 else None)

        midi_file.save(f"{midi_output_dir}/{base_suffix}.mid")


def render_cluster_midi(clusters_csv, output_dir, base_name, frame_tick_map):
    """Render cluster MIDI files (one per algorithm/k, binary 127/0 per cluster)."""
    cluster_df = pd.read_csv(clusters_csv, index_col=0)
    frame_list = cluster_df.index.tolist()

    tick_list = frame_tick_map.ticks_for_frames(frame_list)
    tempo_events = frame_tick_map.tempo_events()
    ticks_per_beat = frame_tick_map.ticks_per_beat

    midi_output_dir = os.path.join(output_dir, "clusters_midi")
    os.makedirs(midi_output_dir, exist_ok=True)

    cluster_cols = [c for c in cluster_df.columns if c.startswith(('kmeans_', 'gmm_'))]

    # Group columns by algorithm + k_value; within a group, key by boxcar period.
    grouped_cols = {}
    for col in cluster_cols:
        parts = col.split('_')
        base_key = f"{parts[0]}_{parts[1]}"
        boxcar_period = parts[2] if len(parts) > 2 and parts[2].startswith('b') else 'b001'
        grouped_cols.setdefault(base_key, {})[boxcar_period] = col

    print("\nGenerating MIDI files from cluster assignments")
    for base_key, tracks_dict in sorted(grouped_cols.items()):
        midi_file = mido.MidiFile(ticks_per_beat=ticks_per_beat)
        k_value = int(base_key.split('_k')[1])

        for boxcar_period in sorted(tracks_dict.keys()):
            col_name = tracks_dict[boxcar_period]
            cluster_assignments = cluster_df[col_name].to_numpy()

            for cluster_num in range(k_value):
                is_first_track = not midi_file.tracks
                values = np.where(cluster_assignments == cluster_num, 127, 0)
                _append_cc_track(
                    midi_file, f"{col_name}_c{cluster_num}", values, tick_list,
                    control=1,
                    tempo_events=tempo_events if is_first_track else None)

        midi_filename = os.path.join(midi_output_dir, f"{base_name}_{base_key}.mid")
        midi_file.save(midi_filename)
        print(f"  Saved {midi_filename} with {len(midi_file.tracks)} tracks")


def write_midi_from_config(config):
    """Render metrics and cluster MIDI for a pipeline config dict."""
    timing = config.get("timing", {})
    metrics = config.get("metrics_processing", {})
    video_name = config["video"]["video_name"]
    preset = config.get("video_processing", {}).get("optical_flow", {}).get("preset", "default")

    beats_per_minute = timing.get("beats_per_minute", 64)
    frames_per_second = timing.get("frames_per_second", 30)
    ticks_per_beat = timing.get("ticks_per_beat", 480)

    # video_name may contain a subdirectory (e.g. "N44/N44_testgi2"): the full
    # path names the output directory, but file names use only the basename.
    output_prefix = f"{video_name}_{preset}"
    output_dir = f"data/output/{output_prefix}"
    name_prefix = f"{os.path.basename(video_name)}_{preset}"

    frame_tick_map = build_frame_tick_map(beats_per_minute, frames_per_second, ticks_per_beat)

    values_csv = f"{output_dir}/{name_prefix}_values.csv"
    if os.path.exists(values_csv):
        print(f"Using metrics values CSV: {values_csv}")
        render_metrics_midi(
            values_csv, output_dir, frame_tick_map,
            cc_number=metrics.get("cc_number", 1),
            filter_periods=metrics.get("filter_periods", [17, 65, 257]),
            block_beats=metrics.get("block_beats", []),
            stretch_values=metrics.get("stretch_values", [8]),
            stretch_centers=metrics.get("stretch_centers", [0.33, 0.67]))
    else:
        print(f"No metrics values CSV at {values_csv}; skipping metrics MIDI")

    clusters_csv = f"{output_dir}/{name_prefix}_clusters.csv"
    if os.path.exists(clusters_csv):
        print(f"Using clusters CSV: {clusters_csv}")
        render_cluster_midi(clusters_csv, output_dir, name_prefix, frame_tick_map)
    else:
        print(f"No clusters CSV at {clusters_csv}; skipping cluster MIDI")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Render MIDI files from already-computed metrics/cluster CSVs, "
                    "using the tempo specification in the config.")
    parser.add_argument("config", help="Path to the pipeline JSON config file (e.g. json/N42.json)")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    write_midi_from_config(config)


if __name__ == "__main__":
    sys.exit(main())
