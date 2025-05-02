# reads in the output of process.py, calculated derived values and puts them out
# both as a csv file and as midi files.

import pandas as pd
import numpy as np
import mido
import os
from scipy.stats import rankdata

def percentile_data(data):
    """
    Transform the vector <data> into a percentile list where 0 is the lowest and 1 the highest.
    """
    ranks = rankdata(data, method='average')
    percentiles = (ranks-1) / (len(data)-1)
    return percentiles

def scale_data(data):
    """
    Scale the vector <data> so that 0 is the lowest and 1 the highest.
    """
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val > min_val:
        scaled_data = (data - min_val) / (max_val - min_val)
    else:
        scaled_data = np.zeros_like(data)
    return scaled_data

def triangular_filter_odd(data, N):
    if N < 1:
        raise ValueError("Filter length N must be at least 1.")
    if N % 2 == 0:
        raise ValueError("Triangular filter requires odd N.")

    data = np.asarray(data)
    half_window = N // 2

    # Create triangular weights
    weights = np.arange(1, half_window + 2)
    weights = np.concatenate([weights, weights[:-1][::-1]])
    weights = weights / weights.sum()  # Normalize to sum to 1

    # Pad data at both ends using edge values
    padded = np.pad(data, pad_width=half_window, mode='edge')

    # Apply convolution
    filtered = np.convolve(padded, weights, mode='valid')
    return filtered


def post_process(csv, prefix, vars, metrics, process_list, ticks_per_beat, beats_per_minute, frames_per_second, cc_number, filter_narrow, filter_wide):

    if not os.path.exists(f"../video_midi/{prefix}"):
        os.makedirs(f"../video_midi/{prefix}")

    master_dict = {}
    for var in vars:
        for metric in metrics:
            key = var + "_" + metric # The base metric is the "_pos" version
            if key not in csv.columns:
                continue
            process_dict = {}

            if "rank" in process_list:
                process_dict[key + "_v"] = csv[key]
                process_dict[key + "_r"] = percentile_data(csv[key])            # first apply a narrow triangular filter
            
            
                        # apply an additional 25 imot wode troamgular filter upon request
            if "filter" in process_list:
                process_dict_copy = process_dict.copy()
                for key in process_dict_copy:
                    process_dict[key + "_f" + str(filter_narrow)] = triangular_filter_odd(process_dict_copy[key], filter_narrow)
                    process_dict[key + "_f" + str(filter_wide)] = triangular_filter_odd(process_dict_copy[key], filter_wide)

            # scale the data to 0-1
            process_dict_copy = process_dict.copy()
            for key in process_dict_copy:
                process_dict[key] = scale_data(process_dict[key])

            if "power" in process_list:
                process_dict_copy = process_dict.copy()
                for key in process_dict_copy:
                    process_dict[key + "_p4"] = process_dict_copy[key] ** 4
                    process_dict[key + "_n4"] = 1 - (1 - process_dict_copy[key]) ** 4

            if "inv" in process_list:
                process_dict_copy = process_dict.copy()
                for key in process_dict_copy:
                    process_dict[key + "_i"] = 1.0 - process_dict_copy[key]

            master_dict.update(process_dict)

            # now write out everything for this var and metric as a single midi file

        ticks_per_frame = ticks_per_beat * beats_per_minute / (60 *frames_per_second)
        frame_count_list = csv.index.tolist()
        for var in vars:
            for metric in metrics:
                for suffix in ["v", "r"]:
                    key = var + "_" + metric  +  "_" + suffix # The base metric is the "_pos" version
                    if key not in master_dict:
                        continue
                    midi_file = mido.MidiFile()
                    for full_key in master_dict:
                        if key not in full_key:
                            continue
                        midi_track = mido.MidiTrack()
                        midi_file.tracks.append(midi_track)

                        midi_track.append(mido.MetaMessage('track_name', name=full_key, time=0))

                        midi_val_base = [round(104 * val) for val in master_dict[full_key]]

                        for i, midi_value in enumerate(midi_val_base):
                            time_tick = 0 if i == 0 else int(ticks_per_frame * (frame_count_list[i] - frame_count_list[i - 1]))
                            midi_track.append(
                                mido.Message('control_change',
                                        control=cc_number,
                                        value=midi_value,
                                        channel=0,
                                        time=time_tick))

                        
                    midi_file.save("../video_midi/" + prefix + "/" + key + ".mid")

    # write out the master xlsx
    master_dict['frame_count_list'] = frame_count_list
    master_df = pd.DataFrame(master_dict)
    # Reorder columns to put frame_count_list first
    cols = ['frame_count_list'] + [col for col in master_df.columns if col != 'frame_count_list']
    master_df = master_df[cols]
    master_df.to_excel(prefix + "_derived.xlsx", index=False)
    print(f"Derived data saved to {prefix}_derived.xlsx")



if __name__ == "__main__":
    prefix = "MzUL2-5jm3f"
    csv = pd.read_csv(prefix + "_basic.csv", index_col=0)

    vars= ["R", "G", "B","Gray","HSV"]
    metric_names = ["avg", "var", "lrg", "xps", "rfl", "rad", "lmd","l10","l90","ee1","ee2","ee3","ed1","ed2","ed3","es1","es2","es3"]
    process_list = ["neg","rank", "power","inv","filter"]
    ticks_per_beat = 480
    beats_per_minute=92
    frames_per_second=30
    cc_number = 1
    filter_narrow = 5
    filter_wide = 25

    post_process(csv, prefix, vars, metric_names, process_list, ticks_per_beat, beats_per_minute, frames_per_second, cc_number, filter_narrow, filter_wide)


