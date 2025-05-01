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


def post_process(csv, prefix, vars, metrics, process_list, ticks_per_frame, cc_number, filter_width):

    if not os.path.exists(f"../video_midi/{prefix}"):
        os.makedirs(f"../video_midi/{prefix}")

    master_dict = {}
    for var in vars:
        for metric in metrics:
            key = var + "_" + metric # The base metric is the "_pos" version
            if key not in csv.columns:
                continue
            process_dict = {}
            triangle_filter = triangular_filter_odd(csv[key], filter_width)
            process_dict[var + "_" + metric + "_p" ] = scale_data(triangle_filter)


            if "rank" in process_list:
                process_dict_copy = process_dict.copy()
                for key in process_dict_copy:
                    normalized_metric = percentile_data(process_dict_copy[key])
                    process_dict[key + "_r"] = normalized_metric

            # apply an additional 25 imot wode troamgular filter upon request
            if "f25" in process_list:
                process_dict_copy = process_dict.copy()
                for key in process_dict_copy:
                    process_dict[key+"_f33"] = triangular_filter_odd(process_dict_copy[key], 33)

            if "neg" in process_list:
                process_dict_copy = process_dict.copy()
                for key in process_dict_copy:
                    process_dict[key + "_n"] = 1.0 - scale_data(process_dict_copy[key])

            if "power" in process_list:
                process_dict_copy = process_dict.copy()
                for key in process_dict_copy:
                    process_dict[key + "_p4"] = process_dict_copy[key] ** 4
                    process_dict[key + "_n4"] = 1 - (1 - process_dict_copy[key]) ** 4

            if "inv" in process_list:
                process_dict_copy = process_dict.copy()
                for key in process_dict_copy:
                    # we only want the inverse of the metric if there is a p4 or n4 version
                    if "_p4" in key or "_n4" in key:
                        process_dict[key + "_i"] = 1.0 - process_dict_copy[key]

            master_dict.update(process_dict)

            # now write out everything for this var and metric as a single midi file


        frame_count_list = csv.index.tolist()
        for key in process_dict:
            midi_file = mido.MidiFile()
            midi_val_base = [round(104 * val) for val in process_dict[key]]
            track_base = mido.MidiTrack()
            track_base.append(mido.MetaMessage('track_name', name=key, time=0))
            for i, midi_value in enumerate(midi_val_base):
                time_tick = 0 if i == 0 else int(ticks_per_frame * (frame_count_list[i] - frame_count_list[i - 1]))
                track_base.append(
                    mido.Message('control_change',
                            control=cc_number,
                            value=midi_value,
                            channel=0,
                            time=time_tick))
            midi_file.tracks.append(track_base)
                    
        midi_file.save("../video_midi/" + prefix + "/" + var + "_" + metric + ".mid")

    # write out the master xlsx
    master_df = pd.DataFrame(master_dict)
    master_df.to_excel(prefix + "_derived.xlsx", index=False)

                
















if __name__ == "__main__":
    prefix = "MzUL2-5jm3f"
    csv = pd.read_csv(prefix + ".csv", index_col=0)

    vars= ["R", "G", "B","Gray","HSV"]
    metric_names = ["avg", "var", "lrg", "xps", "rfl", "rad", "lin","ee1","ee2","ee3","ed1","ed2","ed3","es1","es2","es3"]
    process_list = ["neg","rank", "power","inv","f33"]
    ticks_per_frame = 480
    cc_number = 1
    filter_width = 5

    post_process(csv, prefix, vars, metric_names, process_list, ticks_per_frame, cc_number, filter_width)


