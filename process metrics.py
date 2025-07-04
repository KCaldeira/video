# reads in the output of process.py, calculated derived values and puts them out
# both as a csv file and as midi files.

import pandas as pd
import numpy as np
import mido
import os
from scipy.stats import rankdata
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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
    # check if either max or min is a Nan
    if np.isnan(max_val) or np.isnan(min_val):
        scaled_data = np.zeros_like(data)
    elif max_val > min_val:
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

    # replace NA values with 0
    csv = csv.fillna(0)

    master_dict = {}
    for var in vars:
        for metric in metrics:
            key = var + "_" + metric # The base metric is the "_pos" version
            if key not in csv.columns:
                continue
            process_dict = {}

            process_dict[key + "_v"] = csv[key]

            if "rank" in process_list:
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
                    # process_dict[key + "_n4"] = 1 - (1 - process_dict_copy[key]) ** 4  # This is almost redundant with the _p4_i

            if "inv" in process_list:
                process_dict_copy = process_dict.copy()
                for key in process_dict_copy:
                    process_dict[key + "_i"] = 1.0 - process_dict_copy[key]

            if "derivative" in process_list:  # derivative
                process_dict_copy = process_dict.copy()
                for key in process_dict_copy:
                    if not key.endswith("_i"):
                        process_dict[key + "_dp"] = np.maximum(0, np.gradient(process_dict_copy[key]))
                        process_dict[key + "_dn"] = np.maximum(0, -np.gradient(process_dict_copy[key]))

            master_dict.update(process_dict)

    # convert any Nan's in master_dict to 0
    for key in master_dict:
        master_dict[key] = np.where(np.isnan(master_dict[key]), 0, master_dict[key])
    
    # now write out everything for this var and averaging period and var or ranking as a single midi file
    print(f"Writing out midi files by var, rank/value, and averaging period")
    ticks_per_frame = ticks_per_beat * beats_per_minute / (60 *frames_per_second)
    frame_count_list = csv.index.tolist()
    for var in vars:
        for averaging in ["f25","f5",""]:
            for suffix in ["v", "r"]:
                for derivative in ["","d"]:
                    midi_file = mido.MidiFile()
                    for key in master_dict:
                        if not key.startswith(var):
                            continue
                        if averaging != "" and "_" + averaging+"_" not in key and not key.endswith("_"+averaging):
                            continue
                        if averaging == "" and ("_f25_" in key or "_f5_" in key or key.endswith("_f25") or key.endswith("_f5")):
                            continue
                        if not "_" + suffix + "_" in key and not key.endswith("_"+suffix):
                            continue
                        if derivative == "d" and ("_dp_" not in key and "_dn_" not in key and not key.endswith("_dp") and not key.endswith("_dn")):
                            continue
                        if derivative == "" and ("_dp_" in key or "_dn_" in key or key.endswith("_dp") or key.endswith("_dn")):
                            continue                        


                        midi_track = mido.MidiTrack()
                        midi_file.tracks.append(midi_track)

                        midi_track.append(mido.MetaMessage('track_name', name=key, time=0))

                        midi_val_base = (np.round(104 * master_dict[key])).astype(int).tolist()

                        for i, midi_value in enumerate(midi_val_base):
                            time_tick = 0 if i == 0 else int(ticks_per_frame * (frame_count_list[i] - frame_count_list[i - 1]))
                            midi_track.append(
                                mido.Message('control_change',
                                        control=cc_number,
                                        value=midi_value,
                                        channel=7,
                                        time=time_tick))

                        
                    file_name = "../video_midi/" + prefix + "/" + var + "_" + suffix 
                    if averaging != "":
                        file_name += "_" + averaging
                    if derivative != "":
                        file_name += "_" + derivative   
                    file_name += f"_{filter_narrow}_{filter_wide}.mid"
                    midi_file.save(file_name)

    # now write everything for this metric and postprocessing in a single midi file
    print(f"Writing out midi files by postprocessing methods")
    ticks_per_frame = ticks_per_beat * beats_per_minute / (60 *frames_per_second)
    frame_count_list = csv.index.tolist()

    # now get a list of all of the unique key endings after the second underscore
    suffix_list = ["_".join(key.split("_")[1:]) for key in master_dict.keys()]
    suffix_list = list(set(suffix_list))
    for suffix in suffix_list:
        suffix_key_list = [key for key in master_dict.keys() if key.endswith(suffix)]
        midi_file = mido.MidiFile()

        for key in suffix_key_list:
            midi_track = mido.MidiTrack()
            midi_file.tracks.append(midi_track)
            midi_track.append(mido.MetaMessage('track_name', name=key, time=0))

            midi_val_base = (np.round(104 * master_dict[key])).astype(int).tolist()

            for i, midi_value in enumerate(midi_val_base):
                time_tick = 0 if i == 0 else int(ticks_per_frame * (frame_count_list[i] - frame_count_list[i - 1]))
                midi_track.append(
                    mido.Message('control_change',
                            control=cc_number,
                            value=midi_value,
                            channel=7,
                            time=time_tick))
                    
        midi_file.save("../video_midi/" + prefix + "/" + suffix + f"_{filter_narrow}_{filter_wide}.mid")

    # write out the master xlsx
    print(f"Writing out master xlsx")
    master_dict['frame_count_list'] = frame_count_list
    master_df = pd.DataFrame(master_dict)
    # sort the columns alphabetically
    master_df = master_df.sort_index(axis=1)        
    # Reorder columns to put frame_count_list first
    cols = ['frame_count_list'] + [col for col in master_df.columns if col != 'frame_count_list']
    master_df = master_df[cols]
    master_df.to_excel(prefix + f"_{filter_narrow}_{filter_wide}_derived.xlsx", index=False)
    print(f"Derived data saved to {prefix}_{filter_narrow}_{filter_wide}_derived.xlsx")

    # prepare sorted keys for the plots
    keys_list = list(master_dict.keys())  # Convert keys to list first
    # remove "frame_count_list" from the list
    keys_list = [key for key in keys_list if key != "frame_count_list"]
    
    # Create sort tuples for each key
    sort_tuples = []
    for key in keys_list:

        var_name = key.split("_")[0]

        # Calculate the primary sort values
        is_r = 1 if "_r_" in key or key.endswith("_r") else 0
        is_v = 1 if "_v" in key or key.endswith("_v") else 0
        is_f25 = 1 if "_f25_" in key or key.endswith("_f25") else 0
        is_f5 = 1 if "_f5_" in key or key.endswith("_f5") else 0
        is_f0 = 1 if is_f25 == 0 and is_f5 == 0 else 0
        is_i = 1 if "_i" in key or key.endswith("_i") else 0
        is_p4 = 1 if "_p4" in key or key.endswith("_p4") else 0
        is_n4 = 1 if "_n4" in key or key.endswith("_n4") else 0
        is_dp = 1 if "_dp_" in key or key.endswith("_dp") else 0
        is_dn = 1 if "_dn_" in key or key.endswith("_dn") else 0
        

        # Calculate the primary sort value
        primary_sort = (
                         1e8*is_f0 + 1e7*is_f5 +
                         0*is_r + 1e6*is_v +
                         1e4*is_p4 + 1e5*is_n4 +
                         1e2*is_dp + 1e3*is_dn +
                         1e1*is_i +
                         0)
        
        # Create a tuple with primary sort value and the key itself for alphabetical sorting
        sort_tuples.append((var_name,primary_sort, key))
    
    # Sort first by primary sort value, then alphabetically by key
    sort_tuples.sort(key=lambda x: (x[0], x[1],x[2]))
    
    # Create a dictionary to store sort values for each key
    sort_values = {key: sort_val for prefix,sort_val, key in sort_tuples}
    sorted_keys = [t[2] for t in sort_tuples]  # Extract just the keys in sorted order
    
    # write out a pdf book of plots of each of the metrics
    print(f"Writing out pdf book of plots of each of the metrics")
    pdf = PdfPages(prefix + f"_{filter_narrow}_{filter_wide}_plots.pdf")
    
    plt.rcParams['figure.max_open_warning'] = 50  # Allow more figures before warning

    # Plot 30 plots per page (5 rows x 6 columns)
    for i in range(0, len(sorted_keys), 30):  # Process 30 keys at a time
        plt.figure(figsize=(20, 15))  # Larger figure to accommodate 30 plots
        
        # Get up to 30 keys for this page
        current_keys = sorted_keys[i:i+30]
        
        # Create a 5x6 grid of subplots
        for j, key in enumerate(current_keys):
            plt.subplot(5, 6, j+1)
            plt.plot(master_dict[key])
            # Get the sort value for this key and format it
            sort_val = sort_values[key]
            plt.title(f"{key} {sort_val}", fontsize=8)  # Smaller font size for titles
            plt.grid(True)
            plt.xticks([])  # Remove x-axis ticks for cleaner look
            plt.yticks([])  # Remove y-axis ticks for cleaner look
        
        plt.tight_layout()  # Adjust spacing between subplots
        plt.savefig(pdf, format="pdf")
        plt.close()  # Close the figure to free memory
    
    pdf.close()




if __name__ == "__main__":
    #prefix = "MzUL2-5jm3f"
    #prefix = "N3_M5zulPentAf2-V3A"
    #prefix = "N2_M3toBSy25f-1"
    #prefix = "N6_BSt-3DAf"
    prefix = "N8_M3toM2µa7fC2"

    csv = pd.read_csv(prefix + "_basic.csv", index_col=0)

    vars= ["R", "G", "B","Gray","H000","H060","H120","H180","H240","H300","H360","Hmon"]
    metric_names = ["avg", "var", "lrg", "xps", "rfl", "rad", "lmd","l10","l90","dcd","dcl","ee1","ee2","ee3","ed1","ed2","ed3","es1","es2","es3",
                    "std","int"]
    process_list = ["neg","rank", "power","inv","filter","derivative"]
    ticks_per_beat = 480
    beats_per_minute=84
    frames_per_second=30
    cc_number = 1
    beats_per_midi_event = 1
    filter_narrow = 9 # 9 is about 2 bars if every midi event is a beat
    filter_wide = 65 # 65 is about 16 bars if every midi event is a beat

    post_process(csv, prefix, vars, metric_names, process_list, ticks_per_beat, beats_per_minute, frames_per_second, cc_number, filter_narrow, filter_wide)


