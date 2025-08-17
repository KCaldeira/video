# reads in the output of process.py, calculated derived values and puts them out
# both as a csv file and as midi files.

import pandas as pd
import numpy as np
import mido
import os
from scipy.stats import rankdata
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json

def add_derived_columns(csv, metrics):
    """
    Add derived columns to the CSV dataframe based on existing metrics.
    
    Parameters:
    - csv: pandas DataFrame containing the metrics
    - metrics: list of metric names to check for
    
    Returns:
    - csv: pandas DataFrame with additional derived columns
    """
    # Check for ee0, ee1, ee2 columns and add ee1r, ee2r if they don't exist
    ee_base_columns = ['ee0', 'ee1', 'ee2']
    ee_derived = ['ee1r', 'ee2r']
    
    # Check if any color channel has all the required ee columns
    ee_columns_exist = False
    for color_channel in ["R", "G", "B", "Gray", "S", "V"]:
        ee0_col = f"{color_channel}_ee0"
        ee1_col = f"{color_channel}_ee1"
        ee2_col = f"{color_channel}_ee2"
        if all(col in csv.columns for col in [ee0_col, ee1_col, ee2_col]):
            ee_columns_exist = True
            ee1r_col = f"{color_channel}_ee1r"
            ee2r_col = f"{color_channel}_ee2r"
            
            # Avoid division by zero
            csv[ee1r_col] = np.where(csv[ee0_col] != 0, csv[ee1_col] / csv[ee0_col], 0)
            csv[ee2r_col] = np.where(csv[ee0_col] != 0, csv[ee2_col] / csv[ee0_col], 0)
    
    # Check for es0, es1, es2 columns and add es1r, es2r if they don't exist
    es_base_columns = ['es0', 'es1', 'es2']
    es_derived = ['es1r', 'es2r']
    
    # Check if any color channel has all the required es columns
    es_columns_exist = False
    for color_channel in ["R", "G", "B", "Gray", "S", "V"]:
        es0_col = f"{color_channel}_es0"
        es1_col = f"{color_channel}_es1"
        es2_col = f"{color_channel}_es2"
        if all(col in csv.columns for col in [es0_col, es1_col, es2_col]):
            es_columns_exist = True
            es1r_col = f"{color_channel}_es1r"
            es2r_col = f"{color_channel}_es2r"
            
            # Avoid division by zero
            csv[es1r_col] = np.where(csv[es0_col] != 0, csv[es1_col] / csv[es0_col], 0)
            csv[es2r_col] = np.where(csv[es0_col] != 0, csv[es2_col] / csv[es0_col], 0)
    
    return csv

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


def post_process(csv, prefix, vars, metrics, process_list, ticks_per_beat, beats_per_minute, frames_per_second, cc_number, filter_periods, stretch_values, stretch_centers):

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

            # Always add filter suffixes to all metrics
            if "filter" in process_list:
                process_dict[key + "_v_f001"] = csv[key]
                if "rank" in process_list:
                    process_dict[key + "_r_f001"] = percentile_data(csv[key])
            else:
                process_dict[key + "_v"] = csv[key]
                if "rank" in process_list:
                    process_dict[key + "_r"] = percentile_data(csv[key])
            
            
                        # apply triangular filters for all specified periods (skip f001 as it's already created)
            if "filter" in process_list:
                process_dict_copy = process_dict.copy()
                for key in process_dict_copy:
                    for filter_period in filter_periods:
                        if filter_period == 1:
                            # Skip f001 as it's already created above
                            continue
                        else:
                            # Replace the _f001 suffix with the new filter period
                            new_key = key.replace("_f001", f"_f{filter_period:03d}")
                            process_dict[new_key] = triangular_filter_odd(process_dict_copy[key], filter_period)

            # scale the data to 0-1
            process_dict_copy = process_dict.copy()
            for key in process_dict_copy:
                process_dict[key] = scale_data(process_dict[key])

            if "stretch" in process_list and len(stretch_values) > 0:
                process_dict_copy = process_dict.copy()
                for key in process_dict_copy:
                    x = process_dict_copy[key]
                    # Remove the original key and replace with stretched versions
                    del process_dict[key]
                    # Add stretched versions
                    for stretch_value in stretch_values:
                        for stretch_center in stretch_centers:
                            process_dict[key + "_s" + str(stretch_value) + "-" + str(stretch_center)] = (x / stretch_center)**stretch_value / ((x / stretch_center)**stretch_value + ((1 - x) / (1 - stretch_center))**stretch_value)

            if "inv" in process_list:
                process_dict_copy = process_dict.copy()
                for key in process_dict_copy:
                    process_dict[key + "_i"] = 1.0 - process_dict_copy[key]



            master_dict.update(process_dict)

    # convert any Nan's in master_dict to 0
    for key in master_dict:
        master_dict[key] = np.where(np.isnan(master_dict[key]), 0, master_dict[key])
    
    # now write out everything for this var and averaging period and var or ranking as a single midi file
    print(f"Writing out midi files by var, rank/value, and averaging period")
    ticks_per_frame = ticks_per_beat * beats_per_minute / (60 *frames_per_second)
    frame_count_list = csv.index.tolist()
    
    # Create filter suffixes for averaging loop
    filter_suffixes = [f"f{period:03d}" for period in filter_periods]
    
    for var in vars:
        for averaging in filter_suffixes:
            for suffix in ["v", "r"]:
                for stretch_value in stretch_values:

                        midi_file = mido.MidiFile()
                        for key in master_dict:
                            if not key.startswith(var):
                                continue
                            if averaging != "" and "_" + averaging+"_" not in key and not key.endswith("_"+averaging):
                                continue
                            if not "_" + suffix + "_" in key and not key.endswith("_"+suffix):
                                continue
                            # Check if this key has the specific stretch value (any center)
                            if not f"_s{stretch_value}-" in key:
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
                        if averaging != "f001":
                            file_name += "_" + averaging
                        file_name += f"_s{stretch_value}.mid"
                        midi_file.save(file_name)

    # now write everything for this metric and postprocessing in a single midi file
    print(f"Writing out midi files by postprocessing methods")
    ticks_per_frame = ticks_per_beat * beats_per_minute / (60 *frames_per_second)
    frame_count_list = csv.index.tolist()

    # now get a list of all of the unique key endings after the second underscore
    suffix_list = ["_".join(key.split("_")[1:]) for key in master_dict.keys()]
    # Split on "_s" and keep only the beginning part
    base_suffix_list = []
    for suffix in suffix_list:
        if "_s" in suffix:
            base_suffix = suffix.split("_s")[0]
        else:
            base_suffix = suffix
        base_suffix_list.append(base_suffix)
    
    # Get unique base suffixes
    unique_base_suffixes = list(set(base_suffix_list))
    
    for base_suffix in unique_base_suffixes:
        # Find all keys that start with this base suffix (before any stretch processing)
        suffix_key_list = []
        for key in master_dict.keys():
            key_suffix = "_".join(key.split("_")[1:])
            if "_s" in key_suffix:
                key_base_suffix = key_suffix.split("_s")[0]
            else:
                key_base_suffix = key_suffix
            if key_base_suffix == base_suffix:
                suffix_key_list.append(key)
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
                    
        midi_file.save("../video_midi/" + prefix + "/" + base_suffix + ".mid")

    # write out the master xlsx
    print(f"Writing out master xlsx")
    master_dict['frame_count_list'] = frame_count_list
    master_df = pd.DataFrame(master_dict)
    # sort the columns alphabetically
    master_df = master_df.sort_index(axis=1)        
    # Reorder columns to put frame_count_list first
    cols = ['frame_count_list'] + [col for col in master_df.columns if col != 'frame_count_list']
    master_df = master_df[cols]
    master_df.to_excel(f"../video_midi/{prefix}/{prefix}_derived.xlsx", index=False)
    print(f"Derived data saved to ../video_midi/{prefix}/{prefix}_derived.xlsx")

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
        

        
        is_i = 1 if "_i" in key or key.endswith("_i") else 0

        

        # Calculate the primary sort value
        primary_sort = (
                         0*is_r + 1e6*is_v +
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
    pdf = PdfPages(f"../video_midi/{prefix}/{prefix}_plots.pdf")
    
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
    #prefix = "N8_M3toM2µa7fC2"
    #prefix = "N11_M8zaf-Cdeg-1"
    #prefix = "N9B_M6tonM2ta5f-2"
    #prefix = "N12_sinz2-3j2f"
    #prefix = "N13_Mz10tn3f"
    #prefix = "N18_cosz2-3-An61"
    prefix = "N17_Mz7fo6C2f"

    # Try to load config from JSON if it exists
    config_filename = f"{prefix}_config.json"
    if os.path.exists(config_filename):
        with open(config_filename, 'r') as f:
            config = json.load(f)
        ticks_per_beat = config.get("ticks_per_beat", 480)
        beats_per_minute = config.get("beats_per_minute", 100)
        frames_per_second = config.get("frames_per_second", 30)
        cc_number = config.get("cc_number", 1)
        beats_per_midi_event = config.get("beats_per_midi_event", 1)
        # You can add more parameters as needed
    else:
        ticks_per_beat = 480
        beats_per_minute=100
        frames_per_second=30
        cc_number = 1
        beats_per_midi_event = 1

    csv = pd.read_csv(prefix + "_basic.csv", index_col=0)
    
    vars= ["R", "G", "B","Gray","H000","H060","H120","H180","H240","H300","H360","Hmon"]
    metric_names = ["avg", "lrg", "xps", "rfl", "rad", "dcd","dcl",
                    "ee0","ee1","ee2","ee1r","ee2r","ed0","ed1","ed2","es0","es1","es2","es1r","es2r",
                    "std","int","czd","crd","cmg","cam","crm","ctm","cmv"]
    
    # Add derived columns
    csv = add_derived_columns(csv, metric_names)
    process_list = ["neg","rank", "stretch","inv","filter"]
    filter_periods = [1, 17, 65, 257]  # 1 = no filtering (f001), 9 is about 2 bars if every midi event is a beat at 4/4 (f009), 33 is about 8 bars if every midi event is a beat at 4/4 (f033), 65 is about 16 bars if every midi event is a beat at 4/4 (f065), 129 is about 32 bars if every midi event is a beat at 4/4 (f129)
    stretch_values = [1,  8]  # Values for stretch processing
    stretch_centers = [0.33, 0.5,0.67]  # Centers for stretch processing

    post_process(csv, prefix, vars, metric_names, process_list, ticks_per_beat, beats_per_minute, frames_per_second, cc_number, filter_periods, 
                 stretch_values, stretch_centers)


