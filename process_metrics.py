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

def add_derived_columns(csv):
    """
    Add derived columns to the CSV dataframe based on existing metrics.
    
    Parameters:
    - csv: pandas DataFrame containing the metrics
    
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
    
    # Add rotation metrics (crl, crr, and cra) based on crc metrics
    # Find all columns that contain "_crc" in their name
    crc_columns = [col for col in csv.columns if "_crc" in col]
    
    for crc_col in crc_columns:
        # Extract the base name (everything before "_crc")
        base_name = crc_col.replace("_crc", "")
        
        # Create crl, crr, and cra column names
        crl_col = f"{base_name}_crl"
        crr_col = f"{base_name}_crr"
        cra_col = f"{base_name}_cra"
        
        # Create the new metrics
        # crl = max(crc, 0) - captures positive rotation (counterclockwise)
        csv[crl_col] = np.maximum(csv[crc_col], 0)
        
        # crr = max(-crc, 0) - captures negative rotation (clockwise)
        csv[crr_col] = np.maximum(-csv[crc_col], 0)
        
        # cra = abs(crc) - captures absolute rotation magnitude
        csv[cra_col] = np.abs(csv[crc_col])
    
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


def post_process(csv, prefix, ticks_per_beat, beats_per_minute, frames_per_second, cc_number, filter_periods, stretch_values, stretch_centers, farneback_preset="default"):

    # Create output prefix with farneback preset
    output_prefix = f"{prefix}_{farneback_preset}"
    
    if not os.path.exists(f"../video_midi/{output_prefix}"):
        os.makedirs(f"../video_midi/{output_prefix}")
    
    # Use original prefix for file names (without farneback preset)
    file_prefix = prefix
    
    # replace NA values with 0
    csv = csv.fillna(0)

    master_dict = {}
    
    # Automatically detect variables and metrics from CSV columns
    # Extract all unique variables and metrics from column names
    variables = set()
    metrics = set()
    
    for col in csv.columns:
        if '_' in col:
            parts = col.split('_', 1)  # Split on first underscore only
            if len(parts) == 2:
                var, metric = parts
                variables.add(var)
                metrics.add(metric)
    
    # Process all combinations
    for var in sorted(variables):
        for metric in sorted(metrics):
            key = var + "_" + metric
            if key not in csv.columns:
                continue
            process_dict = {}

            # Create base entries for processing
            raw_entries = {}
            raw_entries[key + "_v"] = csv[key]
            raw_entries[key + "_r"] = percentile_data(csv[key])

            # Scale the data to 0-1
            scaled_entries = {}
            for entry_key, entry_data in raw_entries.items():
                scaled_entries[entry_key] = scale_data(entry_data)

            # Apply filtering first - create filtered entries
            filtered_entries = {}
            for entry_key, entry_data in scaled_entries.items():
                for filter_period in filter_periods:
                    # Create filtered version for all periods (including 1)
                    new_key = entry_key + f"_f{filter_period:03d}"
                    if filter_period == 1:
                        # For period 1, just copy the data (no filtering)
                        filtered_entries[new_key] = entry_data
                    else:
                        # For other periods, apply triangular filtering and rescale to 0-1
                        filtered_data = triangular_filter_odd(entry_data, filter_period)
                        filtered_entries[new_key] = scale_data(filtered_data)

            # Apply stretching to filtered data
            stretched_entries = {}
            for entry_key, entry_data in filtered_entries.items():
                x = entry_data
                # Add stretched versions
                for stretch_value in stretch_values:
                    for stretch_center in stretch_centers:
                        new_key = entry_key + "_s" + str(stretch_value) + "-" + str(stretch_center)
                        stretched_entries[new_key] = (x / stretch_center)**stretch_value / ((x / stretch_center)**stretch_value + ((1 - x) / (1 - stretch_center))**stretch_value)
                
                # Always ensure the special case stretch_value=1, stretch_center=0.5 is included
                special_key = entry_key + "_s1-0.5"
                if special_key not in stretched_entries:
                    stretched_entries[special_key] = (x / 0.5)**1 / ((x / 0.5)**1 + ((1 - x) / (1 - 0.5))**1)

            # Apply inversion
            final_entries = {}
            for entry_key, entry_data in stretched_entries.items():
                # Add original version with _o suffix
                final_entries[entry_key + "_o"] = entry_data
                # Add inverted version with _i suffix
                final_entries[entry_key + "_i"] = 1.0 - entry_data

            # Add final entries to the master dictionary
            process_dict.update(final_entries)



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
    
    for var in sorted(variables):
        for averaging in filter_suffixes:
            for rank_type in ["v", "r"]:
                # Create a list of stretch values to process, including the special case
                stretch_values_to_process = list(stretch_values)
                if 1 not in stretch_values_to_process:
                    stretch_values_to_process.append(1)
                
                # Create a list of stretch centers to process, including the special case
                stretch_centers_to_process = list(stretch_centers)
                if 0.5 not in stretch_centers_to_process:
                    stretch_centers_to_process.append(0.5)
                
                for stretch_value in stretch_values_to_process:
                    for stretch_center in stretch_centers_to_process:
                        midi_file = mido.MidiFile()
                        
                        # Process both inversion types ("o" and "i") in the same MIDI file
                        for inversion_type in ["o", "i"]:
                            for key in master_dict:
                                if  "_" + averaging+"_" not in key:
                                    continue
                                # Check if this key matches the pattern: var_metric_rank_averaging_stretch_inversion
                                if not key.startswith(f"{var}_"):
                                    continue
                                if not f"_{rank_type}_{averaging}_s{stretch_value}-{stretch_center}" in key:
                                    continue
                                if not key.endswith(f"_{inversion_type}"):
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

                        
                        # Create file name: var_rank_averaging_stretch-center.mid (exclude metric and inversion)
                        file_name = f"../video_midi/{output_prefix}/{var}_{rank_type}_{averaging}_s{stretch_value}-{stretch_center}.mid"
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
                    
        midi_file.save("../video_midi/" + output_prefix + "/" + base_suffix + ".mid")

    # write out the master xlsx
    print(f"Writing out master xlsx")
    master_dict['frame_count_list'] = frame_count_list
    master_df = pd.DataFrame(master_dict)
    # sort the columns alphabetically
    master_df = master_df.sort_index(axis=1)        
    # Reorder columns to put frame_count_list first
    cols = ['frame_count_list'] + [col for col in master_df.columns if col != 'frame_count_list']
    master_df = master_df[cols]
    master_df.to_excel(f"../video_midi/{output_prefix}/{file_prefix}_derived.xlsx", index=False)
    print(f"Derived data saved to ../video_midi/{output_prefix}/{file_prefix}_derived.xlsx")

    # Flexible sorting configuration
    # Define the order of fields for sorting (can be easily modified)
    SORT_ORDER = [
        'color_channel',     # field 1: R, G, B, Gray, H000, etc.
        'metric',           # field 2: avg, std, xps, etc.
        'rank_value',        # field 4: r or v
        'smoothing_period',  # field 3: f001, f017, f065, f257
        'inversion',         # field 5: o or i
        'stretching'       # field 6: s1-0.5, s8-0.33, etc.
    ]

    def parse_key_fields(key):
        """Parse key into structured fields for sorting"""
        parts = key.split('_')
        
        # Handle keys with different numbers of parts
        if len(parts) < 3:
            return None  # Skip keys that don't match expected format
            
        fields = {
            'color_channel': parts[0],
            'metric': parts[1],
            'rank_value': parts[2]
        }
        
        # Extract inversion (field 6) - o or i
        inversion = None
        if parts[-1] in ['o', 'i']:
            inversion = parts[-1]
        fields['inversion'] = inversion
        
        # Extract smoothing period (field 4)
        smoothing_period = None
        for part in parts[3:]:
            if part.startswith('f'):
                smoothing_period = part
                break
        fields['smoothing_period'] = smoothing_period
        
        # Extract stretching (field 5)
        stretching = None
        for part in parts[3:]:
            if part.startswith('s'):
                stretching = part
                break
        fields['stretching'] = stretching
        
        return fields

    def get_sort_key(key, sort_order):
        """Generate sort key based on specified order"""
        fields = parse_key_fields(key)
        if fields is None:
            return (None,) * len(sort_order)  # Put unparseable keys at the end
        
        # Create sort tuple with values in specified order
        sort_values = []
        for field_name in sort_order:
            value = fields.get(field_name, '')
            # Handle None values for consistent sorting
            if value is None:
                value = ''
            # Special handling for inversion: put 'o' before 'i'
            if field_name == 'inversion':
                value = '0' if value == 'o' else '1' if value == 'i' else value
            sort_values.append(value)
        
        return tuple(sort_values)

    # Prepare sorted keys for the plots
    keys_list = list(master_dict.keys())
    # Remove "frame_count_list" from the list
    keys_list = [key for key in keys_list if key != "frame_count_list"]
    
    # Sort keys using the new flexible sorting system
    sorted_keys = sorted(keys_list, key=lambda k: get_sort_key(k, SORT_ORDER))
    
    # Create a dictionary to store sort values for display (optional)
    sort_values = {}
    for key in sorted_keys:
        sort_key = get_sort_key(key, SORT_ORDER)
        sort_values[key] = f"{sort_key[0]}-{sort_key[1]}-{sort_key[2]}-{sort_key[3]}-{sort_key[4]}"  # Show first 5 fields
    
    # write out a pdf book of plots of each of the metrics
    print(f"Writing out pdf book of plots of each of the metrics")
    pdf = PdfPages(f"../video_midi/{output_prefix}/{file_prefix}_plots.pdf")
    
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
            plt.title(f"{key}", fontsize=8)  # Smaller font size for titles
            plt.grid(True)
            plt.xticks([])  # Remove x-axis ticks for cleaner look
            plt.yticks([])  # Remove y-axis ticks for cleaner look
        
        plt.tight_layout()  # Adjust spacing between subplots
        plt.savefig(pdf, format="pdf")
        plt.close()  # Close the figure to free memory
    
    pdf.close()




def process_metrics_to_midi(prefix, config=None):
    """
    Process metrics from CSV file and generate MIDI files.
    
    Args:
        prefix (str): The prefix for input/output files
        config (dict, optional): Configuration dictionary. If None, will try to load from {prefix}_config.json
    """
    # Extract farneback preset from config for output naming
    farneback_preset = "default"  # default fallback
    if config and "farneback_preset" in config:
        farneback_preset = config["farneback_preset"]
    
    # Create output prefix with farneback preset
    output_prefix = f"{prefix}_{farneback_preset}"
    # Try to load config from JSON if it exists and config not provided
    if config is None:
        # First try the original config filename
        config_filename = f"{prefix}_config.json"
        if os.path.exists(config_filename):
            with open(config_filename, 'r') as f:
                config = json.load(f)
        else:
            # Try to find config file with farneback preset suffix
            import glob
            config_files = glob.glob(f"{prefix}_*_config.json")
            if config_files:
                # Use the first matching config file
                config_filename = config_files[0]
                with open(config_filename, 'r') as f:
                    config = json.load(f)
                print(f"Using config file: {config_filename}")
            else:
                config = {}
    
    # Extract parameters from config with defaults
    ticks_per_beat = config.get("ticks_per_beat", 480)
    beats_per_minute = config.get("beats_per_minute", 100)
    frames_per_second = config.get("frames_per_second", 30)
    cc_number = config.get("cc_number", 1)
    beats_per_midi_event = config.get("beats_per_midi_event", 1)
    
    # Extract processing parameters from config with defaults
    filter_periods = config.get("filter_periods", [17, 65, 257])
    stretch_values = config.get("stretch_values", [8])
    stretch_centers = config.get("stretch_centers", [0.33, 0.67])

    # Try to find the CSV file with farneback preset suffix
    csv_filename = f"{prefix}_basic.csv"
    if not os.path.exists(csv_filename):
        # Try to find CSV file with farneback preset suffix
        import glob
        csv_files = glob.glob(f"{prefix}_*_basic.csv")
        if csv_files:
            csv_filename = csv_files[0]
            print(f"Using CSV file: {csv_filename}")
        else:
            raise FileNotFoundError(f"Could not find CSV file: {csv_filename} or any matching files with preset suffix")
    
    csv = pd.read_csv(csv_filename, index_col=0)
    
    # Add derived columns
    csv = add_derived_columns(csv)
    # All transformations are now applied by default (no conditional logic needed)

    post_process(csv, prefix, ticks_per_beat, beats_per_minute, frames_per_second, cc_number, filter_periods, 
                 stretch_values, stretch_centers, farneback_preset)

# This script is designed to be called by run_video_processing.py
# For standalone usage, use: python run_video_processing.py <video_name>


