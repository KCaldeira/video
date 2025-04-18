# video to midi code
"""
prompt for ChatGTP:

Here's a Python script that processes every Nth frame of a video file, calculates multiple metrics (average intensity, 
standard deviation, and entropy) for each color color_channel_name (R, G, B, and grayscale) and outputs 24 MIDI files. 
Each metric is stored in two MIDI files: one directly scaled (0-127) and another inverted (127-0).

This script will:

Extract every Nth frame from the video.
Compute various metrics for each color channel.
Map values to MIDI Control Change (CC) messages (both direct and inverted).
Save each metric in a separate MIDI file, with filenames autogenerated.

"""
import cv2
import os
import pandas as pd
import numpy as np
import re
from mido import Message, MidiFile, MidiTrack
from scipy.stats import rankdata
from collections import defaultdict

def information_metric(color_channel, downscale_factor=4):
    """
    Returns a metric of information loss when a color channel from a frame 
    is downscaled and then upscaled.
    
    frame: color channel (e.g., R, G, B, or grayscale)
    downscale_factor: how much to shrink (e.g., 4 means 1/4 size)
    """

    # Original size
    h, w = color_channel.shape[0:2]

    # Downscale and then upscale
    downscaled  = cv2.resize(color_channel, (w // downscale_factor, h // downscale_factor), interpolation=cv2.INTER_AREA)
    restored = cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_LINEAR)

    # Compute mean squared error (MSE) between original and restored image
    mse = np.mean((color_channel - restored) ** 2)

    # Optional: normalize MSE to 0–1 by dividing by max possible value (variance)
    normalized_mse = 1.- mse / np.var(color_channel) 
    # 1 means all information is at coarser spatial scales, 
    # 0 means all information is at finer spatial scales

    return normalized_mse

def tranpose_metric(color_channel, downscale_factor=4):
    """
    Returns a metric of information loss when a color channel from a frame 
    is downscaled and then upscaled.
    
    frame: color channel (e.g., R, G, B, or grayscale)
    downscale_factor: how much to shrink (e.g., 4 means 1/4 size)
    """

    # Original size
    h, w = color_channel.shape[0:2]

    # Downscale and then upscale
    downscaled  = cv2.resize(color_channel, (w // downscale_factor, h // downscale_factor), interpolation=cv2.INTER_AREA)

    # Compute mean squared error (MSE) between original and restored image
    mse = np.mean((downscaled - downscaled[::-1,::-1]) ** 2)

    # Optional: normalize MSE to 0–1 by dividing by max possible value (variance)
    normalized_mse = 1 - mse / np.var(downscaled) 
    # 1 means perfect symmetry at this scale 
    # 0 means no symmetry at this scale

    return normalized_mse

def reflect_metric(color_channel, downscale_factor=4):
    """
    Returns a metric of information loss when a color channel from a frame 
    is downscaled and then upscaled.
    
    frame: color channel (e.g., R, G, B, or grayscale)
    downscale_factor: how much to shrink (e.g., 4 means 1/4 size)
    """

    # Original size
    h, w = color_channel.shape[0:2]

    # Downscale and then upscale
    downscaled  = cv2.resize(color_channel, (w // downscale_factor, h // downscale_factor), interpolation=cv2.INTER_AREA)

    # Compute mean squared error (MSE) between original and vertically reflected image
    mse0 = np.mean((downscaled - downscaled[::-1]) ** 2)

    # Compute mean squared error (MSE) between original and horizontally reflected image
    mse1 = np.mean((downscaled - downscaled[:,::-1]) ** 2)

    # Optional: normalize MSE to 0–1 by dividing by max possible value (variance)
    normalized_mse = 1 -(mse0 + mse1) / (2. * np.var(downscaled) )
    # 1 means perfect symmetry at this scale 
    # 0 means no symmetry at this scale

    return normalized_mse

def radial_symmetry_metric(color_channel, scale_factor):
    """
    Compute radial symmetry metric for a color channel with distance bins of width `scale_factor`.
    """
    # Step 1: Create coordinate grid
    y, x = np.indices(color_channel.shape)
    center_y = (color_channel.shape[0] / 2) - 0.5
    center_x = (color_channel.shape[1] / 2) - 0.5

    # Step 2: Compute radial distance from center for each pixel
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)

    # Step 2b: Bin distances into scale_factor-wide bins
    r_bin = (r / scale_factor).astype(np.int32)

    # Step 3: Compute mean value for each radial bin
    max_bin = r_bin.max()
    radial_mean = np.zeros(max_bin + 1)
    counts = np.bincount(r_bin.ravel())
    sums = np.bincount(r_bin.ravel(), weights=color_channel.ravel())

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        radial_mean[:len(sums)] = np.where(counts != 0, sums / counts, 0)

    # Optional: remove zeros or masked bins if they skew the variance
    # valid = counts > 0
    # return np.var(radial_mean[valid])

    return np.var(radial_mean)


def bgr_to_cmyk(b, g, r):
    """
    Convert BGR to CMYK color space.
    """
    b = b.astype(float) / 255.0
    g = g.astype(float) / 255.0
    r = r.astype(float) / 255.0

    k = 1 - np.max([b, g, r], axis=0) # 1 if black or highest color intensity, 0 if white
    c = (1 - b - k) / (1 - k + 1e-10)
    m = (1 - g - k) / (1 - k + 1e-10)
    y = (1 - r - k) / (1 - k + 1e-10)

    return c, m, y, k

def bgr_to_hsv(b, g, r):
    """
    Convert RGB to HSV for 2D numpy arrays.
    Inputs r, g, b: 2D numpy arrays with values in [0, 255]
    Outputs h in degrees [0, 360), s and v in [0.0, 1.0]
    """
    r = r.astype(np.float32) / 255
    g = g.astype(np.float32) / 255
    b = b.astype(np.float32) / 255

    cmax = np.maximum.reduce([r, g, b])
    cmin = np.minimum.reduce([r, g, b])
    delta = cmax - cmin

    # Hue calculation
    h = np.zeros_like(cmax)

    mask = delta != 0
    r_max = (cmax == r) & mask
    g_max = (cmax == g) & mask
    b_max = (cmax == b) & mask

    h[r_max] = (60 * ((g[r_max] - b[r_max]) / delta[r_max])) % 360
    h[g_max] = (60 * ((b[g_max] - r[g_max]) / delta[g_max]) + 120)
    h[b_max] = (60 * ((r[b_max] - g[b_max]) / delta[b_max]) + 240)

    # Saturation calculation
    s = np.zeros_like(cmax)
    nonzero = cmax != 0
    s[nonzero] = delta[nonzero] / cmax[nonzero]

    # Value
    v = cmax

    return h, s, v

# Circular statistics function to compute standard deviation of angle weighted by saturation
# Hue is assumed to be 0-360 degrees, saturation is 0-1
def weighted_circular_std_deg(hue, saturation):
    """Weighted circular standard deviation in degrees"""
    angles_rad = np.deg2rad(hue)
    weights = np.array(saturation)
    z = weights * np.exp(1j * angles_rad)
    R_w = np.abs(np.sum(z) / np.sum(weights))
    return np.rad2deg(np.sqrt(-2 * np.log(R_w)))


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
    scaled_data = (data - min_val) / (max_val - min_val)
    return scaled_data


def compute_basic_metrics(frame, scale_boundary):
    """
    Compute different intensity-based metrics on R, G, B, and grayscale images.
    Returns a dictionary of results.
    """
    basic_metrics = {}
    
    b, g, r = cv2.split(frame)
    c, m, y, k = bgr_to_cmyk(b, g, r)
    h, s, v = bgr_to_hsv(b, g, r)
  
    # Convert to grayscale
    gray = 255 - cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 255 is black, 0 is white

    # Split into R, G, B channels

    for color_channel_name, color_channel in [("R", r), ("G", g), ("B", b),
                                ("C", c), ("M", m), ("Y", y), ("K", k),
                                ("Gray", gray),("V", v)]:
        avg_intensity = np.mean(color_channel)
        variance = np.var(color_channel)

        # 1 means all information is at finer spatial scales,
        # 0 means all information is at coarser spatial scales
        large_scale_info = information_metric(color_channel, scale_boundary) # fraction info at small and medium scales

        transpose_metric_value = tranpose_metric(color_channel, scale_boundary) # degree of symmettry for flipping around the center point
        # at the specified spatial scale
        reflect_metric_value = reflect_metric(color_channel, scale_boundary) # degree of symmettry for flipping around the center point
        # at the specified spatial scale
        radial_symmetry_metric_value = radial_symmetry_metric(color_channel, scale_boundary) # degree of symmettry for flipping around the center point
        # at the specified spatial scale

        # Store values
        basic_metrics[f"{color_channel_name}_avg"] = avg_intensity
        basic_metrics[f"{color_channel_name}_var"] = variance # note that variance is total info (i.e., diff^2 relative to mean)
        basic_metrics[f"{color_channel_name}_large"] =  large_scale_info # fraction of info at large scales
        basic_metrics[f"{color_channel_name}_transpose"] = transpose_metric_value
        basic_metrics[f"{color_channel_name}_reflect"] = reflect_metric_value
        basic_metrics[f"{color_channel_name}_radial"] = radial_symmetry_metric_value

    #monochromicity metric is the standard deviation of hue weighted by saturation
    basic_metrics["HSV_monochromicity"] = weighted_circular_std_deg(h, s) 

    return basic_metrics

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

# Export metrics to CSV
def export_metrics_to_csv(frame_count_list, metrics, filename):
    """
    Export frame count and metric data to a CSV file using pandas,
    with metrics sorted alphabetically by key.

    Parameters:
    - frame_count_list (list): List of frame counts.
    - metrics (dict): Dictionary where each value is a list of the same length as frame_count_list.
    - filename (str): Name of the CSV file to write.
    """

    # Sort the metric keys alphabetically
    sorted_keys = sorted(metrics.keys())

    # Create a DataFrame using the sorted keys
    data = {'frame_count_list': frame_count_list}
    for key in sorted_keys:
        data[key] = metrics[key]

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

def process_video_to_midi(video_path, 
                          subdir_name, # output prefix 
                          frames_per_second, 
                          beats_per_frame,
                          ticks_per_beat, 
                          beats_per_minute, 
                          cc_number, 
                          midi_channel,
                          scale_boundary,
                          filter_width):
    """
    Process every Nth frame, calculate metrics, and generate multiple MIDI files.
    
    :param video_path: Path to the video file.
    :param output_prefix: Prefix for output MIDI filenames.
    :param frames_per_second (number of frames per second in video)
    :param beats_per_frame (number of beats between each frame that will per processed)
    :param ticks_per_beat (number of midi ticks per beat in DAW)
    :param beats_per_minute (number of beats per minute in DAW)
    :param cc_number: MIDI CC number (default 7 for volume).
    :param channel: MIDI channel (0-15).
    :param scale_boundary: spatial scale for computing metrics
    :param filter_width: width of boxcar filter for smoothing

    """




    ticks_per_frame = ticks_per_beat *( beats_per_minute / 60.) / frames_per_second # ticks per second / frames per second
    # Calculate the frame interval for processing frames
    seconds_per_analysis_frame = beats_per_frame / (beats_per_minute / 60) # beats per frame / beats per second
    frames_per_analysis_frame_real = seconds_per_analysis_frame * frames_per_second
    # Take every Nth frame, where frames_per_interval_real is the floating point non-integer version of N

    frame_count = 0
    frame_count_list = []



    # Define metric categories that get computed by <compute_metrics>
    # and the color channels that get computed
    metric_names = ["avg", "var", "large", "transpose", "reflect", "radial"]
    color_channel_names = ["R", "G", "B", "C", "M", "Y", "K", "Gray","V"]
    basic_metrics = {f"{color_channel_name}_{metric_name}": [] 
               for color_channel_name in color_channel_names for metric_name in metric_names}
    # add metric that is outside of the normal grouping
    basic_metrics["HSV_monochromicity"] = []

    # open rhe video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        k = frame_count / frames_per_analysis_frame_real
        k_rounded = round(k)
        frame_count_good = round(k_rounded * frames_per_analysis_frame_real)
        if frame_count == frame_count_good :
            print ("Processing frame:", frame_count)
            frame_count_list.append(frame_count)

            metric_results = compute_basic_metrics(frame, scale_boundary)

            #append to the dictionary of results
            for key, value in metric_results.items():
                basic_metrics[key].append(value)

        frame_count += 1

    cap.release()

    #now compute derivative metrics that are computed after all frames are processed

    # normalize all metrics to be between 0 and 1, with a percentile mapping
    #iterate over copy to avoid modifying the dictionary while iterating
    metrics = {}
    for key, values in basic_metrics.items():
        metric = np.array(values)
        # max_val = np.max(metric)
        # min_val = np.min(metric)
        # normalized_metric = (metric - min_val) / (max_val - min_val)
        # scale data by rescaling metric form 0 to 1
        scaled_metric = scale_data(metric)
        metrics[f"{key}-pos"] = scaled_metric
        metrics[f"{key}-neg"] = (1.-scaled_metric)
        # rank the data and convert to percentiles
        normalized_metric = percentile_data(metric)
        metrics[f"{key}-Ppos"] = normalized_metric
        metrics[f"{key}-Pneg"] = (1.-normalized_metric)

    # create derived metrics that involve combining metrics, i.e., only symmetry now

    # Create symmetry metric

    target_substrings = {"transpose", "reflect", "radial"}

    # Step 1: Filter relevant keys
    filtered_metrics = [m for m in metrics if any(substr in m for substr in target_substrings)]

    # Step 2: Identify and group by the shared prefix (with substring stripped out)
    grouped = defaultdict(set)

    for metric in filtered_metrics:
        for substr in target_substrings:
            if substr in metric:
                # Remove the substring (and maybe an underscore) to isolate the "base"
                base = metric.replace(f"_{substr}", "").replace(substr, "")
                grouped[base].add(metric)
                break

    # Step 3: Find groups that contain all three substrings
    def contains_all_substrings(group):
        return all(any(substr in key for key in group) for substr in target_substrings)

    complete_sets = [group for group in grouped.values() if contains_all_substrings(group)]

    # Show results
    for group in complete_sets:
        symmetry = np.minimum.reduce([metrics[item] for item in group], axis=0)
        sample_key = next(iter(group))  # Get a sample key from the group
        new_key = re.sub(r"(transpose|reflect|radial)", "symmetry", sample_key)
        metrics[new_key] = symmetry

    metric_name_list = list({
        key.split("_", 1)[1]
        for key in metrics
        if key.startswith(("R_", "G_", "B_","C_", "M_", "Y_"))})

    # create derived metrics that involve combining color channels
    for metric_name in metric_name_list:
        # find minimum of "transpose", "reflect", "radial" metrics
        r = metrics[f"R_{metric_name}"]
        g = metrics[f"G_{metric_name}"]
        b = metrics[f"B_{metric_name}"]
        c = metrics[f"C_{metric_name}"]
        m = metrics[f"M_{metric_name}"]
        y = metrics[f"Y_{metric_name}"]

        minRGB = np.minimum.reduce([r, g, b], axis=0)
        metrics[f"minRGB_{metric_name}"] = minRGB

        minCMY = np.minimum.reduce([c, m, y], axis=0)
        metrics[f"minCMY_{metric_name}"] = minCMY

        maxRGB = np.maximum.reduce([r, g, b], axis=0)
        metrics[f"maxRGB_{metric_name}"] = maxRGB

        maxCMY = np.maximum.reduce([c, m, y], axis=0)
        metrics[f"maxCMY_{metric_name}"] = maxCMY
    
    metrics_copy = metrics.copy()
    for key, values in metrics_copy.items():    
        # add square and square root of metric to give different scaling choices
        metrics[f"{key}-p050"] = np.sqrt(values)
        metrics[f"{key}-p200"] = np.square(values)
        #metrics[f"{key}-Nsqrt"] = (1.-np.sqrt(1.-values))  
        #metrics[f"{key}-Nsquare"] = (1.-np.square(1.-values))
        metrics[f"{key}-p025"] = np.power(values,0.25)
        metrics[f"{key}-p400"] = np.power(values,4)
        #metrics[f"{key}-Nqtr"] = (1.-np.power(1.-values,0.25))  
        #metrics[f"{key}-Nfourth"] = (1.-np.power(1.-values,4))

    # Smooth the data with a boxcar filter
    if filter_width > 1:
        for key, values in metrics.items():
            metrics[key] = triangular_filter_odd(np.array(values), filter_width)    
            
    # renomalize between 0 and 1
    for key, values in metrics.items():
        metric = np.array(values)
        # max_val = np.max(metric)
        # min_val = np.min(metric)
        # normalized_metric = (metric - min_val) / (max_val - min_val)
        # scale data by rescaling metric form 0 to 1
        scaled_metric = scale_data(metric)
        metrics[key] = scaled_metric

    # create inverse of each metric
    metrics.update({f"{k}_inv": 1. - v for k, v in metrics.items()})

    # Export metrics to CSV
    csv_filename = f"{subdir_name}.csv"
    export_metrics_to_csv(frame_count_list, metrics, csv_filename)
    print(f"Metrics exported to {csv_filename}")

    if not os.path.exists(f"./video_midi/{subdir_name}"):
        os.makedirs(f"./video_midi/{subdir_name}")

    # Create MIDI files for each base metric
    midi_files = {}

    for key in metrics:
        # Skip keys that are "_inv" variants
        if key.endswith("_inv"):
            continue

        base_key = key
        inv_key = f"{key}_inv"
        
        color_channel_name, metric_name = base_key.split("_", 1)
        
        # Initialize MIDI file and tracks
        midi_file = MidiFile()
        track_base = MidiTrack()
        midi_file.tracks.append(track_base)

        # Write base metric values
        midi_val_base = [round(104 * val) for val in metrics[base_key]]

        for i, midi_value in enumerate(midi_val_base):
            time_tick = 0 if i == 0 else int(ticks_per_frame * (frame_count_list[i] - frame_count_list[i - 1]))
            track_base.append(
                Message('control_change',
                        control=cc_number,
                        value=midi_value,
                        channel=midi_channel,
                        time=time_tick)
            )

        # If the inverse metric exists, add a second track for it
        if inv_key in metrics:
            track_inv = MidiTrack()
            midi_file.tracks.append(track_inv)

            midi_val_inv = [round(104 * val) for val in metrics[inv_key]]
            for i, midi_value in enumerate(midi_val_inv):
                time_tick = 0 if i == 0 else int(ticks_per_frame * (frame_count_list[i] - frame_count_list[i - 1]))
                track_inv.append(
                    Message('control_change',
                            control=cc_number,
                            value=midi_value,
                            channel=midi_channel,
                            time=time_tick)
                )

        # Save the MIDI file (one file per base key)
        filename = f"video_midi/{subdir_name}/{color_channel_name}_{metric_name}.mid"
        midi_file.save(filename)





# Example usage
#video_file = "Mz3DllgimbrV2.wmv"  #  small test video file
#subdir_name = "Mz3DllgimbrV2" # output prefix
#video_file = "He saw Julias everywhere (MzJuliaV2e).wmv"
#video_file = "Mz3DllgimbrV2B.wmv"
#subdir_name = "Mz3DllgimbrV2B" # output prefix
video_file = "M10zul.wmv"
subdir_name = "M10zul" # output prefix

process_video_to_midi(video_file, 
                      subdir_name, # output prefix
                      frames_per_second=30, 
                      beats_per_frame=1,
                      ticks_per_beat=480, 
                      beats_per_minute=92,  
                      cc_number=7, 
                      midi_channel=0,
                      scale_boundary=12, # scale boundary means divide so 12x12 pixels in a cell
                      filter_width = 5 ) # smooth the data with a triangular filter of this (odd) width
# process_video_to_midi("path_to_your_video.mp4", "output_prefix", nth_frame=30, frames_per_second=30, ticks_per_beat=480, beats_per_minute=120, cc_number=7, channel=0)


