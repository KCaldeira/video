#!/usr/bin/env python3
"""Find local minima and maxima in y_values.py and tempo CSV"""

import sys
sys.path.insert(0, './data/input')

from y_values import y_values
from scipy.signal import find_peaks
import numpy as np
import csv

def format_timestamp(frame_index, fps=30):
    """Convert frame index to MM:SS format"""
    total_seconds = frame_index / fps
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes}:{seconds:05.2f}"

def merge_similar_extrema(indices, values, min_distance=100, min_value_diff=0.5):
    """
    Merge extrema that are close together and have similar values.
    Keep only the most extreme value in each cluster.
    """
    if len(indices) == 0:
        return []

    filtered = []
    i = 0

    while i < len(indices):
        # Start a new cluster
        cluster_start = i
        cluster_indices = [indices[i]]
        cluster_values = [values[indices[i]]]

        # Find all nearby points with similar values
        j = i + 1
        while j < len(indices):
            distance = indices[j] - indices[cluster_start]
            value_diff = abs(values[indices[j]] - cluster_values[0])

            if distance < min_distance and value_diff < min_value_diff:
                cluster_indices.append(indices[j])
                cluster_values.append(values[indices[j]])
                j += 1
            else:
                break

        # Keep the most extreme value in this cluster
        if len(cluster_values) > 0:
            max_idx = np.argmax(np.abs(cluster_values))
            filtered.append(cluster_indices[max_idx])

        i = j

    return np.array(filtered)

print("=" * 70)
print("ANALYSIS 1: y_values.py")
print("=" * 70)

# Calculate a reasonable prominence threshold for y_values
y_range = max(y_values) - min(y_values)
y_prominence = 2.0  # Use absolute value of 2.0

print(f"Data range: {min(y_values):.2f} to {max(y_values):.2f}")
print(f"Using prominence threshold: {y_prominence:.2f}")
print()

# Find local maxima with prominence filter
maxima_indices, _ = find_peaks(y_values, prominence=y_prominence)

# Find local minima (peaks in the inverted signal) with prominence filter
minima_indices, _ = find_peaks([-y for y in y_values], prominence=y_prominence)

# Merge similar extrema
maxima_indices = merge_similar_extrema(maxima_indices, y_values, min_distance=100, min_value_diff=1.0)
minima_indices = merge_similar_extrema(minima_indices, y_values, min_distance=100, min_value_diff=1.0)

print(f"Total frames: {len(y_values)}")
print(f"Total duration: {format_timestamp(len(y_values) - 1)}")
print()

print(f"Local Maxima ({len(maxima_indices)} found):")
print("-" * 70)
for idx in maxima_indices:
    timestamp = format_timestamp(idx)
    value = y_values[idx]
    print(f"Index: {idx:6d}  |  Time: {timestamp}  |  Value: {value:.4f}")

print()
print(f"Local Minima ({len(minima_indices)} found):")
print("-" * 70)
for idx in minima_indices:
    timestamp = format_timestamp(idx)
    value = y_values[idx]
    print(f"Index: {idx:6d}  |  Time: {timestamp}  |  Value: {value:.4f}")

# Combine all extrema and sort by index
all_extrema = []
for idx in maxima_indices:
    timestamp = format_timestamp(idx)
    value = y_values[idx]
    all_extrema.append(('Maximum', idx, timestamp, value))

for idx in minima_indices:
    timestamp = format_timestamp(idx)
    value = y_values[idx]
    all_extrema.append(('Minimum', idx, timestamp, value))

# Sort by index (second element in each tuple)
all_extrema.sort(key=lambda x: x[1])

# Write to CSV
csv_filename = 'extrema_results.csv'
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Type', 'Index', 'Time', 'Value'])

    for extremum in all_extrema:
        extremum_type, idx, timestamp, value = extremum
        writer.writerow([extremum_type, idx, timestamp, f'{value:.4f}'])

print()
print(f"Results written to {csv_filename} (sorted by frame number)")

# Now analyze the tempo CSV
print()
print("=" * 70)
print("ANALYSIS 2: beat_tempos_yvalues_96bpm.csv")
print("=" * 70)

tempo_csv_path = './data/output/beat_tempos_yvalues_96bpm.csv'
tempo_data = []
with open(tempo_csv_path, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        tempo_data.append({
            'beat_index': int(row['beat_index']),
            'bar': int(row['bar']),
            'beat_in_bar': int(row['beat_in_bar']),
            'time_sec': float(row['time_sec']),
            'frame': int(row['frame']),
            'tempo_bpm': float(row['tempo_bpm'])
        })

tempo_values = [d['tempo_bpm'] for d in tempo_data]

# Calculate a reasonable prominence threshold for tempo
tempo_range = max(tempo_values) - min(tempo_values)
tempo_prominence = tempo_range * 0.05  # 5% of range

print(f"Tempo range: {min(tempo_values):.2f} to {max(tempo_values):.2f} bpm")
print(f"Using prominence threshold: {tempo_prominence:.2f} bpm")
print()

# Find local maxima in tempo with prominence filter
tempo_maxima_indices, _ = find_peaks(tempo_values, prominence=tempo_prominence)

# Find local minima in tempo (peaks in the inverted signal) with prominence filter
tempo_minima_indices, _ = find_peaks([-t for t in tempo_values], prominence=tempo_prominence)

# Merge similar extrema - use smaller distance for beats, and tempo difference of 0.1 bpm
tempo_maxima_indices = merge_similar_extrema(tempo_maxima_indices, tempo_values, min_distance=20, min_value_diff=0.1)
tempo_minima_indices = merge_similar_extrema(tempo_minima_indices, tempo_values, min_distance=20, min_value_diff=0.1)

print(f"Total beats: {len(tempo_data)}")
print(f"Total duration: {tempo_data[-1]['time_sec']:.2f} seconds ({tempo_data[-1]['time_sec']/60:.2f} minutes)")
print()

print(f"Local Maxima ({len(tempo_maxima_indices)} found):")
print("-" * 70)
for idx in tempo_maxima_indices:
    beat = tempo_data[idx]
    time_str = format_timestamp(beat['frame'])
    print(f"Beat: {beat['beat_index']:4d}  |  Bar: {beat['bar']:3d}.{beat['beat_in_bar']}  |  Time: {time_str}  |  Tempo: {beat['tempo_bpm']:7.2f} bpm")

print()
print(f"Local Minima ({len(tempo_minima_indices)} found):")
print("-" * 70)
for idx in tempo_minima_indices:
    beat = tempo_data[idx]
    time_str = format_timestamp(beat['frame'])
    print(f"Beat: {beat['beat_index']:4d}  |  Bar: {beat['bar']:3d}.{beat['beat_in_bar']}  |  Time: {time_str}  |  Tempo: {beat['tempo_bpm']:7.2f} bpm")

# Combine tempo extrema and sort by beat index
tempo_extrema = []
for idx in tempo_maxima_indices:
    beat = tempo_data[idx]
    time_str = format_timestamp(beat['frame'])
    tempo_extrema.append(('Maximum', beat['beat_index'], beat['bar'], beat['beat_in_bar'],
                         time_str, beat['frame'], beat['tempo_bpm']))

for idx in tempo_minima_indices:
    beat = tempo_data[idx]
    time_str = format_timestamp(beat['frame'])
    tempo_extrema.append(('Minimum', beat['beat_index'], beat['bar'], beat['beat_in_bar'],
                         time_str, beat['frame'], beat['tempo_bpm']))

# Sort by beat index
tempo_extrema.sort(key=lambda x: x[1])

# Write tempo extrema to CSV
tempo_csv_output = 'tempo_extrema_results.csv'
with open(tempo_csv_output, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Type', 'Beat_Index', 'Bar', 'Beat_in_Bar', 'Time', 'Frame', 'Tempo_BPM'])

    for extremum in tempo_extrema:
        extremum_type, beat_idx, bar, beat_in_bar, time_str, frame, tempo = extremum
        writer.writerow([extremum_type, beat_idx, bar, beat_in_bar, time_str, frame, f'{tempo:.2f}'])

print()
print(f"Tempo results written to {tempo_csv_output} (sorted by beat number)")
