#!/usr/bin/env python3
"""
Standalone utility to convert speed values to MIDI CC tracks.

Reads a Python file containing speed values (variable 's' or 'y_values'),
computes 1/s (inverse), scales to CC range 0-127, and generates MIDI file
with two tracks: normal and inverted.

Usage:
    python speed_to_midi_cc.py <input_file.py> [options]

Example:
    python speed_to_midi_cc.py data/input/N33_speed.py --tempo 108 --fps 30
    python speed_to_midi_cc.py data/input/N33_speed.py --output my_output.mid
"""

import argparse
import numpy as np
import mido
import os


def load_speed_values(filepath):
    """
    Load speed values from a Python file.

    Expects a file with either 's = [...]' or 'y_values = [...]'

    Parameters:
    - filepath: Path to Python file

    Returns:
    - numpy array of speed values
    """
    namespace = {}
    with open(filepath, 'r') as f:
        exec(f.read(), namespace)

    # Try both variable names
    if 's' in namespace:
        values = np.array(namespace['s'])
    elif 'y_values' in namespace:
        values = np.array(namespace['y_values'])
    else:
        raise ValueError(f"Could not find 's' or 'y_values' in {filepath}")

    return values


def compute_inverse_and_scale(values):
    """
    Compute 1/s and scale to 0-127 range.

    Parameters:
    - values: Array of speed values (s)

    Returns:
    - cc_normal: CC values scaled 0-127 (min inverse → 0, max inverse → 127)
    - cc_inverted: CC values inverted (127 - cc_normal)
    """
    # Compute inverse (1/s)
    # Handle potential division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        inverse = 1.0 / values

    # Replace inf/nan with 0
    inverse = np.nan_to_num(inverse, nan=0.0, posinf=0.0, neginf=0.0)

    # Find min and max for scaling
    min_val = np.min(inverse)
    max_val = np.max(inverse)

    print(f"Inverse values: min={min_val:.6f}, max={max_val:.6f}")

    # Scale to 0-127
    if max_val > min_val:
        scaled = (inverse - min_val) / (max_val - min_val)
    else:
        # All values are the same
        scaled = np.ones_like(inverse) * 0.5

    # Convert to integer CC values (0-127)
    cc_normal = np.clip(np.round(scaled * 127), 0, 127).astype(int)

    # Inverted track
    cc_inverted = 127 - cc_normal

    return cc_normal, cc_inverted


def generate_midi_file(cc_normal, cc_inverted, output_path, fps=30.0, tempo_bpm=108, ticks_per_beat=480):
    """
    Generate MIDI file with two CC1 tracks.

    Parameters:
    - cc_normal: Array of CC values (0-127) for normal track
    - cc_inverted: Array of CC values (0-127) for inverted track
    - output_path: Path to output MIDI file
    - fps: Frame rate (frames per second)
    - tempo_bpm: Tempo in beats per minute
    - ticks_per_beat: MIDI resolution (ticks per quarter note)
    """
    n_frames = len(cc_normal)

    # Calculate ticks per frame
    # ticks_per_frame = (tempo_bpm / 60) * (1 / fps) * ticks_per_beat
    ticks_per_frame = int(round((tempo_bpm / 60.0) * (1.0 / fps) * ticks_per_beat))

    print(f"MIDI parameters:")
    print(f"  Tempo: {tempo_bpm} BPM")
    print(f"  Frame rate: {fps} fps")
    print(f"  Ticks per beat: {ticks_per_beat}")
    print(f"  Ticks per frame: {ticks_per_frame}")
    print(f"  Total frames: {n_frames}")
    print(f"  Total duration: {n_frames/fps:.2f} seconds")

    # Create MIDI file
    midi_file = mido.MidiFile(ticks_per_beat=ticks_per_beat)

    # Track 1: CC1 Normal (1/s scaled to 0-127)
    track_normal = mido.MidiTrack()
    midi_file.tracks.append(track_normal)

    # Add track name and tempo
    track_normal.append(mido.MetaMessage('track_name', name='CC1 Normal (1/s)', time=0))
    track_normal.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo_bpm), time=0))

    # Add time signature (4/4)
    track_normal.append(mido.MetaMessage('time_signature', numerator=4, denominator=4,
                                         clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))

    # Add CC events
    track_normal.append(mido.Message('control_change', control=1, value=cc_normal[0], channel=0, time=0))

    for i in range(1, n_frames):
        track_normal.append(mido.Message('control_change', control=1, value=cc_normal[i],
                                        channel=0, time=ticks_per_frame))

    # Track 2: CC1 Inverted (127 - normal)
    track_inverted = mido.MidiTrack()
    midi_file.tracks.append(track_inverted)

    # Add track name
    track_inverted.append(mido.MetaMessage('track_name', name='CC1 Inverted', time=0))

    # Add CC events
    track_inverted.append(mido.Message('control_change', control=1, value=cc_inverted[0], channel=0, time=0))

    for i in range(1, n_frames):
        track_inverted.append(mido.Message('control_change', control=1, value=cc_inverted[i],
                                          channel=0, time=ticks_per_frame))

    # Save MIDI file
    midi_file.save(output_path)
    print(f"\nMIDI file saved to: {output_path}")
    print(f"  Track 1: CC1 Normal (1/s scaled 0-127)")
    print(f"  Track 2: CC1 Inverted (127 - normal)")


def main():
    parser = argparse.ArgumentParser(
        description='Convert speed values to MIDI CC tracks',
        epilog='Reads Python file with speed values, computes 1/s, scales to 0-127, and generates MIDI'
    )

    parser.add_argument('input_file',
                       help='Path to Python file containing speed values (s or y_values)')

    parser.add_argument('-o', '--output',
                       help='Output MIDI file path (default: based on input filename)')

    parser.add_argument('--fps', type=float, default=30.0,
                       help='Frame rate in frames per second (default: 30)')

    parser.add_argument('--tempo', type=float, default=108.0,
                       help='Tempo in beats per minute (default: 108)')

    parser.add_argument('--ticks-per-beat', type=int, default=480,
                       help='MIDI resolution in ticks per beat (default: 480)')

    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Generate output filename from input filename
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        output_dir = 'data/output'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{base_name}_cc_{int(args.tempo)}bpm.mid")

    print(f"Speed to MIDI CC Converter")
    print(f"=" * 50)
    print(f"Input file: {args.input_file}")
    print(f"Output file: {output_path}")
    print()

    # Load speed values
    print("Loading speed values...")
    speed_values = load_speed_values(args.input_file)
    print(f"Loaded {len(speed_values)} frames")
    print(f"Speed values: min={speed_values.min():.4f}, max={speed_values.max():.4f}")
    print()

    # Compute inverse and scale
    print("Computing 1/s and scaling to CC range...")
    cc_normal, cc_inverted = compute_inverse_and_scale(speed_values)
    print(f"CC normal: min={cc_normal.min()}, max={cc_normal.max()}")
    print(f"CC inverted: min={cc_inverted.min()}, max={cc_inverted.max()}")
    print()

    # Generate MIDI
    print("Generating MIDI file...")
    generate_midi_file(cc_normal, cc_inverted, output_path,
                      fps=args.fps, tempo_bpm=args.tempo, ticks_per_beat=args.ticks_per_beat)
    print()
    print("Done!")


if __name__ == "__main__":
    main()
