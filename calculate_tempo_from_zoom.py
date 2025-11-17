import pandas as pd
import numpy as np
import mido

def compute_beat_tempos_from_zoom(
    csv_path,
    tempo_slowest=32.0,
    tempo_fastest=132.0,
    zoom_cutoff_fraction=0.1,
    fps=30.0,
    smooth_beats=10,
    division=480,
    csv_out_path="beat_tempos.csv",
    midi_out_path="tempo_map.mid",
):
    """
    From a zoom-vs-frame CSV, build a tempo map with tempo proportional to zoom rate.

    Algorithm:
      1. Load frame_count_list and Gray_czd (zoom rate)
      2. Clip zoom to percentile range based on zoom_cutoff_fraction
      3. Map zoom percentiles to tempo range [tempo_slowest, tempo_fastest]
      4. Generate beats with unsmoothed tempo
      5. Smooth tempo in beat-space (triangular average over ±smooth_beats)
      6. Regenerate beat times with smoothed tempo
      7. Generate MIDI file with tempo changes

    Inputs:
      - csv_path: CSV with frame_count_list and Gray_czd columns
      - tempo_slowest: minimum tempo in BPM (default 32.0)
      - tempo_fastest: maximum tempo in BPM (default 132.0)
      - zoom_cutoff_fraction: fraction to clip at both ends (default 0.1 = 10%)
      - fps: video frame rate (default 30.0)
      - smooth_beats: half-width in beats for triangular smoothing (default 10)

    Outputs:
      - CSV with beat-level tempo info
      - Standard MIDI file with tempo events at every beat
    """

    # --- Load data ---
    df = pd.read_csv(csv_path)
    frames = df["frame_count_list"].to_numpy()
    zoom = df["Gray_czd"].to_numpy()

    # --- Clip zoom to percentile range ---
    percentile_low = zoom_cutoff_fraction * 100
    percentile_high = (1.0 - zoom_cutoff_fraction) * 100
    zoom_low = np.percentile(zoom, percentile_low)
    zoom_high = np.percentile(zoom, percentile_high)

    zoom_clipped = np.clip(zoom, zoom_low, zoom_high)

    print(f"Original zoom stats: min={zoom.min():.4f}, max={zoom.max():.4f}, mean={zoom.mean():.4f}")
    print(f"Clipping to {percentile_low:.0f}th-{percentile_high:.0f}th percentiles: [{zoom_low:.4f}, {zoom_high:.4f}]")
    print(f"Clipped zoom stats: min={zoom_clipped.min():.4f}, max={zoom_clipped.max():.4f}, mean={zoom_clipped.mean():.4f}")

    # --- Map zoom to tempo (no smoothing yet) ---
    # Map percentile values to tempo range
    # zoom_low (Xth percentile) → tempo_slowest
    # zoom_high (Yth percentile) → tempo_fastest
    tempo_at_frames = tempo_slowest + (zoom_clipped - zoom_low) / (zoom_high - zoom_low) * (tempo_fastest - tempo_slowest)

    # Debug output
    print(f"Tempo range (unsmoothed): {tempo_at_frames.min():.1f} - {tempo_at_frames.max():.1f} BPM")
    print(f"Number of sampled frames: {len(frames)}")

    # --- Interpolate tempo to all video frames ---
    # Create a continuous tempo function for all frames in the video
    video_duration = frames[-1] / fps
    all_frame_nums = np.arange(0, frames[-1] + 1)
    tempo_continuous = np.interp(all_frame_nums, frames, tempo_at_frames)

    # --- Walk through time and place beats ---
    beat_times = []
    beat_frames_list = []
    beat_tempos = []

    # First beat at time 0
    beat_times.append(0.0)
    beat_frames_list.append(0)
    beat_tempos.append(tempo_continuous[0])

    current_time = 0.0
    current_frame = 0
    beat_accumulator = 0.0  # Fraction of a beat accumulated

    dt = 1.0 / fps  # Time per frame

    for frame_num in all_frame_nums:
        tempo_now = tempo_continuous[frame_num]
        beats_per_second = tempo_now / 60.0
        beats_this_frame = beats_per_second * dt
        beat_accumulator += beats_this_frame

        # Check if we've accumulated a full beat
        if beat_accumulator >= 1.0:
            # Record the beat
            beat_times.append(current_time)
            beat_frames_list.append(frame_num)
            beat_tempos.append(tempo_now)
            beat_accumulator -= 1.0

        current_time += dt

    # Convert to numpy arrays
    beat_times = np.array(beat_times)
    beat_frames_array = np.array(beat_frames_list)
    tempo_bpm_unsmoothed = np.array(beat_tempos)

    n_beats = len(beat_times)
    n_bars = n_beats / 4.0

    print(f"Total video duration: {video_duration:.2f} seconds")
    print(f"Total beats generated: {n_beats}")
    print(f"Total bars (at 4/4): {n_bars:.1f}")

    # --- Smooth tempo in beat-space ---
    # IMPORTANT: Must smooth beat duration (seconds per beat), not tempo (beats per minute)
    # to preserve the mean tempo correctly (harmonic mean issue)

    # Convert tempo to beat duration (seconds per beat)
    beat_duration_unsmoothed = 60.0 / tempo_bpm_unsmoothed

    # Apply triangular smoothing over ±smooth_beats with edge reflection
    # Create triangular kernel
    kernel_size = 2 * smooth_beats + 1
    offsets = np.arange(-smooth_beats, smooth_beats + 1)
    kernel = np.maximum(0, smooth_beats + 1 - np.abs(offsets))
    kernel = kernel / kernel.sum()

    # Pad edges by reflection to preserve mean
    pad_width = smooth_beats
    duration_padded = np.pad(beat_duration_unsmoothed, pad_width, mode='reflect')

    # Apply convolution to beat durations
    beat_duration_smoothed = np.convolve(duration_padded, kernel, mode='valid')

    # Convert back to tempo
    tempo_bpm_midi = 60.0 / beat_duration_smoothed

    print(f"Average tempo (unsmoothed): {tempo_bpm_unsmoothed.mean():.1f} BPM, range: {tempo_bpm_unsmoothed.min():.1f}-{tempo_bpm_unsmoothed.max():.1f}")
    print(f"Average tempo (smoothed over ±{smooth_beats} beats): {tempo_bpm_midi.mean():.1f} BPM, range: {tempo_bpm_midi.min():.1f}-{tempo_bpm_midi.max():.1f}")

    # Verify mean preservation
    mean_duration_unsmoothed = beat_duration_unsmoothed.mean()
    mean_duration_smoothed = beat_duration_smoothed.mean()
    print(f"Mean beat duration: unsmoothed={mean_duration_unsmoothed:.3f}s, smoothed={mean_duration_smoothed:.3f}s")

    # --- Build CSV ---
    bars_arr = (np.arange(n_beats) // 4) + 1
    beat_in_bar = (np.arange(n_beats) % 4) + 1

    beat_df = pd.DataFrame({
        "beat_index": np.arange(1, n_beats + 1),
        "bar": bars_arr,
        "beat_in_bar": beat_in_bar,
        "time_sec": beat_times,
        "frame": beat_frames_array,
        "tempo_bpm": tempo_bpm_midi,
    })

    beat_df.to_csv(csv_out_path, index=False)

    # --- Build MIDI tempo map ---
    # Create MIDI file with tempo changes and notes at each beat
    midi_file = mido.MidiFile(ticks_per_beat=division)
    track = mido.MidiTrack()
    midi_file.tracks.append(track)

    # Add track name
    track.append(mido.MetaMessage('track_name', name='Tempo Map', time=0))

    # First tempo event at time 0
    # This tempo controls the duration until beat 1
    first_tempo = tempo_bpm_midi[0]
    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(first_tempo), time=0))

    # Add a note-on event for middle C at the start
    track.append(mido.Message('note_on', note=60, velocity=64, time=0))

    # Add a note-off event one quarter note later
    track.append(mido.Message('note_off', note=60, velocity=0, time=division))

    # Subsequent tempo events: one per beat, delta = one quarter note
    # Tempo at beat k controls duration from beat k to beat k+1
    for t in tempo_bpm_midi[1:]:
        # Tempo change at delta-time = 0 (simultaneous with note-off from previous beat)
        track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(t), time=0))

        # Note-on for middle C
        track.append(mido.Message('note_on', note=60, velocity=64, time=0))

        # Note-off one quarter note later
        track.append(mido.Message('note_off', note=60, velocity=0, time=division))

    # Save MIDI file
    midi_file.save(midi_out_path)

    # Total video duration
    T_total = beat_times[-1]

    return beat_df, csv_out_path, midi_out_path, T_total


if __name__ == "__main__":
    import os

    # Define paths and parameters
    base_dir = os.path.expanduser("~/video/data/output/N30_T7a_default")
    csv_path = os.path.join(base_dir, "N30_T7a_default_basic.csv")

    tempo_slowest = 32.0
    tempo_fastest = 116.0
    zoom_cutoff_fraction = 0.2
    smooth_beats = 16 # 4 bars half-width

    csv_out_path = os.path.join(base_dir, f"beat_tempos_{tempo_slowest:.0f}_{tempo_fastest:.0f}_{smooth_beats:.0f}_{zoom_cutoff_fraction:.2f}.csv")
    midi_out_path = os.path.join(base_dir, f"tempo_map_{tempo_slowest:.0f}_{tempo_fastest:.0f}_{smooth_beats:.0f}_{zoom_cutoff_fraction:.2f}.mid")

    beat_df, csv_path_out, midi_path_out, T_total = compute_beat_tempos_from_zoom(
        csv_path=csv_path,
        tempo_slowest=tempo_slowest,
        tempo_fastest=tempo_fastest,
        zoom_cutoff_fraction=zoom_cutoff_fraction,
        fps=30.0,
        smooth_beats=smooth_beats,
        division=480,
        csv_out_path=csv_out_path,
        midi_out_path=midi_out_path,
    )

    print(f"Total duration (s): {T_total:.2f}")
    print(f"CSV written to: {csv_path_out}")
    print(f"MIDI tempo map written to: {midi_path_out}")
