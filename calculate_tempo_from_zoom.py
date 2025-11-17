import pandas as pd
import numpy as np
import struct

def compute_beat_tempos_from_zoom(
    csv_path,
    bars,
    fps=30.0,
    half_width_seconds=4.0,
    division=480,
    csv_out_path="beat_tempos.csv",
    midi_out_path="tempo_map.mid",
):
    """
    From a zoom-vs-frame CSV, build a tempo map with one tempo value per beat.

    Algorithm:
      1. Load frame_count_list and Gray_czd (zoom rate)
      2. Apply triangular filter to zoom rate (half-width = half_width_seconds)
      3. Compute cumulative sum of filtered zoom
      4. Place beats where cumulative zoom reaches evenly-spaced milestones:
         beat k occurs when cumsum(zoom) = k × total_cumsum / num_beats
      5. Calculate tempo from time between consecutive beats

    Inputs:
      - csv_path: CSV with frame_count_list and Gray_czd columns
      - bars: number of 4/4 bars (so num_beats = bars × 4)
      - fps: video frame rate (default 30.0)
      - half_width_seconds: triangular filter half-width in seconds

    Outputs:
      - CSV with beat-level tempo info
      - Standard MIDI file with tempo events at every beat
    """

    # --- Load data ---
    df = pd.read_csv(csv_path)
    frames = df["frame_count_list"].to_numpy()
    zoom = df["Gray_czd"].to_numpy()

    # --- Triangular smoothing ---
    # Half-width in frames (e.g., 4 seconds × 30 fps = 120 frames)
    half_width_frames = int(round(half_width_seconds * fps))
    offsets = np.arange(-half_width_frames, half_width_frames + 1)
    kernel = (half_width_frames + 1 - np.abs(offsets)).astype(float)
    kernel /= kernel.sum()

    zoom_smooth = np.convolve(zoom, kernel, mode="same")

    # --- Cumulative zoom ---
    cumsum_zoom = np.cumsum(zoom_smooth)
    total_cumsum = cumsum_zoom[-1]

    # --- Place beats at evenly-spaced cumulative zoom milestones ---
    n_beats = 4 * bars
    # Beat k should be at cumulative zoom = k × total_cumsum / n_beats
    # For k = 1, 2, 3, ..., n_beats
    zoom_milestones = np.arange(1, n_beats + 1) * total_cumsum / n_beats

    # Find frame indices where cumsum_zoom crosses each milestone
    beat_frames = np.searchsorted(cumsum_zoom, zoom_milestones)

    # Convert frame indices to times (in seconds)
    beat_times = beat_frames / fps

    # --- Calculate tempo from beat spacing ---
    # Time between consecutive beats
    beat_durations = np.diff(beat_times, prepend=0)
    # First beat duration is from time 0 to first beat

    # Tempo for CSV: represents the tempo of each beat (retrospective)
    # Tempo in BPM = 60 / duration_in_seconds
    tempo_bpm_csv = 60.0 / beat_durations
    tempo_bpm_csv = np.where(beat_durations > 0, tempo_bpm_csv, 120.0)

    # Tempo for MIDI: controls time until NEXT beat (prospective)
    # Duration from beat k to beat k+1
    beat_durations_forward = np.diff(beat_times, append=beat_times[-1])
    tempo_bpm_midi = 60.0 / beat_durations_forward
    tempo_bpm_midi = np.where(beat_durations_forward > 0, tempo_bpm_midi, 120.0)

    # --- Build CSV ---
    bars_arr = (np.arange(n_beats) // 4) + 1
    beat_in_bar = (np.arange(n_beats) % 4) + 1

    beat_df = pd.DataFrame({
        "beat_index": np.arange(1, n_beats + 1),
        "bar": bars_arr,
        "beat_in_bar": beat_in_bar,
        "time_sec": beat_times,
        "frame": beat_frames,
        "tempo_bpm": tempo_bpm_csv,
    })

    beat_df.to_csv(csv_out_path, index=False)

    # --- Build MIDI tempo map ---
    # Standard MIDI: header + 1 track with tempo meta events at each beat.
    def write_varlen(value: int) -> bytes:
        """Encode a MIDI variable-length quantity."""
        buffer = value & 0x7F
        bytes_out = []
        while True:
            bytes_out.insert(0, buffer & 0x7F)
            value >>= 7
            if value > 0:
                buffer = value & 0x7F | 0x80
            else:
                break
        # Set continuation bits for all but last byte
        for i in range(len(bytes_out) - 1):
            bytes_out[i] |= 0x80
        return bytes(bytearray(bytes_out))

    track_events = bytearray()

    # First tempo event at delta-time = 0
    # This tempo controls the duration until beat 1
    first_tempo = tempo_bpm_midi[0]
    mpq = int(round(60_000_000 / first_tempo))  # microseconds per quarter
    track_events += write_varlen(0)
    track_events += bytes([0xFF, 0x51, 0x03,
                           (mpq >> 16) & 0xFF,
                           (mpq >> 8) & 0xFF,
                           mpq & 0xFF])

    # Subsequent tempo events: one per beat, delta = one quarter note
    # Tempo at beat k controls duration from beat k to beat k+1
    for t in tempo_bpm_midi[1:]:
        mpq = int(round(60_000_000 / t))
        # One quarter-note later, in ticks
        track_events += write_varlen(division)
        track_events += bytes([0xFF, 0x51, 0x03,
                               (mpq >> 16) & 0xFF,
                               (mpq >> 8) & 0xFF,
                               mpq & 0xFF])

    # End-of-track meta event
    track_events += write_varlen(0)
    track_events += bytes([0xFF, 0x2F, 0x00])

    # Track chunk
    track_chunk = b"MTrk" + struct.pack(">I", len(track_events)) + track_events

    # Header chunk: format 1, 1 track, 'division' ticks per quarter
    header_chunk = struct.pack(">4sIHHH", b"MThd", 6, 1, 1, division)

    midi_data = header_chunk + track_chunk

    with open(midi_out_path, "wb") as f:
        f.write(midi_data)

    # Total video duration
    T_total = beat_times[-1]

    return beat_df, csv_out_path, midi_out_path, T_total


if __name__ == "__main__":
    import os

    # Define paths
    base_dir = os.path.expanduser("~/video/data/output/N30_T7a_default")
    csv_path = os.path.join(base_dir, "N30_T7a_default_basic.csv")
    csv_out_path = os.path.join(base_dir, "beat_tempos_408bars.csv")
    midi_out_path = os.path.join(base_dir, "tempo_map_408bars.mid")

    beat_df, csv_path_out, midi_path_out, T_total = compute_beat_tempos_from_zoom(
        csv_path=csv_path,
        bars=408,
        fps=30.0,
        half_width_seconds=4.0,
        division=480,
        csv_out_path=csv_out_path,
        midi_out_path=midi_out_path,
    )

    print(f"Total duration (s): {T_total:.2f}")
    print(f"Number of beats: {len(beat_df)}")
    print(f"Tempo range: {beat_df['tempo_bpm'].min():.1f} - {beat_df['tempo_bpm'].max():.1f} BPM")
    print(f"CSV written to: {csv_path_out}")
    print(f"MIDI tempo map written to: {midi_path_out}")
