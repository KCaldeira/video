"""Frame-to-tick mapping for MIDI output.

Everything upstream of MIDI creation is indexed by video frame number and knows
nothing about tempo.  This module is the single place that turns a tempo
specification plus a frame rate into absolute MIDI ticks.

The tempo specification (`beats_per_minute` in the config) is overloaded:

  * a number  -> a constant tempo.
  * a string  -> a path to a MIDI "tempo file" (a conductor track carrying
                 `set_tempo` / `time_signature` meta events, e.g. the files
                 produced by calculate_tempo_from_inverse.py).  The variable
                 tempo map is read back, integrated to relate elapsed seconds to
                 ticks, and every video frame is placed by

                     frame -> (frame - origin) / fps seconds -> tick.

In both cases the first analysed frame maps to tick 0, and per-event rounding is
used so cumulative timing does not drift.
"""

import os

import mido

# MIDI default tempo (120 BPM) that applies before the first set_tempo event.
_DEFAULT_TEMPO_US = 500000


class FrameTickMap:
    """Maps video frame numbers to absolute MIDI ticks and carries the tempo map.

    Attributes:
        ticks_per_beat: resolution the output MIDI files must be written at.
        _tick_for_delta: callable mapping (frame - origin) -> absolute tick (int).
        _tempo_events: list of (abs_tick, mido.MetaMessage) to merge onto the
            first track of every output file (set_tempo, plus time_signature in
            the variable-tempo case).
    """

    def __init__(self, ticks_per_beat, tick_for_delta, tempo_events):
        self.ticks_per_beat = ticks_per_beat
        self._tick_for_delta = tick_for_delta
        self._tempo_events = tempo_events

    def ticks_for_frames(self, frame_list):
        """Return absolute ticks for a list of frame numbers (first frame -> 0)."""
        origin = frame_list[0]
        return [self._tick_for_delta(frame - origin) for frame in frame_list]

    def tempo_events(self):
        """Return [(abs_tick, MetaMessage)] for the conductor information."""
        return self._tempo_events


def _read_tempo_file(path):
    """Read a MIDI tempo file into (ticks_per_beat, tempo_changes, meta_events).

    tempo_changes: sorted list of (abs_tick, tempo_us) from set_tempo events.
    meta_events:   sorted list of (abs_tick, MetaMessage) for set_tempo and
                   time_signature, to be re-emitted into output files.
    """
    midi = mido.MidiFile(path)
    ticks_per_beat = midi.ticks_per_beat

    tempo_changes = []
    meta_events = []
    for track in midi.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == 'set_tempo':
                tempo_changes.append((abs_tick, msg.tempo))
                meta_events.append((abs_tick, msg.copy(time=0)))
            elif msg.type == 'time_signature':
                meta_events.append((abs_tick, msg.copy(time=0)))

    if not tempo_changes:
        raise ValueError(f"Tempo file '{path}' contains no set_tempo events")

    tempo_changes.sort(key=lambda e: e[0])
    # time_signature (order 0) before set_tempo (order 1) at the same tick.
    meta_events.sort(key=lambda e: (e[0], 0 if e[1].type == 'time_signature' else 1))
    return ticks_per_beat, tempo_changes, meta_events


def _build_tick_for_time(ticks_per_beat, tempo_changes):
    """Build a callable seconds -> absolute tick from a piecewise tempo map.

    Each tempo segment is (start_tick, start_seconds, tempo_us).  Within a
    segment the tempo is constant, so seconds and ticks are linearly related.
    Times past the final segment extrapolate with the last tempo.
    """
    if tempo_changes[0][0] != 0:
        tempo_changes = [(0, _DEFAULT_TEMPO_US)] + tempo_changes

    segments = []  # (start_tick, start_seconds, tempo_us)
    cum_seconds = 0.0
    prev_tick, prev_tempo = tempo_changes[0]
    segments.append((prev_tick, 0.0, prev_tempo))
    for tick, tempo_us in tempo_changes[1:]:
        cum_seconds += mido.tick2second(tick - prev_tick, ticks_per_beat, prev_tempo)
        segments.append((tick, cum_seconds, tempo_us))
        prev_tick, prev_tempo = tick, tempo_us

    def tick_for_time(seconds):
        # Find the last segment whose start time is at or before `seconds`.
        start_tick, start_seconds, tempo_us = segments[0]
        for seg in segments:
            if seg[1] <= seconds:
                start_tick, start_seconds, tempo_us = seg
            else:
                break
        offset_ticks = mido.second2tick(seconds - start_seconds, ticks_per_beat, tempo_us)
        return start_tick + offset_ticks

    return tick_for_time


def build_frame_tick_map(tempo_spec, frames_per_second, ticks_per_beat):
    """Build a FrameTickMap from a tempo specification.

    tempo_spec: a number (constant BPM) or a string path to a MIDI tempo file.
    """
    if isinstance(tempo_spec, str):
        if not os.path.exists(tempo_spec):
            raise FileNotFoundError(f"Tempo file not found: {tempo_spec}")
        file_tpb, tempo_changes, meta_events = _read_tempo_file(tempo_spec)
        tick_for_time = _build_tick_for_time(file_tpb, tempo_changes)

        def tick_for_delta(delta_frames):
            return int(round(tick_for_time(delta_frames / frames_per_second)))

        return FrameTickMap(file_tpb, tick_for_delta, meta_events)

    if isinstance(tempo_spec, bool):
        raise TypeError("beats_per_minute must be a number or a path string, not a bool")

    if isinstance(tempo_spec, (int, float)):
        bpm = float(tempo_spec)
        ticks_per_frame = ticks_per_beat * bpm / (60.0 * frames_per_second)
        tempo_events = [(0, mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm), time=0))]

        def tick_for_delta(delta_frames):
            return int(round(ticks_per_frame * delta_frames))

        return FrameTickMap(ticks_per_beat, tick_for_delta, tempo_events)

    raise TypeError(
        f"beats_per_minute must be a number or a path string, got {type(tempo_spec).__name__}"
    )
