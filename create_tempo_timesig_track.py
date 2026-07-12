#!/usr/bin/env python3
"""One-off: build a MIDI tempo + time-signature track for Cubase import.

Each cycle (23.5 s):
    1 beat  in (1/31)*23.5 s  with a 1/4 time signature
    48 beats in (30/31)*23.5 s with a 4/4 time signature
Cycle repeats 30 times.

The conductor track carries set_tempo + time_signature meta events, which
Cubase imports into its Tempo Track and Signature Track respectively.
"""
import mido

TICKS_PER_BEAT = 480
CYCLES = 30
CYCLE_SECONDS = 23.5

# --- segment timings (seconds) ---------------------------------------------
secA = (1 / 31) * CYCLE_SECONDS        # duration of the single 1/4 beat
secB = (30 / 31) * CYCLE_SECONDS       # duration of the 48 beats of 4/4

BEATS_A = 1
BEATS_B = 48
BEATS_PER_CYCLE = BEATS_A + BEATS_B

# --- tempo as microseconds per quarter note (computed directly, no BPM round-trip) ---
tempoA_us = round(secA / BEATS_A * 1_000_000)
tempoB_us = round(secB / BEATS_B * 1_000_000)

bpmA = mido.tempo2bpm(tempoA_us)
bpmB = mido.tempo2bpm(tempoB_us)

# --- build the conductor track ---------------------------------------------
mid = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
track = mido.MidiTrack()
mid.tracks.append(track)
track.append(mido.MetaMessage('track_name', name='Tempo/Sig Map', time=0))

# collect (abs_tick, order, message); order breaks ties so time_sig precedes tempo
events = []
for c in range(CYCLES):
    base = c * BEATS_PER_CYCLE * TICKS_PER_BEAT
    # segment A: 1 beat of 1/4
    events.append((base, 0, mido.MetaMessage('time_signature', numerator=1, denominator=4)))
    events.append((base, 1, mido.MetaMessage('set_tempo', tempo=tempoA_us)))
    # segment B: 48 beats of 4/4, starting one beat later
    b = base + BEATS_A * TICKS_PER_BEAT
    events.append((b, 0, mido.MetaMessage('time_signature', numerator=4, denominator=4)))
    events.append((b, 1, mido.MetaMessage('set_tempo', tempo=tempoB_us)))

events.sort(key=lambda e: (e[0], e[1]))

prev_tick = 0
for abs_tick, _order, msg in events:
    msg.time = abs_tick - prev_tick
    prev_tick = abs_tick
    track.append(msg)

total_ticks = CYCLES * BEATS_PER_CYCLE * TICKS_PER_BEAT
track.append(mido.MetaMessage('end_of_track', time=total_ticks - prev_tick))

OUT = 'tempo_timesig_track.mid'
mid.save(OUT)

# --- report ----------------------------------------------------------------
print(f"Saved: {OUT}")
print(f"Ticks per beat: {TICKS_PER_BEAT}")
print(f"Cycles: {CYCLES}   Beats per cycle: {BEATS_PER_CYCLE}   Total beats: {CYCLES * BEATS_PER_CYCLE}")
print(f"Total length: {CYCLES * CYCLE_SECONDS:.3f} s  ({total_ticks} ticks)")
print()
print("Segment A  (1/4, 1 beat):")
print(f"    duration   = {secA:.6f} s")
print(f"    tempo      = {tempoA_us} us/quarter")
print(f"    tempo      = {bpmA:.6f} BPM")
print()
print("Segment B  (4/4, 48 beats):")
print(f"    duration   = {secB:.6f} s   ({secB / BEATS_B:.6f} s/beat)")
print(f"    tempo      = {tempoB_us} us/quarter")
print(f"    tempo      = {bpmB:.6f} BPM")
print()
print("Per-cycle check:")
print(f"    A: {tempoA_us * BEATS_A / 1e6:.6f} s + B: {tempoB_us * BEATS_B / 1e6:.6f} s"
      f" = {(tempoA_us * BEATS_A + tempoB_us * BEATS_B) / 1e6:.6f} s (target {CYCLE_SECONDS})")
