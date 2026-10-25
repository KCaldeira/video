# Audio Synthesis: Listening Notes and Ideas

A running log of listening impressions, open questions and ideas for
`metrics_to_wav.py`. How the code works is documented in `README.md`
("Audio Synthesis"); this file is for what we have heard, what we think it
means, and what to try next. Newest entries at the top of each section.

---

## Listening log

### 2026-10-24: "rattle" at 16.8 s in `B_r028_odd4_004` was clipping

A buzzing rattle at 16.8 s was not in the file: the channel fader sat at
+6 dB, and 16.8 s is where the file's peak first passes 0.5 (-5.75 dBFS),
so the boosted signal crossed 0 dBFS. The fader back at 0 dB fixed it.
These drones peak up to about -3 dBFS and are bass-heavy (55 Hz partial
up to -9 dBFS), so faders above +3 dB, or two tracks summed (peaks can add
6 dB), can clip at the output. Turn down the Stereo Out fader rather than
limiting each track.

### 2026-10-22: octave pattern, 30 dB weights, exponent -1

Configs: all `json/N52_drone_octave/N52_drone_*` (37 rows, octave pattern 1-8 then 10, 12,
14, 16, 20, ..., `weight_range_db` 30, `strength_exponent` -1, random phases).

- **Success.** It sounds a little like a pipe organ, with people pulling
  stops in and out.
- **Some notes clash with the rest.** In the odd-harmonics case (`odd4`)
  there is an unpleasant peak around 2 kHz (perhaps higher).

  Question: is it the partial furthest from a simple relationship (e.g. not
  a third above some octave) that grates?

  What the spectrum shows (long-term average, `G` renders, partials between
  0.8 and 5 kHz):

  | Render | Loudest partials, 0.8-5 kHz |
  |---|---|
  | `r055_odd4` | 880 (n=16), 1540 (n=28), 1100 (n=20), 1320 (n=24) Hz, all about -38 to -41 dB |
  | `r110_odd4` | 1100 (n=10), 880 (n=8), 1320 (n=12), 1540 (n=14) Hz, about -32 to -38 dB |
  | `r220_odd4` | **1100 (n=5) -13 dB, 1540 (n=7) -16 dB**, then 880 (n=4) and 1320 (n=6) at -28 dB |

  - At 55 and 110 Hz the odd rows (1, 3, 5, 7) all sit below 800 Hz, so
    `odd4` only lowers everything above that relative to the bottom. No
    partial sticks out there.
  - At **220 Hz** `odd4` boosts harmonics 5 and 7, which land at **1100 and
    1540 Hz**, 12 dB above their neighbours. This is very likely the peak
    heard.

  Likely reasons it grates (hypotheses, to test by ear):
  1. **The 7th harmonic family** (7, 14, 28, 56, ...; always **1540 Hz,
     3080 Hz, ...** for roots 55, 110, 220) is the only member of the
     pattern outside the familiar major-chord intervals. It is 31 cents
     flatter than an equal-tempered minor seventh, so it sounds "out of
     tune" to Western ears. This matches the "furthest from a simple
     relationship" idea: 7 is the largest prime in the set.
  2. **5 against 7 is a septimal tritone** (7:5, 583 cents). With both
     boosted (`r220_odd4`), the two loudest upper partials form a tritone.
  3. **7 next to 8 is a narrow gap** (8:7, 231 cents). At about 1.5-1.8 kHz
     the two partials are about 220 Hz apart, inside one critical band, so
     they sound rough. The same pair repeats in every octave (14:16,
     28:32, ...).
  4. **The ear is most sensitive from about 2 to 5 kHz**, so anything
     prominent there is heard as harsh even at modest level.

---

## Ideas to try

- **Set `006`** (rendered 2026-10-24, `json/N52_drone_006/`, 54 files: 6
  channel orderings x roots 27.5, 55, 110 Hz x flat/odd4/even4): the
  user's template (renamed `N52_drone_BGR_r055_flat_006.json`) mixes R, G,
  B and Gray rows in one tone. The five other orderings of R, G, B are each
  tagged by the channel at n = 2, 3, 5, to hear whether it matters which
  colour drives which partial.

- **Set `005`** (2026-10-24, `json/N52_drone_005/`, 27 configs: R/G/B x
  roots 27.5, 55, 110 Hz x flat/odd4/even4; **only the 9 B configs
  rendered**): the user's template `N52_drone_B_r055_flat_005.json`, the
  `004` harmonic numbers with the expressions reassigned to rows. Gains
  alternate by row position as in `004`.

- **Set `004`** (rendered 2026-10-24, `json/N52_drone_004/`, 45 files: R/G/B
  x roots 27.5, 55, 66, 110, 220 Hz x flat/odd4/even4), from the user's
  template `N52_drone_B_r028_flat_004.json`: 14 rows, n = 2, 3, 4, 5, 6, 8,
  10, 12, 16, 18, 20, 24, 27, 32, mostly even. Gains 4.0 and 1.0 alternate
  down the rows (`odd4` starts with 4.0, `even4` with 1.0) rather than
  following odd/even n. No pitch glide (`min_fraction` 1.0). The 55 Hz
  series was added after the others.

- **Set `003`** (rendered 2026-10-23, `json/N52_drone_003/`, 9 files): as
  `002` but root A0 = 27.5 Hz and n = 2, 3, 4, 5, 6, 8 (55, 82.5, 110, 137.5,
  165, 220 Hz; no partial at 27.5 Hz), same expressions in row order, same
  15/16 speed glide. Also rendered at roots 66 Hz (27.5 x 2 x 6/5, an
  octave and a just minor third up), 110 and 220 Hz: 36 files in all.

- **Set `002`** (rendered 2026-10-23, `json/N52_drone_002/`, 9 files: R/G/B
  x flat/odd4/even4): more bass and a subtler glide. Root G#1 =
  51.9130871975 Hz; slowest = 15/16 of it (a just semitone, -111.7 cents),
  so the pitch moves between G#1 and about G1.

- **Speed-driven pitch, set `001`** (rendered 2026-10-23,
  `json/N52_drone_001/`): the `000` set with the fundamental gliding with
  camera speed from `N52_speed.py`: fastest = root, slowest = root x 5/6 (a
  minor third down), even in cents. The pitch curve shows many fast
  excursions of 200-300 cents and long plateaus at the top (fastest).

- **Template `000`** (rendered 2026-10-23, `json/N52_drone_000/`, 18 files:
  R/G/B x 55/110 Hz x flat/odd4/even4): 6 harmonics, every one scaled by
  `C_avg / Gray_avg` (1: the ratio alone; 2-5: times `Gray_std`, `Gray_es1`,
  `Gray_ee1`, `Gray_ed1`; 6: times `C_std / Gray_std`), 4 cents jitter,
  30 dB weights, exponent -1.

- **Harmonics 1-6 only** (rendered 2026-10-22, `json/N52_drone_h6/`, same
  R/G/B x 55/110/220 x flat/odd4/even4 grid): octaves, fifths and the major
  third only; no 7th, nothing crowded up high. To get higher partials,
  overlay a second track rooted higher (an octave or other interval above)
  rather than extending the series. Expressions chosen for the limited set:
  `C_avg`, `C_std`, `C_avg / Gray_avg`, `(C_avg / Gray_avg) * Gray_ed1`,
  `* Gray_ee1`, `* Gray_es1`. `odd4` boosts rows 1, 3, 5 and `even4` rows
  2, 4, 6. Top partial: 330 Hz at 55, 660 Hz at 110, 1320 Hz at 220.

- **Test the 7th-harmonic hypothesis:** render `G_r220_odd4` with the gain
  of the 7-family rows (n = 7, 14, 28, 56, 112, 224, 448, 896) set to 0,
  then to 0.25, and compare by ear.
- **Drop the 7 family altogether:** use a pattern whose odd parts are only
  1, 3 and 5 (8, 10, 12, 16, 20, 24, ...). Every octave then holds a pure
  major triad.
- **Ear-sensitivity weighting:** reduce gains for partials in the 2-5 kHz
  region (a fixed equal-loudness tilt), rather than relying on 1/n alone.
- **Pitch jitter** on top of the octave pattern: probably less needed now
  that the dense upper harmonics are gone; check by ear.
- **MIDI-driven notes:** implemented (`"midi_file"`), awaiting a real
  monophonic MIDI file for N52.

---

## Open questions

- Is 30 dB the right weight range, or should it be larger, and should it
  differ between low and high harmonics?
- Should the gain shaping move from per-row numbers to a few named
  "stops" (e.g. 8', 4', 2 2/3', ... as on an organ)?
