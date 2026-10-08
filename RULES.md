# Kilithattu Kali: formal rules as implemented

This is the rule set the engine (`kilithattu-core`) implements. Where the game is played differently in your
village, open a "Rule question" issue. Numbers are in metres and seconds.

## 1. Field
A rectangle of `n_lanes x n_rows` boxes (default 2 x 5, each 3 m x 3 m: 6 m wide, 15 m long). Transverse lines
L0..L5 run across the field (L0 is the home end, L5 the far end); longitudinal lines run along it. Beyond L0 is
the **home zone** and beyond L5 the **far zone**, each 2 m deep. Both zones are safe from tags.

## 2. Teams
- **Attack: 4 crossers.** Free 2D movement, 8 directions, speed 5.0 m/s.
- **Defence: 5 players.** 4 **guards**, one per inner line L1..L4, each sliding along its own line only; and the **Kili**, who
  moves along the line graph (any chalk line) and never enters a box. At junctions the Kili may turn; a turn is
  allowed within one step (`v*dt`) of a junction, where the position snaps onto the line.

## 3. Time and the clap
The engine runs at 20 ticks per second. Crossers are held behind L0 until the clap, which happens automatically
at 2 s (Classic). The innings lasts 120 s of live play from the clap, or ends earlier when every crosser is out.

## 4. Tagging
A defender within the tag radius (0.5 m) of a crosser puts that crosser out, unless the crosser is in a safe
zone. The test is **swept**: the closest approach during the tick counts, so fast movers cannot tunnel through
a guard. Tags resolve before line crossings in the same tick.

## 5. Trip progress, pacha and uppu
A crosser starts as **pacha** (going out). Progress `kappa` advances only when the *next expected* line is
crossed in the expected direction: L0 up to L5 up (6 lines), then, as **uppu**, L5 down to L0 down (6 lines).
Re-crossing a line already counted earns nothing, so points cannot be farmed by oscillating.

## 6. Fouls and boundaries
- **Foul:** if an uppu and a pacha are in the same box at the end of a tick, the one that *entered* the box that
  tick is out; if both or neither entered, the uppu is out.
- **Side exit:** leaving the field sideways between L0 and L5 puts the crosser out (0.1 m tolerance).

## 7. Scoring
+1 per correct line (12 per round trip) and +6 when the uppu reaches home with the salt: **18 per full trip**
(the reward is normalised so a trip is +1; the defence receives the exact negative, so the game is zero-sum).
A crosser that delivers becomes pacha again and may go again.

## 8. Kilithattu+ toggles (off in Classic)
| Toggle | Effect |
|---|---|
| P1 momentum | acceleration limit; sprint / dash with a stamina bar |
| P2 facing | defenders have a facing; tags only count inside the front half-plane; turning takes 0.3 s and disables tagging while turning |
| P3 vision | observations hide opponents beyond a radius (fog of war) |
| P4 strategic clap | the Kili chooses when to clap inside a 1-5 s window; crossing L0 before the clap is a false start (out) |
| P5 procedural fields | different `n_lanes`, `n_rows`, `n_crossers` (one environment per size; networks are size-independent) |
| P6 calls | a small set of discrete team calls broadcast to teammates |
| P7 human-fair reflexes | wrapper: observation delay, decision interval, sticky actions |

## 9. Balance v1
With the traditional-looking speeds (guard 4.2, Kili 4.8, tag radius 0.6) the scripted attack scored 0 round trips
and almost never got all four out: the defence mirrored every crosser. A small sweep against the scripted bots
chose **guard 3.0 m/s, Kili 3.6 m/s, tag radius 0.5 m**, giving roughly 24 points and 0.44 round trips per innings
with 38% of innings ending all-out. This was tuned for scripted bots only, so treat it as a starting point; experiment E2 re-fits it
with learned agents.

## 10. Determinism and replays
The step function is pure. A `.kreplay` folder stores the rules hash, seed and action log; `kilithattu-eval
verify-replay <folder>` re-simulates and compares a state hash.
