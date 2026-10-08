# Kilithattu-RL

A formal rules engine, reinforcement-learning stack and playable game for **Kilithattu Kali**, the Kerala Onam
line-tag game. See `PLAN.md` / `PROPOSAL.md` for the research plan and `RULES.md` for the implemented rules.

## Packages
| Package | Contents |
|---|---|
| `kilithattu-core` | JAX environment (Classic + P1-P4, P6), rules, replays, plain-Python reference for differential tests |
| `kilithattu-agents` | entity-attention actor/critic, scripted bots, checkpoints, difficulty ladder, human-fair wrapper (P7) |
| `kilithattu-train` | MAPPO, curriculum, self-play league with PFSP and exploiters, FCP partners, behaviour cloning |
| `kilithattu-eval` | matches, Elo and Nash ratings, tournaments, exploitability, behaviour detectors, Optuna balance search, plots |
| `kilithattu-game` | pygame client: play as crosser, guard or Kili against scripted or trained bots; replays |

`web/kilithattu.html` is a small single-file browser version of the game (scripted bots, no install).

## Install
```bash
pip install -r requirements.txt
pip install -e packages/kilithattu-core -e packages/kilithattu-agents -e packages/kilithattu-game
pip install -e packages/kilithattu-train -e packages/kilithattu-eval --no-deps
```
Or `uv sync`. GPU: `pip install "jax[cuda12]"`.

## Play
```bash
kilithattu --side attack --difficulty medium         # run as a crosser (WASD/arrows, Shift dash, Tab switch)
kilithattu --side kili --difficulty easy             # run the Kili (hold a direction, you turn at junctions)
kilithattu --side watch --rules plus                 # watch the bots, Kilithattu+ rules
kilithattu --headless-ticks 400 --side watch         # no window, prints the result
```

## Train and evaluate
```bash
kilithattu-train curriculum --config configs/train/smoke.yaml      # tiny run
kilithattu-train curriculum --config configs/train/curriculum.yaml # real run (use a GPU)
kilithattu-train league --config configs/train/league.yaml
kilithattu-train fcp --config configs/train/fcp.yaml
kilithattu-eval match scripted:crosser scripted:guard
kilithattu-eval verify-replay replays/<game>.kreplay
```
Experiments E1-E7 live in `experiments/` with configs in `configs/eval/`.

## Status: what has and has not been verified
**Tested (passing):** core rules engine (37 tests incl. differential tests against the reference), agents (7), game
(4, headless). A one-crosser MAPPO run learned 3 full round trips per 24 s innings.

**Written but never run in full:** the league, FCP partner mixing, behaviour cloning, curriculum runner,
exploitability, tournaments, balance search and plots. Their test files exist (`-m nightly`) but I did not get to see them
pass; expect some bugs there. No trained checkpoints are included: the difficulty ladder uses scripted bots until you train and
point `configs/game/ladder.yaml` at checkpoints.

**Known limits:** scripted bots assume full observability (do not use with P3); the Kili's P1 uses dash speed but no
acceleration; checkpoints use flax msgpack rather than Orbax; balance v1 was tuned on scripted bots only.
