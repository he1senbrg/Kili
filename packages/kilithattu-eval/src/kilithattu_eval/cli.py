"""``kilithattu-eval`` command line."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from kilithattu_agents import load_policy
from kilithattu_core import load_params
from kilithattu_core.replay import verify_replay

from .matches import play_match


def _pol(ref, side, p):
    return load_policy(ref, side, p)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kilithattu-eval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("match", help="attack policy vs defence policy")
    m.add_argument("attack"); m.add_argument("defence"); m.add_argument("--rules", default="classic"); m.add_argument("-n", type=int, default=64)
    t = sub.add_parser("tournament", help="round robin over agent refs (each used for both sides)")
    t.add_argument("agents", nargs="+"); t.add_argument("--rules", default="classic"); t.add_argument("-n", type=int, default=32); t.add_argument("--out", default="runs/tournament")
    b = sub.add_parser("balance", help="Optuna balance search with scripted bots")
    b.add_argument("--rules", default="classic"); b.add_argument("--trials", type=int, default=20); b.add_argument("-n", type=int, default=32)
    v = sub.add_parser("verify-replay"); v.add_argument("path")
    pl = sub.add_parser("plot"); pl.add_argument("metrics"); pl.add_argument("--out", default="docs/figures/curve.png")
    args = ap.parse_args(argv)
    if args.cmd == "match":
        p = load_params(args.rules)
        print(json.dumps(play_match(p, _pol(args.attack, "attack", p), _pol(args.defence, "defence", p), args.n), indent=2))
    elif args.cmd == "tournament":
        from .tournament import round_robin, save_tournament

        p = load_params(args.rules)
        agents = {a: {"attack": _pol(a, "attack", p), "defence": _pol(a, "defence", p)} for a in args.agents}
        res = round_robin(p, agents, args.n)
        save_tournament(res, args.out)
        for n_, e in zip(res["names"], res["elo"]):
            print(f"{n_:30s} elo {e:7.1f}")
    elif args.cmd == "balance":
        from .balance import search

        study = search(load_params(args.rules), args.trials, args.n)
        print(json.dumps({"best_params": study.best_params, "best_value": study.best_value, "metrics": study.best_trial.user_attrs["metrics"]}, indent=2))
    elif args.cmd == "verify-replay":
        ok = verify_replay(args.path)
        print("OK" if ok else "MISMATCH")
        raise SystemExit(0 if ok else 1)
    elif args.cmd == "plot":
        from .plots import learning_curve

        print(learning_curve(args.metrics, args.out))


if __name__ == "__main__":
    main()
