"""Run directory, JSONL metric log, optional Weights & Biases."""
from __future__ import annotations

import json
import time
from pathlib import Path


class RunLogger:
    def __init__(self, out_dir: str | Path, run_id: str, config: dict | None = None, wandb: bool = False):
        self.dir = Path(out_dir) / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.file = (self.dir / "metrics.jsonl").open("a")
        self.t0 = time.time()
        self.wb = None
        if config is not None:
            (self.dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
        if wandb:
            try:
                import wandb as _w

                self.wb = _w.init(project="kilithattu-rl", name=run_id, config=config)
            except Exception as e:  # pragma: no cover
                print(f"[log] wandb disabled: {e}")

    def __call__(self, step: int, metrics: dict, tag: str = "train") -> None:
        rec = {"tag": tag, "step": step, "time": round(time.time() - self.t0, 1), **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in metrics.items()}}
        self.file.write(json.dumps(rec) + "\n")
        self.file.flush()
        brief = {k: round(v, 3) for k, v in metrics.items() if k in ("ep/points", "ep/trips", "a/ent", "d/ent", "sps", "eval/points", "eval/p_win")}
        print(f"[{tag} {step}] {brief}", flush=True)
        if self.wb:
            self.wb.log({f"{tag}/{k}": v for k, v in metrics.items()}, step=step)
