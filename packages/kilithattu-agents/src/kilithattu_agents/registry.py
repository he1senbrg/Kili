"""Resolve policy references such as ``main@hard``, ``run:<id>/step:<n>``, ``scripted:guard?reaction=0.5``."""
from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs

import yaml

from kilithattu_core import EnvParams

from .checkpoint import load_checkpoint
from .policy import NetworkPolicy
from .scripted import ScriptedAttack, ScriptedDefence


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs" / "game").is_dir():
            return parent
    return Path.cwd()


def load_ladder(path: str | Path | None = None) -> dict:
    path = Path(path) if path else _repo_root() / "configs" / "game" / "ladder.yaml"
    return yaml.safe_load(path.read_text())


def hub_download(repo: str, revision: str, subdir: str, cache: Path) -> Path:
    """Fetch a checkpoint folder from the Hugging Face Hub (optional dependency, cached locally)."""
    from huggingface_hub import snapshot_download

    root = snapshot_download(repo_id=repo, revision=revision, allow_patterns=[f"{subdir}/*"], cache_dir=str(cache))
    return Path(root) / subdir


def load_policy(ref: str, side: str, env_params: EnvParams, deterministic: bool = False, ladder: dict | None = None,
                runs_dir: str | Path = "runs"):
    """Return a Policy for ``side`` ("attack" | "defence")."""
    if ref.startswith("scripted:"):
        name, _, q = ref[len("scripted:"):].partition("?")
        kw = {k: float(v[0]) for k, v in parse_qs(q).items()}
        if side == "attack":
            return ScriptedAttack(env_params, **{k: v for k, v in kw.items() if k in ("aggression", "safety")})
        return ScriptedDefence(env_params, **{k: v for k, v in kw.items() if k in ("reaction", "anticipation")})
    if ref.startswith("run:"):
        run, _, step = ref[4:].partition("/step:")
        path = Path(runs_dir) / run / f"{side}" / (f"step-{step}" if step else "latest")
        params, cfg, _ = load_checkpoint(path, env_params)
        return NetworkPolicy(side, env_params, cfg, params, deterministic)
    if ref.startswith("path:"):
        params, cfg, _ = load_checkpoint(ref[5:], env_params)
        return NetworkPolicy(side, env_params, cfg, params, deterministic)
    name, _, level = ref.partition("@")
    ladder = ladder or load_ladder()
    entry = ladder["levels"][level or "medium"][name][side]
    if "scripted" in entry:
        return load_policy(entry["scripted"], side, env_params, deterministic)
    path = Path(entry["path"]) if "path" in entry else hub_download(ladder["hub_repo"], entry["revision"], entry["subdir"], Path.home() / ".cache" / "kilithattu")
    params, cfg, _ = load_checkpoint(path, env_params)
    return NetworkPolicy(side, env_params, cfg, params, deterministic)
