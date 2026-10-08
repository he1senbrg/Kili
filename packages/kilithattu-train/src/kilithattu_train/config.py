"""Training configuration (dataclasses, merged with YAML and CLI overrides through OmegaConf)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf

from kilithattu_agents import NetConfig


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip: float = 0.2
    vf_clip: float = 0.2
    vf_coef: float = 0.5
    ent_start: float = 0.02
    ent_end: float = 0.003
    max_grad_norm: float = 0.5
    epochs: int = 4
    minibatches: int = 4
    rollout_len: int = 128
    num_envs: int = 64
    shaping_beta: float = 0.1  # potential-based shaping weight (0 disables)
    total_updates: int = 1000


@dataclass
class NetCfg:
    d_model: int = 64
    layers: int = 2
    heads: int = 4
    hidden: int = 128
    gru: bool = True

    def build(self, n_calls: int = 4) -> NetConfig:
        return NetConfig(self.d_model, self.layers, self.heads, self.hidden, self.gru, n_calls)


@dataclass
class StageCfg:
    name: str
    env: dict = field(default_factory=dict)  # EnvParams overrides for the stage
    attack: str = "learn"  # learn | scripted:... | frozen
    defence: str = "scripted:guard?reaction=0.2"
    updates: int = 200
    promote_points: float = 0.0  # mean points per innings needed to leave the stage early
    ent_scale: float = 1.0


@dataclass
class TrainConfig:
    run_id: str = "run"
    seed: int = 0
    rules: str = "classic"
    out_dir: str = "runs"
    side: str = "both"  # both | attack | defence (self-play side selection)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    net: NetCfg = field(default_factory=NetCfg)
    log_every: int = 10
    ckpt_every: int = 100
    eval_episodes: int = 16
    wandb: bool = False
    stages: list[StageCfg] = field(default_factory=list)
    # league
    league_phases: int = 40
    phase_updates: int = 25
    snapshot_every: int = 4
    pfsp_power: float = 2.0
    win_points: float = 18.0
    exploiter_updates: int = 150
    # FCP
    fcp_partners: int = 8
    fcp_partner_updates: int = 60
    fcp_updates: int = 200


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> TrainConfig:
    cfg = OmegaConf.structured(TrainConfig)
    if path:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return OmegaConf.to_object(cfg)
