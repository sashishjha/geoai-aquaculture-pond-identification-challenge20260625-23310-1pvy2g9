"""
config.py — Global configuration object with deterministic hashing.

All experiment switches live here. Serialised to JSON and sha1-hashed into
every artefact filename and log line. A run without a config hash did not happen.
"""
from __future__ import annotations
import hashlib, json, os
from dataclasses import dataclass, asdict, field
from typing import Optional

# ── Global random seed ───────────────────────────────────────────────────────
SEED = 42


@dataclass
class RunConfig:
    """All experiment switches. Change one field per submission."""
    # Feature blocks
    use_absolute_anchors: bool = True        # Phase 3: restore Iter1 absolutes
    use_z_shape_features: bool = False       # Phase 2 z-scored temporal shapes
    use_rank_features: bool = False          # Phase 4 nonparametric ranks/sign changes

    # Domain adaptation
    use_av_weights: bool = False             # DISABLED: loss-level weighting destroys calibration

    # Calibration
    calibration_enabled: bool = False        # Phase 5: prior correction in log-odds space
    calibration_shift: float = 0.0          # total logit shift (derived from OOF t*, not theory)
    pi_t_assumed: float = 0.60              # provisional test prior (Phase 2 uses this before Phase 4)
    lambda_shrinkage: float = 0.5           # shrinkage on total shift

    # Augmentation
    n_augmentation_copies: int = 15         # Iter 1 value; sweep {10,15,25,40} in Phase 3

    # Splitting
    n_cv_folds: int = 5
    cv_block_size: int = 25

    # Reproducibility
    seed: int = SEED
    n_jobs: int = 1  # fixed to ensure XGB/LGBM determinism

    use_rescue_features: bool = False
    use_phenology_features: bool = False
    
    # pseudo label config
    is_pseudo: bool = False  # Phase 9.1: drift-immune phenology features
    drop_features: list[str] = field(default_factory=list)

    def hash(self) -> str:
        """sha1 of sorted JSON representation."""
        d = asdict(self)
        if not d.get("drop_features"):
            d.pop("drop_features", None)
        if not d.get("use_rescue_features"):
            d.pop("use_rescue_features", None)
        s = json.dumps(d, sort_keys=True)
        return hashlib.sha1(s.encode()).hexdigest()[:12]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "RunConfig":
        with open(path) as f:
            return cls(**json.load(f))


# ── Champion baseline (Iteration 1 configuration) ────────────────────────────
CHAMPION_CONFIG = RunConfig(
    use_absolute_anchors=True,
    use_z_shape_features=False,
    use_rank_features=False,
    use_av_weights=False,
    calibration_enabled=False,
    calibration_shift=0.0,
    n_augmentation_copies=15,
)
