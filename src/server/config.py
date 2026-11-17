"""Settings of the Claude MCP server (plan/10a section 1, limits from plan/10 section 12).

Every value can be overridden by an environment variable of the same name prefixed ``SO_``
(``SO_DRAFTS_PER_DAY=5``). The module-level constants are read once at import; tests build
their own ``Config(...)`` or call ``load(env)`` with a dict instead of touching the environment.
Days are UTC dates everywhere in the server.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path

ENV_PREFIX = "SO_"
MIB = 1024 * 1024


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Config:
    DATA_DIR: Path = Path("data/server")
    UPLOAD_DIR: Path | None = None          # default DATA_DIR/uploads
    DB_PATH: Path | None = None             # default DATA_DIR/server.sqlite3
    MIN_DURATION_S: float = 8.0
    MAX_DURATION_S: float = 60.0
    MAX_UPLOAD_BYTES: int = 100 * MIB
    ALLOWED_CONTAINERS: tuple[str, ...] = ("mp4", "mov", "webm")
    UPLOAD_TTL_S: int = 1800
    RESULT_TTL_S: int = 7 * 86400
    DRAFTS_PER_DAY: int = 3
    OWN_KEY_DRAFTS_PER_DAY: int = 10
    DAY_CAP_USD: float = 1.00
    MONTH_CAP_USD: float = 10.00
    EST_COST_PER_DRAFT_USD: float = 0.12
    BUDGET_FACTOR: float = 2.0              # a job is stopped once its recorded cost passes this many estimates
    MAX_JOBS_TOTAL: int = 2
    MAX_JOBS_PER_USER: int = 1
    BAD_TOKEN_MAX: int = 10
    BAD_TOKEN_WINDOW_S: int = 600
    PUBLIC_BASE_URL: str | None = None

    def __post_init__(self) -> None:
        set_ = lambda name, value: object.__setattr__(self, name, value)
        set_("DATA_DIR", Path(self.DATA_DIR))
        set_("UPLOAD_DIR", Path(self.UPLOAD_DIR) if self.UPLOAD_DIR is not None else self.DATA_DIR / "uploads")
        set_("DB_PATH", Path(self.DB_PATH) if self.DB_PATH is not None else self.DATA_DIR / "server.sqlite3")
        set_("ALLOWED_CONTAINERS", tuple(str(c).strip().lower() for c in self.ALLOWED_CONTAINERS))
        if self.PUBLIC_BASE_URL is not None:
            set_("PUBLIC_BASE_URL", str(self.PUBLIC_BASE_URL).rstrip("/") or None)
        for name in _NUMERIC:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ConfigError(f"{name} must be a finite number >= 0")
        if self.BUDGET_FACTOR < 1:
            raise ConfigError("BUDGET_FACTOR must be at least 1")
        if self.MIN_DURATION_S > self.MAX_DURATION_S:
            raise ConfigError("MIN_DURATION_S must not exceed MAX_DURATION_S")
        if not self.ALLOWED_CONTAINERS or not all(self.ALLOWED_CONTAINERS):
            raise ConfigError("ALLOWED_CONTAINERS must name at least one container")


_INT = ("MAX_UPLOAD_BYTES", "UPLOAD_TTL_S", "RESULT_TTL_S", "DRAFTS_PER_DAY", "OWN_KEY_DRAFTS_PER_DAY",
        "MAX_JOBS_TOTAL", "MAX_JOBS_PER_USER", "BAD_TOKEN_MAX", "BAD_TOKEN_WINDOW_S")
_FLOAT = ("MIN_DURATION_S", "MAX_DURATION_S", "DAY_CAP_USD", "MONTH_CAP_USD", "EST_COST_PER_DRAFT_USD",
          "BUDGET_FACTOR")
_PATH = ("DATA_DIR", "UPLOAD_DIR", "DB_PATH")
_NUMERIC = _INT + _FLOAT


def _parse(name: str, raw: str):
    try:
        if name in _INT:
            return int(raw)
        if name in _FLOAT:
            return float(raw)
    except ValueError:
        raise ConfigError(f"{ENV_PREFIX}{name} is not a valid number") from None
    if name in _PATH:
        return Path(raw).expanduser()
    if name == "ALLOWED_CONTAINERS":
        return tuple(part for part in (p.strip() for p in raw.split(",")) if part)
    return raw


def load(env: Mapping[str, str] | None = None, **overrides) -> Config:
    """Config from ``SO_*`` variables in ``env`` (default: the process environment); keyword overrides win."""
    env = os.environ if env is None else env
    values = {f.name: _parse(f.name, env[ENV_PREFIX + f.name]) for f in fields(Config)
              if env.get(ENV_PREFIX + f.name, "").strip()}
    return Config(**{**values, **overrides})


CONFIG = load()
globals().update({f.name: getattr(CONFIG, f.name) for f in fields(Config)})
