"""
Writing `metrics.json` — one record per evaluated sample set.

The metric names are fixed: `w2_pooled` is Distance-W2, `w2_energy` Energy-W2,
`geometric_w2` Geometric-W2.

Records land at `results/<target>/[seedN/]<run_id>/metrics.json`, where `run_id` names
the sample set (`sde_raw`, `sde_is`, `buffer`). Never imports from the core
package (one-directional layering).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

__all__ = ["SCHEMA_VERSION", "json_safe", "git_commit", "write_record"]

#: Bumped when the record's shape changes.
SCHEMA_VERSION = 1

METHOD = "progressive_is_tempering"


def json_safe(x: Any) -> Any:
    """Recursively convert to JSON-safe values.

    NaN and inf become `None` rather than invalid JSON; NumPy / JAX scalars and
    arrays become Python floats, ints and lists; dataclasses become dicts.
    """
    if x is None or isinstance(x, (bool, str)):
        return x
    if is_dataclass(x) and not isinstance(x, type):
        return json_safe(asdict(x))
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, (int, np.integer)):
        return int(x)
    a = np.asarray(x)
    if a.ndim > 0:
        return [json_safe(v) for v in a.tolist()]
    if np.issubdtype(a.dtype, np.integer):
        return int(a)
    v = float(a)
    return v if np.isfinite(v) else None


def git_commit(cwd: Optional[Path] = None) -> Optional[str]:
    """Short hash of HEAD, or `None` if not a git repository or the tree is dirty.

    A hash that does not describe the code that actually ran is worse than no
    hash, because it looks authoritative.
    """
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=cwd,
                             capture_output=True, text=True, timeout=5)
        if rev.returncode != 0:
            return None
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=cwd,
                               capture_output=True, text=True, timeout=5)
        if dirty.returncode != 0 or dirty.stdout.strip():
            return None
        return rev.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def write_record(
    results_dir: Path,
    target: str,
    run_id: str,
    metrics: dict[str, Any],
    seed: int = 0,
    config: Optional[dict[str, Any]] = None,
    notes: Optional[str] = None,
) -> dict[str, Any]:
    """Write `results_dir / run_id / metrics.json` and return the record.

    `results_dir` is the target's folder, e.g. `results/dw4`. `config` holds the
    run's configuration (dataclasses are converted), so a record says what
    produced it. `commit` and `timestamp` are filled in here, so a notebook
    cannot report a stale one.
    """
    results_dir = Path(results_dir)
    record = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "target": target,
        "run_id": run_id,
        "seed": int(seed),
        "metrics": json_safe(metrics),
        "config": json_safe(config) if config is not None else None,
        "commit": git_commit(Path(__file__).resolve().parent),   # the code that ran
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "notes": notes,
    }
    path = results_dir / run_id / "metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")
    return record
