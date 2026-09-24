"""Tests for progressive_is_tempering.utils.report (metrics.json records)."""

import json
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from progressive_is_tempering.utils.report import json_safe, write_record


@dataclass
class _Cfg:
    lr: float = 1e-3
    n: int = 4


def test_json_safe_converts_scalars_arrays_and_non_finite():
    out = json_safe({
        "np_float": np.float32(0.5),
        "jax_float": jnp.asarray(0.25),
        "np_int": np.int64(3),
        "nan": float("nan"),
        "inf": np.inf,
        "arr": jnp.arange(3),
        "none": None,
        "cfg": _Cfg(),
        "tuple": (1, 2.0),
    })
    assert out == {
        "np_float": 0.5, "jax_float": 0.25, "np_int": 3, "nan": None, "inf": None,
        "arr": [0, 1, 2], "none": None, "cfg": {"lr": 1e-3, "n": 4}, "tuple": [1, 2.0],
    }
    json.dumps(out)   # valid JSON


def test_write_record_layout(tmp_path):
    rec = write_record(
        tmp_path / "dw4", "dw4", "sde_is",
        {"w2_pooled": np.float64(0.2), "w2_energy": 0.1, "geometric_w2": float("nan")},
        seed=3, config={"training": _Cfg()}, notes="x",
    )
    path = tmp_path / "dw4" / "sde_is" / "metrics.json"
    on_disk = json.loads(path.read_text())
    assert on_disk == rec
    assert on_disk["method"] == "progressive_is_tempering"
    assert (on_disk["target"], on_disk["run_id"], on_disk["seed"]) == ("dw4", "sde_is", 3)
    assert on_disk["metrics"] == {"w2_pooled": 0.2, "w2_energy": 0.1, "geometric_w2": None}
    assert on_disk["config"] == {"training": {"lr": 1e-3, "n": 4}}
    for key in ("schema_version", "commit", "timestamp", "notes"):
        assert key in on_disk


def test_write_record_overwrites(tmp_path):
    write_record(tmp_path, "gmm", "buffer", {"w2_pooled": 1.0})
    write_record(tmp_path, "gmm", "buffer", {"w2_pooled": 2.0})
    on_disk = json.loads((tmp_path / "buffer" / "metrics.json").read_text())
    assert on_disk["metrics"]["w2_pooled"] == 2.0
