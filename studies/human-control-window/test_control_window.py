"""Regression tests: the code must reproduce the committed results.

The full Monte Carlo takes several minutes, so a few encounters are
re-measured and the summary numbers are recomputed from
results/encounters.csv. If a test fails after a change to the simulator,
either the change is a bug or the study must be run again and its results
committed.
"""

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_study import DESIGNS, MAIN, W_REQ_REF, evaluate, load_rows, share, wilson  # noqa: E402

ROWS = load_rows(os.path.join(HERE, "results", "encounters.csv"))
TEXT = open(os.path.join(HERE, "results", "control_window_results.txt"), encoding="utf8").read()


@pytest.mark.parametrize("k", [0, 1, 500, 999])
def test_encounter_is_reproduced(k):
    row = ROWS[k]
    out = evaluate((row["idx"], row["params"], None))
    assert out["status"] == "ok"
    keys = ["t_wc"] + [f"{q}_{d}" for d in DESIGNS for q in ("t_alert", "t_last", "W")]
    for key in keys:
        if row[key] is None:
            assert out[key] is None, key
        else:
            assert out[key] == pytest.approx(row[key], abs=1e-3), key
    for d in DESIGNS:
        assert out[f"maneuver_{d}"] == row[f"maneuver_{d}"]


@pytest.mark.parametrize("d", list(DESIGNS))
def test_summary_numbers_match_the_csv(d):
    w = [r[f"W_{d}"] for r in ROWS]
    lo, hi = wilson(w, W_REQ_REF)
    assert len(ROWS) == 1000
    assert (f"W_avail >= {W_REQ_REF} s (reference W_req): {share(w, W_REQ_REF):.1f}% "
            f"(95% CI {lo:.1f}-{hi:.1f}%)") in TEXT
    feasible = sum(r[f"t_last_{d}"] is not None for r in ROWS)
    assert f"some recovery still works at the alert: {feasible}/1000" in TEXT


def test_quoted_values():
    w = {d: np.array([r[f"W_{d}"] for r in ROWS]) for d in DESIGNS}
    assert share(w[MAIN], W_REQ_REF) == pytest.approx(73.6, abs=0.05)
    assert share(w[MAIN], 5) == pytest.approx(88.6, abs=0.05)
    assert share(w[MAIN], 25) == pytest.approx(39.3, abs=0.05)
    assert share(w["I45"], W_REQ_REF) == pytest.approx(93.6, abs=0.05)
    # projecting the own flight plan never removes meaningful control here
    a, b = w[MAIN] >= W_REQ_REF - 1e-9, w["I45"] >= W_REQ_REF - 1e-9
    assert (a & ~b).sum() == 0 and (~a & b).sum() == 200
