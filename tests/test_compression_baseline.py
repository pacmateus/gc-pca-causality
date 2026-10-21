"""Tests of the sorted-conditional compression baseline (Section 5, Table 4)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "causality"))

import baselines_compression as bc  # noqa: E402


def test_bz2_matches_paper():
    res = bc.evaluate("bz2")
    assert res["n"] == 102, res["n"]
    assert res["ties"] == 0, res["ties"]
    assert abs(res["acc"] - 74.66) < 0.05, res["acc"]


def test_lzma_matches_paper():
    res = bc.evaluate("lzma")
    assert res["ties"] == 25, res["ties"]
    assert abs(res["acc"] - 62.33) < 0.05, res["acc"]


def test_normaliser_uses_original_order():
    """The bug this guards against: normalising by the sorted sequence gives 70.46%."""
    pairs = bc.load_pairs()
    wrong = 0.0
    w_tot = sum(p["w"] for p in pairs)
    for p in pairs:
        qx, qy = bc.qlin(p["x"]), bc.qlin(p["y"])
        ox, oy = bc.np.argsort(qx, kind="mergesort"), bc.np.argsort(qy, kind="mergesort")
        c_x = bc._delta(qx, ox, "bz2")          # deliberately wrong normaliser
        c_y = bc._delta(qy, oy, "bz2")
        s = bc._delta(qx, oy, "bz2") / max(c_x, 1) - bc._delta(qy, ox, "bz2") / max(c_y, 1)
        wrong += p["w"] * (s > 0)
    wrong = wrong / w_tot * 100
    assert abs(wrong - 70.46) < 0.1, wrong


if __name__ == "__main__":
    for fn in (test_bz2_matches_paper, test_lzma_matches_paper,
               test_normaliser_uses_original_order):
        fn()
        print("PASS", fn.__name__)
