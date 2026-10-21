"""The sorted-conditional compression baseline of Table 4 (bz2 and lzma).

Recipe, with no fitted parameters: quantise both variables linearly to 256
levels, sort one variable by the other, first-difference the sorted sequence as
int8, compress it with an off-the-shelf compressor, and compare the two
directions.  The score is

    s = C(x | sorted by y) / C(x) - C(y | sorted by x) / C(y)

where C(x) is the compressed size of the delta code of x in its original order,
and s > 0 means that x is the cause.

The two normalisers matter: taking the delta code of the *sorted* x instead of the
original order gives 70.46% on the CEP instead of the 74.66% reported, an error
that looks like a weak baseline rather than a bug.

This implementation was written from the algorithm description and reproduces the
published figures exactly: 102/102 per-pair decisions for both compressors, and
the same 25 lzma ties.

    python causality/baselines_compression.py
"""
import bz2
import json
import lzma
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TUE = os.path.join(HERE, "data", "tuebingen")

LEVELS = 256
MAXLEN = 8000
MIN_SAMPLES = 50
EXPECTED = {"bz2": 74.66, "lzma": 62.33}


def load_pairs(directory=TUE):
    """Eligible univariate pairs, cause first, with the official weights."""
    out = []
    with open(os.path.join(directory, "pairmeta.txt")) as fh:
        for line in fh:
            f = line.split()
            if len(f) < 6:
                continue
            pid, c0, c1, e0, e1, w = (int(f[0]), int(f[1]), int(f[2]),
                                      int(f[3]), int(f[4]), float(f[5]))
            if c0 != c1 or e0 != e1:
                continue
            path = os.path.join(directory, "pair%04d.txt" % pid)
            if not os.path.exists(path):
                continue
            d = np.loadtxt(path)
            if d.shape[0] < MIN_SAMPLES:
                continue
            if d.shape[0] > MAXLEN:
                idx = np.linspace(0, d.shape[0] - 1, MAXLEN).astype(int)
                d = d[idx]
            x, y = d[:, c0 - 1], d[:, e0 - 1]
            if not (np.isfinite(x).all() and np.isfinite(y).all()):
                continue
            out.append({"pair": pid, "w": w, "x": x, "y": y})
    return out


def qlin(x, levels=LEVELS):
    x = np.asarray(x, float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi - lo < 1e-12:
        return np.zeros(len(x), np.uint8)
    return np.clip(((x - lo) / (hi - lo) * (levels - 1)).round(),
                   0, levels - 1).astype(np.uint8)


def _csize(payload, coder):
    if coder == "bz2":
        return len(bz2.compress(payload, 9))
    return len(lzma.compress(payload, preset=9 | lzma.PRESET_EXTREME))


def _delta(v, order, coder):
    """Compressed size of the first differences of v taken in the given order."""
    return _csize(np.diff(v[order].astype(np.int16)).astype(np.int8).tobytes(), coder)


def score(x, y, coder="bz2"):
    """Signed score; positive means that x is the cause."""
    qx, qy = qlin(x), qlin(y)
    ox, oy = np.argsort(qx, kind="mergesort"), np.argsort(qy, kind="mergesort")
    # C(x): delta code of x in its own (original) order, not sorted by itself
    c_x = _csize(np.diff(qx.astype(np.int16)).astype(np.int8).tobytes(), coder)
    c_y = _csize(np.diff(qy.astype(np.int16)).astype(np.int8).tobytes(), coder)
    # C(x | sorted by y): x's values in the order induced by y
    c_x_given_y = _delta(qx, oy, coder)
    c_y_given_x = _delta(qy, ox, coder)
    return c_x_given_y / max(c_x, 1) - c_y_given_x / max(c_y, 1)


def evaluate(coder="bz2", pairs=None):
    """Weighted accuracy of one compressor, with the per-pair decisions."""
    pairs = pairs if pairs is not None else load_pairs()
    w = np.array([p["w"] for p in pairs])
    rows, ties = [], 0
    for p in pairs:
        s = score(p["x"], p["y"], coder)
        if abs(s) < 1e-15:
            ties += 1
            right = False
        else:
            right = s > 0
        rows.append({"pair": p["pair"], "w": p["w"], "s": float(s), "right": bool(right)})
    acc = float(sum(r["w"] * r["right"] for r in rows) / w.sum() * 100)
    return {"coder": coder, "acc": acc, "ties": ties, "n": len(rows), "per_pair": rows}


def main():
    pairs = load_pairs()
    print(f"{len(pairs)} eligible pairs, total weight {sum(p['w'] for p in pairs):.3f}")
    out, ok = {}, True
    for coder in ("bz2", "lzma"):
        res = evaluate(coder, pairs)
        out[coder] = res
        tag = "  ok" if abs(res["acc"] - EXPECTED[coder]) < 0.05 else f"  paper {EXPECTED[coder]}"
        ok = ok and abs(res["acc"] - EXPECTED[coder]) < 0.05
        print(f"  {coder:4s}  weighted {res['acc']:6.2f}%   ties {res['ties']:3d}{tag}")
    with open(os.path.join(HERE, "baselines_compression_results.json"), "w") as fh:
        json.dump({k: {"acc": v["acc"], "ties": v["ties"], "per_pair": v["per_pair"]}
                   for k, v in out.items()}, fh, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
