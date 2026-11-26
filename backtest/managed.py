"""Simulate the manual's T1-half / move-to-BE management on saved entries."""
import numpy as np
import pandas as pd
from pathlib import Path

from backtest import load_m1


def sim(entry, stop, target, pos, end, side, O, H, L, C, be=True):
    risk = abs(entry - stop)
    t1 = entry + side * risk
    half = False
    cur_stop = stop
    rsum = 0.0
    for j in range(pos, end):
        # stop before anything else (conservative)
        if (side > 0 and O[j] <= cur_stop) or (side < 0 and O[j] >= cur_stop):
            rsum += (0.5 if half else 1.0) * (side * (O[j] - entry) / risk)
            return rsum, "stop"
        if not half and ((side > 0 and (O[j] >= t1 or H[j] >= t1)) or
                        (side < 0 and (O[j] <= t1 or L[j] <= t1))):
            px = t1 if (side > 0 and O[j] < t1) or (side < 0 and O[j] > t1) else O[j]
            rsum += 0.5 * (side * (px - entry) / risk)
            half = True
            if be:
                cur_stop = entry
            # same-bar ambiguity: runner stop first if BE was touched
            if be and ((side > 0 and L[j] <= entry) or (side < 0 and H[j] >= entry)):
                rsum += 0.5 * (side * (entry - entry) / risk)
                return rsum, "be"
            if (side > 0 and H[j] >= target) or (side < 0 and L[j] <= target):
                px2 = target if (side > 0 and O[j] < target) or (side < 0 and O[j] > target) else O[j]
                rsum += 0.5 * (side * (px2 - entry) / risk)
                return rsum, "target"
            continue
        if half and ((side > 0 and (O[j] >= target or H[j] >= target)) or
                     (side < 0 and (O[j] <= target or L[j] <= target))):
            px2 = target if (side > 0 and O[j] < target) or (side < 0 and O[j] > target) else O[j]
            rsum += 0.5 * (side * (px2 - entry) / risk)
            return rsum, "target"
        if (side > 0 and L[j] <= cur_stop) or (side < 0 and H[j] >= cur_stop):
            rsum += (0.5 if half else 1.0) * (side * (cur_stop - entry) / risk)
            return rsum, "stop"
    rsum += (0.5 if half else 1.0) * (side * (C[end - 1] - entry) / risk)
    return rsum, "time"


def stats(label, r):
    r = np.asarray(r)
    rng = np.random.default_rng(3)
    idx = rng.integers(0, len(r), (4000, len(r)))
    lo, hi = np.percentile(r[idx].mean(axis=1), [2.5, 97.5])
    pf = r[r > 0].sum() / abs(r[r < 0].sum()) if (r < 0).any() else np.inf
    print(f"{label:<34} n={len(r)}  win={(r>0).mean():.1%}  avgR={r.mean():+.3f}  "
          f"95%CI[{lo:+.3f},{hi:+.3f}]  PF={pf:.2f}")


def main():
    df = pd.read_csv(Path(__file__).resolve().parent / "trades_fixed.csv")
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True).dt.tz_convert("America/New_York")
    m1 = load_m1()
    O, H, L, C = (m1[c].values for c in ("open", "high", "low", "close"))
    for be in (True, False):
        rs, reasons = [], []
        for _, t in df.iterrows():
            pos = int(m1.index.get_indexer([t.entry_time])[0])
            end_ts = pd.Timestamp(f"{t.date} 16:00", tz="America/New_York")
            ec = m1.index[m1.index >= end_ts]
            end = int(m1.index.get_indexer([ec[0]])[0]) if len(ec) else len(O)
            if pos < 0 or pos >= end:
                continue
            side = 1 if t.bias == "long" else -1
            r, why = sim(t.entry, t.stop, t.target, pos, end, side, O, H, L, C, be=be)
            rs.append(r - t.cost_r)
            reasons.append(why)
        stats(f"T1 half + {'BE' if be else 'original stop'} @12bps", rs)
        print("   exits:", pd.Series(reasons).value_counts().to_dict())


if __name__ == "__main__":
    main()
