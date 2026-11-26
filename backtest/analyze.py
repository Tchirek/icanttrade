"""Post-hoc analysis: cost scenarios and direction baselines from saved trades."""
import numpy as np
import pandas as pd
from pathlib import Path

from backtest import load_m1, build, simulate_long, simulate_short


def boot_ci(x, n=4000, seed=7):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n, len(x)))
    means = x[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def show(label, r):
    r = np.asarray(r)
    lo, hi = boot_ci(r)
    pf = r[r > 0].sum() / abs(r[r < 0].sum()) if (r < 0).any() else np.inf
    print(f"{label:<42} n={len(r):<5} win={(r>0).mean():.1%}  avgR={r.mean():+.3f}  "
          f"95%CI[{lo:+.3f},{hi:+.3f}]  PF={pf:.2f}")


def main():
    df = pd.read_csv(Path(__file__).resolve().parent / "trades_fixed.csv")
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True).dt.tz_convert("America/New_York")
    print(f"trades: {len(df)}  avg risk width={df.risk_pct.mean():.3%}  median={df.risk_pct.median():.3%}")
    print(f"gross R: mean={df.R.mean():+.3f}  median={df.R.median():+.3f}  "
          f"MFE>=1R={(df.mfe>=1).mean():.1%}  MFE>=2R={(df.mfe>=2).mean():.1%}")
    show("gross (zero cost)", df.R)
    for bps in (12, 6, 2, 0):
        r = df.R - df.cost_r * (bps / 12.0)
        show(f"net @ {bps} bps round trip", r)

    # direction baselines at the actual entry times
    m1 = load_m1()
    m5, d1_prev, asian, london, last_sh, last_sh_i, last_sl, last_sl_i, m1_start = build(m1)
    O, H, L, C = (m1[c].values for c in ("open", "high", "low", "close"))
    rng = np.random.default_rng(11)
    flips, coins = [], []
    for _, t in df.iterrows():
        pos = int(m1.index.get_indexer([t.entry_time])[0])
        if pos < 0:
            continue
        end_ts = pd.Timestamp(f"{t.date} 16:00", tz="America/New_York")
        ec = m1.index[m1.index >= end_ts]
        end_pos = int(m1.index.get_indexer([ec[0]])[0]) if len(ec) else len(O)
        if pos >= end_pos:
            continue
        risk = abs(t.entry - t.stop)
        side = -1 if t.bias == "long" else 1
        stop = t.entry - side * risk
        target = t.entry + side * 2 * risk
        exit_px, _, reason, _, _ = (simulate_long if side > 0 else simulate_short)(
            t.entry, stop, target, pos, end_pos, O, H, L, C)
        r = side * (exit_px - t.entry) / risk - t.cost_r
        flips.append(r)
        side2 = 1 if rng.random() < 0.5 else -1
        stop2 = t.entry - side2 * risk
        target2 = t.entry + side2 * 2 * risk
        exit2, _, _, _, _ = (simulate_long if side2 > 0 else simulate_short)(
            t.entry, stop2, target2, pos, end_pos, O, H, L, C)
        coins.append(side2 * (exit2 - t.entry) / risk - t.cost_r)

    show("flip direction, same entry time", flips)
    show("random side, same entry time", coins)

    # distribution of R and holding time
    print("\nR quantiles:", np.round(np.percentile(df.R_net, [1, 5, 25, 50, 75, 95, 99]), 2))
    hold = (pd.to_datetime(df.exit_time, utc=True) - pd.to_datetime(df.entry_time, utc=True)).dt.total_seconds() / 60
    print(f"holding minutes: median={hold.median():.0f}  mean={hold.mean():.0f}  p90={hold.quantile(.9):.0f}")
    print("\nby pool:")
    print(df.groupby("pool")["R_net"].agg(["count", "mean"]).round(3).to_string())


if __name__ == "__main__":
    main()
