"""Mechanical backtest of the ICT "Model A" core described in manual.md.

Pipeline: HTF bias (prior-day premium/discount) -> sweep of a low/high pool
-> displacement + MSS -> fresh FVG in the outer half of the leg -> limit
entry at the FVG near edge -> stop beyond the sweep extreme -> target at the
nearest opposite pool (>= 2R) or fixed 2R.

All timestamps are New York local time. Executions are simulated on 1-minute
bars; if stop and target are both touched inside one minute, the stop wins.
"""
from __future__ import annotations

import argparse
import pickle
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
CACHE = BASE / "cache"


@dataclass
class Cfg:
    disp_atr: float = 1.0          # displacement: body >= k * ATR(14)
    disp_body_frac: float = 0.6    # or body / range >= this
    reclaim_bars: int = 3          # sweep must reclaim within N 5m bars
    mss_window: int = 12           # bars allowed from sweep to MSS
    entry_valid: int = 12          # bars a resting limit stays alive
    stop_atr: float = 0.1          # stop buffer beyond sweep extreme
    min_rr: float = 2.0
    target_mode: str = "pool"      # "pool" (nearest >= min_rr) or "fixed"
    cost_bps: float = 5.0          # fee per side, bps
    slip_bps: float = 1.0          # slippage per side, bps
    require_open_side: bool = True  # sweep must be on the manipulation side of 09:30 open
    one_per_day: bool = True
    max_risk_pct: float = 0.03     # reject absurd stops (>3% of price)
    min_risk_pct: float = 0.0005   # reject tiny stops (<0.05%)


def load_m1() -> pd.DataFrame:
    files = sorted(CACHE.glob("BTCUSDT-1m-*.pkl"))
    if not files:
        raise SystemExit("no cached data; run download.py first")
    frames = []
    for f in files:
        with f.open("rb") as fh:
            frames.append(pickle.load(fh))
    m1 = pd.concat(frames).sort_index()
    m1 = m1[~m1.index.duplicated(keep="first")]
    m1.index = m1.index.tz_convert("America/New_York")
    return m1[["open", "high", "low", "close", "volume"]].astype(float)


def build(m1: pd.DataFrame):
    m5 = m1.resample("5min").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"), volume=("volume", "sum"),
    ).dropna()
    prev = m5["close"].shift(1)
    tr = pd.concat(
        [m5["high"] - m5["low"], (m5["high"] - prev).abs(), (m5["low"] - prev).abs()],
        axis=1,
    ).max(axis=1)
    m5["atr"] = tr.rolling(14).mean()

    d1 = m1.resample("1D").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
    ).dropna()
    d1_prev = d1.shift(1)

    asian = m1.between_time("00:00", "07:59").resample("1D").agg(
        high=("high", "max"), low=("low", "min")
    )
    london = m1.between_time("02:00", "04:59").resample("1D").agg(
        high=("high", "max"), low=("low", "min")
    )

    # 3-bar fractals, confirmed one bar later
    h, l = m5["high"].values, m5["low"].values
    n = len(m5)
    sh = np.zeros(n, bool)
    sl = np.zeros(n, bool)
    sh[1:-1] = (h[1:-1] > h[:-2]) & (h[1:-1] > h[2:])
    sl[1:-1] = (l[1:-1] < l[:-2]) & (l[1:-1] < l[2:])

    sh_price = np.full(n, np.nan)
    sh_idx = np.full(n, np.nan)
    sl_price = np.full(n, np.nan)
    sl_idx = np.full(n, np.nan)
    idx = np.arange(n)
    # swing at bar i is only usable at bar i+1
    sh_price[2:] = np.where(sh[1:-1], h[1:-1], np.nan)
    sh_idx[2:] = np.where(sh[1:-1], idx[1:-1], np.nan)
    sl_price[2:] = np.where(sl[1:-1], l[1:-1], np.nan)
    sl_idx[2:] = np.where(sl[1:-1], idx[1:-1], np.nan)
    last_sh = pd.Series(sh_price).ffill().values
    last_sh_i = pd.Series(sh_idx).ffill().values
    last_sl = pd.Series(sl_price).ffill().values
    last_sl_i = pd.Series(sl_idx).ffill().values

    m1_start = m1.index.get_indexer(m5.index)
    if (m1_start < 0).any():
        raise RuntimeError("m5 bar not aligned to m1")
    return m5, d1_prev, asian, london, last_sh, last_sh_i, last_sl, last_sl_i, m1_start


def simulate_long(entry, stop, target, fill_pos, end_pos, O, H, L, C):
    mfe = mae = 0.0
    risk = entry - stop
    for j in range(fill_pos, end_pos):
        mfe = max(mfe, (H[j] - entry) / risk)
        mae = min(mae, (L[j] - entry) / risk)
        if O[j] <= stop:
            return O[j], j, "stop", mfe, mae
        if O[j] >= target:
            return O[j], j, "target", mfe, mae
        if L[j] <= stop:
            return stop, j, "stop", mfe, mae
        if H[j] >= target:
            return target, j, "target", mfe, mae
    j = end_pos - 1
    return C[j], j, "time", mfe, mae


def simulate_short(entry, stop, target, fill_pos, end_pos, O, H, L, C):
    mfe = mae = 0.0
    risk = stop - entry
    for j in range(fill_pos, end_pos):
        mfe = max(mfe, (entry - L[j]) / risk)
        mae = min(mae, (entry - H[j]) / risk)
        if O[j] >= stop:
            return O[j], j, "stop", mfe, mae
        if O[j] <= target:
            return O[j], j, "target", mfe, mae
        if H[j] >= stop:
            return stop, j, "stop", mfe, mae
        if L[j] <= target:
            return target, j, "target", mfe, mae
    j = end_pos - 1
    return C[j], j, "time", mfe, mae


def run(cfg: Cfg, start: str | None = None, end: str | None = None):
    m1 = load_m1()
    if start:
        m1 = m1.loc[start:]
    if end:
        m1 = m1.loc[:end]
    m5, d1_prev, asian, london, last_sh, last_sh_i, last_sl, last_sl_i, m1_start = build(m1)

    O = m1["open"].values
    H = m1["high"].values
    L = m1["low"].values
    C = m1["close"].values

    O5, H5, L5, C5 = (m5[c].values for c in ("open", "high", "low", "close"))
    ATR5 = m5["atr"].values
    body = np.abs(C5 - O5)
    rng = np.maximum(H5 - L5, 1e-12)

    d1_map = {ts.date(): row for ts, row in d1_prev.iterrows()}
    asia_map = {ts.date(): row for ts, row in asian.iterrows()}
    lon_map = {ts.date(): row for ts, row in london.iterrows()}
    day_pos = m5.groupby(m5.index.date).indices

    trades, skips = [], Counter()
    cost_rate = (cfg.cost_bps + cfg.slip_bps) / 10_000.0

    for d in sorted(day_pos):
        positions = day_pos[d]
        # session bars 09:30..11:55 NY
        times = m5.index[positions]
        mask = (times.hour == 9) & (times.minute >= 30) | (times.hour == 10) | ((times.hour == 11) & (times.minute <= 25))
        bars = positions[mask]
        if len(bars) < 20:
            continue
        first, last = bars[0], bars[-1]
        if m5.index[first].hour != 9 or m5.index[first].minute != 30:
            continue
        if not (np.isfinite(ATR5[first]) and ATR5[first] > 0):
            continue
        prev = d1_map.get(d)
        if prev is None or not np.isfinite(prev["high"]):
            continue
        ash = asia_map.get(d)
        loh = lon_map.get(d)
        pdh, pdl = float(prev["high"]), float(prev["low"])
        if pdh <= pdl:
            continue
        open930 = O5[first]
        bias = "long" if open930 < (pdh + pdl) / 2 else "short"

        # session end = first 1m bar at/after 12:00
        end_ts = pd.Timestamp(f"{d} 16:00", tz="America/New_York")
        end_candidates = m1.index[m1.index >= end_ts]
        end_pos = int(m1.index.get_indexer([end_candidates[0]])[0]) if len(end_candidates) else len(O)

        consumed_low: set[float] = set()
        consumed_high: set[float] = set()
        i = 0
        while i < len(bars):
            b = bars[i]
            side = 1 if bias == "long" else -1
            if bias == "long":
                pools = [(pdl, "PDL"), (float(ash["low"]) if ash is not None else np.nan, "ASIA_L"),
                         (float(loh["low"]) if loh is not None else np.nan, "LON_L")]
                if np.isfinite(last_sl[b]) and b - last_sl_i[b] <= 288:
                    pools.append((float(last_sl[b]), "SWING_L"))
                pools = sorted([(p, n) for p, n in pools if np.isfinite(p)], reverse=True)
            else:
                pools = [(pdh, "PDH"), (float(ash["high"]) if ash is not None else np.nan, "ASIA_H"),
                         (float(loh["high"]) if loh is not None else np.nan, "LON_H")]
                if np.isfinite(last_sh[b]) and b - last_sh_i[b] <= 288:
                    pools.append((float(last_sh[b]), "SWING_H"))
                pools = sorted([(p, n) for p, n in pools if np.isfinite(p)])

            hit = None
            for p, name in pools:
                if p in (consumed_low if bias == "long" else consumed_high):
                    continue
                if bias == "long":
                    if L5[b] < p:
                        # same-bar reclaim or reclaim within N bars
                        if C5[b] > p:
                            hit = (i, p, name, float(L5[b]), b)
                        else:
                            for q in range(i + 1, min(i + 1 + cfg.reclaim_bars, len(bars))):
                                if C5[bars[q]] > p:
                                    hit = (i, p, name, float(L5[b:bars[q] + 1].min()), bars[q])
                                    break
                else:
                    if H5[b] > p:
                        if C5[b] < p:
                            hit = (i, p, name, float(H5[b]), b)
                        else:
                            for q in range(i + 1, min(i + 1 + cfg.reclaim_bars, len(bars))):
                                if C5[bars[q]] < p:
                                    hit = (i, p, name, float(H5[b:bars[q] + 1].max()), bars[q])
                                    break
                if hit:
                    break

            if not hit:
                i += 1
                continue

            start_i, pool, pool_name, sweep_ext, sweep_bar = hit
            (consumed_low if bias == "long" else consumed_high).add(pool)
            if cfg.require_open_side:
                if bias == "long" and sweep_ext >= open930:
                    i = start_i + 1
                    continue
                if bias == "short" and sweep_ext <= open930:
                    i = start_i + 1
                    continue

            # reference swing for MSS
            if bias == "long":
                ref = last_sh[sweep_bar]
                ref_i = last_sh_i[sweep_bar]
                if not np.isfinite(ref) or sweep_bar - ref_i > 288 or ref <= sweep_ext:
                    i = start_i + 1
                    continue
            else:
                ref = last_sl[sweep_bar]
                ref_i = last_sl_i[sweep_bar]
                if not np.isfinite(ref) or sweep_bar - ref_i > 288 or ref >= sweep_ext:
                    i = start_i + 1
                    continue

            fill = None
            for k in range(sweep_bar + 1, min(sweep_bar + 1 + cfg.mss_window, last + 1)):
                if bias == "long" and not (C5[k] > ref and C5[k] > O5[k]):
                    continue
                if bias == "short" and not (C5[k] < ref and C5[k] < O5[k]):
                    continue
                disp = False
                for t in range(sweep_bar, k + 1):
                    if body[t] >= cfg.disp_atr * ATR5[t] or body[t] / rng[t] >= cfg.disp_body_frac:
                        disp = True
                        break
                if not disp:
                    continue

                # fresh FVG in the outer half of the leg
                legs = []
                for t in range(sweep_bar + 2, k + 1):
                    if bias == "long":
                        if L5[t] > H5[t - 2]:
                            bottom, top = float(H5[t - 2]), float(L5[t])
                            if (bottom + top) / 2 <= (sweep_ext + C5[k]) / 2:
                                continue
                            if t < k and np.min(L5[t + 1:k + 1]) <= top:
                                continue
                            legs.append((top, bottom, t))
                    else:
                        if H5[t] < L5[t - 2]:
                            bottom, top = float(H5[t]), float(L5[t - 2])
                            if (bottom + top) / 2 >= (sweep_ext + C5[k]) / 2:
                                continue
                            if t < k and np.max(H5[t + 1:k + 1]) >= bottom:
                                continue
                            legs.append((bottom, top, t))
                if not legs:
                    continue
                if bias == "long":
                    entry, fvg_bottom, _ = max(legs, key=lambda x: x[0])
                    fvg_far = fvg_bottom
                    stop = sweep_ext - cfg.stop_atr * ATR5[k]
                    risk = entry - stop
                else:
                    entry, fvg_far, _ = min(legs, key=lambda x: x[0])
                    stop = sweep_ext + cfg.stop_atr * ATR5[k]
                    risk = stop - entry
                if not np.isfinite(risk) or risk <= 0:
                    continue
                if not (cfg.min_risk_pct <= risk / entry <= cfg.max_risk_pct):
                    continue

                # target: nearest opposite *major* pool at least min_rr away, not yet taken
                if bias == "long":
                    cands = [pdh, float(ash["high"]) if ash is not None else np.nan,
                             float(loh["high"]) if loh is not None else np.nan]
                    cands = [p for p in cands if np.isfinite(p) and p >= entry + cfg.min_rr * risk
                             and p > np.max(H5[first:k + 1])]
                    target = min(cands) if cands else np.nan
                else:
                    cands = [pdl, float(ash["low"]) if ash is not None else np.nan,
                             float(loh["low"]) if loh is not None else np.nan]
                    cands = [p for p in cands if np.isfinite(p) and p <= entry - cfg.min_rr * risk
                             and p < np.min(L5[first:k + 1])]
                    target = max(cands) if cands else np.nan

                if cfg.target_mode == "pool":
                    if not np.isfinite(target):
                        skips["no_pool_2r"] += 1
                        continue
                else:
                    target = entry + side * cfg.min_rr * risk

                fill = dict(sweep=pool_name, sweep_ext=sweep_ext, mss_bar=k, entry=entry,
                            fvg_far=fvg_far, stop=stop, risk=risk, target=target,
                            open930=open930, bias=bias)
                break

            if not fill:
                i = start_i + 1
                continue

            # resting limit, then 1m execution
            k = fill["mss_bar"]
            entry, stop, target = fill["entry"], fill["stop"], fill["target"]
            fill_pos = None
            for m in range(k + 1, min(k + 1 + cfg.entry_valid, last + 1)):
                pos0 = int(m1_start[m])
                for j in range(pos0, pos0 + 5):
                    if j >= end_pos:
                        break
                    if bias == "long" and L[j] <= entry:
                        fill_pos = j
                        break
                    if bias == "short" and H[j] >= entry:
                        fill_pos = j
                        break
                if fill_pos is not None:
                    break
                if bias == "long" and C5[m] < fill["fvg_far"]:
                    skips["invalid_before_fill"] += 1
                    break
                if bias == "short" and C5[m] > fill["fvg_far"]:
                    skips["invalid_before_fill"] += 1
                    break

            if fill_pos is None:
                skips["no_fill"] += 1
                i = start_i + 1
                continue

            if bias == "long":
                exit_px, exit_j, reason, mfe, mae = simulate_long(entry, stop, target, fill_pos, end_pos, O, H, L, C)
                r = (exit_px - entry) / fill["risk"]
            else:
                exit_px, exit_j, reason, mfe, mae = simulate_short(entry, stop, target, fill_pos, end_pos, O, H, L, C)
                r = (entry - exit_px) / fill["risk"]
            cost_r = 2 * cost_rate * entry / fill["risk"]
            trades.append(dict(
                date=d, bias=bias, pool=fill["sweep"], sweep_ext=fill["sweep_ext"],
                entry_time=m1.index[fill_pos], entry=entry, stop=stop, target=target,
                risk_pct=fill["risk"] / entry, exit_time=m1.index[exit_j],
                exit=exit_px, reason=reason, R=r, R_net=r - cost_r,
                mfe=mfe, mae=mae, cost_r=cost_r,
                mss_time=m5.index[k], open930=open930,
            ))
            if cfg.one_per_day:
                break
            i = start_i + 1

    return pd.DataFrame(trades), skips


def summarize(df: pd.DataFrame, label: str):
    if df.empty:
        print(f"{label}: no trades")
        return {}
    x = df["R_net"].values
    wins = x > 0
    sort_col = "entry_time" if "entry_time" in df.columns else "date"
    eq = np.cumsum(df.sort_values(sort_col)["R_net"].values)
    dd = float(np.max(np.maximum.accumulate(eq) - eq)) if len(eq) else 0
    gap = eq[-1] if len(eq) else 0
    out = {
        "label": label, "n": len(df), "win_rate": wins.mean(),
        "avg_R": x.mean(), "median_R": float(np.median(x)),
        "gross_avg_R": df["R"].mean(),
        "avg_cost_R": df["cost_r"].mean() if "cost_r" in df.columns else np.nan,
        "profit_factor": (x[x > 0].sum() / abs(x[x < 0].sum())) if (x < 0).any() else np.inf,
        "max_dd_R": dd, "total_R": gap,
        "avg_mfe": df["mfe"].mean() if "mfe" in df.columns else np.nan,
        "avg_mae": df["mae"].mean() if "mae" in df.columns else np.nan,
        "target_rate": (df["reason"] == "target").mean(),
        "stop_rate": (df["reason"] == "stop").mean(),
        "time_rate": (df["reason"] == "time").mean(),
        "avg_risk_pct": df["risk_pct"].mean() if "risk_pct" in df.columns else np.nan,
    }
    rng = np.random.default_rng(7)
    if len(x) > 1:
        n_boot = 2000 if len(x) <= 5000 else 400
        idx = rng.integers(0, len(x), (n_boot, len(x)))
        means = x[idx].mean(axis=1)
        out["ci_low"], out["ci_high"] = np.percentile(means, [2.5, 97.5])
    else:
        out["ci_low"] = out["ci_high"] = np.nan
    print(f"\n[{label}]  n={out['n']}  win={out['win_rate']:.1%}  avgR_net={out['avg_R']:.3f} "
          f"(95%CI {out['ci_low']:.3f}..{out['ci_high']:.3f})  avgR_gross={out['gross_avg_R']:.3f}  "
          f"cost={out['avg_cost_R']:.3f}R  PF={out['profit_factor']:.2f}  maxDD={out['max_dd_R']:.1f}R  "
          f"target/stop/time={out['target_rate']:.0%}/{out['stop_rate']:.0%}/{out['time_rate']:.0%}")
    return out


def baseline_random(m1, m5, d1_prev, asian, london, m1_start, cfg, n_per_day=10, seed=0, random_side=False):
    rng = np.random.default_rng(seed)
    O, H, L, C = (m1[c].values for c in ("open", "high", "low", "close"))
    O5, H5, L5, C5 = (m5[c].values for c in ("open", "high", "low", "close"))
    ATR5 = m5["atr"].values
    d1_map = {ts.date(): row for ts, row in d1_prev.iterrows()}
    day_pos = m5.groupby(m5.index.date).indices
    out = []
    for d in sorted(day_pos):
        positions = day_pos[d]
        times = m5.index[positions]
        mask = (times.hour == 9) & (times.minute >= 30) | (times.hour == 10) | ((times.hour == 11) & (times.minute <= 25))
        bars = positions[mask]
        if len(bars) < 10:
            continue
        if not (np.isfinite(ATR5[bars[0]]) and ATR5[bars[0]] > 0):
            continue
        prev = d1_map.get(d)
        if prev is None:
            continue
        bias = "long" if O5[bars[0]] < (prev["high"] + prev["low"]) / 2 else "short"
        end_ts = pd.Timestamp(f"{d} 16:00", tz="America/New_York")
        end_candidates = m1.index[m1.index >= end_ts]
        end_pos = int(m1.index.get_indexer([end_candidates[0]])[0]) if len(end_candidates) else len(O)
        for _ in range(n_per_day):
            b = int(rng.choice(bars))
            if not np.isfinite(ATR5[b]):
                continue
            side = 1 if (not random_side and bias == "long") or (random_side and rng.random() < 0.5) else -1
            entry = C5[b]
            stop = entry - side * ATR5[b]
            target = entry + side * cfg.min_rr * ATR5[b]
            start = int(m1_start[b]) + 5
            if start >= end_pos:
                continue
            exit_px, _, reason, mfe, mae = (simulate_long if side > 0 else simulate_short)(
                entry, stop, target, start, end_pos, O, H, L, C)
            r = side * (exit_px - entry) / ATR5[b]
            cost_r = 2 * (cfg.cost_bps + cfg.slip_bps) / 10_000 * entry / ATR5[b]
            out.append(dict(R_net=r - cost_r, R=r, reason=reason, date=d, cost_r=cost_r, mfe=mfe, mae=mae))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--target", default="pool", choices=["pool", "fixed"])
    ap.add_argument("--disp", type=float, default=1.0)
    ap.add_argument("--reclaim", type=int, default=3)
    ap.add_argument("--entry-valid", type=int, default=12)
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = Cfg(target_mode=args.target, disp_atr=args.disp, reclaim_bars=args.reclaim,
              entry_valid=args.entry_valid)
    trades, skips = run(cfg, args.start, args.end)
    print("skips:", dict(skips))
    if not trades.empty:
        trades.to_csv(BASE / (args.out or f"trades_{args.target}.csv"), index=False)
        summarize(trades, f"Model A / {args.target} / {args.start or 'all'}..{args.end or 'all'}")
        for y, g in trades.groupby(trades["entry_time"].dt.year):
            summarize(g, f"year {y}")
        summarize(trades[trades.bias == "long"], "long")
        summarize(trades[trades.bias == "short"], "short")

        if args.baseline:
            m1 = load_m1()
            m5, d1_prev, asian, london, *_rest, m1_start = build(m1)
            base = baseline_random(m1, m5, d1_prev, asian, london, m1_start, cfg, seed=1, n_per_day=3)
            summarize(base, "baseline: random entry, bias side, 1 ATR stop, 2R")
            base2 = baseline_random(m1, m5, d1_prev, asian, london, m1_start, cfg, seed=2, random_side=True, n_per_day=3)
            summarize(base2, "baseline: random entry, random side")

            # flip actual entries
            flips = []
            for _, t in trades.iterrows():
                side = -1 if t.bias == "long" else 1
                risk = abs(t.entry - t.stop)
                stop = t.entry + side * risk
                target = t.entry - side * cfg.min_rr * risk
                O, H, L, C = (m1[c].values for c in ("open", "high", "low", "close"))
                pos = int(m1.index.get_indexer([t.entry_time])[0])
                end_ts = pd.Timestamp(f"{t.date} 16:00", tz="America/New_York")
                end_candidates = m1.index[m1.index >= end_ts]
                end_pos = int(m1.index.get_indexer([end_candidates[0]])[0]) if len(end_candidates) else len(O)
                if pos >= end_pos:
                    continue
                exit_px, _, reason, mfe, mae = (simulate_long if side > 0 else simulate_short)(
                    t.entry, stop, target, pos, end_pos, O, H, L, C)
                r = side * (exit_px - t.entry) / risk
                cost_r = 2 * (cfg.cost_bps + cfg.slip_bps) / 10_000 * t.entry / risk
                flips.append(dict(R_net=r - cost_r, R=r, reason=reason, date=t.date, cost_r=cost_r, mfe=mfe, mae=mae))
            summarize(pd.DataFrame(flips), "baseline: flip actual direction")


if __name__ == "__main__":
    main()
