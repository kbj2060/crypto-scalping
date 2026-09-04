#!/usr/bin/env python3
"""추세 신호 8종 v1 -- **TRAIN 전용 임계값 변형 탐색** + 선택 변형의 VAL/OOS 단일 확인 (2026-09-04).

첫 스크린(`research_eth_trend_signals_v1_screen_20260904.py`)의 임계값은 사전 지정이었고 VAL/OOS를 이미 1회 봤다.
여기서는 신호별 임계값 변형 2~3개를 **TRAIN에서만** 평가해 규칙(n≥300, P≥0.53, 차이 CI 하한>0)으로 하나를 고르고,
그 변형만 VAL/OOS를 **두 번째 조회**로 본다(문서에 '2차 조회'로 표기 — 다중성 방어는 TRAIN 선택 + 두 창 동시 요구).
변형 축은 신호마다 '강도 임계값' 하나만 움직인다(형태는 고정). 결과는 report_variants.json.
"""
from __future__ import annotations
import importlib.util, json, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("tv1", ROOT / "scripts/research_eth_trend_signals_v1_screen_20260904.py"); TV = importlib.util.module_from_spec(_s); _s.loader.exec_module(TV)
OUT = TV.OUT
VARIANTS = {
    "quarter_hour_boundary_flow": [("imbz1.5_rng0.8", 1.5, 0.8), ("imbz2.0_rng0.5", 2.0, 0.5), ("imbz1.0_rng1.0", 1.0, 1.0)],
    "session_open_range_breakout": [("volz0.5", 0.5), ("volz0.0", 0.0), ("volz1.5", 1.5)],
    "oi_confirmed_breakout": [("oi0.001_volz0.5", 0.001, 0.5), ("oi0.002_volz0.0", 0.002, 0.0), ("oi0.000_volz1.0", 0.0, 1.0)],
    "funding_squeeze": [("fz1.0", 1.0), ("fz0.75", 0.75), ("fz1.25", 1.25)],
    "regime_pullback_resume": [("bounce0.5", 0.5), ("bounce1.0", 1.0), ("bounce0.75", 0.75)],
    "btc_leadlag": [("zb1.5_lag0.5", 1.5, 0.5), ("zb2.0_lag0.7", 2.0, 0.7), ("zb1.5_lag0.7", 1.5, 0.7)],
    "spot_led_move": [("mv1.5", 1.5), ("mv2.0", 2.0), ("mv1.0_lead0.5atr", 1.0)],
    "liquidity_vacuum": [("mv0.5_w0.7", 0.5, 0.7), ("mv0.7_w0.75", 0.7, 0.75), ("mv0.5_w0.8", 0.5, 0.8)],
}


def main():
    t0 = time.time()
    kl = TV.load_klines(TV.KL); n = len(kl)
    o, h, l, c, v, tb = (kl[x].to_numpy(float) for x in ("open", "high", "low", "close", "volume", "taker_buy_base"))
    ts = kl["timestamp"]; close_ts = ts + pd.Timedelta(minutes=5)
    prev = np.r_[np.nan, c[:-1]]; tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev))); atr = pd.Series(tr).rolling(14, min_periods=14).mean().to_numpy()
    vol_z = TV.roll_z(kl["volume"]).to_numpy(); imb = (2 * tb - v) / np.where(v > 0, v, np.nan); imb_z = TV.roll_z(pd.Series(imb)).to_numpy(); rng_atr = (h - l) / atr
    ret3 = c / np.r_[np.nan, np.nan, np.nan, c[:-3]] - 1.0; mv3 = (c - np.r_[np.nan, np.nan, np.nan, c[:-3]]) / atr
    hi48 = pd.Series(h).shift(1).rolling(48).max().to_numpy(); lo48 = pd.Series(l).shift(1).rolling(48).min().to_numpy()
    hi24 = pd.Series(h).shift(1).rolling(24).max().to_numpy(); lo24 = pd.Series(l).shift(1).rolling(24).min().to_numpy()
    btc = TV.load_klines(TV.KL_BTC, "btc_"); bc = kl[["timestamp"]].merge(btc[["timestamp", "btc_close"]], on="timestamp", how="left")["btc_close"].ffill().to_numpy()
    bret3 = bc / np.r_[np.nan, np.nan, np.nan, bc[:-3]] - 1.0; z_e = TV.roll_z(pd.Series(ret3)).to_numpy(); z_b = TV.roll_z(pd.Series(bret3)).to_numpy()
    spot = TV.load_klines(TV.KL_SPOT, "spot_"); sc = kl[["timestamp"]].merge(spot[["timestamp", "spot_close"]], on="timestamp", how="left")["spot_close"].ffill().to_numpy(); sret3 = sc / np.r_[np.nan, np.nan, np.nan, sc[:-3]] - 1.0
    met = TV.load_metrics(); m = pd.merge_asof(pd.DataFrame({"ts": close_ts}), met, on="ts", direction="backward"); oi = m["sum_open_interest_value"].to_numpy(float); oi_chg6 = oi / np.r_[[np.nan] * 6, oi[:-6]] - 1.0
    fu = TV.load_funding(); fz = pd.merge_asof(pd.DataFrame({"ts": close_ts}), fu, on="ts", direction="backward")["funding_z"].to_numpy(float)
    reg = pd.read_parquet(TV.REG); reg["timestamp"] = pd.to_datetime(reg["timestamp"]); rg = kl[["timestamp"]].merge(reg[["timestamp", "regime_oof"]], on="timestamp", how="left")["regime_oof"].fillna(-1).astype(int).to_numpy()
    bd = TV.load_bookdepth(); b = pd.merge_asof(pd.DataFrame({"ts": close_ts}), bd, on="ts", direction="backward"); b1h = pd.merge_asof(pd.DataFrame({"ts": close_ts - pd.Timedelta(hours=1)}), bd, on="ts", direction="backward")
    up1, dn1, up1h, dn1h = (x.to_numpy(float) for x in (b["up1"], b["dn1"], b1h["up1"], b1h["dn1"]))
    minute = close_ts.dt.minute.to_numpy(); boundary = np.isin(minute, [0, 15, 30, 45]); tod = (ts.dt.hour * 60 + ts.dt.minute).to_numpy()
    lo_ref = pd.Series(l).shift(6).rolling(30).min().to_numpy(); hi_ref = pd.Series(h).shift(6).rolling(30).max().to_numpy()
    bounce_up = pd.Series(h).shift(1).rolling(5).max().to_numpy() - lo_ref; bounce_dn = hi_ref - pd.Series(l).shift(1).rolling(5).min().to_numpy()
    spot_lead = (sret3 - ret3) * c / atr                       # 현물 선도폭(ATR 단위)

    def orb(volz):
        up = np.zeros(n, bool); dn = np.zeros(n, bool)
        for start_min in (0, 480, 810):
            for s0 in np.flatnonzero(tod == start_min):
                if s0 + 30 >= n: continue
                rh, rl = h[s0:s0 + 6].max(), l[s0:s0 + 6].min()
                for i in range(s0 + 6, min(s0 + 30, n)):
                    if vol_z[i] >= volz and c[i] > rh: up[i] = True; break
                    if vol_z[i] >= volz and c[i] < rl: dn[i] = True; break
        return up, dn

    def trig(name, var):
        p = var[1:]
        if name == "quarter_hour_boundary_flow": return boundary & (imb_z >= p[0]) & (rng_atr >= p[1]), boundary & (imb_z <= -p[0]) & (rng_atr >= p[1])
        if name == "session_open_range_breakout": return orb(p[0])
        if name == "oi_confirmed_breakout": return (c > hi48) & (oi_chg6 >= p[0]) & (vol_z >= p[1]), (c < lo48) & (oi_chg6 >= p[0]) & (vol_z >= p[1])
        if name == "funding_squeeze": return (fz <= -p[0]) & (c > hi24), (fz >= p[0]) & (c < lo24)
        if name == "regime_pullback_resume": return (rg == 0) & (bounce_dn >= p[0] * atr) & (c > hi_ref), (rg == 1) & (bounce_up >= p[0] * atr) & (c < lo_ref)
        if name == "btc_leadlag": return (z_b >= p[0]) & (z_e < p[1] * z_b), (z_b <= -p[0]) & (z_e > p[1] * z_b)
        if name == "spot_led_move":
            if var[0].endswith("lead0.5atr"): return (mv3 >= p[0]) & (spot_lead >= 0.05), (mv3 <= -p[0]) & (spot_lead <= -0.05)
            return (mv3 >= p[0]) & (sret3 >= ret3), (mv3 <= -p[0]) & (sret3 <= ret3)
        if name == "liquidity_vacuum": return (mv3 >= p[0]) & (up1 <= p[1] * up1h), (mv3 <= -p[0]) & (dn1 <= p[1] * dn1h)
        raise KeyError(name)

    D = pd.read_parquet(TV.FRAME, columns=["pos", "is_downside", "timestamp", "split", "net_bp", "net_bp_flip"]); D["timestamp"] = pd.to_datetime(D["timestamp"])
    key = D.set_index(["timestamp", "is_downside"]); rng = np.random.default_rng(7)

    def day_ci(x, t):
        d = pd.Series(np.asarray(x, float), index=pd.DatetimeIndex(t).normalize()); days = d.index.unique().to_numpy(); g = d.groupby(level=0)
        sums = g.sum().reindex(days).to_numpy(); cnts = g.count().reindex(days).to_numpy(); out = np.empty(400)
        for k in range(400):
            j = rng.integers(0, len(days), len(days)); out[k] = sums[j].sum() / max(cnts[j].sum(), 1)
        return round(float(np.percentile(out, 2.5)), 2), round(float(np.percentile(out, 97.5)), 2)

    def stats(mask_up, mask_dn, windows):
        rows = []
        for mask, isd in ((mask_up, 1), (mask_dn, 0)):
            ff = TV.first_fire(np.nan_to_num(mask.astype(float)).astype(bool)); idx = np.flatnonzero(ff); idx = idx[(ts.iloc[idx] >= TV.START).to_numpy()]
            r = key.reindex(pd.MultiIndex.from_arrays([ts.iloc[idx].to_numpy(), np.full(len(idx), isd)], names=["timestamp", "is_downside"]))
            r = r[np.isfinite(r["net_bp"].to_numpy())].reset_index(); rows.append(r)
        A = pd.concat(rows); out = {}
        for w in windows:
            s = A[A.split == w]
            if len(s) < 20: out[w] = {"n": int(len(s))}; continue
            diff = (s.net_bp - s.net_bp_flip).to_numpy(); lo, hi = day_ci(diff, s.timestamp); days = pd.DatetimeIndex(s.timestamp).normalize().nunique()
            out[w] = {"n": int(len(s)), "per_day": round(len(s) / max(days, 1), 2), "p": round(float((s.net_bp > s.net_bp_flip).mean()), 3), "sig_bp": round(float(s.net_bp.mean()), 2),
                      "opp_bp": round(float(s.net_bp_flip.mean()), 2), "diff_bp": round(float(diff.mean()), 2), "ci": [lo, hi]}
        return out

    rep = {"note": "TRAIN-only variant selection; chosen variant VAL/OOS = second look (first look was the prereg screen)", "signals": {}}
    print(f"{'signal':>28s} {'variant':>18s} | {'TRAIN n':>7s} {'/d':>4s} {'P':>5s} {'diff':>6s} {'CI':>16s} {'ok':>3s}")
    for name, vars_ in VARIANTS.items():
        res = {}
        for var in vars_:
            up, dn = trig(name, var); st = stats(up, dn, ("TRAIN",))["TRAIN"]; res[var[0]] = st
            ok = st.get("n", 0) >= 300 and st.get("p", 0) >= 0.53 and st.get("ci", [-1])[0] > 0
            st["train_ok"] = bool(ok)
            print(f"{name:>28s} {var[0]:>18s} | {st.get('n',0):7d} {st.get('per_day',0):4.1f} {st.get('p',float('nan')):5.3f} {st.get('diff_bp',float('nan')):6.2f} {str(st.get('ci')):>16s} {'✓' if ok else '-':>3s}")
        oks = [k for k, s_ in res.items() if s_["train_ok"]]
        chosen = max(oks, key=lambda k: res[k]["ci"][0]) if oks else None       # TRAIN CI 하한이 가장 높은 변형
        conf = None
        # 규칙 경계(TRAIN P<0.53이지만 차이 CI 하한>0)인 신호도 후속(메타라벨·파일럿) 재검을 위해 CI 하한 최고 변형을 내보낸다
        export = chosen or max(res, key=lambda k: res[k].get("ci", [-99])[0])
        var = next(v_ for v_ in vars_ if v_[0] == export); up, dn = trig(name, var)
        rows_exp = []
        for mask, isd in ((up, 1), (dn, 0)):
            ff = TV.first_fire(np.nan_to_num(mask.astype(float)).astype(bool)); idx = np.flatnonzero(ff); idx = idx[(ts.iloc[idx] >= TV.START).to_numpy()]
            rows_exp.append(pd.DataFrame({"timestamp": ts.iloc[idx].to_numpy(), "is_downside": isd}))
        pd.concat(rows_exp).to_parquet(OUT / f"triggers_{name}.parquet", index=False)
        if chosen:
            conf = stats(up, dn, ("VAL", "OOS"))
            vo = int(conf["VAL"].get("diff_bp", -1) > 0) + int(conf["OOS"].get("diff_bp", -1) > 0)
            verdict = "PASS" if vo == 2 else ("WEAK" if vo == 1 else "REJECT")
            print(f"{'':>28s} -> chosen {chosen}: VAL n{conf['VAL'].get('n')} P {conf['VAL'].get('p')} diff {conf['VAL'].get('diff_bp')} {conf['VAL'].get('ci')} | OOS n{conf['OOS'].get('n')} P {conf['OOS'].get('p')} diff {conf['OOS'].get('diff_bp')} {conf['OOS'].get('ci')} => {verdict}")
        else:
            verdict = "REJECT(train)"
            print(f"{'':>28s} -> no TRAIN-passing variant")
        rep["signals"][name] = {"train_variants": res, "chosen": chosen, "exported_variant": export, "confirm": conf, "verdict": verdict}
    (OUT / "report_variants.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str))
    print("verdicts:", {k: v["verdict"] for k, v in rep["signals"].items()}); print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
