#!/usr/bin/env python3
"""4h 변동성 12라운드 — QX4 + 옵션 ATM IV·흐름 잔차 보정. 사전 등록 docs/prereg/vol4h_round12_option_iv_20261025.md.

    python3 scripts/research_vol4h_round12_option_iv_20261025.py          # 한 번만
QX4 는 4라운드 동결 예측(pred_*_r4_ETH) 그대로. 보정 계수는 검증 2026-01~05 에서만. 실주문 없음.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_vol4h_forecast_horserace_20261025 as V          # noqa: E402
import research_deribit_option_flow_direction_20261025 as O     # noqa: E402

OUT = Path("/home/kbj20/crypto-scalping/tmp/vol4h_r12_20261025")
H1 = 3_600_000


def hourly(tr, grid):
    """tr: timestamp·iv·d·T(일)·dnb·amount → 정시 d 마다 [d−1h,d) IV 중앙값, [d−4h,d) 순매수·거래량 피쳐."""
    hb = (tr.timestamp // H1 + 1) * H1
    atm = tr.d.abs().between(.35, .65) & tr["T"].between(1 / 24, 10)
    iv = tr[atm].groupby(hb[atm]).iv.median().reindex(grid).ffill(limit=6)
    dnb4 = tr.groupby(hb).dnb.sum().reindex(grid, fill_value=0.0).rolling(4).sum()
    vol4 = tr.groupby(hb).amount.sum().reindex(grid, fill_value=0.0).rolling(4).sum()
    z = dnb4 / dnb4.rolling(720, min_periods=720).std().shift(1)
    return pd.DataFrame(dict(iv=iv, x2=np.log1p(z.abs()), x3=np.log(vol4 / vol4.rolling(720, min_periods=720).mean().shift(1))), index=grid)


def selftest():
    g = np.arange(0, 12) * H1
    tr = pd.DataFrame(dict(timestamp=[10 * H1 - 1, 10 * H1], iv=[50.0, 70.0], d=[.5, .5], T=[2.0, 2.0], dnb=[1.0, 2.0], amount=[1.0, 1.0]))
    h = hourly(tr, g)
    assert h.iv[10 * H1] == 50.0 and h.iv[11 * H1] == 70.0            # 09:59:59 체결 → 10:00 결정에 포함, 10:00:00 체결은 11:00 부터


def main():
    selftest()
    rp = OUT / "report.json"; assert not rp.exists(), "판정은 한 번만"
    OUT.mkdir(parents=True, exist_ok=True)
    Pv, Pt = (pd.read_parquet(V.OUT / f"pred_{s}_r4_ETH.parquet") for s in ("val", "test"))
    tr = pd.concat([pd.read_parquet(f) for f in sorted((O.OUT / "trades").glob("*.parquet"))], ignore_index=True)
    tr = tr[tr.instrument_name.str.match(r"^ETH-\d+[A-Z]{3}\d{2}-\d+-[CP]$") & ~tr.block & ~tr.liq].reset_index(drop=True)
    exp = pd.to_datetime(tr.instrument_name.str.split("-").str[1], format="%d%b%y").astype("int64") // 10**6 + 8 * H1
    tr["T"] = (exp - tr.timestamp) / 86_400_000; tr["d"] = O.deltas(tr); tr = tr[np.isfinite(tr.d)]
    tr["dnb"] = np.where(tr.direction == "buy", 1, -1) * tr.d * tr.amount
    grid = np.arange(pd.Timestamp("2025-12-01").value // 10**6, pd.Timestamp("2026-10-25").value // 10**6, H1)
    Hf = hourly(tr, grid)

    def X(P, col):
        d = (P.index + pd.Timedelta("5min")).astype("int64") // 10**6
        assert (d % H1 == 0).all(), "결정 시각은 정시"
        f = Hf.reindex(d); yh = P[col].to_numpy()
        x1 = np.nan_to_num(np.log((f.iv.to_numpy() / 100) ** 2 * 4 / 8766) - yh)
        return yh, np.c_[np.ones(len(P)), x1], np.c_[np.ones(len(P)), x1, np.nan_to_num(f.x2.to_numpy()), np.nan_to_num(f.x3.to_numpy())], f
    yv, yt = Pv.y.to_numpy(), Pt.y.to_numpy()
    hv, Iv, Ov, fv = X(Pv, "v_QX4"); ht, It, Ot, ft = X(Pt, "t_QX4")
    bI = np.linalg.lstsq(Iv, yv - hv, rcond=None)[0]; bO = np.linalg.lstsq(Ov, yv - hv, rcond=None)[0]
    pv = dict(QX4=hv, QX4I=hv + Iv @ bI, QX4O=hv + Ov @ bO); pt = dict(QX4=ht, QX4I=ht + It @ bI, QX4O=ht + Ot @ bO)
    c = {m: float(np.log(np.mean(np.exp(yv - pv[m])))) for m in pv}
    L = {m: V.qlike(yt, pt[m] + c[m]) for m in pt}
    rep = dict(n_val=len(yv), n_test=len(yt), iv_coverage=dict(val=float(fv.iv.notna().mean()), test=float(ft.iv.notna().mean())),
               beta_I=bI.tolist(), beta_O=bO.tolist(), models={})
    for m in pt:
        r = dict(c=c[m], val_qlike=float(V.qlike(yv, pv[m] + c[m]).mean()), test_qlike=float(L[m].mean()),
                 test_mse=float(((yt - pt[m]) ** 2).mean()), test_corr=float(np.corrcoef(yt, pt[m])[0, 1]))
        if m != "QX4":
            r["dq_vs_QX4"] = V.day_ci(Pt.index, L[m] - L["QX4"])
            r["monthly_dq"] = {str(k): round(float(v), 4) for k, v in pd.Series(L[m] - L["QX4"], index=Pt.index).groupby(Pt.index.to_period("M")).mean().items()}
        rep["models"][m] = r
    q = rep["models"]["QX4I"]["dq_vs_QX4"]
    rep["G1"] = bool(q[0] < 0 and q[2] < 0); rep["G2"] = bool(bI[1] > 0)
    rep["verdict"] = "QX4I 후보" if rep["G1"] and rep["G2"] else "불통과 — 옵션 IV 축 닫음"
    rp.write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str)); print(json.dumps(rep, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
