"""**테이프 재생성** — unified 화석의 «주문흐름·고래 7열」을 원자료에서 전 구간으로 다시 만든다 (2026-09-14).

## 왜
`rl_training_*_unified.csv`(2026-02-28 까지, 재생산 불가)의 주문흐름·고래 12열이 gym 라벨에서
쌍둥이 증분 **+0.040/+0.107**(덮인 VAL·OOS 1~2월)을 냈다. 쪼개 보니 이득은 **테이프에서 재생성
가능한 7열**에 있었고(+0.047/+0.085) 구성식 불명 5열은 +0.010 이었다.
그런데 내가 앞서 같은 테이프 parquet 로 만든 `tape` 16열은 같은 구간에서 **−0.011** 이었다.
⇒ 차이는 «자료원」이 아니라 **«어떤 변환을 했나»**. unified 7열에 있고 내 16열에 없던 모양은 셋:
   ① 고래/소매 **비율**(대형 거래량 ÷ 소형 거래량)  ② **1차 차분**(가속도)  ③ **Amihud 비유동성**.
여기서 그 셋을 테이프(2024-04-20~2026-09-05, **TEST 포함**)로 만들어 3창 규칙으로 판정한다.

## 판정 (실행 전 고정 — §12 와 같은 규칙)
군이 상태에 들어가려면 43열 기준 대비 쌍둥이 표본외 IC 증분이 **OOS·TEST 둘 다 ≥ +0.01**.
⚠️unified 결과는 «후보 지목」이지 근거가 아니다 — 그 창(VAL+1~2월)은 이미 봤으므로, 판정은
**한 번도 안 본 TEST** 가 실질적으로 정한다.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import research_eth_rl_gym_direction_ppo_20260914 as P  # noqa: E402
import rl_gym_direction_env_20260914 as G  # noqa: E402

TAPE = ROOT / "data/research/eth_tape_1m_20260906.parquet"
OUTP = P.OUT / "tape_rebuild_features.parquet"


def z(s: pd.Series, w: int = 288) -> pd.Series:
    r = s.rolling(w, min_periods=max(20, w // 4))
    return (s - r.mean()) / r.std().replace(0, np.nan)


def build(d: pd.DataFrame) -> pd.DataFrame:
    t = pd.read_parquet(TAPE, columns=["ts", "n_trades", "volume", "signed_vol", "signed_notional",
                                       "notional", "lg_vol", "xl_vol", "px_last", "px_first"])
    t["ts"] = pd.to_datetime(t["ts"]); t["bin"] = t["ts"].dt.floor("5min")
    g = t.groupby("bin")
    a = pd.DataFrame({
        "n_trades": g["n_trades"].sum(), "volume": g["volume"].sum(),
        "signed_vol": g["signed_vol"].sum(), "signed_notional": g["signed_notional"].sum(),
        "notional": g["notional"].sum(), "lg_vol": g["lg_vol"].sum(), "xl_vol": g["xl_vol"].sum(),
        "px_last": g["px_last"].last(), "px_first": g["px_first"].first()}).reset_index().rename(columns={"bin": "ts"})
    big = a["lg_vol"] + a["xl_vol"]
    small = (a["volume"] - big).clip(lower=1e-9)
    f = pd.DataFrame()
    # ① 고래/소매 비율 (unified whale_retail_ratio) -- 내 16열엔 «비중(share)»만 있었고 «비율」이 없었다
    f["tr_whale_retail"] = np.log((big / small).clip(lower=1e-9))
    f["tr_whale_retail_z"] = z(f["tr_whale_retail"])
    # net taker ratio (레벨) + ② 그 1차 차분 = 가속도 (unified taker_acceleration)
    ntr = a["signed_vol"] / a["volume"].replace(0, np.nan)
    f["tr_net_taker"] = ntr
    f["tr_taker_accel"] = ntr.diff()
    # ② OFI 가속도 (unified ofi_acceleration) -- 부호 명목의 1차 차분을 명목으로 정규화
    ofi = a["signed_notional"] / a["notional"].replace(0, np.nan)
    f["tr_ofi"] = ofi
    f["tr_ofi_accel"] = ofi.diff()
    # 거래 강도 (unified trade_intensity) = 건수 / 롤링 평균
    f["tr_intensity"] = a["n_trades"] / a["n_trades"].rolling(288, min_periods=72).mean()
    # 대형 비중(내 16열에도 있었으나 같은 판에서 비교하려 포함)
    f["tr_big_share"] = big / a["volume"].replace(0, np.nan)
    # ③ Amihud 비유동성 = |수익| / 명목, z
    ret = np.log(a["px_last"] / a["px_first"].replace(0, np.nan)).abs()
    f["tr_amihud_z"] = z(np.log((ret / a["notional"].replace(0, np.nan)).clip(lower=1e-18)))
    f.insert(0, "ts", a["ts"])
    out = d[["timestamp"]].merge(f, left_on="timestamp", right_on="ts", how="left").drop(columns=["ts"])
    return out.reset_index(drop=True)


def main() -> int:
    d, sm, win, S43, cols43, _ = P.prepare(G.DEFAULT_FAMILIES)
    if OUTP.exists():
        F = pd.read_parquet(OUTP)
    else:
        print("테이프 재집계 …", flush=True)
        F = build(d); F.to_parquet(OUTP)
        # 인과성 절단검사: 앞부분만으로 다시 만들어도 겹치는 구간이 같아야 한다
        cut = int(win["VAL"][1])
        F2 = build(d.iloc[:cut].reset_index(drop=True))
        A = F.iloc[:cut].drop(columns=["timestamp"]).to_numpy(float); B = F2.drop(columns=["timestamp"]).to_numpy(float)
        mk = np.isfinite(A) & np.isfinite(B)
        assert np.allclose(A[mk], B[mk], rtol=1e-9, atol=1e-9), "재생성 피쳐가 미래 행에 의존한다"
        print(f"  저장 {OUTP} · 절단검사 통과", flush=True)
    cs = [c for c in F.columns if c != "timestamp"]
    z_ = np.load(P.OUT / "direction_labels.npz", allow_pickle=True)
    lab = {k: (z_[f"{k}_idx"], z_[f"{k}_y"]) for k in z_["names"]}
    ti, ty = lab["TRAIN"]; EV = ("VAL", "OOS", "TEST")
    for w in ("TRAIN",) + EV:
        print(f"  {w} 커버리지 {100*np.isfinite(F[cs[0]].to_numpy(float)[lab[w][0]]).mean():.0f}%")
    nx = G.fit_normalizer(F, cs, *win["TRAIN"]); SF = G.build_state(F, nx)
    from sklearn.ensemble import HistGradientBoostingRegressor
    def twin(add):
        Xtr = S43[ti] if add is None else np.hstack([S43[ti], add[ti]])
        mm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                           min_samples_leaf=200, l2_regularization=1.0,
                                           random_state=P.SEEDS[0]).fit(Xtr, ty)
        return {w: float(spearmanr(mm.predict(S43[lab[w][0]] if add is None
                                              else np.hstack([S43[lab[w][0]], add[lab[w][0]]])),
                                   lab[w][1]).statistic) for w in EV}
    b = twin(None); t = twin(SF)
    inc = {w: t[w] - b[w] for w in EV}
    print(f"\n기준 43열      : " + " · ".join(f"{w} {b[w]:+.4f}" for w in EV))
    print(f"+ 재생성 {len(cs)}열 : " + " · ".join(f"{w} {t[w]:+.4f}" for w in EV))
    print(f"증분           : " + " · ".join(f"{w} {inc[w]:+.4f}" for w in EV))
    ok = inc["OOS"] >= 0.01 and inc["TEST"] >= 0.01
    print(f"⇒ 규칙(OOS·TEST 둘 다 ≥ +0.01): {'⭐통과' if ok else '탈락'}")
    print("\n열별 IC (TRAIN/VAL/OOS/TEST · 부호유지)")
    per = []
    for j, c in enumerate(cs):
        r = {"feature": c}
        for w in ("TRAIN",) + EV:
            x = SF[lab[w][0], j]
            r[w] = float(spearmanr(x, lab[w][1]).statistic) if np.std(x) > 0 else np.nan
        r["sign_keep"] = int(sum(np.sign(r[w]) == np.sign(r["TRAIN"]) for w in EV))
        per.append(r)
        print(f"   {c:<22}{r['TRAIN']:>+8.4f}{r['VAL']:>+8.4f}{r['OOS']:>+8.4f}{r['TEST']:>+8.4f}  {r['sign_keep']}/3")
    rep = {"base": b, "with": t, "inc": inc, "pass": bool(ok), "per_feature": per, "cols": cs}
    json.dump(rep, open(P.OUT / "report_tape_rebuild.json", "w"), ensure_ascii=False, indent=1, default=float)
    print(f"저장 {P.OUT / 'report_tape_rebuild.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
