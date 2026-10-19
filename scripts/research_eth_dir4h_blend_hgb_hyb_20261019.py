"""ETH 4h 방향: HGB(65피쳐) + hyb(GRU 원시 24h 창 + 65피쳐) 섞기 -- GPU 전 가장 싼 시험 (2026-10-19, 사용자 «HGB+GRU 결합»).

근거: research_eth_dir4h_reverify_20261019 B -- hyb − mlp +14.0bp/일 [+3.6,+25.4]·IC +.023 = 원시 창에 65피쳐 밖 정보가 있다.
  hyb 단독은 HGB 를 못 넘었다(−7.5 [−26.7,+8.8]). 두 예측이 다른 정보를 담으면 섞은 것이 HGB 단독을 넘어야 한다.
입력 = 저장된 매달 표본 밖 예측(재학습 없음): pred_base.npz(HGB 시드 3 평균) · pred_v2_hyb.npz(시드 5 평균 + 시드별 pred_s).
  hyb 는 y/σ 로 학습했다 → ×σ(sig = P_rv1d·√48) 로 bp 눈금 복원(리뷰 MEDIUM-1).

사전 등록(계산 전 고정):
  섞기 = 0.5·z(HGB) + 0.5·z(hyb×σ), z = 그 모델 예측의 직전 30일(닫힌 왼쪽) 평균·표준편차 -- 비중·창은 튜닝하지 않는다.
  매매 = 각 예측 자신의 직전 30일 20/80 분위 문턱(reverify B 와 같은 공통 규칙) · 위상 8 · 비용 4bp · 5일 블록 부트스트랩.
  통과 = ① 2025~ 섞기 − HGB 일 손익 차 95% CI 하한 > 0
         ② 2023-24 같은 차의 평균 > 0 (부호 일치)
         ③ hyb 시드 5개 각각으로 섞은 것의 2025~ 차가 5개 중 4개 이상 양수.
  셋 다 → 서버 GPU 로 hyb 제대로 키우기(논문 설정·σ 타깃·시드 보강) · 하나라도 실패 → 4h 방향 DL 축 종료.
  진단(판정 무관): IC(섞기·HGB·hyb×σ) · 비중 0.3/0.7 · hyb 원 눈금 섞기.

결과(2026-10-19, tmp/dir4h_blend_20261019/report.json) = 불통과 → 4h 방향 DL 축 종료(GPU 안 씀):
  2025~ 일 손익: HGB +12.6 [−4.5,+31.8] · hyb×σ +4.1 · 섞기 +7.9 [−8.2,+24.9]
  ① 섞기 − HGB 2025~ −4.7 [−15.0,+4.8] ✗ (2025 −0.9 · 2026 −10.0) · ② 2023-24 +4.8 [−4.4,+13.5] ✓ · ③ 시드 2/5 양수 ✗
  진단: 비중 .3/.7·원 눈금 섞기도 전부 2025~ 음수(−4.7~−5.8) · IC 2025~ HGB .063 > 섞기 .051 · 2023-24 는 hyb .032 > HGB .012
  → 2023-24 에서만 hyb·섞기가 앞선다 = 65피쳐·HGB 가 2025~ 로 골라진 몫(리뷰 HIGH-1)과 맞다. 최근 우선 규칙으로 판정은 2025~.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_eth_dir4h_reverify_20261019 as rv  # noqa: E402  (block_ci·phase_daily·roll30_q·summ·diff·ic_stats·pearson 재사용)

OF = rv.DL / "dir4h_otherfeats_20261018"
OUT = ROOT / "tmp/dir4h_blend_20261019"


def rz30(p: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """직전 30일(닫힌 왼쪽 = 지금 행 제외) 평균·표준편차로 표준화 -- 인과."""
    s = pd.Series(p, index=pd.DatetimeIndex(ts))
    r = s.rolling("30D", closed="left", min_periods=50)
    return ((s - r.mean()) / r.std()).to_numpy()


def blend(a: np.ndarray, b: np.ndarray, ts: np.ndarray, w: float = 0.5) -> np.ndarray:
    return w * rz30(a, ts) + (1 - w) * rz30(b, ts)


def pnl(p, ts, y, a, b) -> pd.Series:
    return rv.phase_daily(p, *rv.roll30_q(p, ts), ts, y, a, b)


def main() -> dict:
    Z = np.load(rv.DL / "dir4h_seq_v2_20261018/data.npz")
    ts, y, sig = Z["ts"].astype("datetime64[m]").astype("datetime64[ns]"), Z["y"].astype(float), Z["sig"].astype(float)
    hgb = np.load(OF / "pred_base.npz")["pred"].astype(float)
    H = np.load(OF / "pred_v2_hyb.npz")
    hyb, hyb_s = H["pred"].astype(float), H["pred_s"].astype(float)
    hs = hyb * sig
    arms = {"hgb": hgb, "hyb_sig": hs, "blend": blend(hgb, hs, ts),
            "diag_w0.3": blend(hgb, hs, ts, 0.3), "diag_w0.7": blend(hgb, hs, ts, 0.7), "diag_raw": blend(hgb, hyb, ts)}
    out = {"coverage": {k: [str(pd.Timestamp(ts[np.isfinite(v)].min())), str(pd.Timestamp(ts[np.isfinite(v)].max()))] for k, v in arms.items()},
           "pnl": {}, "diff_vs_hgb": {}, "ic": {}}
    D = {}
    for nm, p in arms.items():
        out["pnl"][nm] = {}
        for lab, a, b in rv.PERIODS:
            D[nm, lab] = pnl(p, ts, y, a, b); out["pnl"][nm][lab] = rv.summ(D[nm, lab])
        print(nm, json.dumps(out["pnl"][nm], ensure_ascii=False), flush=True)
    for nm in arms:
        if nm != "hgb":
            out["diff_vs_hgb"][nm] = {lab: rv.diff(D[nm, lab], D["hgb", lab]) for lab, _, _ in rv.PERIODS}
    a, b = rv.PERIODS[0][1:]
    out["seed_diff_2025~"] = [round(float((pnl(blend(hgb, hyb_s[i] * sig, ts), ts, y, a, b) - D["hgb", "2025~"]).mean()), 2) for i in range(len(hyb_s))]
    for lab, a, b in rv.PERIODS:
        m = (ts >= np.datetime64(a)) & (ts < np.datetime64(b)) & np.isfinite(y) & np.all([np.isfinite(arms[k]) for k in ("hgb", "hyb_sig", "blend")], 0)
        out["ic"][lab] = {k: round(float(rv.pearson(rv.ic_stats(arms[k][m], y[m], pd.to_datetime(ts[m]).floor("D")).sum(0))), 4)
                          for k in ("hgb", "hyb_sig", "blend")}
    d25, d23 = out["diff_vs_hgb"]["blend"]["2025~"], out["diff_vs_hgb"]["blend"]["2023-24"]
    out["verdict"] = {"c1_ci_lo_gt0": d25["ci"][0] > 0, "c2_2023_24_mean_gt0": d23["mean"] > 0,
                      "c3_seeds_pos": sum(x > 0 for x in out["seed_diff_2025~"]), "c3_ok": sum(x > 0 for x in out["seed_diff_2025~"]) >= 4}
    out["verdict"]["pass"] = out["verdict"]["c1_ci_lo_gt0"] and out["verdict"]["c2_2023_24_mean_gt0"] and out["verdict"]["c3_ok"]
    print(json.dumps({k: out[k] for k in ("diff_vs_hgb", "seed_diff_2025~", "ic", "verdict", "coverage")}, ensure_ascii=False, indent=1), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / "report.json", "w"), ensure_ascii=False, indent=1)
    return out


def selfcheck() -> None:
    ts = pd.date_range("2025-01-01", periods=40 * 288, freq="5min").to_numpy()
    p = np.random.default_rng(0).normal(5, 2, len(ts))
    z = rz30(p, ts)
    p2 = p.copy(); p2[-1] = 1e6                       # 마지막 값을 바꿔도 그 이전 z 는 그대로 = 인과
    z2 = rz30(p2, ts)
    assert np.allclose(z[:-1], z2[:-1], equal_nan=True) and z2[-1] > 100
    assert abs(np.nanmean(z[-288 * 5:])) < 0.2 and abs(np.nanstd(z[-288 * 5:]) - 1) < 0.1
    bl = blend(p, 2 * p + 3, ts)                        # 같은 신호의 선형 변환끼리 섞으면 z 그대로
    assert np.allclose(bl[~np.isnan(bl)], z[~np.isnan(z)])
    print("selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else main()
