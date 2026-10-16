"""박스 유지 확률 (2026-10-16 사용자 «①부터 재검증하고 시안») -- 지금가 ± 0.5×(지난 W분 고저폭) 띠를 다음 W분 안 안 벗어날 확률.

근거 scripts/research_eth_range_hold_multiscale_20261016.py(1분봉 4.7년): 2022~24 학습 로지스틱, 2025·2026 따로 채점 —
AUC 30분 .77/.77 · 1h .77/.76 · 4h .78/.77, 보정 오차 10분위 최대 3~6.4pp(1h 2026 5.1 · 4h 2025 6.4), 하위·상위 10% 유지 3~5% → 61~71%.
기하(변동성 대비 띠 거리) 하나가 AUC .74 — 나머지(최근 1h/24h 변동성 비 · 시각 · 주말 · 창폭)가 +.02~.04.
🔴방향 정보가 아니다(어느 쪽으로 깨질지는 모른다) · 24h 창은 횡보 구간 보정 오차 8~10pp 라 뺐다(전체 상태로는 4h 와 비슷한 5~6pp).
계수 재현: python scripts/research_eth_range_hold_multiscale_20261016.py --fit-coef
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

COEF = json.loads((Path(__file__).with_name("box_hold_coef_20261016.json")).read_text("utf-8"))
WIN = {"30m": 30, "1h": 60, "4h": 240}
NEED = 1441 + 1          # σ24 = 직전 1440 개 1분 수익 → 1분봉 1441개 + 여유


def payload(k1m: pd.DataFrame) -> dict:
    """k1m: 닫힌 1분봉(오름차순) timestamp(봉 시작, UTC)·high·low·close. 마지막 봉 종가 시점의 확률."""
    if k1m is None or len(k1m) < NEED:
        return {"ok": False, "reason": f"1분봉 {0 if k1m is None else len(k1m)}개 < {NEED}"}
    h, l, c = (k1m[x].to_numpy(float) for x in ("high", "low", "close"))
    r = np.diff(np.log(c[-1441:]))
    s24, s1h = float(np.sqrt(np.mean(r ** 2))), float(np.sqrt(np.mean(r[-60:] ** 2)))
    t0 = pd.Timestamp(k1m["timestamp"].iloc[-1])
    t0 = t0.tz_localize("UTC") if t0.tzinfo is None else t0.tz_convert("UTC")
    hr = t0.hour + t0.minute / 60
    px = float(c[-1])
    out = {"ok": True, "asof_ms": int(t0.value // 10 ** 6) + 60_000, "px": px, "scales": {}}
    for name, W in WIN.items():
        rng = float(h[-W:].max() - l[-W:].min())
        if not (rng > 0 and s24 > 0):
            out["scales"][name] = {"ok": False}
            continue
        x = {"log_ds": np.log(0.5 * rng / px / (s24 * np.sqrt(W))), "log_vr": np.log(max(s1h, 1e-6) / s24),
             "log_rng": np.log(rng / px * 1e4), "hs": np.sin(2 * np.pi * hr / 24), "hc": np.cos(2 * np.pi * hr / 24),
             "wkd": float(t0.dayofweek >= 5)}
        cf = COEF[name]
        z = cf["intercept"] + sum(cf["coef"][k] * v for k, v in x.items())
        out["scales"][name] = {"ok": True, "p": round(float(1 / (1 + np.exp(-z))), 4), "base": round(cf["train_base"], 4),
                               "q20": round(cf["q20"], 4), "q80": round(cf["q80"], 4),   # 화면 «깨질 쪽»(< q20) · «머물 쪽»(> q80) 문턱 = 학습 예측 분위
                               "up": round(px + 0.5 * rng, 2), "dn": round(px - 0.5 * rng, 2), "half_pct": round(0.5 * rng / px * 100, 3),
                               "until_ms": out["asof_ms"] + W * 60_000}
    return out


if __name__ == "__main__":
    # 자체점검: 연구 스크립트 build() 의 결정 행 특징으로 낸 확률 = payload 확률(같은 1분봉) -- 서빙/연구 정의 패리티
    import importlib.util
    spec = importlib.util.spec_from_file_location("rh", Path(__file__).resolve().parents[1] / "scripts/research_eth_range_hold_multiscale_20261016.py")
    rh = importlib.util.module_from_spec(spec); spec.loader.exec_module(rh)
    rng_ = np.random.default_rng(3)
    n = 4000
    c = 2400 * np.exp(np.cumsum(rng_.normal(0, 6e-4, n)))
    hh, ll = c * (1 + rng_.uniform(0, 4e-4, n)), c * (1 - rng_.uniform(0, 4e-4, n))
    t = (pd.Timestamp("2026-10-17 00:00").value // 10 ** 6) + np.arange(n) * 60_000      # 토요일 → 주말 칸 확인
    D = rh.build(pd.DataFrame({"t": t, "h": hh, "l": ll, "c": c}))
    for name, W in WIN.items():
        row = D[D.scale == name].iloc[-1]                      # 연구는 미래 H분이 있어야 행이 생긴다 → 그 시점까지 자른 1분봉으로 비교
        j = int(np.flatnonzero(pd.to_datetime(t, unit="ms") == row.ts)[0])
        k1m = pd.DataFrame({"timestamp": pd.to_datetime(t[:j + 1], unit="ms", utc=True), "high": hh[:j + 1], "low": ll[:j + 1], "close": c[:j + 1]})
        sc = payload(k1m)["scales"][name]
        cf = COEF[name]
        z = cf["intercept"] + sum(cf["coef"][k] * row[k] for k in cf["coef"])
        assert abs(sc["p"] - 1 / (1 + np.exp(-z))) < 1e-4, (name, sc["p"], 1 / (1 + np.exp(-z)))
        assert sc["dn"] < c[j] < sc["up"]
    assert D.wkd.any() and not D.wkd.all()                     # 토·일·월에 걸친 표본 -- 주말 칸이 실제로 갈린다
    assert payload(pd.DataFrame({"timestamp": [], "high": [], "low": [], "close": []}))["ok"] is False
    print("selftest OK -- 30분·1h·4h 확률 = 연구 build() 특징으로 낸 확률(패리티) · 띠가 현재가를 감쌈 · 짧은 입력 거부")
