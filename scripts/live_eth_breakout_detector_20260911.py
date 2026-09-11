#!/usr/bin/env python3
"""ETH **추세 전환** 경보·탐지 라이브 계산 (2026-09-11) — 읽기 전용.

2026-09-11 압축 게이트 제거로 «횡보→» 전제가 빠졌다 — 전환의 77%는 압축을 거치지 않는다.

사용자 실계좌 손실 패턴에서 출발: 압축 구간 상단에서 숏 페이드 → 상방 돌파 → -535.96 USDT
(전체 손실의 95.7%). **방향은 예측하지 않는다** — 돌파가 온다/왔다만 말한다.
방향 축은 이 저장소에서 닫혔다(MFE-MAE rho 0.000) — 여기서 다시 열지 않는다.

경보(예측)  피쳐 3종을 **각자 독립 신호등**으로 켠다(합치지 않는다 — 지평이 달라 뜻이 다르다).
            압축 구간에서만 감시. 새 진입 억제용.
            ⚠️경보 쪽 lift 5.75/3.65/3.06x 는 **자명한 대리 타깃**(앞 24봉 실현변동성) 값이다 —
              atr_pct 단독으로 lift 7.01 이 나오는 라벨이었다. 경보는 모델로 교체 예정.
탐지(즉시)  피쳐 2종(거래대금·체결속도) z288 q90 **AND**. 보유 중 반대 방향 돌파면 즉시 청산용.
            2026-09-11 **압축 게이트 제거**(감시창 + 압축봉 전용 임계 둘 다). 아래 §정정 참조.

성적(ETH 2026-01~09, 압축 봉 29,778 · 전환 431건 · 앞만 보는 타깃 · 순환이동 귀무):
  경보  3봉지속 z2016 q99 → 앞 2시간 상위5% (전역분위 연구 7.71x / **인과 임계 5.75x**)
        전반 8.56x / 후반 8.58x · 63셀 전부 2x 이상 · 셀 순위 상관 +0.708
  경보  인과 임계로 재현: 3봉지속 5.75x(하루 1.6회) · 체결속도 3.65x · 거래대금 3.06x
        경보 ON 시 앞 2시간 상위5% 확률 28.75%(기저 5%) · 실제 선행 중앙 15분
  탐지  2종 AND q90. volexp 포함 3종이면 느려진다(12봉 롤링이라 구조적 지연) — 뺀 채로 둔다.

🔴**수치 정정 + 게이트 제거 (2026-09-11)**
  옛 표기 «포착률 98.4% · 지연 5분 · 진행률 6.08%» 는 **volexp 1.80 교차**의 포착률이었다.
  사용자가 실제로 손실을 보는 «큰 이동»(앞 24봉 최대이탈 상위 5%) 기준으로 다시 재면:
      압축 게이트 있음(옛 배포)   사건 포착 47.9 / 50.5 / 52.8%  · 발동 32.8회/일
      압축 게이트 없음(현재)      사건 포착 82.3 / 82.3 / 82.9%  · 발동 25.1회/일
  **더 많이 잡으면서 덜 울린다** — 게이트는 기저만 낮추고 실력도 깎고 있었다.
  (VAL/OOS/FWD = 2025-09~12 / 2026-01~03 / 2026-04~. 사건 751/572/1046건)

⚠️리드타임을 주장하지 않는다 — 앞선 측정의 "-25분"은 탐색 창 폭의 산물이었다.
⚠️입력은 공개 kline 의 n(체결 건수)·quote_volume 뿐이다. 호가는 기여 없음(166건 검정).
⚠️단일 자산·단일 연도. 전환 정의(압축<0.7 → 확장>=1.8)는 이 저장소 것이다.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import eth_breakout_features33_20260911 as F33            # noqa: E402  학습과 **같은** 빌더

KLINES = "https://fapi.binance.com/fapi/v1/klines"
# 예고 모델(사이드카). 디렉터리가 없으면 예고 없이 그대로 돈다 -- 되돌리기 = 디렉터리 삭제.
PREWARN_DIR = ROOT / "data" / "live" / "eth_breakout_prewarn_artifact"
PREWARN_Q = 0.90              # 확률의 후행 분위 -- 커버리지 10%
# 지속시간(2026-09-11 사용자 요청): 한 번 울리면 그 신호의 «수명» 동안 게이지를 채워 둔다.
# 값은 각 신호가 **주장하는 창**과 같게 맞춘다 -- 경보는 «앞으로 30분», 탐지는 판정창 15분.
SUSTAIN = {"prewarn": 6, "detect": 3}


def _sustain(fired: np.ndarray, i: int, bars: int) -> tuple[bool, int]:
    """봉 i 기준 (활성인가, 남은 봉수). 마지막 발동 이후 bars 봉 동안 활성으로 본다."""
    w = fired[max(i - bars + 1, 0):i + 1]
    if not w.any():
        return (False, 0)
    since = len(w) - 1 - int(np.flatnonzero(w)[-1])
    return (True, bars - since)
_MODELS: dict[str, Any] = {}


def _prewarn_models() -> tuple[list, dict] | tuple[None, None]:
    """아티팩트를 **한 번만** 읽는다. 없으면 (None, None) -- 호출부가 예고를 건너뛴다."""
    if "v" not in _MODELS:
        try:
            import joblib
            meta = json.loads((PREWARN_DIR / "meta.json").read_text(encoding="utf-8"))
            models = [joblib.load(PREWARN_DIR / f"hgb_{sd}.joblib") for sd in meta["seeds"]]
            _MODELS["v"] = (models, meta)
        except Exception:                                  # 신뢰경계: 아티팩트 부재/손상
            _MODELS["v"] = (None, None)
    return _MODELS["v"]
SYMBOL = "ETHUSDT"
FETCH_BARS = 4200            # z2016(7일) + 압축분위 창 + 여유
COMPRESS = 0.70              # volexp < 0.70 = 압축(횡보) — 이 구간에서만 감시
QWIN = 2016                  # 임계 분위를 재는 후행 창(인과)
# 탐지: 2종 AND · z288 · q90
# volexp 는 뺐다 — 전환의 **정의**(volexp>=1.8 교차)에 쓰인 양이라 기여가 정의상 보장된
# 것이지 정보가 아니고, 넣으면 진행률이 6.08% → 11.51% 로 느려진다(12봉 롤링이라 구조적 지연).
DETECT = [("거래대금", "qv", 288, 0.90), ("체결속도", "n", 288, 0.90)]
HIST_BARS = 48               # 화면 띠 길이(4시간) — 다른 특화감지기와 같은 눈금
_CACHE: dict[str, Any] = {}


def _fetch(limit: int = FETCH_BARS) -> pd.DataFrame:
    out, end = [], None
    while len(out) < limit:
        p = {"symbol": SYMBOL, "interval": "5m", "limit": 1500}
        if end is not None:
            p["endTime"] = end
        r = requests.get(KLINES, params=p, timeout=15)
        r.raise_for_status()
        k = r.json()
        if not k:
            break
        out = k + out
        end = k[0][0] - 1
        if len(k) < 1500:
            break
    d = pd.DataFrame(out, columns=["ot", "o", "h", "l", "c", "v", "ct", "qv", "n", "tbv", "tbq", "x"])
    d["timestamp"] = pd.to_datetime(d.ot, unit="ms")
    for cc in ("o", "h", "l", "c", "qv", "tbq"):   # tbq 누락 시 피쳐가 터진다
        d[cc] = d[cc].astype(float)
    d["n"] = d["n"].astype(int)
    return d.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def compute_signals(d: pd.DataFrame) -> dict[str, Any]:
    """kline 프레임 하나로 경보·탐지를 계산한다. 마지막 행이 현재 봉이다."""
    c = d["c"].to_numpy(float)
    lr = np.diff(np.log(c), prepend=np.log(c[0]))
    volexp = (pd.Series(lr).rolling(12).std() / pd.Series(lr).rolling(288).std()).to_numpy()
    comp = volexp < COMPRESS

    def _z(col: str, w: int) -> np.ndarray:
        s = pd.to_numeric(d[col], errors="coerce")
        return ((s - s.rolling(w).mean()) / s.rolling(w).std()).to_numpy()

    def _thr_all_at(x: np.ndarray, q: float, j: int) -> float:
        """**탐지** 임계 — 압축 봉이 아니라 **전 봉**에서 후행 창 분위를 낸다.

        2026-09-11 압축 게이트 제거와 함께 바뀌었다. 게이트가 있던 판은 임계도 압축 봉만
        모아 쟀는데, 둘을 같이 풀어야 실측 우위가 나온다(사건 포착 50.5%→82.3%,
        발동 32.8→25.1회/일 — 더 많이 잡으면서 **덜** 울린다).
        """
        v = x[:j + 1][np.isfinite(x[:j + 1])][-QWIN:]
        return float(np.nanquantile(v, q)) if len(v) >= 200 else np.inf

    def _thr_all(x: np.ndarray, q: float) -> float:
        return _thr_all_at(x, q, len(d) - 1)

    i = len(d) - 1
    out: dict[str, Any] = {"timestamp": str(d["timestamp"].iloc[i]), "close": float(c[i]),
                           "volexp": float(volexp[i]) if np.isfinite(volexp[i]) else None,
                           "compressed": bool(comp[i])}
    # ── 예고: «앞으로 30분 이내에 탐지기가 발동하나» (HGB 5시드 동결 앙상블)
    #    2026-09-11 옛 경보 신호등 3종을 교체했다 -- 그 lift 5.75x 는 «앞 24봉 실현변동성»
    #    이라는 자명한 대리 타깃 값이었고(atr_pct 단독 7.01), 전환 기준으로는 무작위였다.
    models, meta = _prewarn_models()
    prewarn: dict[str, Any] = {"available": False}
    pw_hist = np.zeros(len(d), bool)
    if models:
        feat = F33.build_features(d.rename(columns={
            "o": "open", "h": "high", "l": "low", "c": "close",
            "qv": "quote_volume", "n": "trades", "tbq": "taker_buy_quote"}))
        Xf = feat[meta["features"]].to_numpy(np.float32)
        good = np.isfinite(Xf).all(axis=1)
        sc = np.full(len(d), np.nan)
        if good.any():
            sc[good] = np.mean([m.predict_proba(Xf[good])[:, 1] for m in models], axis=0)
        # 임계도 **후행 분위**다(규칙과 같은 인과 규약). 지금 임계로 과거를 칠하지 않는다.
        pth = pd.Series(sc).rolling(QWIN, min_periods=200).quantile(PREWARN_Q).shift(1).to_numpy()
        pw_hist = np.isfinite(sc) & np.isfinite(pth) & (sc >= pth)
        p_act, p_left = _sustain(pw_hist, i, SUSTAIN["prewarn"])
        prewarn = {"active": p_act, "sustain_left_min": p_left * 5,
                   "sustain_min": SUSTAIN["prewarn"] * 5,
                   "available": bool(np.isfinite(sc[i])),
                   "proba": None if not np.isfinite(sc[i]) else round(float(sc[i]), 4),
                   "threshold": None if not np.isfinite(pth[i]) else round(float(pth[i]), 4),
                   "on": bool(pw_hist[i]), "horizon": "앞으로 30분 이내",
                   "base_rate": meta.get("base_rate_train"),
                   "precision": meta.get("oos_precision_cov10_worst_seed")}
    dets = []
    for label, col, w, q in DETECT:
        x = _z(col, w)
        thr = _thr_all(x, q)
        on = bool(np.isfinite(x[i]) and x[i] >= thr)
        dets.append({"name": label, "on": on,
                     "z": None if not np.isfinite(x[i]) else round(float(x[i]), 3),
                     "threshold": None if not np.isfinite(thr) else round(thr, 3)})
    out["prewarn"] = prewarn
    # 지속 판정은 최근 몇 봉만 필요하다 -- 전 구간에 _thr_all_at 를 돌리면 O(n x 창) 이라 느리다
    _zc = {(col, w): _z(col, w) for _lb, col, w, _q in DETECT}
    det_fire = np.zeros(len(d), bool)
    for j in range(max(i - max(SUSTAIN.values()), 0), i + 1):
        det_fire[j] = all(bool(np.isfinite(_zc[(col, w)][j])
                               and _zc[(col, w)][j] >= _thr_all_at(_zc[(col, w)], q, j))
                          for _lb, col, w, q in DETECT)
    d_act, d_left = _sustain(det_fire, i, SUSTAIN["detect"])
    out["detect"] = {"signals": dets, "count": sum(x["on"] for x in dets),
                     "on": all(x["on"] for x in dets),      # AND
                     "active": d_act, "sustain_left_min": d_left * 5,
                     "sustain_min": SUSTAIN["detect"] * 5}
    out["state"] = ("돌파 진행" if out["detect"]["on"] else
                    ("돌파 예고" if prewarn.get("on") else "미발동"))

    # ── 화면 계약 (규약 §1~3). 톤은 4색 안에서만 쓴다.
    #    **카드 2장**으로 나눴다(2026-09-11 사용자 결정) — 각 카드가 축 하나씩 갖는다.
    #      경보기: warn / neutral  (예고 확률)
    #      탐지기: bad  / neutral  (발동 여부)
    #    그래서 띠도 둘이다. 한 띠에 warn·bad 를 섞으면 어느 카드의 색인지 못 읽는다.
    detect_v = [_z(col, w) for _, col, w, _q in DETECT]
    hist, pw, times = [], [], []
    for j in range(max(i - HIST_BARS + 1, 0), i + 1):
        det_j = all(bool(np.isfinite(x[j]) and x[j] >= _thr_all_at(x, dd[3], j))
                    for x, dd in zip(detect_v, DETECT))
        hist.append("bad" if _sustain(det_fire, j, SUSTAIN["detect"])[0] else "neutral")
        pw.append("warn" if _sustain(pw_hist, j, SUSTAIN["prewarn"])[0] else "neutral")
        times.append(str(pd.Timestamp(d["timestamp"].iloc[j]).tz_localize("UTC").isoformat()))
    out["tone"] = hist[-1]
    out["subText"] = "돌파 발동" if out["detect"]["active"] else "미발동"
    prewarn["tone"] = pw[-1]
    prewarn["subText"] = ("돌파 예고" if (prewarn.get("on") or prewarn.get("active"))
                          else ("미발동" if prewarn.get("available") else "웜업"))
    prewarn["history"], prewarn["times"] = pw, times
    out["history"], out["times"] = hist, times
    return out


def get_signals(ttl: int = 60) -> dict[str, Any]:
    now = time.time()
    if _CACHE.get("t", 0) + ttl > now and "v" in _CACHE:
        return _CACHE["v"]
    try:
        v = compute_signals(_fetch())
        v["ok"] = True
    except Exception as e:                                   # 신뢰경계: 외부 API
        v = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    _CACHE.update(t=now, v=v)
    return v


def _mk(n: int = 3400, quiet: int = 120, seed: int = 0) -> pd.DataFrame:
    """뒤쪽 quiet 봉만 조용하게 만들어 volexp<0.7(압축)을 성립시킨다."""
    rng = np.random.default_rng(seed)
    sd = np.r_[np.full(n - quiet, 1.0), np.full(quiet, 0.15)]
    c = 2000 + np.cumsum(rng.normal(0, 1, n) * sd)
    cnt = rng.normal(5000, 250, n)
    return pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=n, freq="5min"),
                         "o": c, "h": c + 0.5, "l": c - 0.5, "c": c,
                         "qv": cnt * 1000.0, "n": cnt.astype(int),
                         "tbq": cnt * 500.0})     # 테이커 매수 대금 -- 피쳐 빌더가 요구한다


def _self_check() -> None:
    d = _mk()
    base = compute_signals(d)
    assert base["compressed"] is True, base                     # 뒤쪽이 조용하므로 압축
    assert base["detect"]["on"] is False, base["detect"]
    assert "alert" not in base, sorted(base)                    # 옛 경보 3종은 제거됐다
    assert "prewarn" in base, sorted(base)

    # 체결 건수만 급등 — 거래대금은 그대로다. 탐지는 2종 **AND** 라 켜지면 안 된다.
    # (첫 판에서는 n·qv 를 같이 올려 한쪽만 올리는 경우를 한 번도 시험하지 못했다.)
    a = d.copy()
    a.loc[a.index[-3:], "n"] = 60000
    ra = compute_signals(a)
    assert ra["compressed"] is True, ra                         # 가격이 안 움직였으니 여전히 압축
    assert ra["detect"]["count"] == 1, ra["detect"]             # 한 쪽만 — AND 미성립
    assert ra["detect"]["on"] is False, ra["detect"]

    b = a.copy()                                                # 거래대금까지 + 가격 돌파
    b.loc[b.index[-3:], "qv"] = 6.0e7
    b.loc[b.index[-1], ["o", "h", "l", "c"]] = float(b["c"].iloc[-1]) + 14.0
    rb = compute_signals(b)
    assert rb["compressed"] is False, rb                        # 돌파 봉은 압축이 아니고
    assert rb["detect"]["on"] is True, rb["detect"]             # 2종 AND 가 켜진다
    assert rb["state"] == "돌파 진행", rb["state"]

    # 2026-09-11 압축 게이트 제거의 핵심 검사 — **압축이 한 번도 없던 구간**에서도 탐지가
    # 켜져야 한다. 옛 판은 `watch`(직전 1시간 내 압축)가 없으면 무조건 꺼졌다.
    e = _mk(quiet=0)
    e.loc[e.index[-3:], "n"] = 60000
    e.loc[e.index[-3:], "qv"] = 6.0e7
    re_ = compute_signals(e)
    assert re_["compressed"] is False, re_
    assert re_["detect"]["on"] is True, re_["detect"]
    assert "watch" not in re_, sorted(re_)

    # ⭐사이드카 계약: 아티팩트가 없어도 **탐지는 그대로 돌아야** 한다(되돌리기 = 디렉터리 삭제)
    global PREWARN_DIR
    keep = PREWARN_DIR
    PREWARN_DIR = ROOT / "data" / "live" / "__없는_디렉터리__"
    _MODELS.clear()
    try:
        rn = compute_signals(b)
        assert rn["detect"]["on"] is True, rn["detect"]          # 탐지는 그대로
        assert rn["prewarn"]["available"] is False, rn["prewarn"]  # 예고만 빠진다
        assert rn["prewarn"]["subText"] == "웜업", rn["prewarn"]
    finally:
        PREWARN_DIR = keep
        _MODELS.clear()

    # 화면 계약: 카드 2장이라 띠도 2개 · 각 띠는 자기 색만 쓴다
    # ⚠️base(평탄 합성)는 지속창 안에 우연히 발동이 들 수 있다 -- 고정값 대신 **내부 정합**을 본다
    assert rb["tone"] == "bad" and rb["subText"] == "돌파 발동", (rb["tone"], rb["subText"])
    for r in (base, rb):
        want = "bad" if r["detect"]["active"] else "neutral"
        assert r["tone"] == want, (r["tone"], r["detect"])
        assert r["subText"] == ("돌파 발동" if r["detect"]["active"] else "미발동"), r["subText"]
        assert len(r["history"]) == len(r["times"]) == HIST_BARS, len(r["history"])
        assert r["history"][-1] == r["tone"], (r["history"][-1], r["tone"])
        assert set(r["history"]) <= {"bad", "neutral"}, set(r["history"])   # 탐지 띠는 2색
        pwh = r["prewarn"].get("history")
        if pwh is not None:
            assert len(pwh) == HIST_BARS, len(pwh)
            assert set(pwh) <= {"warn", "neutral"}, set(pwh)                # 경보 띠는 2색
    print("self-check OK  (경보3종 제거 · AND 게이트 · 비압축 탐지 · 띠 2개 분리 · 색 분리)")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    else:
        import json
        print(json.dumps(get_signals(), ensure_ascii=False, indent=2))
