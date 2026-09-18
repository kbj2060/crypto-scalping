#!/usr/bin/env python3
"""Zeus Baseline v4 **섀도우 러너** — 주문을 내지 않는다. 원장만 쓴다. (2026-09-18)

사양: `docs/zeus/README.md §3` · 판정 기준: `docs/zeus/shadow_prereg_v4_20260918.md`
아티팩트: `data/live/zeus_v4_shadow_20260918/{model.pt,meta.json}` (파리티 최대편차 0.000e+00)

## 🔴피쳐 원천 — 봇 스냅샷은 쓸 수 없다 (2026-09-18 실측)
처음엔 봇이 봉마다 쓰는 `decision_feature_snapshot.jsonl`(215열)을 쓰려 했다. **틀렸다.**
그 파일의 행은 **형성 중인 봉**이다 — 연구 프레임과 5,000봉을 대조하니
`volume` 이 연구의 **2.2%**, `trades` 3.0%, `taker_buy_base` 1.65% 인데 `open` 은 52.9%가
완전 일치하고 `close` 는 1.1%만 일치한다(봉이 열린 직후 몇 초 시점의 지문).
그대로 쓰면 방향 일치율 **67.3%** · 점수 상관 0.56 으로, **우리가 평가한 모델이 아니다.**
⇒ 원천은 **연구 프레임과 같은 경로**다: BV 패널(ETH·BTC 완결 5분봉 + OI/LSR) + 복구 펀딩
→ `FeatureEngineer.process` → `_with_raw_state12` → balnobb 아티팩트 중앙값 채움.
`--parity` 가 그 동등성을 «연구 parquet 과 직접 대조»해 증명한다(가정하지 않는다).

## 인과성 — 임계값은 «자기 자신»을 안 본다
게이트는 직전 1,000 후보 점수의 0.85 분위다. 현재 봉의 점수는 **임계값을 계산한 뒤에**
버퍼에 넣는다(연구 코드의 `shift(1)` 과 같다). 버퍼는 디스크에 영속돼 재시작을 넘긴다
(사전등록 R 항목).

## 배리어 — 연구와 같은 규약
intrabar 고·저가, **동시 접촉 시 SL 우선**, 시간청산 없음, 청산 봉까지 슬롯 점유
(재진입은 그 다음 봉부터). 진입가는 **결정 봉의 종가**.

🔴주문 없음. 🔴사양을 바꾸면 표본이 무효다(사전등록 §2).
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

CODE = Path(__file__).resolve().parents[1]
# 🔴데이터 루트는 «코드 루트»와 다를 수 있다 -- 워크트리에서 돌리면 data/ 가 없다.
# 서버 배포본에서는 둘이 같다(스크립트가 저장소 루트에 있다).
ROOT = Path(os.environ.get("ZEUS_ROOT") or Path.home() / "crypto-scalping")
for _p in (CODE, CODE / "scripts", ROOT, ROOT / "scripts"):
    sys.path.insert(0, str(_p))
import train_eval_omega1_2_tabm_3head_20260603 as tabm            # noqa: E402
from retrain_clean_regime_hmm_raw_state12_20260517 import _with_raw_state12  # noqa: E402
from features.engineering import FeatureEngineer                            # noqa: E402

ART = ROOT / "data/live/zeus_v4_shadow_20260918"
PANEL = ROOT / "data/binance_vision/panel"
FUNDING = ROOT / "tmp/omega461_longwindow_20260917/funding_2021_2026.csv"
BALNOBB = ROOT / "tmp/eth_regime_balnobb_20260910/model.joblib"
WARMUP_BARS = 10000         # ⭐프레임 빌더가 준 웜업(2021-12 한 달)과 같은 규모. 짧으면
                            #   긴 롤링(288·2016봉) 피쳐가 어긋난다 -- --parity 가 잡는다.
WARM = 200                  # 롤링 분위 최소 관측(그 전에는 확장창 분위)


def log(*a): print(*a, flush=True)


def load_art(device):
    b = torch.load(ART / "model.pt", map_location=device, weights_only=False)
    spec = json.loads((ART / "meta.json").read_text())["spec"]
    models = []
    for sd in b["state_dicts"]:
        m = tabm.ThreeHeadTabM(b["input_dim"], cfg=tabm.ThreeHeadConfig(**b["cfg"])).to(device)
        m.load_state_dict(sd); m.eval(); models.append(m)
    return models, b["scaler"], list(b["base_cols"]), spec


_OHLC = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume",
         "trades", "taker_buy_base", "taker_buy_quote"]


def _panel(sym: str, n: int) -> pd.DataFrame:
    """BV 패널의 마지막 n봉. 파생 3열 계산은 프레임 빌더와 **한 글자도 다르지 않다**."""
    d = pd.read_parquet(PANEL / f"{sym}USDT.parquet")
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True).dt.tz_localize(None)
    d = d.drop_duplicates("timestamp").sort_values("timestamp").tail(n)
    typ = (d.high + d.low + d.close) / 3.0
    d["open"] = d.close.shift(1).fillna(d.close)
    d["quote_volume"] = d.volume * typ
    d["taker_buy_quote"] = d.get("taker_buy_base", pd.Series(np.nan, index=d.index)) * typ
    return d.reset_index(drop=True)


def build_frame(n_bars: int) -> pd.DataFrame:
    """연구 프레임(`build_omega461_longwindow_frame_realfunding_20260917.py`)과 같은 조립."""
    pe = _panel("ETH", n_bars)
    eth = pe[_OHLC].copy()
    eth = eth.merge(pe[["timestamp", "sum_open_interest", "sum_toptrader_long_short_ratio",
                        "count_long_short_ratio"]], on="timestamp", how="left")
    eth["sum_open_interest_value"] = eth.sum_open_interest * eth.close
    eth = eth.merge(_panel("BTC", n_bars)[["timestamp", "close", "volume", "quote_volume"]].rename(
        columns={"close": "close_btc", "volume": "volume_btc", "quote_volume": "quote_volume_btc"}),
        on="timestamp", how="left")
    fr = pd.read_csv(FUNDING); fr["timestamp"] = pd.to_datetime(fr["timestamp"])
    eth = pd.merge_asof(eth.sort_values("timestamp"),
                        fr[["timestamp", "last_funding_rate"]].sort_values("timestamp"),
                        on="timestamp", direction="backward")
    F = FeatureEngineer().process(eth.drop(columns=["close_btc", "volume_btc", "quote_volume_btc"]).copy(),
                                  eth[["timestamp", "close_btc", "volume_btc", "quote_volume_btc"]].copy())
    if "timestamp" not in F.columns:
        F["timestamp"] = eth["timestamp"].to_numpy()
    F = _with_raw_state12(F)
    import joblib
    med = joblib.load(BALNOBB)["feature_medians"]          # 빌더와 같은 채움 규약
    for c, v in med.items():
        if c in F.columns:
            F[c] = F[c].fillna(v)
    return F


def heads(models, x, device):
    """6시드 확률 평균. 연구 캐시와 같은 계약(softmax 후 k 평균, 그 뒤 시드 평균)."""
    D = Q = None
    with torch.no_grad():
        t = torch.from_numpy(x).to(device)
        for m in models:
            o = m(t)
            d = torch.softmax(o["direction"], -1).mean(1).cpu().numpy()
            q = torch.softmax(o["quality"], -1).mean(1).cpu().numpy()
            D = d if D is None else D + d
            Q = q if Q is None else Q + q
    return D / len(models), Q / len(models)


_COVERED = [False]


def scores(F: pd.DataFrame, models, scaler, base_cols, device):
    """프레임 전체의 (방향, 점수)를 **한 번에** 낸다.

    ⭐봉마다 전체 프레임을 다시 변환하면 O(n²)이고, 배치로 내도 **행별 결과는 동일**하다 --
    피쳐는 롤링(행 i 는 i 이하만 본다)이고 모델은 행 단위다. 연구 채점기와 같은 계산이다.
    """
    f = _with_raw_state12(F)
    if not _COVERED[0]:
        # 🔴`_base_input` 은 없는 열을 reindex 로 NaN->0 으로 «조용히» 채운다. 원천 스키마가
        #   바뀌어 열이 빠지면 모델이 0을 먹는데 에러가 안 난다 -- 여기서 크게 터뜨린다.
        miss = [c for c in base_cols if c not in f.columns]
        assert not miss, f"프레임에 입력 열 {len(miss)}개가 없다(조용한 0-채움 방지): {miss[:8]}"
        _COVERED[0] = True
    x = tabm._standardize_apply(tabm._base_input(f, base_cols), scaler)
    D, _Q = heads(models, x, device)
    da = D.argmax(1)
    return da, D[np.arange(len(D)), da] - D[:, 0]


def threshold(buf: list[float], q: float, window: int) -> float:
    """직전 후보들만 본다 -- 현재 점수는 «호출 뒤에» 넣는다(인과)."""
    if len(buf) < 50:
        return float("inf")
    v = buf[-window:] if len(buf) >= WARM else buf
    return float(np.quantile(v, q))


def step(st: dict, row: dict, da: int, score: float, spec: dict, thr: float) -> list[dict]:
    """한 봉 처리. 반환: 원장에 쓸 사건들. 배리어는 intrabar · SL 우선."""
    ev, ts = [], str(row["timestamp"])
    hi, lo, cl = float(row["high"]), float(row["low"]), float(row["close"])
    pos = st.get("pos")
    if pos:
        e, sgn = pos["entry"], pos["side"]
        tp = e * (1 + sgn * spec["tp"]); sl = e * (1 - sgn * spec["sl"])
        hit_sl = (lo <= sl) if sgn > 0 else (hi >= sl)
        hit_tp = (hi >= tp) if sgn > 0 else (lo <= tp)
        if hit_sl or hit_tp:                       # 🔴동시 접촉이면 SL 우선(연구와 동일)
            px, why = (sl, "SL") if hit_sl else (tp, "TP")
            ev.append({"t": ts, "ev": "exit", "why": why, "px": px, "side": sgn,
                       "entry": e, "bars": int(pos["bars"]) + 1,
                       "bp": float(sgn * (px - e) / e * 1e4)})
            st["pos"] = None
            return ev                              # 청산 봉은 점유 -- 재진입은 다음 봉부터
        pos["bars"] = int(pos["bars"]) + 1
        return ev
    if da != 0:
        fired = bool(np.isfinite(thr) and score >= thr)
        ev.append({"t": ts, "ev": "cand", "da": da, "score": score, "thr": thr, "fired": fired})
        if fired:
            st["pos"] = {"entry": cl, "side": 1 if da == 1 else -1, "bars": 0, "t": ts}
            ev.append({"t": ts, "ev": "entry", "px": cl, "side": st["pos"]["side"]})
    return ev


def parity(models, scaler, base_cols, device, bars: int) -> int:
    """⭐«가정하지 않고 증명한다» — 이 러너가 만든 프레임을 **연구 parquet 과 직접 대조**한다.

    러너와 연구가 같은 열 이름을 쓴다는 건 아무것도 보장하지 않는다(2026-09-18: 봇 스냅샷은
    열 이름이 전부 같았지만 형성 중인 봉이라 방향 일치율이 67.3% 였다). 여기서는
    **표준화 단위 차이**(모델이 실제로 느끼는 차이)와 **최종 결정 일치율**을 잰다.
    """
    import train_eval_omega461_parent_zig075_longwindow_20260917 as E
    df, _ = E.load()
    F = build_frame(bars)
    m = df[["timestamp"] + base_cols].merge(F[["timestamp"] + base_cols], on="timestamp",
                                            suffixes=("_r", "_l"), how="inner")
    assert len(m) > 1000, f"겹치는 봉이 너무 적다: {len(m)}"
    sd = np.asarray(scaler["std"], float)
    worst = []
    for i, c in enumerate(base_cols):
        a = pd.to_numeric(m[c + "_r"], errors="coerce").to_numpy(float)
        b = pd.to_numeric(m[c + "_l"], errors="coerce").to_numpy(float)
        k = np.isfinite(a) & np.isfinite(b)
        worst.append((c, float((np.abs(a[k] - b[k]) / max(sd[i], 1e-9)).mean()) if k.sum() else np.nan))
    w = sorted(worst, key=lambda x: -(x[1] if np.isfinite(x[1]) else -1))
    log(f"겹치는 봉 {len(m):,} [{m.timestamp.min()} ~ {m.timestamp.max()}]")
    log("표준화 단위 평균차 상위 10열:")
    for c, v in w[:10]:
        log(f"  {c:34s} {v:.4f}")
    mean_z = float(np.nanmean([v for _c, v in worst]))
    # 결정 일치율 -- 마지막 3,000봉에서 두 프레임의 (방향, 점수)를 비교한다
    tail_ts = m.timestamp.tail(3000)
    xr = tabm._standardize_apply(tabm._base_input(
        df[df.timestamp.isin(tail_ts)][base_cols], base_cols), scaler)
    xl = tabm._standardize_apply(tabm._base_input(
        F[F.timestamp.isin(tail_ts)][base_cols], base_cols), scaler)
    Dr, _ = heads(models, xr, device); Dl, _ = heads(models, xl, device)
    agree = float((Dr.argmax(1) == Dl.argmax(1)).mean())
    sr = Dr[np.arange(len(Dr)), Dr.argmax(1)] - Dr[:, 0]
    sl = Dl[np.arange(len(Dl)), Dl.argmax(1)] - Dl[:, 0]
    corr = float(np.corrcoef(sr, sl)[0, 1])
    log(f"\n전체 평균 z차이 {mean_z:.4f} · 방향 일치율 {agree*100:.2f}% · 점수 상관 {corr:.4f}")
    ok = mean_z < 0.02 and agree > 0.999 and corr > 0.999
    log("⭐파리티 통과 -- 이 러너는 연구가 평가한 것과 같은 피쳐를 만든다." if ok else
        "🔴파리티 실패 -- 섀도우를 켜면 «평가하지 않은 모델»을 재게 된다.")
    return 0 if ok else 1


def run(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="새 봉만 처리하고 종료(cron 용)")
    ap.add_argument("--sleep", type=int, default=300)
    ap.add_argument("--parity", action="store_true", help="연구 프레임과 대조만 하고 종료")
    ap.add_argument("--bars", type=int, default=WARMUP_BARS)
    a = ap.parse_args(argv)
    dev = torch.device("cpu")
    models, scaler, base_cols, spec = load_art(dev)
    log(f"v4 섀도우 · 입력 {len(base_cols)}+{len(tabm.POS_COLS)}열 · 모델 {len(models)}개 · "
        f"q={spec['rollq_q']} 창 {spec['rollq_window']} · "
        f"TP{spec['tp']*100:g}%/SL{spec['sl']*100:g}% · 🔴주문 없음")
    if a.parity:
        return parity(models, scaler, base_cols, dev, a.bars)
    sp = ART / "state.json"
    st = json.loads(sp.read_text()) if sp.exists() else {"buf": [], "pos": None, "last_ts": None}
    led = ART / "ledger.jsonl"
    while True:
        F = build_frame(a.bars)
        DA, SC = scores(F, models, scaler, base_cols, dev)
        last = pd.Timestamp(st["last_ts"]) if st.get("last_ts") else None
        for i in range(300, len(F)):
            ts = F.timestamp.iloc[i]
            if last is not None and ts <= last:
                continue
            da, score = int(DA[i]), float(SC[i])
            thr = threshold(st["buf"], float(spec["rollq_q"]), int(spec["rollq_window"]))
            ev = step(st, {"timestamp": str(ts), "high": float(F.high.iloc[i]),
                           "low": float(F.low.iloc[i]), "close": float(F.close.iloc[i])},
                      da, score, spec, thr)
            if da != 0:                            # ⭐임계값을 «쓴 뒤에» 넣는다(인과)
                st["buf"].append(score)
                st["buf"] = st["buf"][-int(spec["rollq_window"]) * 2:]
            st["last_ts"] = str(ts)
            if ev:
                with led.open("a") as f:
                    for e in ev:
                        f.write(json.dumps(e, ensure_ascii=False) + "\n")
        sp.write_text(json.dumps(st))
        if a.once:
            break
        time.sleep(a.sleep)
    return 0


def _selfcheck():
    """프레임워크 없는 자체점검 -- 배리어 판정과 게이트 인과성만 본다."""
    spec = {"tp": 0.015, "sl": 0.007}
    # ① 한 봉에서 TP·SL 을 둘 다 치면 SL 이 이긴다
    st = {"pos": {"entry": 100.0, "side": 1, "bars": 0, "t": "x"}}
    ev = step(st, {"timestamp": "t", "high": 102.0, "low": 99.0, "close": 100.0}, 0, 0.0, spec, 1.0)
    assert ev and ev[0]["why"] == "SL", ev
    assert abs(ev[0]["bp"] - (-70.0)) < 1e-6, ev
    # ② 숏의 배리어는 방향이 뒤집힌다
    st = {"pos": {"entry": 100.0, "side": -1, "bars": 0, "t": "x"}}
    ev = step(st, {"timestamp": "t", "high": 100.2, "low": 98.0, "close": 99.0}, 0, 0.0, spec, 1.0)
    assert ev and ev[0]["why"] == "TP" and abs(ev[0]["bp"] - 150.0) < 1e-6, ev
    # ③ 임계값 미달이면 진입하지 않는다 · 초과면 한다
    st = {"pos": None}
    ev = step(st, {"timestamp": "t", "high": 1, "low": 1, "close": 1}, 1, 0.10, spec, 0.20)
    assert st["pos"] is None and ev[0]["fired"] is False, ev
    ev = step(st, {"timestamp": "t", "high": 1, "low": 1, "close": 1}, 2, 0.30, spec, 0.20)
    assert st["pos"]["side"] == -1 and ev[-1]["ev"] == "entry", ev
    # ④ 워밍업 전 임계값은 inf -- 아무것도 발화하지 않는다
    assert threshold([0.5] * 10, 0.85, 1000) == float("inf")
    assert abs(threshold([0.0, 1.0] * 100, 0.85, 1000) - 1.0) < 1e-9
    print("자체점검 4/4 통과")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck(); raise SystemExit(0)
    raise SystemExit(run())
