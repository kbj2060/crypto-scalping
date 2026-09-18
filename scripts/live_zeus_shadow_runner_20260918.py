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
import argparse, ctypes, gc, json, os, sys, time, urllib.request
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
from build_binance_vision_panel_20260915 import day as _vision_day          # noqa: E402

ART = ROOT / "data/live/zeus_v4_shadow_20260918"
PANEL = ROOT / "data/binance_vision/panel"
FUNDING = ROOT / "tmp/omega461_longwindow_20260917/funding_2021_2026.csv"
BALNOBB = ROOT / "tmp/eth_regime_balnobb_20260910/model.joblib"
FAPI = "https://fapi.binance.com"
TAIL = ART / "rest_tail.parquet"   # 🔴러너 «전용» 캐시. 공유 패널 parquet 은 건드리지 않는다
                                   #   (패널 빌더는 범위를 통째로 덮어쓰므로 부분 갱신이 불가).
WEIGHT_CAP = 1500                  # 바이낸스 선물 한도 2400/분을 트레이딩 봇과 «공유»한다
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



# ─────────────────────────── 실시간 꼬리 (2026-09-18) ───────────────────────────
# 패널은 하루 한 번 갱신이라 그대로 쓰면 최대 ~24h 지연이다. 세 원천을 «같은 스키마»로 잇는다:
#   ①BV 패널(과거)  ②data.binance.vision 일별 파일(패널 끝 ~ 어제)  ③REST(오늘)
# ②는 패널 빌더의 `day()` 를 **그대로 임포트**해 파싱이 한 글자도 다르지 않다.
# 🔴REST 메트릭(OI/LSR)은 최근 **41.7h** 만 준다 -- ②가 그 앞을 메우지 못하면 거기서 멈춘다.

_PCOLS = ["timestamp", "high", "low", "close", "volume", "trades", "taker_buy_base",
          "sum_open_interest", "sum_toptrader_long_short_ratio", "count_long_short_ratio"]


def _get(url: str):
    """REST 한 번. ⭐가중치는 트레이딩 봇과 공유하므로 헤더를 읽고 한도 근처면 멈춘다."""
    with urllib.request.urlopen(url, timeout=30) as r:
        used = int(r.headers.get("x-mbx-used-weight-1m", 0) or 0)
        body = json.loads(r.read())
    if used > WEIGHT_CAP:
        raise RuntimeError(f"바이낸스 가중치 {used} > {WEIGHT_CAP} -- 이번 주기는 건너뛴다")
    return body


def _rest_bars(sym: str) -> pd.DataFrame:
    """REST 로 최근 완결 5분봉 + OI/LSR. 🔴**형성 중인 봉은 반드시 버린다**(2026-09-18 사고)."""
    k = _get(f"{FAPI}/fapi/v1/klines?symbol={sym}&interval=5m&limit=1000")
    d = pd.DataFrame(k, columns=["open_time", "open", "high", "low", "close", "volume",
                                 "close_time", "quote_volume", "trades", "taker_buy_base",
                                 "taker_buy_quote", "ignore"])
    now_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    d = d[d["close_time"].astype("int64") < now_ms]          # ⭐완결된 봉만
    d["timestamp"] = pd.to_datetime(d["open_time"].astype("int64"), unit="ms")
    for c in ("high", "low", "close", "volume", "trades", "taker_buy_base"):
        d[c] = d[c].astype(float)
    out = d[["timestamp", "high", "low", "close", "volume", "trades", "taker_buy_base"]].copy()
    for col, ep, fld in (("sum_open_interest", "openInterestHist", "sumOpenInterest"),
                         ("sum_toptrader_long_short_ratio", "topLongShortPositionRatio", "longShortRatio"),
                         ("count_long_short_ratio", "globalLongShortAccountRatio", "longShortRatio")):
        m = pd.DataFrame(_get(f"{FAPI}/futures/data/{ep}?symbol={sym}&period=5m&limit=500"))
        m["timestamp"] = pd.to_datetime(m["timestamp"].astype("int64"), unit="ms")
        m[col] = m[fld].astype(float)
        out = out.merge(m[["timestamp", col]], on="timestamp", how="left")
    return out[_PCOLS]


def live_tail(sym: str, panel_end: pd.Timestamp) -> pd.DataFrame:
    """패널 끝 이후를 «패널과 같은 스키마»로 만든다. 캐시에 누적해 다음 주기를 싸게 만든다."""
    cache = pd.read_parquet(TAIL) if TAIL.exists() else pd.DataFrame(columns=_PCOLS + ["sym"])
    cache = cache[cache.get("sym", pd.Series(dtype=str)) == sym] if len(cache) else cache
    parts = [cache[_PCOLS]] if len(cache) else []
    have_to = max([panel_end] + ([pd.Timestamp(cache.timestamp.max())] if len(cache) else []))
    # ②일별 파일 -- 어제까지. 하루치는 288봉이고 캐시에 남으므로 매번 받지 않는다.
    for d in pd.date_range((have_to + pd.Timedelta("5min")).normalize(),
                           (pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta("1D")).normalize(),
                           freq="D"):
        got = _vision_day(sym, d.strftime("%Y-%m-%d"))
        if got is not None:
            parts.append(got.reindex(columns=_PCOLS))
            log(f"  [{sym}] vision {d.date()} {len(got)}봉")
    parts.append(_rest_bars(sym))                                  # ③오늘
    out = pd.concat([x for x in parts if len(x)], ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    # 🔴같은 봉이 여러 원천에서 온다. `keep="last"` 로 고르면 **메트릭이 NaN 인 REST 행**이
    #   메트릭이 실제로 있는 vision 행을 이긴다(2026-09-18 실측: OI/LSR 43.2% 결측).
    #   열마다 «비어 있지 않은 첫 값»을 취해 합친다 -- OHLCV 는 원천 간 100% 일치 확인됨.
    out = (out[out.timestamp > panel_end].sort_values("timestamp")
           .groupby("timestamp", as_index=False).first().reset_index(drop=True))
    keep = out.copy(); keep["sym"] = sym
    other = pd.read_parquet(TAIL) if TAIL.exists() else pd.DataFrame(columns=_PCOLS + ["sym"])
    other = other[other["sym"] != sym] if len(other) else other
    TAIL.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([other, keep], ignore_index=True).to_parquet(TAIL, index=False)
    # 🔴격자에 구멍이 있으면 «조용히» 중앙값으로 채워지므로 여기서 터뜨린다.
    if len(out) > 1:
        gap = out.timestamp.diff().dropna()
        bad = gap[gap != pd.Timedelta("5min")]
        assert bad.empty, f"[{sym}] 5분 격자 불연속 {len(bad)}곳 (첫 구멍 {out.timestamp[bad.index[0]-1]})"
    # 🔴결측을 그냥 넘기면 프레임 빌더가 balnobb 중앙값으로 «조용히» 채워 딴 피쳐가 된다.
    #   러너가 계속 돌면 캐시가 메워주므로 자가회복한다 -- 그때까지는 거부한다.
    miss = out[_PCOLS[7:]].isna().mean()
    assert not (miss > 0.02).any(), (
        f"[{sym}] OI/LSR 결측 {miss.round(3).to_dict()} -- REST 메트릭은 41.7h 만 주고 "
        f"그 앞 구간의 vision 일별 파일이 아직 없다. 파일이 올라오면 자가회복한다.")
    return out


_OHLC = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume",
         "trades", "taker_buy_base", "taker_buy_quote"]


def _panel(sym: str, n: int, live: bool = False) -> pd.DataFrame:
    """BV 패널의 마지막 n봉(+ live 면 오늘까지). 파생 3열은 프레임 빌더와 **한 글자도 같다**."""
    d = pd.read_parquet(PANEL / f"{sym}USDT.parquet")
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True).dt.tz_localize(None)
    d = d.drop_duplicates("timestamp").sort_values("timestamp")
    if live:
        d = pd.concat([d, live_tail(f"{sym}USDT", d.timestamp.max())], ignore_index=True)
        d = d.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    d = d.tail(n)
    typ = (d.high + d.low + d.close) / 3.0
    d["open"] = d.close.shift(1).fillna(d.close)
    d["quote_volume"] = d.volume * typ
    d["taker_buy_quote"] = d.get("taker_buy_base", pd.Series(np.nan, index=d.index)) * typ
    return d.reset_index(drop=True)


def build_frame(n_bars: int, live: bool = False) -> pd.DataFrame:
    """연구 프레임(`build_omega461_longwindow_frame_realfunding_20260917.py`)과 같은 조립."""
    pe = _panel("ETH", n_bars, live)
    eth = pe[_OHLC].copy()
    eth = eth.merge(pe[["timestamp", "sum_open_interest", "sum_toptrader_long_short_ratio",
                        "count_long_short_ratio"]], on="timestamp", how="left")
    eth["sum_open_interest_value"] = eth.sum_open_interest * eth.close
    eth = eth.merge(_panel("BTC", n_bars, live)[["timestamp", "close", "volume", "quote_volume"]].rename(
        columns={"close": "close_btc", "volume": "volume_btc", "quote_volume": "quote_volume_btc"}),
        on="timestamp", how="left")
    fr = pd.read_csv(FUNDING); fr["timestamp"] = pd.to_datetime(fr["timestamp"])
    if live:                      # 🔴CSV 는 연구 아티팩트다 -- 덮어쓰지 않고 «메모리에서만» 잇는다
        try:
            fj = _get(f"{FAPI}/fapi/v1/fundingRate?symbol=ETHUSDT&limit=100")
            nf = pd.DataFrame({"timestamp": pd.to_datetime([x["fundingTime"] for x in fj], unit="ms"),
                               "last_funding_rate": [float(x["fundingRate"]) for x in fj]})
            fr = (pd.concat([fr, nf], ignore_index=True).drop_duplicates("timestamp", keep="last")
                  .sort_values("timestamp"))
        except Exception as e:
            log(f"  🔴펀딩 REST 실패({type(e).__name__}) -- CSV 마지막값이 그대로 전방 채움된다")
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


try:                                   # 🔴2026-09-18 서버 24GB OOM 의 지문: 논리적 leak 이
    _LIBC = ctypes.CDLL("libc.so.6")   # 아니라 glibc arena 미반환. 주기 루프에는 필수다.
except OSError:
    _LIBC = None


def _trim():
    gc.collect()
    if _LIBC is not None:
        _LIBC.malloc_trim(0)


def scores(F: pd.DataFrame, models, scaler, base_cols, device, i0: int = 0):
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
    # 🔴피쳐는 프레임 «전체»로 만들어야 롤링이 맞지만, 추론은 필요한 행만 하면 된다.
    #   주기 루프에서 2만봉을 매번 6모델로 다시 미는 건 순수 낭비다(행별 결과는 동일).
    x = tabm._standardize_apply(tabm._base_input(f, base_cols), scaler)[i0:]
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



def verify_rest(bars: int) -> int:
    """⭐라이브 원천이 패널과 «같은 봉»인지 증명한다. 가정하지 않는다.

    ①vision 일별 파일 vs 패널 — 패널이 이미 덮은 날로 대조(완전 일치여야 한다)
    ②REST klines vs 패널 — 겹치는 완결 봉으로 대조(완전 일치여야 한다)
    이 둘이 통과해야 «패널 + 라이브 꼬리»가 한 계열이라고 말할 수 있다.
    """
    sym = "ETHUSDT"
    pan = pd.read_parquet(PANEL / f"{sym}.parquet")
    pan["timestamp"] = pd.to_datetime(pan["timestamp"], utc=True).dt.tz_localize(None)
    d0 = (pan.timestamp.max() - pd.Timedelta("1D")).normalize()
    v = _vision_day(sym, d0.strftime("%Y-%m-%d"))
    assert v is not None, f"vision 파일 없음: {d0.date()}"
    v["timestamp"] = pd.to_datetime(v["timestamp"])
    m = pan.merge(v, on="timestamp", suffixes=("_p", "_v"))
    cols = ["high", "low", "close", "volume", "trades", "taker_buy_base", "sum_open_interest"]
    log(f"① vision {d0.date()} vs 패널 — 겹치는 봉 {len(m):,}")
    ok1 = True
    for c in cols:
        a = pd.to_numeric(m[c + "_p"], errors="coerce").to_numpy(float)
        b = pd.to_numeric(m[c + "_v"], errors="coerce").to_numpy(float)
        k = np.isfinite(a) & np.isfinite(b)
        eq = float(np.mean(np.isclose(a[k], b[k], rtol=1e-9))) * 100 if k.sum() else float("nan")
        ok1 &= eq > 99.99
        log(f"   {c:30s} 완전일치 {eq:6.2f}%")
    r = _rest_bars(sym)
    m2 = pan.merge(r, on="timestamp", suffixes=("_p", "_r"))
    log(f"\n② REST klines vs 패널 — 겹치는 완결 봉 {len(m2):,}")
    ok2 = len(m2) > 50
    for c in ["high", "low", "close", "volume", "trades", "taker_buy_base"]:
        a = pd.to_numeric(m2[c + "_p"], errors="coerce").to_numpy(float)
        b = pd.to_numeric(m2[c + "_r"], errors="coerce").to_numpy(float)
        k = np.isfinite(a) & np.isfinite(b)
        eq = float(np.mean(np.isclose(a[k], b[k], rtol=1e-9))) * 100 if k.sum() else float("nan")
        ok2 &= eq > 99.99
        log(f"   {c:30s} 완전일치 {eq:6.2f}%")
    log("\n⭐라이브 원천 검증 통과 -- 패널과 같은 계열이다." if (ok1 and ok2) else
        "🔴검증 실패 -- 라이브 꼬리가 패널과 다른 봉이다. 이어붙이면 안 된다.")
    return 0 if (ok1 and ok2) else 1



def run(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="새 봉만 처리하고 종료(cron 용)")
    ap.add_argument("--sleep", type=int, default=300)
    ap.add_argument("--parity", action="store_true", help="연구 프레임과 대조만 하고 종료")
    ap.add_argument("--verify-rest", action="store_true", help="라이브 원천이 패널과 같은 봉인지 증명")
    ap.add_argument("--live", action="store_true", help="패널 뒤에 vision 일별 + REST 오늘치를 잇는다")
    ap.add_argument("--bars", type=int, default=WARMUP_BARS)
    a = ap.parse_args(argv)
    if a.verify_rest:
        return verify_rest(a.bars)
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
        F = build_frame(a.bars, live=a.live)
        last = pd.Timestamp(st["last_ts"]) if st.get("last_ts") else None
        i0 = max(300, int((F.timestamp <= last).sum())) if last is not None else 300
        DA, SC = scores(F, models, scaler, base_cols, dev, i0)
        for i in range(i0, len(F)):
            ts = F.timestamp.iloc[i]
            if last is not None and ts <= last:
                continue
            da, score = int(DA[i - i0]), float(SC[i - i0])
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
        log(f"[{pd.Timestamp.utcnow():%Y-%m-%d %H:%M}Z] 마지막 봉 {st['last_ts']} · "
            f"버퍼 {len(st['buf'])} · 포지션 {'있음' if st['pos'] else '없음'}")
        _trim()
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
