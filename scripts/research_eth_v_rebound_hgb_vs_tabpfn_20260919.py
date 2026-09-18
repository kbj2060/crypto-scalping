#!/usr/bin/env python3
"""V자 급등락: **TabPFN -> HGB 교체 타당성 측정** (2026-09-19)

라이브 워커는 5분마다 동결 컨텍스트 18,000행에 TabPFN 을 **다시 fit** 한다(서버 로그 실측
6.37~8.14초 · RSS 1.45GB + swap 753MB · CUDA 점유). 극점 탐지기가 2026-09-16 에 같은 교체를
이미 성공시켰다(TabPFN 1.08GB/5.14s -> HGB 4.7MB/0.47s). 그게 V자에서도 되는지 잰다.

## 설계 -- 왜 이 조립인가
피쳐  라이브 스크립트의 `_build_features`/`_every_bar_rows` 를 **그대로 import** 한다.
      재구현하면 라이브와 어긋난다. 모집단도 라이브와 같은 **매 봉 x 양측면**이다.
      ⚠️모집단은 아래 `--pop` 참조. 09-12 재학습 스크립트는 9트리거 후보풀을 써서 "절대 AUC 를
      배포 0.6942 와 비교하지 말라"고 못박았는데, `--pop matched` 는 그 0.6942 를 낳은 모집단
      자체를 복원하므로 비교가 성립한다(실측 재현: TRAIN 184,207행 @14.64% vs 기록 182,969
      @14.63% · VAL 37,145행 vs 기록 "VAL 전체봉 37k").
라벨  `research_eth_v_rebound_close_anchor_retrain_20260912.py` 의 `build_labels`(extreme
      앵커 = 배포본)·`parity_gate`·`auc` 를 그대로 import 한다.
      🔴파리티 < 0.99 면 즉시 중단한다 -- 라벨이 다르면 그 위 숫자는 전부 무의미하다.
팔    H_ctx  : HGB + 동결 18,000행 컨텍스트  <- 라이브 TabPFN 과 **완전히 같은 TRAIN**
      H_full : HGB + 매 봉 TRAIN 전체        <- HGB 가 실제로 쓸 수 있는 TRAIN(1회 학습이라
                                               TabPFN 의 컨텍스트 크기 제약을 안 받는다)
      기준선 : 발동봉 레인지/ATR 단일피쳐 (2026-09-11 사용자 지시 -- 항상 같이 낸다)
창    저장소 표준 split. VAL 2025-09-01~12-31 · OOS 2026-01-01~03-31 · FWD 2026-04-01~

## TabPFN 팔은 왜 여기 없나
dev 에 CUDA 도 `tabpfn` 도 없다(CPU 작업은 dev · GPU 는 서버, 2026-09-17 사용자 지시).
그래서 이 스크립트는 채점할 행을 `score_rows_<pop>.parquet` 로 떨어뜨리고, **서버에서 같은
파일에** `--tabpfn-score` 로 TabPFN 을 돌린다. 두 팔이 **같은 행·같은 라벨**을 보므로 맞대결이
성립한다. 🔴CUDA 잡에 `ulimit -v` 를 걸면 cudaGetDeviceCount 가 OOM 으로 죽는다(가상주소 공간을
수십 GB 예약한다) -- 대신 청크를 1024 로 묶고 OMP_NUM_THREADS=2 로 돌린다.

    dev    : python scripts/research_eth_v_rebound_hgb_vs_tabpfn_20260919.py --pop matched
    server : python <this> --tabpfn-score tmp/v_rebound_hgb_20260919/score_rows_matched.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
for _p in (CODE, CODE / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import live_eth_sweep_v_rebound_signal_20260829 as LIVE                      # noqa: E402
import research_eth_v_rebound_close_anchor_retrain_20260912 as R12           # noqa: E402

# 데이터 루트는 코드 루트와 다르다(워크트리에 data/ 가 없거나 비어 있다). 09-12 스크립트가
# 이미 정한 dev 루트를 쓰되, 없으면 이 계정의 홈으로 떨어진다 -- `--tabpfn-score` 는 서버에서
# 돌기 때문이다(거기선 /home/kbj20 가 없다).
ROOT = R12.ROOT if (R12.ROOT / "data").is_dir() else Path.home() / "crypto-scalping"
FROZEN = ROOT / ("data/labels/eth_5m_v_rebound_every_bar_20260901/"
                 "tabpfn_train_context_frozen_every_bar_20260901.csv")
KL = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
OUT = CODE / "tmp/v_rebound_hgb_20260919"
KL_FROM = "2023-06-01"     # 동결 컨텍스트 시작(2024-01-05) - 864봉 워밍업에 넉넉한 여유
LIVE_ROWS = (LIVE.HISTORY_BARS + 1 + LIVE.BADGE_HORIZON_BARS) * 2   # 라이브 1사이클 채점 행수
EVAL_CAP = 15000           # 창당 TabPFN 채점 상한 -- 2026-09-01 원본과 같은 값(실거래 기기 보호)

# ── 모집단 ────────────────────────────────────────────────────────────────────
# 2026-09-01 의 0.6942 는 **3상태 라벨**에서 나왔다: v_rebound / chop / **ambiguous(버림)**.
# 그 빌더(research_eth_v_rebound_every_bar_scoring_feasibility_20260901.py, 09-10 ponytail-audit
# 로 삭제됨 -- `git show f2eb2377^:...` 로 복원해 확인)의 마지막 두 줄이 근거다:
#     label = where(status=="v_rebound", 1, where(status=="chop", 0, NaN));  labeled = label.notna()
# R12 의 이진 라벨은 ambiguous 를 **0 으로 흡수**한다. 그래서 같은 창에서 행수가 1.8배가 되고
# 라벨률이 절반이 된다(양성/일은 거의 같다). 두 모집단을 다 낸다:
#   matched : 3상태, ambiguous 제외, START 2024-01-01  <- 0.6942 와 **비교 가능한** 쪽
#   live    : 이진, 모든 봉 x 양측면                    <- 라이브가 **실제로 채점하는** 쪽
CHOP_MULT = 1.0            # 복원한 label_variant() 의 상수. ATR_MULT/T_SUSTAIN 은 R12 와 공유.
MATCHED_START = "2024-01-01"


def log(m):
    print(m, flush=True)


def build_every_bar(kl: pd.DataFrame) -> pd.DataFrame:
    """라이브와 같은 매 봉 x 양측면 피쳐 프레임. 라이브 함수 두 개를 그대로 쓴다."""
    frame = LIVE._build_features(kl)
    # ponytail: `sig` 는 `_every_bar_rows` 안에서 **표시용 `triggers` 문자열에만** 쓰인다
    # (모델 피쳐 23개 중 어느 것도 여기 의존하지 않는다 -- 라이브 함수 본문 확인). 전부 False 로
    # 넣어 compute_signals 재계산(라이브 1사이클의 대부분)을 건너뛴다.
    # 업그레이드 경로: 트리거별 하위분석이 필요해지면 그때 compute_signals 를 실제로 태운다.
    sig = pd.DataFrame({f"{s}_{n}": False for s in ("bottom", "top") for n in LIVE.NAMED_TRIGGERS},
                       index=frame.index)
    cand = LIVE._every_bar_rows(frame, sig, len(frame))
    carry = [c for c in LIVE.FEATURES
             if c not in ("is_downside", "sweep_penetration_atr", "flow_aligned_delta_z")]
    cand = cand.merge(frame[["timestamp"] + carry], on="timestamp", how="left")
    return cand.dropna(subset=LIVE.FEATURES).reset_index(drop=True)


def load_klines_full() -> pd.DataFrame:
    """피쳐용 전체 열 5분봉. R12.load_klines() 는 라벨용이라 high/low/close 만 읽는다."""
    d = pd.read_csv(KL, usecols=["timestamp", "open", "high", "low", "close", "volume",
                                 "quote_volume", "trades", "taker_buy_base"],
                    parse_dates=["timestamp"])
    d = d.dropna().drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    if d["timestamp"].dt.tz is None:
        d["timestamp"] = d["timestamp"].dt.tz_localize("UTC")
    return d


def fast_mult(kl: pd.DataFrame) -> pd.DataFrame:
    """빠른 다리 배수. R12.build_labels 안에 있지만 반환되지 않아 **같은 창으로** 다시 낸다.
    같은 값인지는 아래 `is_v == R12 의 y` 자체점검이 매 실행마다 확인한다."""
    h, l, c = kl.high.to_numpy(), kl.low.to_numpy(), kl.close.to_numpy()
    atr, n = kl.atr14.to_numpy(), len(kl)

    def fwd(x, k, how):                                  # R12.build_labels 의 fwd 와 동일
        r = pd.Series(x[::-1]).rolling(k, min_periods=k)
        v = (r.max() if how == "max" else r.min()).to_numpy()[::-1]
        out = np.full(n, np.nan)
        out[:n - k] = v[1:n - k + 1]
        return out

    fmax_c, fmin_c = fwd(c, R12.FAST_BARS, "max"), fwd(c, R12.FAST_BARS, "min")
    return pd.DataFrame({"timestamp": kl.timestamp,
                         "mult_down": (fmax_c - l) / atr, "mult_up": (h - fmin_c) / atr})


def assemble(pop: str) -> tuple[pd.DataFrame, dict[str, np.ndarray], float]:
    cut = pd.Timestamp(KL_FROM, tz="UTC")
    # 라벨용(R12, atr14 포함)과 피쳐용(전체 열)을 **각각 전체 이력에서 만든 뒤** 자른다 --
    # 먼저 자르면 864봉 롤링 지표의 워밍업이 날아간다.
    lab_kl = R12.load_klines()
    kl = load_klines_full()
    kl = kl[kl.timestamp >= cut].reset_index(drop=True)
    log(f"[데이터] 5분봉 {len(kl):,} · {kl.timestamp.min():%Y-%m-%d} ~ {kl.timestamp.max():%Y-%m-%d}")

    lab = R12.build_labels(lab_kl)
    agree = R12.parity_gate(lab)
    if agree < 0.99:
        raise SystemExit(f"🔴 파리티 실패({agree:.4f}) -- 라벨 재현이 다르다. 중단한다.")
    log("✅ 파리티 통과 -- 아래 숫자는 배포 라벨과 같은 정의 위에 있다.")

    t = time.time()
    D = build_every_bar(kl)
    log(f"[모집단] 매 봉 x 양측면 {len(D):,}행 ({time.time() - t:.0f}s) · "
        f"{D.timestamp.min():%Y-%m-%d} ~ {D.timestamp.max():%Y-%m-%d}")

    D = D.merge(lab[["timestamp", "y_extreme_down", "y_extreme_up", "rng_atr"]],
                on="timestamp", how="left")
    D = D.merge(fast_mult(lab_kl), on="timestamp", how="left")
    dn = D.is_downside.to_numpy() == 1
    D["y"] = np.where(dn, D.y_extreme_down, D.y_extreme_up)
    D = D[np.isfinite(D.y.to_numpy())].reset_index(drop=True)

    # 3상태(2026-09-01 모집단): v_rebound=1 · chop=0 · ambiguous=버림.
    mult = np.where(D.is_downside.to_numpy() == 1, D.mult_down, D.mult_up)
    D["y3"] = np.where(D.y.to_numpy() == 1, 1.0, np.where(mult < CHOP_MULT, 0.0, np.nan))
    # 자체점검: 내가 다시 낸 배수의 v_rebound 경계가 R12 의 이진 라벨과 **정확히** 같아야 한다.
    is_v = (mult >= R12.ATR_MULT) & np.isfinite(mult)
    bad = int((is_v & (D.y.to_numpy() == 0)).sum())
    assert (D.y.to_numpy() == 1).sum() <= is_v.sum(), "fast_mult 재계산이 R12 와 어긋난다"
    log(f"[자체점검] fast_mult>=1.5 인데 y=0 인 행 {bad:,} (= giveback 조건으로 떨어진 것) · "
        f"y=1 인데 fast_mult<1.5 인 행 {int(((D.y.to_numpy() == 1) & ~is_v).sum()):,} (0 이어야 한다)")

    if pop == "matched":
        D = D[np.isfinite(D.y3.to_numpy())
              & (D.timestamp >= pd.Timestamp(MATCHED_START, tz="UTC")).to_numpy()]
        D = D.reset_index(drop=True).assign(y=lambda f: f.y3)
    log(f"[모집단:{pop}] {len(D):,}행")

    ts = D.timestamp
    win = {}
    for nm, a, b in R12.WINDOWS:
        m = np.ones(len(D), bool)
        if a:
            m &= (ts >= pd.Timestamp(a, tz="UTC")).to_numpy()
        if b:
            m &= (ts <= pd.Timestamp(b + " 23:59:59", tz="UTC")).to_numpy()
        win[nm] = m
    for nm, _, _ in R12.WINDOWS:
        log(f"  {nm:>6} {int(win[nm].sum()):>8,}행 · 라벨률 {D.y.to_numpy()[win[nm]].mean():>6.2%}")
    return D, win, agree


def frozen_context() -> pd.DataFrame:
    ctx = pd.read_csv(FROZEN)
    ctx["timestamp"] = pd.to_datetime(ctx["timestamp"], utc=True)
    return ctx


def score_table(name: str, P: np.ndarray, D: pd.DataFrame, win: dict, cap: np.ndarray) -> dict:
    """`cap` = TabPFN 팔이 채점하는 **바로 그 행들**(창당 EVAL_CAP). 맞대결은 이 줄로 한다.
    전체 창 숫자도 같이 남긴다 -- 부분추출이 운이었는지 보려면 둘을 나란히 봐야 한다."""
    y, base = D.y.to_numpy(int), D.rng_atr.to_numpy()
    out = {}
    log(f"  {'창':>6} {'n':>7} {'기저':>7} {'모델':>8} {'단일피쳐':>9} {'순증분':>8}   (전체창)")
    for nm, _, _ in R12.WINDOWS:
        if nm == "TRAIN" or win[nm].sum() < 200:
            continue
        m, full = win[nm] & cap, win[nm]
        am, ab = R12.auc(P[m], y[m]), R12.auc(base[m], y[m])
        fm = R12.auc(P[full], y[full])
        out[nm] = {"n": int(m.sum()), "base_rate": float(y[m].mean()),
                   "model_auc": am, "single_feature_auc": ab, "delta": am - ab,
                   "n_full": int(full.sum()), "model_auc_full_window": fm}
        log(f"  {nm:>6} {m.sum():>7,} {y[m].mean():>6.2%} {am:>8.4f} {ab:>9.4f} {am - ab:>+8.4f}"
            f"   {fm:.4f} (n={full.sum():,})")
    return out


def run_hgb(Xtr, ytr, Xall, seeds) -> tuple[np.ndarray, float, float]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    models, preds, fit_s = [], [], 0.0
    for sd in seeds:
        m = HistGradientBoostingClassifier(          # 09-12 재학습 스크립트와 **같은** 하이퍼파라미터
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15, random_state=sd)
        t = time.time()
        m.fit(Xtr, ytr)
        fit_s += time.time() - t
        models.append(m)
        preds.append(m.predict_proba(Xall)[:, 1])
    # 라이브 1사이클 = LIVE_ROWS 행을 **앙상블 전체**로 채점. 5회 평균.
    t = time.time()
    for _ in range(5):
        for m in models:
            m.predict_proba(Xall[:LIVE_ROWS])
    live_s = (time.time() - t) / 5
    return np.mean(preds, axis=0), fit_s, live_s, preds


def main_dev(pop: str) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    D, win, agree = assemble(pop)
    X, y = D[LIVE.FEATURES].to_numpy(float), D.y.to_numpy(int)

    ctx = frozen_context()
    Xc, yc = ctx[LIVE.FEATURES].to_numpy(float), ctx["label"].to_numpy(int)
    log(f"\n[동결 컨텍스트] {len(ctx):,}행 · 라벨률 {yc.mean():.2%} "
        f"({ctx.timestamp.min():%Y-%m-%d} ~ {ctx.timestamp.max():%Y-%m-%d})")

    # TabPFN 팔이 볼 행을 **먼저** 고른다 -- 두 팔이 같은 행을 봐야 맞대결이다.
    D = D.assign(window=np.select([win["VAL"], win["OOS"], win["FWD"]],
                                  ["VAL", "OOS", "FWD"], default=""))
    cap_idx = (D[D.window.to_numpy() != ""].groupby("window", group_keys=False)
               .apply(lambda g: g.sample(min(len(g), EVAL_CAP), random_state=20260919)).index)
    cap = np.zeros(len(D), bool)
    cap[np.asarray(cap_idx)] = True
    log(f"[채점 표본] 창당 최대 {EVAL_CAP:,} · 합계 {int(cap.sum()):,}행 (TabPFN 팔과 공유)")

    rep = {"parity": agree, "population": pop, "eval_cap": EVAL_CAP,
           "n": int(len(D)), "seeds": R12.SEEDS, "features": LIVE.FEATURES,
           "live_rows_per_cycle": LIVE_ROWS, "arms": {},
           # 🔴context_report.json 의 0.6942(TabPFN)/0.6953(GBM 프록시)는 **이 숫자와 비교하면
           # 안 된다**. 그쪽 VAL 은 15,000행이고 TRAIN 은 182,969행(라벨률 14.63%)인데 여기서
           # 재구성한 매 봉 x 양측면은 그 1.8배 행수에 라벨률이 절반이다(양성/일은 거의 같다 --
           # 즉 저쪽이 «거의 전부 음성인» 행 뭉치를 빼고 있었다). 맞대결은 오직 아래 TabPFN 팔과
           # 한다 -- **같은 parquet 의 같은 행**을 본다.
           "not_comparable_to": {"source": "data/labels/eth_5m_v_rebound_every_bar_20260901/"
                                           "context_report.json",
                                 "tabpfn_val_auc": 0.6942, "gbm_proxy_reference": 0.6953,
                                 "reason": "their VAL n=15,000 / TRAIN n=182,969 @14.63% vs "
                                           "this rebuild's every-bar population"}}

    for name, Xtr, ytr in (("H_ctx", Xc, yc), ("H_full", X[win["TRAIN"]], y[win["TRAIN"]])):
        log(f"\n══ {name} ══  TRAIN {len(ytr):,}행 · 라벨률 {ytr.mean():.2%}")
        P, fit_s, live_s, per_seed = run_hgb(Xtr, ytr, X, R12.SEEDS)
        rep["arms"][name] = {"train_n": int(len(ytr)), "train_rate": float(ytr.mean()),
                             "fit_sec_total": fit_s, "live_cycle_sec": live_s,
                             # 🔴시드폭 -- 앙상블 평균만 보고 팔을 비교하면 «분산 감소»를 실력으로
                             # 읽는다(2026-09-18 zeus v7 전례). TabPFN 은 단일 시드라 특히 중요하다.
                             "per_seed_auc": {nm: [R12.auc(ps[win[nm] & cap],
                                                           D.y.to_numpy(int)[win[nm] & cap])
                                                   for ps in per_seed]
                                              for nm in ("VAL", "OOS", "FWD")},
                             "windows": score_table(name, P, D, win, cap)}
        for nm in ("VAL", "OOS", "FWD"):
            a = rep["arms"][name]["per_seed_auc"][nm]
            log(f"  시드폭 {nm}: {min(a):.4f} ~ {max(a):.4f} (앙상블 "
                f"{rep['arms'][name]['windows'][nm]['model_auc']:.4f})")
        log(f"  학습 {fit_s:.1f}s(시드 {len(R12.SEEDS)}개 합) · "
            f"라이브 1사이클 예측 {live_s * 1000:.1f}ms ({LIVE_ROWS}행)")

    # is_downside 는 FEATURES 안에 이미 있다 -- 여기서 다시 넣으면 parquet 이 중복 열로 죽는다.
    keep = ["timestamp", "y", "rng_atr", "window"] + LIVE.FEATURES
    dump = D.loc[cap, keep].sort_values("timestamp").reset_index(drop=True)
    dump.to_parquet(OUT / f"score_rows_{pop}.parquet", index=False)
    (OUT / f"report_hgb_{pop}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    log(f"\n산출물 {OUT / f'report_hgb_{pop}.json'} · {OUT / f'score_rows_{pop}.parquet'} ({len(dump):,}행)")
    log("다음: 서버에서 같은 parquet 에 --tabpfn-score 를 돌려 맞대결 숫자를 붙인다.")
    return 0


def main_tabpfn(rows_path: Path) -> int:
    """서버 전용. 동결 컨텍스트에 TabPFN 을 fit 하고 **dev 가 떨어뜨린 그 행**을 채점한다."""
    from tabpfn import TabPFNClassifier
    rows = pd.read_parquet(rows_path)
    ctx = frozen_context()
    log(f"[TabPFN] 컨텍스트 {len(ctx):,}행 · 채점 {len(rows):,}행")

    clf = TabPFNClassifier(device="cuda", random_state=20260829, ignore_pretraining_limits=True)
    t = time.time()
    clf.fit(ctx[LIVE.FEATURES], ctx["label"].to_numpy())
    fit_s = time.time() - t

    P = np.empty(len(rows))
    t = time.time()
    for i in range(0, len(rows), 1024):               # 작은 청크 -- 실거래 기기다(2026-09-17 BSOD 는
        # VRAM 고갈이었다). 라이브 워커가 한 사이클에 쓰는 122행의 8배 남짓으로 묶는다.
        sl = slice(i, i + 1024)
        P[sl] = clf.predict_proba(rows[LIVE.FEATURES].iloc[sl])[:, 1]
    pred_s = time.time() - t
    t = time.time()
    clf.predict_proba(rows[LIVE.FEATURES].iloc[:LIVE_ROWS])
    live_s = time.time() - t

    y, base = rows.y.to_numpy(int), rows.rng_atr.to_numpy()
    out = {"fit_sec": fit_s, "predict_sec_all": pred_s, "live_cycle_sec": live_s, "windows": {}}
    log(f"  fit {fit_s:.1f}s · 전체 채점 {pred_s:.1f}s · 라이브 1사이클 {live_s:.2f}s ({LIVE_ROWS}행)")
    log(f"  {'창':>6} {'n':>9} {'기저':>7} {'모델':>8} {'단일피쳐':>9} {'순증분':>8}")
    for nm in ("VAL", "OOS", "FWD"):
        m = (rows.window.to_numpy() == nm)
        if m.sum() < 200:
            continue
        am, ab = R12.auc(P[m], y[m]), R12.auc(base[m], y[m])
        out["windows"][nm] = {"n": int(m.sum()), "base_rate": float(y[m].mean()),
                              "model_auc": am, "single_feature_auc": ab, "delta": am - ab}
        log(f"  {nm:>6} {m.sum():>9,} {y[m].mean():>6.2%} {am:>8.4f} {ab:>9.4f} {am - ab:>+8.4f}")
    dst = rows_path.parent / "report_tabpfn.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    log(f"\n산출물 {dst}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", choices=("matched", "live"), default="matched",
                    help="matched=3상태(2026-09-01 과 비교 가능) · live=이진(라이브가 채점하는 모집단)")
    ap.add_argument("--tabpfn-score", type=Path, default=None,
                    help="서버 전용: dev 가 떨어뜨린 score_rows.parquet 을 TabPFN 으로 채점")
    a = ap.parse_args()
    raise SystemExit(main_tabpfn(a.tabpfn_score) if a.tabpfn_score else main_dev(a.pop))
