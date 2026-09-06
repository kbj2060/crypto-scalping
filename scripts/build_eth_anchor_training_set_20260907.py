#!/usr/bin/env python3
"""앵커 방향 예측 **학습셋 정제** (2026-09-07).

사용자: *"양쪽 다 거의 안 움직인 봉은 제거하고 라벨 데이터를 최대한 잘 깎아서 모델한테 잘 먹여야해."*
       *"등락폭은 ±1%로 해줘."*

## ⭐정제의 제1원칙 — 세 가지를 구분한다
(A) **라벨 미정의**  H봉 안에 어느 배리어도 안 닿음 -> 정답이 없다. 제거는 **필연**이지
    편향이 아니다. 단 라이브에서는 그 봉에도 진입하게 되므로 **그 몫을 반드시 별도 회계**한다
    (이 모델은 "결정된 건 중 방향"만 맞힌다).
(B) **인과 필터**  진입 시점 정보만으로 결정 -> **양쪽 창 모두** 적용 가능, 라이브에도 적용 가능.
(C) **라벨 품질 필터**  미래를 봐야 알 수 있음(시간 여백 등) -> **TRAIN 전용**. VAL/OOS 에 쓰면
    평가 모집단이 라이브와 달라져 성능이 부풀려진다. 이 구분을 흐리면 "쉬운 것만 남기고
    잘 맞혔다"가 된다.

## 라벨 (사용자 지정)
진입 `open[t+1]`. **고정 ±1%** 대칭 배리어. **1분봉**으로 먼저 닿는 쪽.
  y = 1 지속 먼저 · 0 페이드 먼저 · NaN 미결정/동시
H = 48(4h, 결정 70.9%) 를 기본, H = 96(8h, 결정 87.7%) 를 변형으로 둔다.
±1% = 100bp 로 왕복 비용 10bp 의 10배라 비용 문턱 문제가 없다
([[feedback_atr_normalized_label_low_volatility_bp_floor_check]] 회피).

## 그 밖의 정제
  퍼징   split 경계에서 라벨 창(최대 H*5분)이 다음 split 을 넘어가는 행 제거 (AFML purging)
  엠바고 TRAIN 끝 H봉은 VAL 로 새지 않도록 버린다
  가중치 겹치는 라벨 창의 **고유도**(1/동시겹침) — 중복 계수 방지 (AFML Ch.4)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SRC = ROOT / "tmp/eth_anchor_direction_labels_20260907/direction_labels.parquet"
OUT = ROOT / "tmp/eth_anchor_training_set_20260907"
PCT = 1.0                      # 사용자 지정 등락폭 ±1%
H_MAIN, H_ALT = 48, 96
ANCHOR = "any3/Wc3"
MARGIN_MIN_TRAIN = 5           # TRAIN 전용 라벨 품질 필터: 두 배리어 터치 시각 차 >= 5분
ADV_MAX = 0.5                  # 깨끗 판정: 배리어 닿기 **전** 반대 방향 최대이탈 < 0.5%
END_MIN = 1.0                  # 깨끗 판정: 판정창 **끝점**이 그 방향으로 1.0% 이상
EFF_MIN = 0.40                 # 깨끗 판정: **경로 효율** = 끝점진행 / 창내진폭 >= 0.40
#   ⭐2026-09-07 사용자 지적 3차: 닿기전 역행과 끝점만 보면 "닿은 뒤 크게 왕복한" 건이 남는다
#   (10-11 05:50 은 창내 진폭 14.8% 인데 순진행은 그 16%, 12-06 01:05 은 26%).
#   Kaufman efficiency ratio 형태 -- 1에 가까우면 곧게 갔고 0에 가까우면 왔다갔다 제자리.
#   지목 2건 0.16/0.26 vs 나머지 8건 0.54~0.93 으로 **단일 지표가 완전 분리**한다.
SPLIT_EDGES = {"TRAIN": ("2024-01-01", "2025-08-31 23:59:59"),
               "VAL": ("2025-09-01", "2025-12-31 23:59:59"),
               "OOS": ("2026-01-01", "2026-03-31 23:59:59"),
               "HOLDOUT_SPENT": ("2026-04-01", "2099-01-01")}


def path_stats(L: pd.DataFrame, pct: float, H: int, O, Hh, Lo, C):
    """닿기 전 반대이탈(%)과 판정창 끝점(%). 사용자 지적(2026-09-07): 현재 라벨은 '먼저 닿았나'만
    보고 '그 뒤 유지되나'를 안 본다 -- 1% 닿자마자 진입가로 되돌아온 건도 지속 승으로 부른다."""
    n = len(C); lim = H * 5.0
    i = L["bar_idx"].to_numpy()
    okw = (i + 1 + H) < n
    e = O[np.minimum(i + 1, n - 1)]
    w = np.arange(H)[None, :] + np.minimum(i + 1, n - 1 - H)[:, None]
    hi, lo, cl = Hh[w], Lo[w], C[w]
    sgn_cont = np.where(L["side"].to_numpy() == "bottom", -1.0, 1.0)     # 지속 방향 가격 부호
    endc = (cl[:, -1] - e) / e * 100 * sgn_cont                          # 지속 방향 기준 끝점 %
    def pre(t, cont):
        out = np.full(len(i), np.nan)
        j = np.where(np.isfinite(t) & (t < lim), np.nan_to_num(t // 5, nan=-1).astype(int), -1)
        for k in np.flatnonzero((j >= 0) & okw):
            a, b = hi[k, :j[k] + 1], lo[k, :j[k] + 1]
            out[k] = (a.max() - e[k]) / e[k] * 100 if cont[k] < 0 else (e[k] - b.min()) / e[k] * 100
        return out
    pre_c = pre(L[f"hit_cont_min_P{pct:g}"].to_numpy(float), sgn_cont)
    pre_f = pre(L[f"hit_fade_min_P{pct:g}"].to_numpy(float), -sgn_cont)
    rng_pct = (hi.max(axis=1) - lo.min(axis=1)) / e * 100          # 창내 전체 진폭 %
    return pd.DataFrame({"end_cont_pct": np.where(okw, endc, np.nan),
                         "range_pct": np.where(okw, rng_pct, np.nan),
                         "pre_adv_cont": pre_c, "pre_adv_fade": pre_f, "win_ok": okw})


def derive(L: pd.DataFrame, pct: float, H: int) -> pd.DataFrame:
    lim = H * 5.0
    f = L[f"hit_fade_min_P{pct:g}"].to_numpy(float)
    c = L[f"hit_cont_min_P{pct:g}"].to_numpy(float)
    fi = np.where(np.isfinite(f) & (f < lim), f, np.inf)
    ci = np.where(np.isfinite(c) & (c < lim), c, np.inf)
    y = np.where(np.isinf(fi) & np.isinf(ci), np.nan, (ci < fi).astype(float))
    y = np.where(L[f"ambig_P{pct:g}"].to_numpy(bool), np.nan, y)
    out = pd.DataFrame({
        "y": y,
        "t_win_min": np.minimum(fi, ci),                              # 승자 도달(분), 미결정 inf
        "margin_min": np.abs(np.where(np.isinf(fi), lim, fi) - np.where(np.isinf(ci), lim, ci)),
        "hold_bars": np.where(np.isfinite(y), np.minimum(fi, ci) / 5.0, H),   # 라벨 창 길이(봉)
    })
    return out


def uniqueness(bar_idx: np.ndarray, span_bars: np.ndarray) -> np.ndarray:
    """AFML 고유도: 각 라벨 창이 다른 라벨 창과 겹치는 정도의 역수."""
    n = len(bar_idx)
    order = np.argsort(bar_idx)
    bi, sp = bar_idx[order], np.maximum(span_bars[order], 1.0)
    end = bi + sp
    ov = np.ones(n)
    j0 = 0
    for k in range(n):
        while bi[j0] + sp[j0] <= bi[k]:
            j0 += 1
        ov[k] = np.sum((bi[j0:k + 1] < end[k]) & (end[j0:k + 1] > bi[k]))
    w = np.empty(n)
    w[order] = 1.0 / np.maximum(ov, 1.0)
    return w


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    import build_eth_anchor_label_dataset_20260907 as B
    L = pd.read_parquet(SRC)
    A = L[L.anchor == ANCHOR].reset_index(drop=True)
    eth = B._load_kl(B.ETH_KL); btc = B._load_kl(B.BTC_KL); fund = B._load_funding()
    tmax = min(eth["timestamp"].max(), btc["timestamp"].max(), fund["calc_time"].max())
    eth = eth[eth["timestamp"] <= tmax].reset_index(drop=True)
    O, Hh, Lo, C = (eth[c].to_numpy(float) for c in ("open", "high", "low", "close"))
    print(f"[1/5] 앵커 {ANCHOR}: {len(A):,}행", flush=True)

    rep = []
    for H in (H_MAIN, H_ALT):
        D = pd.concat([A, derive(A, PCT, H), path_stats(A, PCT, H, O, Hh, Lo, C)], axis=1)
        D["H"] = H

        # ⭐3클래스 + 품질 가중치 (사용자 지정 2026-09-07: "2와 3을 같이 두고")
        #   y3 = 2 지속승(깨끗) · 0 되돌림승(깨끗) · 1 혼재(결정됐으나 안 깨끗, 또는 미결정)
        #   깨끗 = 닿기 전 반대이탈 < ADV_MAX% **그리고** 창 끝점이 그 방향 END_MIN% 이상
        #   y_bin = D1(먼저 닿기) 이진 라벨 -- 미결정은 NaN. 옵션 ③(가중 이진)용.
        yb = D["y"].to_numpy(float)
        endc = D["end_cont_pct"].to_numpy(float)
        rngp = D["range_pct"].to_numpy(float)
        eff = np.abs(endc) / np.maximum(rngp, 1e-9)                  # 경로 효율 (방향은 endc 부호가 담당)
        D["path_eff"] = eff
        clean_c = (yb == 1) & (D["pre_adv_cont"].to_numpy(float) < ADV_MAX) & (endc >= END_MIN) & (eff >= EFF_MIN)
        clean_f = (yb == 0) & (D["pre_adv_fade"].to_numpy(float) < ADV_MAX) & (-endc >= END_MIN) & (eff >= EFF_MIN)
        D["y3"] = np.where(clean_c, 2, np.where(clean_f, 0, 1)).astype(float)
        D["y_bin"] = yb
        D["is_clean"] = clean_c | clean_f
        # 품질 점수 q in [0,1]: 역행이 작을수록·끝점이 멀리 갈수록 1. 둘 다 좋아야 높다(곱).
        adv = np.where(yb == 1, D["pre_adv_cont"], D["pre_adv_fade"]).astype(float)
        end_dir = np.where(yb == 1, endc, -endc).astype(float)
        q_adv = np.clip(1.0 - adv / ADV_MAX, 0.0, 1.0)
        q_end = np.clip(end_dir / END_MIN, 0.0, 1.0)
        q_eff = np.clip(eff / EFF_MIN, 0.0, 1.0)                     # 효율도 품질 점수의 한 축
        D["q_label"] = np.where(np.isfinite(yb), q_adv * q_end * q_eff, 0.0)

        # (A) 미결정: 이진 라벨이 없다. 3클래스에서는 '혼재'로 살려두고, 이진 학습에서만 제외한다.
        undecided = ~np.isfinite(D["y"].to_numpy())
        D["undecided"] = undecided
        D = D[D["win_ok"].to_numpy()].reset_index(drop=True)

        # 퍼징: 라벨 창이 자기 split 을 넘어가면 제거
        end_ts = D["timestamp"] + pd.to_timedelta(np.where(D["undecided"], H, D["hold_bars"]) * 5, unit="m")
        keep = np.ones(len(D), bool)
        for sp, (a, b) in SPLIT_EDGES.items():
            m = D["split"].to_numpy() == sp
            keep &= ~(m & (end_ts > pd.Timestamp(b)).to_numpy())
        purged = int((~keep).sum())
        D = D[keep].reset_index(drop=True)

        # 엠바고: TRAIN 마지막 H봉은 버린다
        emb = (D["split"] == "TRAIN") & (D["timestamp"] > pd.Timestamp(SPLIT_EDGES["TRAIN"][1]) - pd.Timedelta(minutes=5 * H))
        n_emb = int(emb.sum())
        D = D[~emb].reset_index(drop=True)

        # 고유도 가중치 (split 안에서)
        D["w_uniq"] = np.nan
        for sp in D["split"].unique():
            m = (D["split"] == sp).to_numpy()
            D.loc[m, "w_uniq"] = uniqueness(D.loc[m, "bar_idx"].to_numpy(), D.loc[m, "hold_bars"].to_numpy())

        # (C) TRAIN 전용 라벨 품질 필터 -- 플래그로만 둔다(적용은 학습 코드에서 선택)
        D["train_quality_ok"] = (D["margin_min"] >= MARGIN_MIN_TRAIN) | (D["split"] != "TRAIN")
        D["w_final"] = D["w_uniq"] * D["q_label"]        # 옵션③: 고유도 x 라벨품질

        D.to_parquet(OUT / f"train_set_P{PCT:g}_H{H}.parquet", index=False)
        for sp in ("TRAIN", "VAL", "OOS", "HOLDOUT_SPENT"):
            s = D[D.split == sp]
            if not len(s):
                continue
            rep.append({"H": H, "split": sp, "n": len(s),
                        "지속승(y3=2)": float((s.y3 == 2).mean()), "혼재(y3=1)": float((s.y3 == 1).mean()),
                        "되돌림승(y3=0)": float((s.y3 == 0).mean()),
                        "깨끗n": int(s.is_clean.sum()), "이진n": int(np.isfinite(s.y_bin).sum()),
                        "지속비율(이진)": float(np.nanmean(s.y_bin)), "q평균": float(s.q_label.mean()),
                        "효율중앙": float(s.path_eff.median()),
                        "유효표본": s.w_uniq.sum(), "고유도평균": s.w_uniq.mean(),
                        "서로다른날": s.timestamp.dt.floor("D").nunique(),
                        "중앙보유(봉)": s.hold_bars.median()})
        print(f"[H={H}] 미결정 {int(undecided.sum()):,} ({undecided.mean()*100:.1f}%, 혼재로 보존) · "
              f"퍼징 {purged} · 엠바고 {n_emb} · 잔존 {len(D):,} · 깨끗 {int(D.is_clean.sum()):,}", flush=True)

    R = pd.DataFrame(rep)
    print("\n=== 최종 학습셋 ===")
    print(R.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    R.to_csv(OUT / "summary.csv", index=False)
    (OUT / "contract.json").write_text(json.dumps({
        "anchor": ANCHOR, "barrier_pct": PCT, "H_main": H_MAIN, "H_alt": H_ALT,
        "label": "y=1 지속 배리어 먼저 · 0 페이드 먼저 · 미결정은 제거(라벨 미정의)",
        "resolution": "1분봉", "entry": "open[t+1]", "barrier": "entry*(1±0.01), 대칭",
        "purged": True, "embargo_bars": H_MAIN,
        "clean_rule": f"닿기전 역행<{ADV_MAX}% AND 끝점>={END_MIN}% AND 경로효율>={EFF_MIN}",
        "train_quality_filter": f"margin_min>={MARGIN_MIN_TRAIN}분 (TRAIN 전용 플래그 train_quality_ok)",
        "weights": "w_uniq = AFML 고유도(1/동시겹침)",
        "⚠️": "미결정 제거분은 라이브에서 시간청산이 된다 -- 이 모델의 평가에 포함되지 않으므로 별도 회계 필요",
    }, indent=2, ensure_ascii=False))
    print(f"\n저장: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
