"""상태 서술 프레임: 미래 수익률을 쓰지 않는다. 각 원천의 «지금 상태»를 분위로 이산화하고
(1) 점유율·에피소드 지속시간 (2) 결합 상황의 정의·빈도·지속 (3) 그 상황에 동시에 나타나는 다른 원천의
상태 분포(리프트 = 조건부/무조건부) (4) 직전 300초에 무엇이 있었나 (5) 60초 뒤 어느 상황으로 옮겨가나(전이).
활동 임계값은 같은 UTC 시간대 안의 분위(계절성 제거). 출력 json 은 아티팩트가 읽는다."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
exec(open(ROOT / "scripts/research_rt5_1s_panel_analyze_20260920.py").read().split("# ── 1. 원천별 단독")[0])  # noqa: 피쳐 재사용
OUT = ROOT / "tmp/rt_probe_20260920/state.json"
R: dict = {}
hr = pd.to_datetime(P.index, unit="s").hour


def pct_in_hour(s: pd.Series) -> pd.Series:
    return s.groupby(hr).rank(pct=True)


def tri(s: pd.Series, lo="↓", mid="—", hi="↑", q=(0.2, 0.8), by_hour=False) -> pd.Series:
    r = pct_in_hour(s) if by_hour else s.rank(pct=True)
    out = pd.Series(np.where(r <= q[0], lo, np.where(r >= q[1], hi, mid)), index=s.index, dtype=object)
    out[s.isna()] = None
    return out


# ── 원천별 상태 ─────────────────────────────────────────────────────────────
S = pd.DataFrame(index=P.index)
S["price"] = tri(P.dmid60, "하락", "횡보", "상승")
S["oi"] = tri(P.oi_d60, "감소", "보합", "증가")
S["taker"] = tri(P.tr_imb60, "매도우위", "중립", "매수우위")
S["whale"] = tri(P.wh_net60, "고래매도", "고래중립", "고래매수")
S["retail"] = tri(P.rt_net60, "리테일매도", "리테일중립", "리테일매수")
S["book"] = tri(P.dd_imb10, "매도벽", "호가중립", "매수벽")
S["deep"] = tri(P.dd_imb50, "깊은매도벽", "깊은중립", "깊은매수벽")
S["act"] = tri(P.tr_vol60, "조용", "보통", "활발", by_hour=True)
S["absorb"] = tri(P.fp_absorb60, "효율적", "흡수보통", "흡수")
qi = tri(P.bt_qi10, "QI매도", "QI중립", "QI매수"); ofi = tri(P.dd_ofi10, "OFI매도", "OFI중립", "OFI매수")
S["micro"] = np.select([(qi == "QI매수") & (ofi == "OFI매수"), (qi == "QI매도") & (ofi == "OFI매도"),
                        (qi == "QI매수") & (ofi == "OFI매도"), (qi == "QI매도") & (ofi == "OFI매수")],
                       ["동조매수", "동조매도", "갈림(QI매수·OFI매도)", "갈림(QI매도·OFI매수)"], "중립")
S.loc[qi.isna() | ofi.isna(), "micro"] = None
lq = P.lq_now_tot.fillna(-1)
S["liq"] = np.select([lq < 0, lq == 0, (P.liq_long > 0) & (P.liq_short == 0), (P.liq_short > 0) & (P.liq_long == 0), (P.liq_long > 0) & (P.liq_short > 0)],
                     [None, "청산없음", "롱청산", "숏청산", "양쪽청산"], None)
big = P.lq_now_tot >= P.lq_now_tot[P.lq_now_tot > 0].quantile(0.95)
S.loc[big & (S.liq == "롱청산"), "liq"] = "롱청산버스트"; S.loc[big & (S.liq == "숏청산"), "liq"] = "숏청산버스트"


def episodes(s: pd.Series) -> dict:
    """상태별 점유율·에피소드(연속 구간) 수·중앙/평균 지속초."""
    v = s.dropna(); out = {}
    if v.empty:
        return out
    chg = (v != v.shift()) | (v.index.to_series().diff() != 1)
    grp = chg.cumsum()
    ep = v.groupby(grp).agg(["first", "size"])
    for st, g in ep.groupby("first"):
        out[str(st)] = dict(share=round(float((v == st).mean()), 4), episodes=int(len(g)),
                            median_sec=float(g["size"].median()), mean_sec=round(float(g["size"].mean()), 1),
                            p90_sec=float(g["size"].quantile(0.9)))
    return out


R["streams"] = {c: episodes(S[c]) for c in S.columns}

# ── 결합 상황 사전 ───────────────────────────────────────────────────────────
SITUATIONS = {
    "OI×가격 (60초)": {
        "신규 롱 유입 (가격↑ OI↑)": (S.price == "상승") & (S.oi == "증가"),
        "숏 커버 (가격↑ OI↓)": (S.price == "상승") & (S.oi == "감소"),
        "신규 숏 유입 (가격↓ OI↑)": (S.price == "하락") & (S.oi == "증가"),
        "롱 이탈/청산 (가격↓ OI↓)": (S.price == "하락") & (S.oi == "감소"),
    },
    "테이커×OI (60초)": {
        "공격적 신규 롱 (매수우위 OI↑)": (S.taker == "매수우위") & (S.oi == "증가"),
        "숏 커버 매수 (매수우위 OI↓)": (S.taker == "매수우위") & (S.oi == "감소"),
        "공격적 신규 숏 (매도우위 OI↑)": (S.taker == "매도우위") & (S.oi == "증가"),
        "롱 투매 (매도우위 OI↓)": (S.taker == "매도우위") & (S.oi == "감소"),
    },
    "청산×가격 (현재 분·60초)": {
        "하락 중 롱 청산": (S.price == "하락") & S.liq.isin(["롱청산", "롱청산버스트"]),
        "상승 중 숏 청산": (S.price == "상승") & S.liq.isin(["숏청산", "숏청산버스트"]),
        "하락인데 숏 청산 (역방향)": (S.price == "하락") & S.liq.isin(["숏청산", "숏청산버스트"]),
        "상승인데 롱 청산 (역방향)": (S.price == "상승") & S.liq.isin(["롱청산", "롱청산버스트"]),
        "양쪽 청산 (휩쏘)": S.liq == "양쪽청산",
    },
    "고래×리테일 (60초)": {
        "동조 매수": (S.whale == "고래매수") & (S.retail == "리테일매수"),
        "동조 매도": (S.whale == "고래매도") & (S.retail == "리테일매도"),
        "고래 매수·리테일 매도 (고래가 받음)": (S.whale == "고래매수") & (S.retail == "리테일매도"),
        "고래 매도·리테일 매수 (리테일이 받음)": (S.whale == "고래매도") & (S.retail == "리테일매수"),
    },
    "테이커×호가벽 (60초·±10bp)": {
        "매수우위 & 매수벽 (밀어올림)": (S.taker == "매수우위") & (S.book == "매수벽"),
        "매수우위 & 매도벽 (벽을 때림)": (S.taker == "매수우위") & (S.book == "매도벽"),
        "매도우위 & 매도벽 (눌러내림)": (S.taker == "매도우위") & (S.book == "매도벽"),
        "매도우위 & 매수벽 (벽을 때림)": (S.taker == "매도우위") & (S.book == "매수벽"),
    },
    "흡수×호가 (60초)": {
        "흡수 & 매수벽 (매도를 받아냄)": (S.absorb == "흡수") & (S.book == "매수벽"),
        "흡수 & 매도벽 (매수를 받아냄)": (S.absorb == "흡수") & (S.book == "매도벽"),
        "효율적 상승 (적은 양으로 크게)": (S.absorb == "효율적") & (S.price == "상승"),
        "효율적 하락": (S.absorb == "효율적") & (S.price == "하락"),
    },
    "미시 호가 (10초)": {
        "동조매수 (QI·OFI 매수)": S.micro == "동조매수",
        "동조매도": S.micro == "동조매도",
        "갈림": S.micro.str.startswith("갈림", na=False),
    },
}

PROFILE_COLS = ["price", "oi", "taker", "liq", "whale", "book", "deep", "act", "absorb", "micro"]
PAST = {"직전300초 가격(bp)": P.dmid300, "직전60초 가격(bp)": P.dmid60, "직전300초 순테이커(ETH)": P.tr_net300,
        "직전300초 ΔOI(ETH)": P.oi_d300, "직전분 청산합($)": P.lq_prev_tot, "직전60초 거래량(ETH)": P.tr_vol60}
uncond = {c: S[c].value_counts(normalize=True).to_dict() for c in PROFILE_COLS}
R["uncond"] = uncond
R["situations"] = {}
for fam, d in SITUATIONS.items():
    R["situations"][fam] = {}
    fam_masks = {k: v.fillna(False) for k, v in d.items()}
    for name, m in fam_masks.items():
        m = m.astype(bool)
        base = m.copy()
        for c in PROFILE_COLS:
            pass
        ep = episodes(pd.Series(np.where(m, "in", "out"), index=S.index)).get("in", {})
        prof = {}
        for c in PROFILE_COLS:
            vc = S.loc[m, c].value_counts(normalize=True)
            prof[c] = {str(k): dict(p=round(float(v), 3), lift=round(float(v / uncond[c].get(k, np.nan)), 2)) for k, v in vc.items()}
        past = {k: dict(median=round(float(v[m].median()), 2), p25=round(float(v[m].quantile(.25)), 2), p75=round(float(v[m].quantile(.75)), 2),
                        all_median=round(float(v.median()), 2)) for k, v in PAST.items() if v[m].notna().sum() > 100}
        # 60초 뒤 같은 가족 안에서 어느 상황인가 (전이). 어느 것도 아니면 '기타'
        nxt = {}
        idx_next = S.index[m] + 60
        valid = idx_next.isin(S.index)
        for name2, m2 in fam_masks.items():
            nxt[name2] = round(float(m2.reindex(idx_next[valid]).fillna(False).mean()), 3)
        nxt["기타/중립"] = round(1 - sum(nxt.values()), 3)
        R["situations"][fam][name] = dict(share=round(float(m.mean()), 4), n_sec=int(m.sum()), episode=ep, profile=prof, past=past, next60=nxt)

# ── 원천 간 동시 상태 일치표 (상태 프레임의 «상관») ───────────────────────────
def agree(a: str, b: str, pos_a: str, neg_a: str, pos_b: str, neg_b: str) -> dict:
    x = S[a]; y = S[b]
    m = x.isin([pos_a, neg_a]) & y.isin([pos_b, neg_b])
    same = ((x == pos_a) & (y == pos_b)) | ((x == neg_a) & (y == neg_b))
    return dict(n=int(m.sum()), same=round(float(same[m].mean()), 3))


R["agreement"] = {
    "가격60 vs 테이커60": agree("price", "taker", "상승", "하락", "매수우위", "매도우위"),
    "가격60 vs OI60": agree("price", "oi", "상승", "하락", "증가", "감소"),
    "테이커60 vs OI60": agree("taker", "oi", "매수우위", "매도우위", "증가", "감소"),
    "고래60 vs 리테일60": agree("whale", "retail", "고래매수", "고래매도", "리테일매수", "리테일매도"),
    "테이커60 vs 호가벽10bp": agree("taker", "book", "매수우위", "매도우위", "매수벽", "매도벽"),
    "가격60 vs 호가벽10bp": agree("price", "book", "상승", "하락", "매수벽", "매도벽"),
    "가격60 vs 깊은벽50bp": agree("price", "deep", "상승", "하락", "깊은매수벽", "깊은매도벽"),
    "가격60 vs 청산(현재분)": agree("price", "liq", "상승", "하락", "숏청산", "롱청산"),
    "테이커60 vs 미시호가10s": agree("taker", "micro", "매수우위", "매도우위", "동조매수", "동조매도"),
}

# 시간대 계절성 (상태 기준선)
seas = P.groupby(hr)[["tr_vol60", "tr_nn", "bt_n", "dd_churn", "rv60", "lq_now_tot"]].median()
R["seasonality"] = (seas / seas.median()).round(2).to_dict(orient="index")
R["meta"] = dict(n_sec=int(len(P)), start=str(pd.to_datetime(P.index.min(), unit="s")), end=str(pd.to_datetime(P.index.max(), unit="s")),
                 oi_cover=round(float(P.oi_last.notna().mean()), 3), whale_cover=round(float(P.tr_whale_buy_qty.notna().mean()), 3),
                 liq_burst_usd=float(P.lq_now_tot[P.lq_now_tot > 0].quantile(0.95)))
OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else str(o)), encoding="utf-8")
print("->", OUT)
print(json.dumps(R["streams"]["oi"], ensure_ascii=False)); print(json.dumps(R["agreement"], ensure_ascii=False))
for fam, d in R["situations"].items():
    print("\n##", fam)
    for k, v in d.items():
        print(f"  {k:36} share {v['share']:.3f} ep {v['episode'].get('episodes')} med {v['episode'].get('median_sec')}s p90 {v['episode'].get('p90_sec')}s  next60 {v['next60']}")
