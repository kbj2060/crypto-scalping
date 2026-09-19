#!/usr/bin/env python3
"""SMC 구조(BOS/CHoCH) · 브레이커 · CRT · TBS — 사전등록 검정 (2026-09-19).

사용자 *"smc도 추가로 조사해줘 / crt, tbs도 연구해줘"*.

## 선행 영수증 — SMC 대부분은 이미 닫혀 있다 (재파생 방지)

`docs/eth_project_wide_idea_map_20260824.md:100` · `eth_ict2022_ob_smt_po3_component_evidence_20260824.md`
· `eth_liquidity_sweep_axis_closed_20260914.md` · `eth_ict3_killzone_fvg_equalhighs_20260916.md`:

  오더블록 1.07x/1.14x CLOSED(존이 봉의 29%를 덮음) · FVG 0.88x CLOSED(발동 27%) · iFVG 0.48x 역예측
  · SMT raw 3.12x(sweep 과 **동급**, 상위호환 아님) · Po3/Judas OOS 붕괴 · 킬존 0 엣지
  · 프리미엄/디스카운트(OTE 61.8~78.6% 골든포켓) 0.88~1.27x 무근거
  · liquidity_sweep 은 1,741일로 CLOSED(비겹침블록 t −1.13, 메커니즘 필터 0/14)
  · EQH/EQL · 전일고저 되찾기 4라운드 CLOSED(되찾기 필터가 **빼고**, TP/SL 판은 유의하게 음수)

즉 SMC 의 «존 회귀»·«컨펌 게이트»·«세션 앵커»·«되돌림 비율» 네 계열은 전부 측정 끝이다.
이 스크립트가 다루는 것은 **그 격자에서 빠져 있던 4개**뿐이다:

  A. 구조 전환 BOS/CHoCH — 스윙 갱신/실패로 «추세 계속 vs 전환»을 가르는 SMC 의 뼈대. 미측정.
  B. 브레이커/미티게이션 블록 — «실패한 OB» 존. plain OB(1.07x)와 달리 «깨진 뒤 리테스트» 조건. 미측정.
  C. CRT (Candle Range Theory) — 직전 **HTF 봉 레인지**를 쓸고 되돌아오면 반대 극단이 타깃.
     기존 sweep(48봉 롤링 스윙)·Po3(아시아 세션)과 **앵커가 다르다**(고정 시간격자 봉 경계). 미측정.
  D. TBS (Time-Based Setups) — 🔴**용어 모호**. 이 스크립트는 CRT 와 같이 가르쳐지는 계열의 지배적
     독법인 «quarterly theory»(사이클을 4등분, Q1 축적 → Q2 조작 → Q3 분배)로 구현한다.
     다른 독법(Three Bar Setup 등)이었다면 이 결과는 그 독법에 대한 판정이 아니다 — 명시한다.
     ⚠️5분봉 해상도상 22.5분 사분면(90분 사이클의 1/4)은 정수봉이 아니라 만들 수 없다.
     격자는 4h→1h · 24h→6h 두 개만 쓴다.

## 이번 설계가 선행 라운드에서 물려받는 것 (같은 함정 재방문 금지)

1. **막는 것은 a 가 아니라 b 다.** 09-16 이 정확도 우위 +3~6pp(z 5.5)에도 전 셀 손익분기 미달임을
   보였다. 그래서 모든 셀에 `acc_needed`(손익분기 적중률)와 `여유pp`를 1급 컬럼으로 싣는다.
2. **존이 차트를 덮으면 신호가 아니라 배경이다.** OB 29% · FVG 27% 전례. `fire_pct`(전체 봉 대비
   발동 비율)를 싣고 10% 초과는 표에 ⚠️를 찍는다.
3. **컨펌 쌓기는 희석하거나 무효다**(3회 재현). 교집합 팔을 넣되 단품과 **같은 표**에서 비교한다.
4. **되찾기(reclaim) 필터는 공짜가 아니다.** 09-16 4R 에서 D0(터치만) > D1(되찾기)이었다.
   CRT/TBS 의 «되돌아 마감» 조건을 켠 팔과 끈 팔을 **쌍으로** 넣는다.
5. ⭐**앵커 특이성 대조군(이 라운드의 핵심 신규).** CRT/TBS 의 주장은 «그 시간 경계가 특별하다»는
   것이다. 격자를 **반 칸 밀기만 한** 대조군을 같은 표에 넣는다 — 밀어도 같은 숫자가 나오면
   정보는 «봉 경계»가 아니라 «최근 레인지 가장자리»에 있는 것이고, 그건 이미 닫힌 sweep 축이다.
6. **0 이 아니라 «잴 수 없다»를 구분한다.** 셀마다 일군집 SE 로 MDE(80% 검정력)를 낸다.
   09-16 4R 에서 MDE 6.5~34bp 였다 — 비용선(5.52bp) 아래 참엣지는 이 표본에서 원리적으로 확인 불가다.
7. **다중검정 귀무.** 발동 인덱스 순환이동 B회로 «우연히 몇 셀이 통과하는가»의 분포를 낸다.

## 인과성 (CLAUDE.md 사건-라벨 경계 계약)

발동 봉 i 는 봉 i 까지의 정보만 쓴다 → 진입 `open[i+1]` → 라벨 탐색은 i+1 부터. 피쳐 창 끝(i) <
라벨 시작(i+1) 이므로 같은 봉을 공유하지 않는다. 스윙 피벗은 **확정 지연 k 봉**을 반영해
p+k 이후에만 쓴다(중심 롤링 최대값은 미래를 보므로 그대로 쓰면 누수다). HTF 레인지는 **완결된**
직전 봉만 쓴다.

## 청산 컨벤션 — 두 벌을 나란히 낸다

(i) 고정지평 종가 H ∈ {12,24,48,144} — 선행 모든 영수증과 직접 비교 가능.
(ii) **네이티브 배리어** — CRT/TBS 는 «반대 극단이 타깃»이라 레벨 청산이 본래형이다.
     intrabar 고가/저가로 판정하고 **같은 봉에서 둘 다 닿으면 SL 우선**(보수적).
     이 저장소 라이브 배리어 컨벤션(`omega4_6_1_live.py::evaluate_exit` 의 bar_high/low_move)과
     09-16 4R 의 E 검정과 같다. 🔴선행 결과는 배리어가 b 를 **더 줄인다**고 말한다 — 기대 낮다.

## 사전등록 판정 (실행 전 고정, 변형 탐색 금지)

  PASS = 블록초과 > 5.52bp(양다리 peg) AND 초과 CI95 가 0 배제 AND |t_블록| >= 2
         AND 전·후반 부호 일치 AND BH-FDR q < 0.10
         AND ⭐앵커이동 대조군 대비 초과가 더 클 것(C/D 계열만 — 앵커 특이성)
  그 외 CLOSED 또는 «검정력 부족»(MDE 표기).

## 데이터

서버 로컬 CSV `binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv`(2021-12~, ~495k봉/1,741일).
🔴작성 세션(claude/dashboard-gex-orderflow-ict-gmdmt6)의 컨테이너는 egress 정책상
data.binance.vision · api.binance.com 이 **차단**돼 실데이터를 못 받았다. 그래서 이 파일은
`--selftest`(발동식 단위검증)와 `--synthetic`(랜덤워크 전 구간 파이프라인 + 귀무 보정)까지만
검증돼 있고, **실측 숫자는 서버에서 이 스크립트를 돌려야 나온다**:

    python scripts/research_eth_smc_crt_tbs_20260919.py            # 실측
    python scripts/research_eth_smc_crt_tbs_20260919.py --synthetic 120000   # 귀무 보정

출력: tmp/eth_smc_crt_tbs_20260919/cells.csv · barrier.csv · summary.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path: sys.path.insert(0, str(_p))

# 하네스 재구현 금지 — 09-14/09-16 라운드에서 그대로 가져온다
from research_eth_liquidity_sweep_block_independence_20260914 import (  # noqa: E402
    gross_bp, thin_nonoverlap, day_cluster_boot, load, CSV, BTC)
from research_eth_ict_killzone_fvg_eqh_20260916 import (  # noqa: E402
    day_boot, null_hit, null_mean, bh_fdr, COST_PEG)

OUT = ROOT / "tmp/eth_smc_crt_tbs_20260919"
HORIZONS = [12, 24, 48, 144]        # 1h · 2h · 4h · 12h — 선행 라운드와 동일
WARMUP = 900                        # 지표 웜업 — 선행 라운드와 동일
PIVOT_K = 6                         # 스윙 피벗 확정 지연(30분). 커질수록 느리고 굵은 스윙
DISPLACE_ATR = 1.0                  # 변위 임계 — eth_ict2022 OB 연구와 동일 기준
ZONE_LIFE = 48                      # 존 수명(4시간) — FVG/OB 관례와 동일
BACKGROUND_PCT = 10.0               # 발동이 전체 봉의 이 % 를 넘으면 «배경» 경고
NBOOT_SHIFT = 200                   # 순환이동 다중검정 귀무 횟수
MDE_Z = 2.802                       # 1.96 + 0.842 = 양측 5% · 검정력 80%
RNG = np.random.default_rng(20260919)


# ────────────────────────────────────────────────────────────────────────────
# A. 구조 — 인과 스윙 피벗 → BOS / CHoCH
# ────────────────────────────────────────────────────────────────────────────
def causal_swing_levels(high: np.ndarray, low: np.ndarray, k: int = PIVOT_K
                        ) -> tuple[np.ndarray, np.ndarray]:
    """확정 지연 k 를 반영한 «마지막 확정 스윙고/저» 레벨.

    프랙탈 피벗: high[p] 가 [p-k, p+k] 의 최대면 스윙고. 이 사실은 p+k 봉이 닫혀야 알 수 있으므로
    **p+k 부터만** 쓴다(중심 롤링을 그대로 쓰면 k 봉 미래참조다 — 이 저장소 5.16/5.21 계열 사고).
    반환 배열의 i 번째 값 = 봉 i 시점에 «이미 확정된» 가장 최근 스윙 레벨(없으면 NaN).
    """
    n = len(high)
    w = 2 * k + 1
    hs = pd.Series(high); ls = pd.Series(low)
    is_hi = (hs == hs.rolling(w, center=True, min_periods=w).max()).to_numpy()
    is_lo = (ls == ls.rolling(w, center=True, min_periods=w).min()).to_numpy()
    sh = np.full(n, np.nan); sl = np.full(n, np.nan)
    # 피벗 p 는 p+k 에 «알려진다» — 그 인덱스에 값을 놓고 앞으로 채운다
    idx_hi = np.flatnonzero(is_hi); idx_lo = np.flatnonzero(is_lo)
    known_hi = idx_hi + k; known_lo = idx_lo + k
    ok = known_hi < n; sh[known_hi[ok]] = high[idx_hi[ok]]
    ok = known_lo < n; sl[known_lo[ok]] = low[idx_lo[ok]]
    return (pd.Series(sh).ffill().to_numpy(), pd.Series(sl).ffill().to_numpy())


def bos_choch(close: np.ndarray, sh: np.ndarray, sl: np.ndarray
              ) -> dict[str, np.ndarray]:
    """구조 돌파를 «계속(BOS)»과 «전환(CHoCH)»으로 가른다.

    상태 = 마지막 구조 돌파의 방향. 상방 돌파(close > 확정 스윙고)가 나왔을 때
      · 직전 상태가 상승이면 BOS(계속)  · 하락이었으면 CHoCH(전환)
    돌파가 유지되는 동안은 발동하지 않는다(전이 봉 1회만) — «존 터치»류가 배경이 되는 병리 회피.
    """
    n = len(close)
    out = {k: np.zeros(n, bool) for k in ("bos_up", "bos_dn", "choch_up", "choch_dn")}
    state = 0          # +1 상승구조 · -1 하락구조 · 0 미정
    armed_up = armed_dn = True
    for i in range(n):
        hi_lvl, lo_lvl = sh[i], sl[i]
        if np.isnan(hi_lvl) or np.isnan(lo_lvl):
            continue
        up = close[i] > hi_lvl
        dn = close[i] < lo_lvl
        if up and armed_up:
            key = "bos_up" if state >= 0 else "choch_up"
            out[key][i] = True
            state, armed_up, armed_dn = 1, False, True
        elif dn and armed_dn:
            key = "bos_dn" if state <= 0 else "choch_dn"
            out[key][i] = True
            state, armed_dn, armed_up = -1, False, True
        if not up:
            armed_up = True
        if not dn:
            armed_dn = True
    return out


# ────────────────────────────────────────────────────────────────────────────
# B. 브레이커 / 미티게이션 블록
# ────────────────────────────────────────────────────────────────────────────
def breaker_touch(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                  atr_price: np.ndarray, sh: np.ndarray, sl: np.ndarray,
                  *, life: int = ZONE_LIFE, disp: float = DISPLACE_ATR
                  ) -> tuple[np.ndarray, np.ndarray]:
    """«실패한 OB» = 브레이커 존의 리테스트.

    강세 브레이커(롱): 하락 구조에서 마지막 **양봉**이 저점을 못 지키고(그 아래로 변위) 무너졌다가,
    이후 상방 구조돌파가 나오면 그 양봉 레인지가 지지로 바뀐다는 주장. 구현:
      1) 봉 p 가 양봉이고, p 이후 `disp`×ATR 이상 **하락** 변위로 깨진다 → 후보 존 [low[p], high[p]]
      2) 그 뒤 상방 구조돌파(close > 확정 스윙고)가 나온다  ← 여기서 «브레이커»가 된다
      3) 돌파 후 `life` 봉 안에 가격이 그 존에 **다시 닿으면** 발동(터치 관례는 FVG/OB 연구와 동일)
    약세 미러. plain OB(1.07x)와의 차이는 (1)의 «깨짐»과 (2)의 «구조돌파» 두 조건이다.

    ⭐존이 차트를 덮는 병리(OB 29% · FVG 27%)를 정의 단계에서 막는다 — SMC 교본도 «변위 직전의
    **마지막** 캔들»이라고 말한다. 그래서 (a) 구조돌파 시점에 깨져 있던 후보 중 **가장 최근 것
    하나만** 존으로 삼고, (b) **첫 터치 1회**만 발동하고 그 존을 소멸시킨다. 합성 랜덤워크
    예비주행에서 이 두 조건 없이는 발동이 봉의 10.4%(30건/일)로 배경이 됐다.
    """
    n = len(close)
    fire_up = np.zeros(n, bool); fire_dn = np.zeros(n, bool)
    # 후보 = [봉, zone_lo, zone_hi, 깨졌나]. «깨짐»은 **한 번 켜지면 유지되는 상태**다 —
    # 존이 깨진 봉과 구조돌파가 난 봉은 보통 다르기 때문(예비판에서 이걸 같은 봉으로 요구해
    # 발동이 0 이 됐다).
    up_cand: list[list] = []; dn_cand: list[list] = []
    act_up: list[tuple[int, float, float]] = []     # (만료봉, lo, hi) 구조돌파로 활성화된 존
    act_dn: list[tuple[int, float, float]] = []
    for i in range(n):
        a = atr_price[i]
        if np.isnan(a) or a <= 0:
            continue
        # 1) 후보 등록 — 직전 봉이 양/음봉
        if i > 0:
            (up_cand if close[i - 1] > open_[i - 1] else dn_cand).append(
                [i - 1, low[i - 1], high[i - 1], False])
        up_cand = [c for c in up_cand if i - c[0] <= life]
        dn_cand = [c for c in dn_cand if i - c[0] <= life]
        # 2) 깨짐 — 존 아래(위)로 disp×ATR 변위. 상태로 남긴다
        for c in up_cand:
            if close[i] < c[1] - disp * a: c[3] = True
        for c in dn_cand:
            if close[i] > c[2] + disp * a: c[3] = True
        # 3) 구조돌파가 나면 **가장 최근에 깨진 존 하나만** 활성화(교본의 «마지막 캔들»)
        if not np.isnan(sh[i]) and close[i] > sh[i]:
            br = [c for c in up_cand if c[3]]
            if br:
                c = max(br, key=lambda z: z[0])
                act_up.append((i + life, c[1], c[2]))
                up_cand.remove(c)              # 한 존이 반복 활성화되지 않게
        if not np.isnan(sl[i]) and close[i] < sl[i]:
            br = [c for c in dn_cand if c[3]]
            if br:
                c = max(br, key=lambda z: z[0])
                act_dn.append((i + life, c[1], c[2]))
                dn_cand.remove(c)
        act_up = [z for z in act_up if z[0] >= i]
        act_dn = [z for z in act_dn if z[0] >= i]
        # 4) 리테스트 — **첫 터치 1회**만 발동하고 그 존은 소멸시킨다
        hit_up = [z for z in act_up if low[i] <= z[2] and high[i] >= z[1]]
        if hit_up:
            fire_up[i] = True
            act_up = [z for z in act_up if z not in hit_up]
        hit_dn = [z for z in act_dn if low[i] <= z[2] and high[i] >= z[1]]
        if hit_dn:
            fire_dn[i] = True
            act_dn = [z for z in act_dn if z not in hit_dn]
    return fire_up, fire_dn


# ────────────────────────────────────────────────────────────────────────────
# C/D. 고정 시간격자 — CRT(직전 HTF 봉 레인지) · TBS(사이클 4분면)
# ────────────────────────────────────────────────────────────────────────────
def grid_id(ts: pd.Series, period: str, offset_min: int = 0) -> np.ndarray:
    """UTC 고정격자의 봉 번호. offset_min 으로 격자를 통째로 민다(앵커 특이성 대조군)."""
    shifted = ts - pd.Timedelta(minutes=offset_min)
    return shifted.dt.floor(period).astype("int64").to_numpy()


def prior_cell_range(gid: np.ndarray, high: np.ndarray, low: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    """각 봉에서 본 «직전 격자칸의 고/저». 완결된 칸만 쓴다(현재 칸은 아직 안 끝났다)."""
    df = pd.DataFrame({"g": gid, "h": high, "l": low})
    agg = df.groupby("g", sort=True).agg(ph=("h", "max"), pl=("l", "min"))
    agg_prev = agg.shift(1)                       # 직전 칸
    return (agg_prev["ph"].reindex(gid).to_numpy(),
            agg_prev["pl"].reindex(gid).to_numpy())


def crt_fire(gid: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
             ph: np.ndarray, pl: np.ndarray, *, reclaim: bool
             ) -> tuple[np.ndarray, np.ndarray]:
    """CRT: 직전 HTF 봉 레인지의 한쪽을 쓸면 반대 극단이 타깃.

    롱(강세 CRT) = 직전 칸 저점 `pl` 아래로 쓸고 **그 봉 종가가 다시 위로**(reclaim=True).
    reclaim=False 는 09-16 4R 의 D0(터치만) 대조군 — 되찾기 조건이 정보를 더하는지 보는 쌍.
    칸당 **첫 발동 1회만**(CRT 는 봉당 한 번 성립하는 설정이지 연속 상태가 아니다).
    """
    n = len(close)
    sweep_lo = (low < pl) & (~np.isnan(pl))
    sweep_hi = (high > ph) & (~np.isnan(ph))
    if reclaim:
        sweep_lo &= close > pl
        sweep_hi &= close < ph
    up = np.zeros(n, bool); dn = np.zeros(n, bool)
    seen_lo: set[int] = set(); seen_hi: set[int] = set()
    for i in range(n):
        g = gid[i]
        if sweep_lo[i] and g not in seen_lo:
            up[i] = True; seen_lo.add(g)
        if sweep_hi[i] and g not in seen_hi:
            dn[i] = True; seen_hi.add(g)
    return up, dn


def tbs_fire(ts: pd.Series, high: np.ndarray, low: np.ndarray, close: np.ndarray,
             *, cycle: str, quarter: str, offset_min: int = 0, reclaim: bool = True
             ) -> tuple[np.ndarray, np.ndarray]:
    """TBS(quarterly theory): Q1 축적 → **Q2 가 Q1 레인지를 조작(스윕)** → Q3 분배.

    발동 = Q2 의 **마지막 봉**(그래야 진입 `open[i+1]` 이 정확히 Q3 시가다).
    롱 = Q2 안에서 Q1 저점 아래를 쓸었고, Q2 종가가 Q1 저점 위로 되돌아옴(reclaim).
    격자를 offset_min 만큼 밀면 «그 시간 경계가 특별한가»의 대조군이 된다.
    """
    n = len(close)
    cid = grid_id(ts, cycle, offset_min)
    qid = grid_id(ts, quarter, offset_min)
    df = pd.DataFrame({"c": cid, "q": qid, "h": high, "l": low, "cl": close})
    df["qn"] = df.groupby("c")["q"].transform(lambda s: s.rank(method="dense").astype(int))
    up = np.zeros(n, bool); dn = np.zeros(n, bool)
    for _, cyc in df.groupby("c", sort=True):
        q1 = cyc[cyc.qn == 1]; q2 = cyc[cyc.qn == 2]
        if len(q1) == 0 or len(q2) == 0:
            continue
        lo1, hi1 = q1["l"].min(), q1["h"].max()
        last = q2.index[-1]
        swept_lo = bool((q2["l"] < lo1).any()); swept_hi = bool((q2["h"] > hi1).any())
        if reclaim:
            swept_lo &= bool(q2["cl"].iloc[-1] > lo1)
            swept_hi &= bool(q2["cl"].iloc[-1] < hi1)
        if swept_lo and not swept_hi:
            up[last] = True
        if swept_hi and not swept_lo:
            dn[last] = True
    return up, dn


# ────────────────────────────────────────────────────────────────────────────
# 네이티브 배리어 청산 (CRT/TBS 의 본래형: 반대 극단 TP · 스윕 극단 SL)
# ────────────────────────────────────────────────────────────────────────────
def barrier_bp(op: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
               idx: np.ndarray, tp: np.ndarray, sl: np.ndarray, *, long: bool,
               max_bars: int) -> np.ndarray:
    """진입 open[i+1] · intrabar 판정 · **같은 봉에서 둘 다 닿으면 SL 우선**(보수적).

    시간초과는 close[i+max_bars]. 라이브 배리어 컨벤션(evaluate_exit 의 bar_high/low_move)과 동일.
    """
    out = np.empty(len(idx))
    n = len(close)
    for m, i in enumerate(idx):
        e = op[i + 1]
        t, s = tp[m], sl[m]
        end = min(i + max_bars, n - 1)
        px = close[end]
        for j in range(i + 1, end + 1):
            if long:
                if low[j] <= s: px = s; break
                if high[j] >= t: px = t; break
            else:
                if high[j] >= s: px = s; break
                if low[j] <= t: px = t; break
        raw = (px - e) / e * 1e4
        out[m] = raw if long else -raw
    return out


# ────────────────────────────────────────────────────────────────────────────
# 검정력 · 다중검정 귀무
# ────────────────────────────────────────────────────────────────────────────
def mde_bp(vals: np.ndarray, days: np.ndarray) -> float:
    """일군집 SE 기반 최소검출효과(양측 5% · 검정력 80%). «0 이다»와 «못 잰다»를 가른다."""
    uniq, inv = np.unique(days, return_inverse=True)
    k = len(uniq)
    if k < 3:
        return float("nan")
    means = np.array([vals[inv == j].mean() for j in range(k)])
    return float(MDE_Z * means.std(ddof=1) / np.sqrt(k))


def shift_null_passes(op: np.ndarray, cl: np.ndarray, fires: list[np.ndarray],
                      lo: int, hi: int, longs: list[bool], nulls: dict,
                      H: int, B: int = NBOOT_SHIFT) -> tuple[float, float]:
    """발동 인덱스를 순환이동해 «우연히 비용선을 넘는 셀 수»의 귀무분포를 만든다.

    건수·군집 구조는 보존하고 «가격과의 정렬»만 파괴한다(09-16 2R 과 동일 장치).
    """
    n = len(cl)
    counts = np.zeros(B)
    for b in range(B):
        sh = int(RNG.integers(1, n))
        c = 0
        for fire, long in zip(fires, longs):
            idx = (np.flatnonzero(fire) + sh) % n
            idx = idx[(idx >= lo) & (idx <= hi)]
            if len(idx) < 30:
                continue
            g = gross_bp(op, cl, idx, H, long)
            if g.mean() - nulls[(H, long)] > COST_PEG:
                c += 1
        counts[b] = c
    return float(counts.mean()), float(np.percentile(counts, 95))


# ────────────────────────────────────────────────────────────────────────────
def selftest() -> None:
    """발동식 단위검증 — 실데이터 없이 «의도한 봉에서만» 켜지는지 본다."""
    # 1) 인과 스윙: 피벗은 k 봉 뒤에야 알려진다
    h = np.array([1, 2, 5, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], float)
    l = np.ones(15, float)
    sh, sl = causal_swing_levels(h, l, k=2)
    assert np.isnan(sh[3]), "피벗 확정 전에 레벨이 보이면 미래참조"
    assert sh[4] == 5.0, f"p=2 피벗은 p+k=4 에 알려져야 한다: {sh[4]}"

    # 2) BOS/CHoCH: 첫 상방돌파는 (직전 상태가 하락이면) CHoCH, 이어지는 돌파는 BOS
    close = np.array([3, 3, 3, 3, 3, 6, 3, 3, 7, 3], float)
    shv = np.full(10, 5.0); slv = np.full(10, 1.0)
    r = bos_choch(close, shv, slv)
    assert r["bos_up"][5] and not r["choch_up"][5], "state=0 출발의 첫 상방은 BOS"
    assert r["bos_up"][8], "재무장 후 두 번째 상방돌파도 BOS"
    assert not r["bos_up"][6], "돌파 유지 구간은 재발동하지 않는다"

    # 2b) 브레이커: 깨진 양봉 존이 구조돌파 뒤 **첫 터치 1회**만 발동한다
    o_ = np.array([10, 10, 10, 10, 10, 10, 10, 10], float)
    c_ = np.array([11, 6, 6, 20, 11, 11, 11, 11], float)   # p=0 양봉 → 하락변위 → 상방 구조돌파
    h_ = np.array([11, 10, 10, 20, 12, 12, 12, 12], float)
    l_ = np.array([10, 6, 6, 10, 10, 10, 10, 10], float)   # 4·5·6 봉이 존[10,11] 재터치
    a_ = np.full(8, 1.0)
    fu, _ = breaker_touch(o_, h_, l_, c_, a_, np.full(8, 12.0), np.full(8, 1.0), life=6, disp=1.0)
    assert fu.sum() == 1, f"첫 터치 1회만 발동해야 한다: {np.flatnonzero(fu)}"

    # 3) CRT: 직전 칸 저점을 쓸고 종가가 되돌아온 봉 1회만
    ts = pd.Series(pd.date_range("2026-01-01", periods=24, freq="5min"))
    gid = grid_id(ts, "1h")
    hh = np.full(24, 10.0); ll = np.full(24, 9.0); cc = np.full(24, 9.5)
    ll[13] = 8.0; cc[13] = 9.4          # 두 번째 칸에서 직전 칸 저점(9.0) 하향 쓸고 복귀
    ll[15] = 8.5; cc[15] = 9.4          # 같은 칸 두 번째 — 발동하면 안 된다
    ph, pl = prior_cell_range(gid, hh, ll)
    up, dn = crt_fire(gid, hh, ll, cc, ph, pl, reclaim=True)
    assert up[13] and not up[15], "칸당 1회만 발동해야 한다"
    assert not up[:12].any(), "첫 칸은 직전 칸이 없어 발동 불가"
    up0, _ = crt_fire(gid, hh, ll, cc, ph, pl, reclaim=False)
    assert up0[13], "터치만 대조군은 같은 봉을 포함해야 한다(진부분집합 관계)"

    # 4) TBS: Q2 마지막 봉에서만 발동 → 진입이 Q3 시가
    ts2 = pd.Series(pd.date_range("2026-01-01", periods=48, freq="5min"))
    h2 = np.full(48, 10.0); l2 = np.full(48, 9.0); c2 = np.full(48, 9.5)
    l2[15] = 8.0                                    # Q2(01:00~01:55) 안에서 Q1 저점 하향
    up2, _ = tbs_fire(ts2, h2, l2, c2, cycle="4h", quarter="1h")
    fired = np.flatnonzero(up2)
    assert len(fired) == 1 and fired[0] == 23, f"Q2 마지막 봉이어야 한다: {fired}"
    assert ts2.iloc[fired[0] + 1].hour == 2, "진입 봉이 Q3 시가여야 한다"

    # 5) 배리어: SL 우선 규칙
    op = np.array([100, 100, 100, 100], float)
    hi_ = np.array([100, 110, 110, 110], float)
    lo_ = np.array([100, 90, 90, 90], float)
    cl_ = np.array([100, 100, 100, 100], float)
    r1 = barrier_bp(op, hi_, lo_, cl_, np.array([0]), np.array([105.0]), np.array([95.0]),
                    long=True, max_bars=3)
    assert r1[0] < 0, "같은 봉에서 TP·SL 둘 다 닿으면 SL 우선이어야 한다"

    # 6) MDE 는 표본이 늘면 줄어든다
    d = np.repeat(np.arange(50), 4)
    v = RNG.normal(0, 10, 200)
    assert mde_bp(v, d) > mde_bp(np.tile(v, 4), np.repeat(np.arange(200), 4))
    print("selftest OK", flush=True)


def synthetic_frame(n: int) -> pd.DataFrame:
    """랜덤워크 5분봉 — 파이프라인 전 구간 + «귀무에서 통과 0» 보정용. 엣지가 있으면 안 된다."""
    r = RNG.normal(0, 0.0012, n)
    px = 2000 * np.exp(np.cumsum(r))
    hi = px * (1 + np.abs(RNG.normal(0, 0.0008, n)))
    lo = px * (1 - np.abs(RNG.normal(0, 0.0008, n)))
    op = np.r_[px[0], px[:-1]]
    return pd.DataFrame({
        "timestamp": pd.date_range("2022-01-01", periods=n, freq="5min"),
        "open": op, "high": np.maximum.reduce([hi, op, px]),
        "low": np.minimum.reduce([lo, op, px]), "close": px,
        "volume": np.abs(RNG.normal(100, 20, n)),
        "taker_buy_base": np.abs(RNG.normal(50, 10, n)),
    })


def atr_price_of(df: pd.DataFrame, length: int = 288) -> np.ndarray:
    """ATR(가격단위). 배리어·변위 임계에 쓴다 — Wilder RMA, 라이브 규약과 같은 식."""
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean().to_numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--synthetic", type=int, default=0,
                    help="랜덤워크 N봉으로 귀무 보정 실행(실데이터 없이 파이프라인 검증)")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return 0
    selftest()
    OUT.mkdir(parents=True, exist_ok=True)

    print("[1/5] 데이터 …", flush=True)
    if a.synthetic:
        kl = synthetic_frame(a.synthetic); sweep_arms = {}
        print(f"  🔴합성 랜덤워크 {len(kl):,}봉 — 귀무 보정 전용, 실측 아님", flush=True)
    else:
        kl = load(CSV)
        import live_evidence_signal_dashboard_20260823 as EV  # noqa: E402
        sig = EV.compute_signals(kl, btc_df=load(BTC) if BTC.exists() else None, funding_df=None)
        kl = sig
        sweep_arms = {"bot": sig["bottom_liquidity_sweep"].fillna(False).to_numpy(bool),
                      "top": sig["top_liquidity_sweep"].fillna(False).to_numpy(bool)}
        print(f"  {len(kl):,}봉  {kl.timestamp.iloc[0]} ~ {kl.timestamp.iloc[-1]}", flush=True)

    ts = kl["timestamp"]
    op = kl["open"].to_numpy(float); cl = kl["close"].to_numpy(float)
    hi = kl["high"].to_numpy(float); lo_ = kl["low"].to_numpy(float)
    atr = atr_price_of(kl)
    day = ts.dt.floor("D").astype("int64").to_numpy()
    half = ts.iloc[len(kl) // 2]
    n = len(kl); lo, hi_i = WARMUP, n - max(HORIZONS) - 2
    span = (ts.iloc[hi_i] - ts.iloc[lo]).total_seconds() / 86400
    print(f"  평가 {span:.0f}일", flush=True)

    print("[2/5] 발동식 …", flush=True)
    sh, sl = causal_swing_levels(hi, lo_, PIVOT_K)
    st = bos_choch(cl, sh, sl)
    brk_up, brk_dn = breaker_touch(op, hi, lo_, cl, atr, sh, sl)

    # CRT — 격자 3종 × (되찾기 on/off) + 앵커이동 대조군
    crt: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    crt_lvl: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}   # (gid, ph, pl)
    for period, off in (("1h", 0), ("4h", 0), ("24h", 0), ("4h", 120)):
        gid = grid_id(ts, period, off)
        ph, pl = prior_cell_range(gid, hi, lo_)
        tag = f"{period}" + (f"+{off}m" if off else "")
        for rc in (True, False):
            crt[f"{tag}{'' if rc else '_touchonly'}"] = crt_fire(gid, hi, lo_, cl, ph, pl, reclaim=rc)
        crt_lvl[tag] = (gid, ph, pl)

    tbs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cyc, q, off in (("4h", "1h", 0), ("24h", "6h", 0), ("4h", "1h", 30)):
        tag = f"{cyc}/{q}" + (f"+{off}m" if off else "")
        tbs[tag] = tbs_fire(ts, hi, lo_, cl, cycle=cyc, quarter=q, offset_min=off)

    ARMS: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("A_bos", st["bos_up"], st["bos_dn"]),
        ("A_choch", st["choch_up"], st["choch_dn"]),
        ("B_breaker", brk_up, brk_dn),
        ("C_crt_1h", *crt["1h"]),
        ("C_crt_4h", *crt["4h"]),
        ("C_crt_24h", *crt["24h"]),
        ("C_crt_4h_touchonly", *crt["4h_touchonly"]),
        ("C_crt_4h_SHIFT(대조)", *crt["4h+120m"]),
        ("D_tbs_4h_q1h", *tbs["4h/1h"]),
        ("D_tbs_24h_q6h", *tbs["24h/6h"]),
        ("D_tbs_4h_SHIFT(대조)", *tbs["4h/1h+30m"]),
        ("E_crt4h_x_choch", crt["4h"][0] & st["choch_up"], crt["4h"][1] & st["choch_dn"]),
        ("E_crt4h_x_breaker", crt["4h"][0] & brk_up, crt["4h"][1] & brk_dn),
    ]
    if sweep_arms:
        ARMS.append(("Z_sweep_plain(대조)", sweep_arms["bot"], sweep_arms["top"]))

    print("[3/5] 고정지평 검정 …", flush=True)
    NULL = {(H, lg): null_mean(op, cl, lo, hi_i, H, lg) for H in HORIZONS for lg in (True, False)}
    NHIT = {(H, lg): null_hit(op, cl, lo, hi_i, H, lg) for H in HORIZONS for lg in (True, False)}
    rows = []
    for name, f_long, f_short in ARMS:
        for side, fire, long in (("bottom", f_long, True), ("top", f_short, False)):
            idx = np.flatnonzero(np.nan_to_num(fire).astype(bool))
            idx = idx[(idx >= lo) & (idx <= hi_i)]
            if len(idx) < 30:
                rows.append(dict(arm=name, side=side, H=0, n_all=len(idx), note="표본부족")); continue
            for H in HORIZONS:
                g = gross_bp(op, cl, idx, H, long)
                nul = NULL[(H, long)]
                kept = thin_nonoverlap(idx, H)
                ex_k = gross_bp(op, cl, kept, H, long) - nul
                t_ex = (float(ex_k.mean() / (ex_k.std(ddof=1) / np.sqrt(len(ex_k))))
                        if len(ex_k) > 2 and ex_k.std(ddof=1) > 0 else np.nan)
                clo, chi, p = day_boot(g - nul, day[idx])
                tsa = ts.iloc[idx].to_numpy()
                h1 = g[tsa < np.datetime64(half)]; h2 = g[tsa >= np.datetime64(half)]
                win = float(g[g > 0].mean()) if (g > 0).any() else np.nan
                loss = float(-g[g <= 0].mean()) if (g <= 0).any() else np.nan
                acc_need = (loss + COST_PEG) / (win + loss) if win == win and loss == loss else np.nan
                hit = float((g > 0).mean())
                rows.append(dict(
                    arm=name, side=side, H=H, n_all=len(idx), n_block=len(kept),
                    per_day=round(len(idx) / span, 2),
                    fire_pct=round(100 * len(idx) / (hi_i - lo), 2),
                    gross=round(float(g.mean()), 2), null=round(float(nul), 2),
                    excess=round(float(g.mean() - nul), 2),
                    ci_lo=round(clo, 2), ci_hi=round(chi, 2), p=round(p, 4),
                    excess_block=round(float(ex_k.mean()), 2), t_block=round(t_ex, 2),
                    mde=round(mde_bp(g - nul, day[idx]), 2),
                    h1=round(float(h1.mean()), 2) if len(h1) else np.nan,
                    h2=round(float(h2.mean()), 2) if len(h2) else np.nan,
                    hit=round(hit, 4), null_hit=round(NHIT[(H, long)], 4),
                    hit_edge_pp=round(100 * (hit - NHIT[(H, long)]), 2),
                    win=round(win, 2), loss=round(loss, 2),
                    acc_needed=round(acc_need, 4),
                    # ⭐여유 = 실제 적중 − 손익분기 적중. 09-16 의 벽이 정확히 이 칸이었다
                    margin_pp=round(100 * (hit - acc_need), 2) if acc_need == acc_need else np.nan))
        print(f"  · {name}", flush=True)

    D = pd.DataFrame(rows)
    ok = D.H > 0
    D.loc[ok, "q"] = bh_fdr(D.loc[ok, "p"].to_numpy())
    D["PASS"] = (ok & (D.excess_block > COST_PEG) & (D.ci_lo > 0) & (D.t_block.abs() >= 2)
                 & (np.sign(D.h1) == np.sign(D.h2)) & (D.q < 0.10))
    D["background"] = D.fire_pct > BACKGROUND_PCT
    D.to_csv(OUT / "cells.csv", index=False)

    print("[4/5] 네이티브 배리어(CRT 본래형: TP=반대 극단 · SL=스윕 극단) …", flush=True)
    brows = []
    for tag in ("1h", "4h", "24h"):
        gid, ph, pl = crt_lvl[tag]
        bars = {"1h": 12, "4h": 48, "24h": 288}[tag]
        for side, fire, long in (("bottom", crt[tag][0], True), ("top", crt[tag][1], False)):
            idx = np.flatnonzero(fire); idx = idx[(idx >= lo) & (idx <= hi_i - bars)]
            if len(idx) < 30:
                continue
            tp = ph[idx] if long else pl[idx]
            slv = lo_[idx] if long else hi[idx]
            g = barrier_bp(op, hi, lo_, cl, idx, tp, slv, long=long, max_bars=bars)
            clo, chi, p = day_boot(g, day[idx])
            hit = float((g > 0).mean())
            brows.append(dict(arm=f"CRT_{tag}", side=side, n=len(idx),
                              gross=round(float(g.mean()), 2),
                              net=round(float(g.mean()) - COST_PEG, 2),
                              ci_lo=round(clo, 2), ci_hi=round(chi, 2), p=round(p, 4),
                              hit=round(hit, 4),
                              win=round(float(g[g > 0].mean()), 2) if (g > 0).any() else np.nan,
                              loss=round(float(-g[g <= 0].mean()), 2) if (g <= 0).any() else np.nan,
                              mde=round(mde_bp(g, day[idx]), 2)))
    B = pd.DataFrame(brows)
    if len(B):
        B.to_csv(OUT / "barrier.csv", index=False)

    print("[5/5] 다중검정 귀무(순환이동) …", flush=True)
    fires = [f for _, fl, fs in ARMS for f in (fl, fs)]
    longs = [lg for _ in ARMS for lg in (True, False)]
    sh_mean, sh_p95 = shift_null_passes(op, cl, fires, lo, hi_i, longs, NULL, H=48)

    obs48 = int(((D.H == 48) & (D.excess > COST_PEG)).sum())
    summary = dict(
        mode="synthetic" if a.synthetic else "real",
        bars=int(n), span_days=round(span, 1), arms=len(ARMS), cells=int(ok.sum()),
        pass_cells=int(D.PASS.sum()),
        cost_line=COST_PEG,
        h48_cells_over_cost=obs48,
        shift_null_mean=round(sh_mean, 2), shift_null_p95=round(sh_p95, 2),
        background_arms=sorted(D.loc[D.background, "arm"].unique().tolist()),
        median_mde_bp=round(float(D.loc[ok, "mde"].median()), 2),
    )
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    show = ["arm", "side", "H", "n_all", "per_day", "fire_pct", "gross", "excess",
            "excess_block", "t_block", "mde", "hit_edge_pp", "acc_needed", "margin_pp", "PASS"]
    print("\n" + "=" * 150)
    print(D.loc[D.H == 48, show].to_string(index=False))
    if len(B):
        print("\n[배리어 — CRT 본래형]")
        print(B.to_string(index=False))
    print("\n[요약]", json.dumps(summary, ensure_ascii=False))
    print(f"\n산출: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
