"""«상황 읽기 · 30분» — 2026-09-21 사용자가 고른 트레이더 읽기를 그대로 규칙으로 옮긴 것.

원문(대시보드 한 장을 보고 한 읽기):
  ①상승 중 OI↓ + 숏 청산 = 스퀴즈(연료 소진형) ②최대 델타·최대 거래량 봉 = 클라이맥스
  ③현재 봉 고래↓·리테일↑·OI↓ = 분배 ④호가 불균형은 지속률이 낮으면 믿지 않는다
  ⑤가치영역(거래량 무게중심)이 되돌림의 자석 ⑥청산맵 저항 가깝고 지지 멀면 아래 쿠션 없음
  → 시나리오 셋(되돌림/지속/플러시)에 확률, 그리고 **생각을 바꾸는 신호**(사용자: "이게 맞았다").

🔴이 확률은 **휴리스틱**이다(점수표를 정규화한 것). 검증된 값이 아니라 «그 읽기»를 숫자로 적은 것이고,
   그래서 예측을 전부 장부(`situation_log.jsonl`)에 남겨 30분 뒤 결과와 맞춘다 -- 적중률은 화면이 보여준다.
   점수표는 아래 SCORES 한 곳에만 있다. 사용자가 쓰면서 고치면 된다.
ponytail: 부호 규약 -- 매수·매수벽·상승이 양수. 이동 방향(상승/하락)에 대해 대칭으로 뒤집는다.
"""
from __future__ import annotations
import re
from typing import Any

WINDOW = 6                 # «이동»을 재는 완결 봉 수 (30분)
MOVE_THR_FRAC = 0.35       # |이동| > 이 비율 × 창 고저폭 이면 추세. 절대 bp 가 아니라 창 상대
# 🔴2026-09-22 레짐에 **슈미트 트리거**. 단일 임계(0.35)면 분자(이동)와 분모(창폭)가 둘 다 매 봉
#   움직여서 선을 스칠 때마다 d 가 뒤집히고, d 가 뒤집히면 이름·목표·사전확률·순위가 **한꺼번에**
#   바뀐다(사용자 «추세 하락이라다가 갑자기 횡보 위로 이탈»). 4.7년 494,766 결정봉 실측:
#     현행 0.35      변경 36.1%(평균 14분마다) · 그중 **45%가 10~15분 안에 원래대로 되돌아옴**
#     0.45 / 0.25    변경 25.8%(−28%)          · 되돌림 **28%** · 직접반전 1.46→1.57% · 추세비중 59→60%
#   추세 비중이 안 변하므로 BASE 의 기하 기저율 {54,26,20} 도 그대로다(세 설정 모두 동일 실측).
#   ⭐방향 전환에도 ENTER 를 요구한다 -- 안 그러면 히스테리시스가 중립을 건너뛰는 직접 반전을
#     늘린다(첫 구현에서 1.46→2.56% 로 악화됐다).
#   🔴«최소 체류 시간»은 넣지 않는다 -- 09-20 호가 방아쇠 연구에서 역효과였다(얼린 뒤 중립을
#     건너뛰어 직접 반전 4.75→17.75/시간). 공짜인 것은 슈미트 트리거뿐이었다.
TREND_ENTER, TREND_EXIT = 0.45, 0.25
CLIMAX_RECENT = 2          # 최대 델타 봉이 마지막 몇 봉 안에 있어야 «클라이맥스»
REJECT_FRAC = 0.5          # 거부 봉: 반대 델타가 창 최대 델타의 이 비율 이상
CUR_MIN_ELAPSED_S = 60     # 현재 봉 시그니처는 이만큼 지나야 읽는다(반쪽 봉 방지)
OBI_SIDE = 0.30            # |obi| ≥ 이면 «벽»
PERSIST_THIN, PERSIST_THICK = 0.30, 0.40
NEAR_RES_BP = 60.0         # 저항이 이 안이면 «가깝다»
FAR_SUP_RATIO = 3.0        # 지지 거리 > 저항 거리 × 이 배수면 «아래 쿠션 없음»
ACT_HOT = 0.80             # 활동 분위
SYM_K = 0.5                # 대칭 라벨 배리어 = ±k × 창 고저폭. k=1.0 은 30분에 63%가 미도달이라 너무 멀다(09-21 실측)
FUNDING_NEUTRAL = 0.0001   # 바이낸스 펀딩의 이자 성분(0.01%/8h). 이만큼 더 벗어나야 «쏠림»(음수 = 숏 과밀)
BASIS_PCT = 0.75           # |Δ베이시스(창 동안)| 이 링 분포의 이 분위 이상이면 «선물 주도/현물 주도»

# 2026-09-22 사용자 요청 «용어가 복잡하다». 기하로 보면 셋은 서로 다른 현상이 아니라 **한 축**이다:
#   B = 이동 방향으로 더 · A = 반대로 조금(중앙 18bp) · C = 반대로 많이, 출발점까지(중앙 52bp).
#   옛 이름(되돌림·가치영역 재방문 / 플러시·베이스 재방문)은 A 와 C 가 «같은 방향의 얕은 것과
#   깊은 것»이라는 사실을 오히려 감췄다. 원문 용어는 names_long 으로 남겨 툴팁에 띄운다.
SHORT_NAMES = {"A": "조금 되돌림", "B": "더 간다", "C": "출발점까지"}
RANGE_NAMES = {"A": "유지", "B": "위로 이탈", "C": "아래로 이탈"}

# 점수표 -- (근거 라벨 → {시나리오: 점수}). A=되돌림 B=지속 C=플러시(이동 반대쪽 과잉)
SCORES: dict[str, dict[str, int]] = {
    "스퀴즈": {"A": 20, "C": 5}, "신규유입": {"B": 20}, "클라이맥스": {"A": 15, "C": 5},
    "전환탐지": {"A": 5, "C": 5}, "분배": {"A": 10, "C": 10}, "축적": {"B": 10}, "거부봉": {"A": 5, "C": 10},
    # 🔴지속률 0.30~0.40 은 라벨에 «얇음»도 «두꺼움»도 안 붙는데 점수는 얇음(B+3)으로 먹고 있었다
    #   (실측 동방향 벽 47건 중 18건). 두 값의 중간으로 밴드를 명시한다.
    "벽_동방향_두꺼움": {"B": 10}, "벽_동방향_중간": {"B": 6}, "벽_동방향_얇음": {"B": 3}, "벽_역방향": {"A": 5},
    "가치영역_밖": {"A": 5}, "저항근접": {"A": 5, "B": -5}, "쿠션없음": {"C": 5}, "활발": {"B": 5, "C": 5},
    # 횡보 전용 거울상. 추세에서는 방향 게이트 때문에 한쪽만 발동하지만 횡보는 둘 다 가능하고,
    # 그때 «지지 근접»이 상단 이탈(B)을 깎거나 «저항이 멀다»가 아무 효과도 못 내던 비대칭이 있었다.
    "지지근접": {"A": 5, "C": -5}, "쿠션없음_위": {"B": 5},
    # 2026-09-21 마크가격 스트림: 펀딩 = «어느 쪽이 갇혔나», 베이시스 = «누가 주도하나»(선물 프리미엄 확장 = 취약)
    "펀딩_반대쏠림": {"A": 10}, "선물주도": {"A": 8}, "현물주도": {"B": 8},
    # BTC 같은 창 이동: 시장 전체가 같이 갔으면 지속, ETH 만 갔으면 되돌림 쪽
    "BTC_동행": {"B": 8}, "BTC_단독": {"A": 8},
    # CVD 역행(흡수): 가격은 이동 방향으로 갔는데 창 누적 델타는 반대 = 공격 체결이 아니라 수동 흡수가 만든 이동.
    # 테이프는 «같은 초» 정보라(09-20 실측) 작게만 준다
    "CVD_역행": {"A": 5},
}
# 🔴2026-09-22 추세도 기하 기저율로 다시 깔았다(횡보에 BASE_RANGE 로 한 것과 같은 일).
#   균등(34/33/33)이 문제의 뿌리였다 -- 추세의 기하는 균등이 아니다. 4.7년 5분 패널의
#   **추세 창 291,596개** × 라이브가 실제로 쓰는 목표 거리 182쌍 = 5,307만 판정에서
#   닿은 것만 분모로 A 53.6 · B 26.3 · C 20.1% 다(미도달 37%, 판정불가 1.1% 제외).
#   거리 사분위를 바꿔도 A 56~67 · B 16~21 · C 15~23 으로 **순서가 안 뒤집힌다**.
#   ⭐이건 원장의 «결과»를 한 번도 쓰지 않았다 -- 거리만 빌렸다. 독립 창 14개에 계수를
#     맞추는 과적합과 다른 범주다(§14-6 이 금지한 것은 후자다).
#   ⚠️사전확률 교체일 뿐 **거리 맹점의 해결이 아니다** -- 점수표는 여전히 «30분에 그
#     거리를 갈 수 있나»를 안 나눈다(미도달이 거리에 따라 23~60% 로 움직인다).
#   재현: scripts/research_situation_geometry_base_rates_20260922.py (같은 코드로 횡보를
#   돌리면 BASE_RANGE 의 근거인 «유지 16.0%»가 16.1% 로 재현된다 -- 자를 먼저 검증했다).
BASE = {"A": 54, "B": 26, "C": 20}
# 🔴횡보는 기저율이 전혀 다르다. «레인지 유지»는 배리어가 아니라 **잔여**(둘 다 안 닿음)이고,
#   4.7년 5분 패널 **203,577건**에서 실제로 유지되는 비율은 **16.0%**(연도별 15.1~17.3%로 안정).
#   34 를 주면 카드가 «레인지 유지»를 1순위로 부르고(원장 횡보 147분 중 141건=96%) 실제로는 8% 만 일어난다.
#   양쪽 다 닿는 11.1% 는 상/하로 반씩 배분했다(37.1+5.5 / 35.8+5.5).
#   ⭐이건 **라벨 정의의 기하 기저율**이지 이 표본을 맞춘 값이 아니다 -- §14-6 이 금지한
#   «작은 표본으로 계수 맞추기»와 다른 범주다(독립 창 15개로는 어떤 계수도 판정 못 한다).
BASE_RANGE = {"A": 16, "B": 43, "C": 41}
# 🔴2026-09-22 그 16 은 **상수가 아니다**. 이탈 배리어가 창 고저 그 자체라 창이 넓으면 배리어가 멀고,
#   같은 패널에서 유지율이 창 폭을 따라 단조로 움직인다(PRE-OOS 179,563건 -> OOS 19,411건 재현):
#     range_bp   <20   20~30  30~40  40~55  55~75  >=75
#     유지        5.6    8.0   10.8   13.7   17.4   26.1  %   (OOS 6.4 10.0 11.8 19.0 23.4 30.7)
#   상수 16 은 좁은 창에서 3배 과대, 넓은 창에서 크게 과소다. garch 분위를 고정해도 5칸 전부에서
#   +11.6~+22.0pp 로 살아남는다(scripts/research_situation_within_regime_feature_screen_20260922.py ctrl).
#   B·C 는 BASE_RANGE 의 43:41 비율을 그대로 나눠 갖는다 -- 창 폭은 «이탈하나»를 말하지 «어느 쪽»은 말하지 않는다
#   (같은 패널에서 횡보의 **방향**을 가르는 피쳐는 152개 중 하나도 없었다).
HOLD_BY_RANGE = ((20, 6), (30, 8), (40, 11), (55, 14), (75, 17), (float("inf"), 26))


def base_range(rng_bp: float) -> dict[str, int]:
    """횡보 사전확률. «유지»(A)는 창 폭에 따라 6~26%. rng_bp=창 고저폭(bp)."""
    a = next(v for edge, v in HOLD_BY_RANGE if rng_bp < edge)
    rest = 100 - a
    b = round(rest * BASE_RANGE["B"] / (BASE_RANGE["B"] + BASE_RANGE["C"]))
    return {"A": a, "B": b, "C": rest - b}


def _ahead(px: float | None, mid: float, above: bool) -> float | None:
    """목표가 현재가의 **기대 방향**에 있어야 한다. 이미 지나갔으면 None(해당 없음).
    🔴그러지 않으면 첫 봉에서 공짜로 닿는다 -- 09-21 검토 실측으로 d≠0 예측의 **18.6%**가
      목표를 선점한 상태였고, 해결 1,055건 중 130건(12%)이 예측 시점에 이미 결정돼 있었다."""
    if px is None:
        return None
    return px if ((px > mid) if above else (px < mid)) else None


def _sign(x: float | None) -> int:
    return 0 if x is None else (1 if x > 0 else -1 if x < 0 else 0)


def classify(inp: dict[str, Any]) -> dict[str, Any]:
    """inp:
      bars   완결 5분봉 오래된→최신, 각 {time, high, low, close, delta, vol, whale_net, retail_net,
             oi_delta(None 허용), liq_long, liq_short}
      levels {가격: 거래량} 창 전체(가치영역용)
      cur    {elapsed_s, whale_net, retail_net, oi_delta, delta}
      book   {obi, persist_share}   act_pct   sr {res, sup, res_bp, sup_bp}
      breakout {detect_on, prewarn_on}   mid
      deriv  {funding, basis_bp, basis_d_bp, basis_thr_bp} (마크가격 링, 없으면 생략)
      btc    {move_bp, range_bp} 같은 창의 BTC (없으면 생략). 추세 판정은 ETH 와 같은 창 상대 규칙
      sr.sup_levels / sr.res_levels  청산맵 [{price, weight_pct}] 가까운 순 (플러시 목표용, 없으면 베이스)
    반환: 라벨·근거·시나리오(확률+목표)·뒤집기 신호(현재 판정 포함)."""
    bars = [b for b in inp.get("bars", []) if b.get("close")]
    if len(bars) < WINDOW + 1:
        return {"ok": False, "reason": f"완결 봉 {len(bars)} < {WINDOW + 1}"}
    w = bars[-WINDOW:]
    mid = float(inp.get("mid") or w[-1]["close"])
    labels: list[str] = []
    ev: dict[str, Any] = {}

    # ── 이동 ──
    ev["mid"] = mid
    rng_bp = (max(b["high"] for b in w) - min(b["low"] for b in w)) / mid * 1e4
    move_bp = (w[-1]["close"] - bars[-WINDOW - 1]["close"]) / bars[-WINDOW - 1]["close"] * 1e4
    # 슈미트 트리거. 이전 레짐은 호출자가 넘긴다(이 함수는 순수하게 둔다).
    ratio = abs(move_bp) / rng_bp if rng_bp > 0 else 0.0
    sgn = 1 if move_bp > 0 else (-1 if move_bp < 0 else 0)
    prev = int(inp.get("prev_dir") or 0)
    if prev == 0:
        d = sgn if ratio > TREND_ENTER else 0
    elif sgn == prev:
        d = 0 if ratio < TREND_EXIT else prev
    else:                                   # 부호가 뒤집혔다 -- 반전에도 ENTER 를 요구한다
        d = sgn if ratio > TREND_ENTER else (0 if ratio < TREND_EXIT else prev)
    # 다음에 상태를 바꿀 문턱과 거기까지의 여유. 화면이 «곧 바뀔 수 있나»를 보이는 데 쓴다.
    thr_next = TREND_EXIT if (d != 0 and sgn == d) else TREND_ENTER
    margin = (ratio - thr_next) if (d != 0 and sgn == d) else (thr_next - ratio)
    thr = thr_next * rng_bp
    ev.update(move_bp=round(move_bp, 1), range_bp=round(rng_bp, 1), dir=d,
              move_ratio=round(ratio, 3), thr_next=thr_next, margin=round(margin, 3))
    up, dn = "상승", "하락"
    labels.append({1: f"{up} {move_bp:+.0f}bp", -1: f"{dn} {move_bp:+.0f}bp", 0: f"횡보 (±{thr:.0f}bp 안)"}[d])

    # ── 연료: 이동 구간의 OI 와 청산 ──
    ois = [b.get("oi_delta") for b in w if b.get("oi_delta") is not None]
    oi_sum = sum(ois) if ois else None
    oi_agree = (sum(1 for x in ois if _sign(x) == _sign(oi_sum)) / len(ois)) if ois else 0.0
    liq_l, liq_s = sum(b.get("liq_long") or 0 for b in w), sum(b.get("liq_short") or 0 for b in w)
    ev.update(oi_sum=None if oi_sum is None else round(oi_sum, 1), oi_agree=round(oi_agree, 2),
              liq_long=round(liq_l), liq_short=round(liq_s))
    fuel = "모름"
    if d != 0 and oi_sum is not None:
        if _sign(oi_sum) < 0:
            fuel = "스퀴즈" if d > 0 else "롱이탈"
            side_liq = liq_s if d > 0 else liq_l
            other = liq_l if d > 0 else liq_s
            labels.append(("숏 스퀴즈" if d > 0 else "롱 청산·이탈") + f" (OI {oi_sum:+.0f}, 봉 {oi_agree:.0%} 같은 부호)"
                          + (" · 청산 동반" if side_liq > other and side_liq > 0 else ""))
        else:
            fuel = "신규유입"
            labels.append(("신규 롱 유입" if d > 0 else "신규 숏 유입") + f" (OI {oi_sum:+.0f})")
    ev["fuel"] = fuel

    # ── 클라이맥스 · 거부 봉 ──
    deltas = [b.get("delta") or 0.0 for b in w]
    vols = [b.get("vol") or 0.0 for b in w]
    imax = max(range(len(w)), key=lambda i: abs(deltas[i]))
    climax = (d != 0 and imax >= len(w) - CLIMAX_RECENT and _sign(deltas[imax]) == d and vols[imax] >= max(vols) * 0.999)
    if climax:
        labels.append(f"클라이맥스 (델타 {deltas[imax]:+.0f}, 창 최대 거래량)")
    cvd = sum(deltas)
    cvd_div = d != 0 and _sign(cvd) == -d
    if cvd_div:
        labels.append(f"CVD 역행 (창 델타 {cvd:+.0f} vs {'상승' if d > 0 else '하락'}) · 흡수")
    last = w[-1]
    reject = (d != 0 and _sign(last.get("delta") or 0) == -d and abs(last.get("delta") or 0) >= REJECT_FRAC * abs(deltas[imax]))
    if reject:
        labels.append(f"거부 봉 (마지막 봉 델타 {last['delta']:+.0f})")
    bo = inp.get("breakout") or {}
    if bo.get("detect_on"):
        labels.append("전환 탐지 켜짐")
    elif bo.get("prewarn_on"):
        labels.append("전환 예고")
    ev.update(cvd=round(cvd), cvd_div=cvd_div, climax=climax, reject=reject,
              last_delta=round(last.get("delta") or 0.0, 1), max_delta=round(deltas[imax], 1), breakout_detect=bool(bo.get("detect_on")), breakout_prewarn=bool(bo.get("prewarn_on")))

    # ── 현재 봉 시그니처 ──
    cur = inp.get("cur") or {}
    sig = "미판정"
    if (cur.get("elapsed_s") or 0) >= CUR_MIN_ELAPSED_S:
        wn, rn, oc = cur.get("whale_net") or 0.0, cur.get("retail_net") or 0.0, cur.get("oi_delta")
        if wn < 0 and rn > 0:
            sig = "분배"
        elif wn > 0 and rn < 0:
            sig = "축적"
        else:
            sig = "동조" if _sign(wn) == _sign(rn) and wn != 0 else "중립"
        labels.append(f"현재 봉 {sig} (고래 {wn:+.0f} · 리테일 {rn:+.0f}"
                      + (f" · OI {oc:+.0f}" if oc is not None else "") + ")")
    # 🔴판정 결과(cur_sig)만 남기면 나중에 CUR_MIN_ELAPSED_S·부호 규칙을 바꿔볼 수 없다 -- 원본을 같이 남긴다
    ev.update(cur_sig=sig, cur_elapsed_s=cur.get("elapsed_s"), cur_whale=cur.get("whale_net"),
              cur_retail=cur.get("retail_net"), cur_oi=cur.get("oi_delta"))

    # ── 호가 질 ──
    book = inp.get("book") or {}
    obi, pers = book.get("obi"), book.get("persist_share")
    wall = 0 if obi is None or abs(obi) < OBI_SIDE else _sign(obi)
    thick = None if pers is None else (pers >= PERSIST_THICK)
    thin = None if pers is None else (pers < PERSIST_THIN)
    if wall:
        labels.append(("매수벽" if wall > 0 else "매도벽") + f" {obi:+.2f}"
                      + (" · 얇음(믿지 말 것)" if thin else " · 두꺼움" if thick else " · 보통")
                      + (f" · 지속 {pers:.0%}" if pers is not None else ""))
    ev.update(wall=wall, obi=obi, persist=pers)

    # ── 가치영역 ──
    levels = {float(k): float(v) for k, v in (inp.get("levels") or {}).items() if v}
    va_lo = va_hi = None
    if len(levels) >= 3:
        top = sorted(levels.items(), key=lambda kv: -kv[1])[:2]
        va_lo, va_hi = min(p for p, _ in top), max(p for p, _ in top)
        pos = "위" if mid > va_hi else ("아래" if mid < va_lo else "안")
        labels.append(f"가치영역 {va_lo:.0f}~{va_hi:.0f} 의 {pos}")
    ev.update(va_lo=va_lo, va_hi=va_hi)

    # ── S/R 비대칭 ──
    sr = inp.get("sr") or {}
    res_bp, sup_bp = sr.get("res_bp"), sr.get("sup_bp")
    near_res = res_bp is not None and 0 < res_bp <= NEAR_RES_BP and d >= 0
    near_sup = sup_bp is not None and 0 < sup_bp <= NEAR_RES_BP and d <= 0
    no_cushion = (res_bp and sup_bp and (sup_bp > FAR_SUP_RATIO * res_bp if d >= 0 else res_bp > FAR_SUP_RATIO * sup_bp))
    if near_res or near_sup:
        labels.append(("저항" if near_res else "지지") + f" {(res_bp if near_res else sup_bp):.0f}bp 근접")
    # 횡보는 위아래가 따로다 -- 거울상을 따로 잰다(추세에서는 no_cushion 하나로 충분하다)
    no_cushion_up = bool(d == 0 and res_bp and sup_bp and res_bp > FAR_SUP_RATIO * sup_bp)
    if no_cushion or no_cushion_up:
        labels.append("이동 반대쪽 청산 쿠션 없음" if d != 0
                      else ("위쪽 청산 쿠션 없음" if no_cushion_up else "아래쪽 청산 쿠션 없음"))
    act = inp.get("act_pct")
    hot = act is not None and act >= ACT_HOT
    if hot:
        labels.append(f"활발 (시간대 상위 {100 - int(act * 100)}%) · 큰 움직임 임박")
    # near_*/no_cushion/hot 은 전부 임계 통과 결과다. 원본(거리·활동분위)이 없으면 임계를 못 바꾼다.
    ev.update(near_res=near_res, near_sup=near_sup, no_cushion=bool(no_cushion), no_cushion_up=no_cushion_up, hot=hot,
              act_pct=act, res_bp=res_bp, sup_bp=sup_bp)

    # ── 펀딩 · 베이시스 (마크가격 스트림) ──
    dv = inp.get("deriv") or {}
    fr, bas, bd, bthr = dv.get("funding"), dv.get("basis_bp"), dv.get("basis_d_bp"), dv.get("basis_thr_bp")
    crowd = 0 if fr is None or abs(fr - FUNDING_NEUTRAL) < FUNDING_NEUTRAL else _sign(fr - FUNDING_NEUTRAL)
    trapped = d != 0 and crowd == -d                       # 이동 반대편이 과밀 = 갇힌 쪽이 연료
    lead = 0 if (d == 0 or bd is None or bthr is None or abs(bd) < bthr) else (1 if _sign(bd) == d else -1)
    if fr is not None:
        labels.append(f"펀딩 {fr * 100:+.4f}%" + {1: " (롱 쏠림)", -1: " (숏 쏠림)", 0: ""}[crowd] + (" · 갇힘" if trapped else "")
                      + (f" · 베이시스 {bas:+.1f}bp" if bas is not None else "")
                      + (f" (창 {bd:+.1f} · " + {1: "선물 주도", -1: "현물 주도", 0: "중립"}[lead] + ")" if bd is not None else ""))
    ev.update(funding=fr, crowd=crowd, trapped=trapped, basis_bp=bas, basis_d_bp=bd, lead=lead,
              basis_thr_bp=bthr)   # 임계(링 p75)도 남긴다 -- lead 만으론 분위를 재조정할 수 없다

    # ── BTC 같은 창 이동 ──
    btc = inp.get("btc") or {}
    bm, brg = btc.get("move_bp"), btc.get("range_bp")
    d_btc = None if bm is None or brg is None else (1 if bm > MOVE_THR_FRAC * brg else -1 if bm < -MOVE_THR_FRAC * brg else 0)
    btc_rel = None
    if d != 0 and d_btc is not None:
        btc_rel = "동행" if d_btc == d else "단독"
        labels.append(f"BTC 동행 {bm:+.0f}bp (시장 전체)" if btc_rel == "동행" else f"ETH 단독 이동 (BTC {bm:+.0f}bp)")
    ev.update(btc_move_bp=None if bm is None else round(bm, 1), btc_dir=d_btc, btc_rel=btc_rel)

    # ── 시나리오 점수 ──
    sc = dict(base_range(rng_bp) if d == 0 else BASE)
    why: list[tuple[str, dict[str, int]]] = []

    def add(key: str) -> None:
        for k, v in SCORES[key].items():
            sc[k] += v
        why.append((key, SCORES[key]))

    if fuel == "스퀴즈" or fuel == "롱이탈":
        add("스퀴즈")
    elif fuel == "신규유입":
        add("신규유입")
    if climax: add("클라이맥스")
    if bo.get("detect_on"): add("전환탐지")
    if sig == "분배" and d > 0 or sig == "축적" and d < 0: add("분배")     # 이동 반대편이 받는 중
    if sig == "축적" and d > 0 or sig == "분배" and d < 0: add("축적")     # 이동 편이 더 사는 중
    if reject: add("거부봉")
    if cvd_div: add("CVD_역행")
    if wall and d != 0:
        if wall == d: add("벽_동방향_두꺼움" if thick else "벽_동방향_얇음" if thin else "벽_동방향_중간")
        else: add("벽_역방향")
    if va_lo is not None and (mid > va_hi or mid < va_lo): add("가치영역_밖")
    if d == 0:
        # 횡보에서 B=상단 이탈·C=하단 이탈이므로 근접한 쪽이 **그쪽 이탈**을 깎아야 한다.
        if near_res: add("저항근접")      # 위가 막혀 있다 → 상단 이탈 −5
        if near_sup: add("지지근접")      # 아래가 받쳐 있다 → 하단 이탈 −5 (전에는 B 를 깎고 있었다)
        if no_cushion: add("쿠션없음")    # 아래 쿠션 없음 → 하단 이탈 +5
        if no_cushion_up: add("쿠션없음_위")   # 위 쿠션 없음 → 상단 이탈 +5 (전에는 영원히 0)
    else:
        if near_res or near_sup: add("저항근접")
        if no_cushion: add("쿠션없음")
    if hot: add("활발")
    if trapped: add("펀딩_반대쏠림")
    if lead > 0: add("선물주도")
    if lead < 0: add("현물주도")
    if btc_rel == "동행": add("BTC_동행")
    if btc_rel == "단독": add("BTC_단독")
    tot = sum(max(v, 1) for v in sc.values())
    prob = {k: round(max(v, 1) / tot * 100) for k, v in sc.items()}

    # ── 목표 ──
    # 플러시 목표 = 이동이 시작되기 **전** 30분의 극값(«베이스»). 원문 읽기의 «00:15~00:20 베이스»가 그것이다.
    # 이동 창 안의 극값을 쓰면 되돌림 목표와 겹쳐 뜻이 없어진다. 앞 창이 없으면 이동 창 앞 절반으로 떨어진다.
    pre = bars[-2 * WINDOW:-WINDOW] or w[: len(w) // 2]
    base_px = min(b["low"] for b in pre) if d > 0 else max(b["high"] for b in pre)
    res_px, sup_px = sr.get("res"), sr.get("sup")
    if d >= 0:
        cont = res_px if (res_px and res_px > mid) else max(b["high"] for b in w)
        names = SHORT_NAMES
        names_long = {"A": "되돌림 · 가치영역 재방문", "B": "지속 · 저항 테스트", "C": "플러시 · 베이스 재방문"}
    else:
        cont = sup_px if (sup_px and sup_px < mid) else min(b["low"] for b in w)
        names = SHORT_NAMES
        names_long = {"A": "되돌림 · 가치영역 재방문", "B": "지속 · 지지 테스트", "C": "역스퀴즈 · 고점 재방문"}
    if d == 0:
        names = RANGE_NAMES
        names_long = {"A": "레인지 유지", "B": "상단 이탈", "C": "하단 이탈"}
        cont = max(b["high"] for b in w); base_px = min(b["low"] for b in w)   # 횡보는 창 자체가 레인지
    # 🔴청산 군집을 플러시 목표로 쓰던 코드를 제거했다(09-21 아침에 넣고 저녁에 되돌림).
    #   실측: 군집 목표는 |거리| 중앙 **156.7bp** 에 경로 도달 **0/444**, 베이스 목표는 59.4bp 에 250/646(38.7%).
    #   청산맵이 «어디서 청산이 터지나»를 맞히는 것은 사실이지만(§9), 그 자리가 **30분 안에 닿는 자리는 아니다**.
    #   목표는 «닿을 수 있는 곳»이어야 한다 -- 아니면 그 시나리오는 정의상 일어나지 않는다.
    ev["c_target_src"] = "베이스"
    # 🔴A(되돌림) 목표는 가치영역의 **먼 변**이다. 가까운 변을 쓰면 현재가에서 중앙 1bp 라
    #   시나리오 경주가 아니라 «가장 가까운 목표가 이긴다»가 채점된다 -- 09-21 장부 실측으로
    #   밖에 있을 때도 최근접이 81% 이겼고 A 가 73% 였다. 먼 변은 «띠를 통과했다»로 뜻이 분명하고
    #   B·C 와 거리가 가장 비슷해진다(재채점: A 73→40%, 거리 중앙 10.7→20.7bp).
    #   횡보(d=0)의 «레인지 유지»는 배리어가 아니라 **잔여**다 -- 목표 없음(둘 다 안 닿으면 A).
    a_raw = None if (va_lo is None or d == 0) else (va_lo if d > 0 else va_hi)
    # 셋 다 «아직 안 지나간» 쪽에 있어야 채점이 성립한다. A 는 이동 반대, B 는 이동 쪽, C 는 이동 반대.
    a_px = _ahead(a_raw, mid, above=(d < 0))
    b_px = _ahead(round(float(cont), 2), mid, above=(d >= 0))
    c_px = _ahead(round(float(base_px), 2), mid, above=(d < 0))
    targets = {"A": a_px, "B": b_px, "C": c_px}
    # 🔴표시 전용. targets 의 None 은 «이 목표는 채점하지 않는다»는 뜻이라(P0 수정) 되돌리면 안 된다.
    #   그런데 숫자를 지우면 «어떻게 지나갔는지»를 화면에서 못 본다(2026-09-22 사용자). 원래 기하값을
    #   따로 실어 보내고, 화면은 그 값에 «지남»을 붙여 그린다. resolve/_touched 는 targets 만 본다.
    targets_raw = {"A": (None if a_raw is None else round(float(a_raw), 2)),
                   "B": round(float(cont), 2), "C": round(float(base_px), 2)}
    # 하나라도 선점됐으면 이 예측은 «배리어 경주»로 답할 수 없다 -- 적중률 집계에서 뺀다(표시는 그대로).
    passed = [k for k, raw in (("A", a_raw), ("B", cont), ("C", base_px))
              if raw is not None and targets[k] is None and not (k == "A" and a_raw is None)]
    scorable = not passed
    # 대칭 라벨(학습의 주 타깃): 거리가 같아 근접 편향이 없다. 표시는 Y_geo, 학습·판정은 이쪽.
    rng_px = rng_bp / 1e4 * mid
    sym = {"k": SYM_K, "up": round(mid + SYM_K * rng_px, 2), "dn": round(mid - SYM_K * rng_px, 2)}
    # 부호를 남긴다(abs 로 지우면 «선점»을 나중에 다시 못 잰다). +는 현재가 위, −는 아래.
    dist_bp = {k: (None if targets[k] is None else round((targets[k] - mid) / mid * 1e4, 1)) for k in ("A", "B", "C")}

    # ── 생각을 바꾸는 신호 (실시간 판정) ──
    lastb = w[-1]; prevb = w[-2]
    cur_ok = (cur.get("elapsed_s") or 0) >= CUR_MIN_ELAPSED_S
    if d > 0:
        flips = [
            ("OI 가 가격 상승과 함께 양수 전환 (신규 롱)", (lastb.get("oi_delta") or 0) > 0 and lastb["close"] > prevb["close"], "B"),
            ("고래 순수급 양수 전환 (현재 봉)", cur_ok and (cur.get("whale_net") or 0) > 0, "B"),
            ("매수벽 지속률 ≥ 40%", wall > 0 and bool(thick), "B"),
            ("가치영역 하단 이탈 + OI 감소", va_lo is not None and mid < va_lo and (lastb.get("oi_delta") or 0) < 0, "C"),
            ("저항 돌파 + OI 증가", bool(res_px) and mid > res_px and (lastb.get("oi_delta") or 0) > 0, "B"),
        ]
    elif d < 0:
        flips = [
            ("OI 가 가격 하락과 함께 양수 전환 (신규 숏)", (lastb.get("oi_delta") or 0) > 0 and lastb["close"] < prevb["close"], "B"),
            ("고래 순수급 음수 전환 (현재 봉)", cur_ok and (cur.get("whale_net") or 0) < 0, "B"),
            ("매도벽 지속률 ≥ 40%", wall < 0 and bool(thick), "B"),
            ("가치영역 상단 이탈 + OI 감소", va_hi is not None and mid > va_hi and (lastb.get("oi_delta") or 0) < 0, "C"),
            ("지지 붕괴 + OI 증가", bool(sup_px) and mid < sup_px and (lastb.get("oi_delta") or 0) > 0, "B"),
        ]
    else:
        flips = [
            ("OI 증가 + 상단 근접", (lastb.get("oi_delta") or 0) > 0 and mid >= max(b["high"] for b in w) * 0.999, "B"),
            ("OI 증가 + 하단 근접", (lastb.get("oi_delta") or 0) > 0 and mid <= min(b["low"] for b in w) * 1.001, "C"),
            ("활발 전환", bool(hot), "B"),
        ]
    # 🔴화면의 «현재 상황» 막대는 이 임계들을 기준선으로 그린다. 클라이언트가 같은 숫자를
    #   다시 선언하면 여기를 고치는 날 조용히 어긋난다 -- 쓰는 쪽에 보낸다.
    ev["thr_ui"] = {"obi": OBI_SIDE, "reject": REJECT_FRAC, "act": ACT_HOT, "near_bp": NEAR_RES_BP,
                    "cushion": FAR_SUP_RATIO, "persist_thin": PERSIST_THIN, "persist_thick": PERSIST_THICK,
                    "funding": FUNDING_NEUTRAL}
    return {"ok": True, "dir": d, "labels": labels, "evidence": ev, "prob": prob, "names": names, "names_long": names_long,
            "regime": {"ratio": round(ratio, 3), "thr": thr_next, "margin": round(margin, 3),
                       "enter": TREND_ENTER, "exit": TREND_EXIT},
            "targets": targets, "targets_raw": targets_raw,
            "sym": sym, "dist_bp": dist_bp, "scorable": scorable, "passed": passed,
            "why": [{"근거": k, **v} for k, v in why],
            "flips": [{"signal": s, "on": bool(o), "toward": t} for s, o, t in flips]}


_NUM = re.compile(r"[-+]?\d[\d.,]*%?")


def log_key(res: dict[str, Any]) -> tuple:
    """장부에 «새 예측»으로 기록할지 가르는 상태 서명 = 라벨의 **종류** + 1순위 시나리오.
    🔴라벨의 숫자(고래 −33 · 펀딩 +0.004%)를 지운다 -- 숫자를 두면 5초마다 다른 키가 되어 3시간에 1,742건이
    쌓였고(09-21 실측), 메모리 상한 300건이 30분 안에 밀려나 해결 시점에 남는 예측이 0건이었다."""
    return (tuple(_NUM.sub("#", x) for x in res["labels"]), max(res["prob"], key=res["prob"].get))


def _touched(d: int, c: dict[str, float], tg: dict[str, Any]) -> list[str]:
    """이 봉에서 성립한 시나리오들. 🔴A·C 와 B 의 판정 기준이 **다르고, 그게 의도다**(사용자 결정 09-21):
      A(되돌림)·C(플러시) = «그 자리에 **닿았나**» → 고가/저가. 레벨 도달은 꼬리로도 성립한다.
      B(지속)     = «저항을 **뚫었나**» → **종가**. 저항을 꼬리로 스치는 것은 돌파가 아니라 거부다.
    이 구분이 SCORES["저항근접"]={"A":+5,"B":−5} 와 뜻이 맞는다 -- 가까운 저항은 «닿기»는 쉽고 «뚫기»는 어렵다.
    (그 전에는 B 도 고가 기준이라 «저항이 가까우면 B 를 깎는다»가 라벨과 정반대였다.)"""
    hit = []
    if tg.get("A") is not None and ((d > 0 and c["low"] <= tg["A"]) or (d < 0 and c["high"] >= tg["A"])):
        hit.append("A")
    if tg.get("B") is not None and ((d >= 0 and c["close"] > tg["B"]) or (d < 0 and c["close"] < tg["B"])):
        hit.append("B")
    if tg.get("C") is not None and ((d >= 0 and c["low"] <= tg["C"]) or (d < 0 and c["high"] >= tg["C"])):
        hit.append("C")
    return hit


def resolve(pred: dict[str, Any], candles: list[dict[str, float]], horizon_s: int = 1800,
            bar_s: int = 60) -> str | None:
    """예측 뒤 horizon 안에 어느 목표가 **먼저** 닿았나. 아직이면 None.
    🔴봉은 **1분**이어야 한다 -- 5분봉이면 창 앞 최대 5분(30분의 17%)이 통째로 버려지고 봉 안 순서를
      몰라 오채점된다(09-21 실측: 같은 장부를 5분↔1분으로 풀면 13.4%가 달라졌다).
    'amb' = 같은 봉에 둘 이상(판정 불가) · 'none' = 아무것도 안 닿음 · 횡보의 A 는 잔여."""
    t0, t1 = pred["ts"], pred["ts"] + horizon_s
    if not candles or candles[0]["time"] > t0 or candles[-1]["time"] + bar_s < t1:
        return None                      # 창의 **앞뒤 양쪽**을 덮어야 한다. 머리를 안 보면 조용히 오답을 확정한다
    d, tg = pred.get("dir", 0), pred.get("targets") or {}
    if isinstance(tg.get("A"), (list, tuple)):    # 09-21 이전 장부(띠)는 먼 변으로 읽는다
        tg = dict(tg, A=(tg["A"][0] if d > 0 else tg["A"][1] if d < 0 else None))
    # 🔴«첫 터치가 이긴다»로 가르면 **C 가 영원히 0** 이다 -- 추세에서 A(가치영역 먼 변)와 C(베이스)는
    #   둘 다 이동 반대쪽이고 A 가 더 가깝다(실측 1,090건 전부 같은 쪽, 1,071건에서 A 가 더 가까움).
    #   C 에 닿으려면 A 를 지나야 하므로 선착 규칙이 C 를 구조적으로 불가능하게 만든다(실측 C 0건,
    #   경로로 세면 22.9% 도달). 시나리오는 «배타적 배리어»가 아니라 **같은 쪽에서 중첩**된다.
    #   ⇒ 쪽끼리는 선착으로 가르고, **이긴 쪽 안에서는 더 깊이 간 것**이 그 창의 답이다.
    win = [c for c in candles if t0 < c["time"] < t1]
    first: dict[str, int] = {}
    for i, c in enumerate(win):
        for k in _touched(d, c, tg):
            first.setdefault(k, i)
    if not first:
        return "A" if d == 0 else "none"
    mid = pred.get("mid")
    if not mid:
        return min(first, key=lambda k: (first[k], {"C": 0, "B": 1, "A": 2}[k]))   # 옛 항목: 쪽을 모른다
    side = lambda k: 1 if tg[k] > mid else -1                                       # noqa: E731
    earliest = min(first.values())
    front = [k for k in first if first[k] == earliest]
    if len({side(k) for k in front}) > 1:
        return "amb"                       # 같은 봉에 **양쪽** -- 봉 안 순서를 모른다
    s_win = side(front[0])
    opp = [first[k] for k in first if side(k) != s_win]
    cut = min(opp) if opp else len(win)    # 반대쪽이 넘겨받기 전까지만 «더 깊이»를 인정한다
    same = [k for k in first if side(k) == s_win and first[k] < cut] or front
    return max(same, key=lambda k: abs(tg[k] - mid))


def resolve_sym(pred: dict[str, Any], candles: list[dict[str, float]], horizon_s: int = 1800,
                bar_s: int = 60) -> str | None:
    """대칭 라벨 = ±k×창고저폭 중 먼저 닿은 쪽. **거리가 같아 근접 편향이 없다** -- 학습의 주 타깃이고
    «엔진이 방향을 맞히나»를 정직하게 재는 유일한 지표다."""
    sym = pred.get("sym") or {}
    if not (sym.get("up") and sym.get("dn")):
        return None
    t0, t1 = pred["ts"], pred["ts"] + horizon_s
    if not candles or candles[0]["time"] > t0 or candles[-1]["time"] + bar_s < t1:
        return None
    for c in candles:
        if c["time"] <= t0 or c["time"] >= t1:
            continue
        up, dn = c["high"] >= sym["up"], c["low"] <= sym["dn"]
        if up and dn:
            return "amb"
        if up:
            return "up"
        if dn:
            return "down"
    return "none"


def path_targets(pred: dict[str, Any], candles: list[dict[str, float]], horizon_s: int = 1800,
                 bar_s: int = 60) -> dict[str, float] | None:
    """연속 타깃(학습 표본용): 창 끝 수익 · 창 안 최대 상승폭 · 최대 하락폭, 전부 bp.
    🔴«유리/불리»(MFE/MAE)로 적지 않는다 -- 포지션이 없으므로 방향 규약이 생기는 순간 해석이 갈린다.
      위/아래로 적어 두면 어느 방향 학습에도 그대로 쓸 수 있다. 부호 유지(위로 못 갔으면 up_bp 는 음수)."""
    mid, t0, t1 = pred.get("mid"), pred["ts"], pred["ts"] + horizon_s
    if not mid or not candles or candles[0]["time"] > t0 or candles[-1]["time"] + bar_s < t1:
        return None                       # resolve 와 같은 커버리지 계약(앞뒤 양쪽)
    win = [c for c in candles if t0 < c["time"] < t1]   # t0 봉은 결정 시점 체결을 품는다(경계 계약)
    if not win:
        return None
    bp = lambda px: round((px - mid) / mid * 1e4, 1)   # noqa: E731
    return {"ret_bp": bp(win[-1]["close"]), "up_bp": bp(max(c["high"] for c in win)),
            "dn_bp": bp(min(c["low"] for c in win))}


def implied_dir(entry: dict[str, Any]) -> str | None:
    """1순위 시나리오가 함의하는 가격 방향. 되돌림·플러시는 이동 반대, 지속은 이동 쪽."""
    d, prob = entry.get("dir", 0), entry.get("prob") or {}
    if not prob:
        return None
    t = max(prob, key=prob.get)
    if d == 0:
        return {"B": "up", "C": "down"}.get(t)
    if t == "B":
        return "up" if d > 0 else "down"
    return "down" if d > 0 else "up"


def calibration(entries: list[dict[str, Any]], horizon_s: int = 1800) -> dict[str, Any]:
    """해결된 예측들의 «말한 확률 vs 실제 빈도»와 1순위 적중률."""
    # 'amb'(판정 불가)·목표 선점(scorable=False)·옛 스키마(sym 없음)는 뺀다. 🔴옛 스키마를 섞으면
    # 배포 직후 화면이 «가까운 변·5분봉» 결과와 새 결과를 한 숫자로 합친다(09-21 검토).
    # 표본 카운터는 **해결 전에도** 낸다 -- 첫 30분과 조용한 구간에 화면이 «몇 건 쌓였나»를 말해야 한다.
    new = [e for e in entries if "feat" in e]   # 학습 표본의 정의 = 피쳐가 실린 줄(P1 이후)
    ts = [e["ts"] for e in new if e.get("ts")]
    out: dict[str, Any] = {
        "n": 0, "samples": len(new),
        "amb": round(100 * sum(1 for e in new if e.get("outcome") == "amb") / max(len(new), 1)),
        "pending": sum(1 for e in new if e.get("outcome") is None),
        "with_path": sum(1 for e in new if e.get("path")),
        "span_h": round((max(ts) - min(ts)) / 3600, 1) if len(ts) > 1 else 0.0,
    }
    done = [e for e in new if e.get("outcome") in ("A", "B", "C", "none") and e.get("scorable") is not False]
    if not done:
        return out
    out["n"] = len(done)
    # 🔴엔진은 «none»에 확률을 주지 않는다. 그런데 happened 의 분모에 none 을 넣으면 said 합은 100 인데
    #   happened 합은 85 가 되어, SCORES 를 어떻게 바꿔도 닫히지 않는 가짜 «보정 격차»가 생긴다.
    #   엔진 확률은 «뭔가 닿았다는 조건 아래»의 것이므로 실제 빈도도 같은 조건에서 센다. none 은 따로 낸다.
    def tally(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
        hit = [e for e in rows if e["outcome"] in ("A", "B", "C")]
        return {k: {"said": round(sum(e["prob"][k] for e in rows) / len(rows)),
                    "happened": (round(100 * sum(1 for e in hit if e["outcome"] == k) / len(hit))
                                 if hit else 0)} for k in ("A", "B", "C")}

    out.update(tally(done))   # 전체(옛 화면 호환 -- 배포 창에서 옛 app.js 가 이 키를 읽는다)
    # 🔴2026-09-22 **레짐별로 가른다.** A/B/C 는 횡보와 추세에서 «뜻이 다른 라벨»이다
    #   (추세 A = 조금 되돌림 / 횡보 A = 잔여로서의 유지). 합쳐 놓고 지금 레짐의 이름을 붙이면
    #   화면이 거짓말을 한다 -- 게다가 둘은 **반대로** 틀린다(09-21 실측: 횡보 A 말한 37 대 실제 2.9%,
    #   추세 A 말한 35 대 실제 64.6%). 합계가 얌전해 보이는 건 상쇄 때문이다.
    #   행 이름을 여기서 같이 실어 보낸다 -- 화면이 «지금» 이름으로 옛 집계를 칠하는 게 문제였다.
    out["by_regime"] = []
    for kind, labels, rows in (("추세", SHORT_NAMES, [e for e in done if e.get("dir")]),
                               ("횡보", RANGE_NAMES, [e for e in done if not e.get("dir")])):
        if not rows:
            continue
        t = tally(rows)
        # 🔴«1순위 적중»·«미도달»도 같은 이유로 레짐별이다. 특히 미도달은 **횡보에서 구조적으로 0**
        #   이다 -- 횡보 A 는 잔여라 아무것도 안 닿으면 그게 A 이고, 그 0 이 추세 값을 희석한다
        #   (실측 전체 17% = 추세 26% + 횡보 0%). 1순위 적중도 전체 28% = 추세 40% + 횡보 4% 였다.
        top = sum(1 for e in rows if e["outcome"] == max(e["prob"], key=e["prob"].get))
        out["by_regime"].append({
            "kind": kind, "n": len(rows),
            "top_hit": round(100 * top / len(rows)),
            "none": round(100 * sum(1 for e in rows if e["outcome"] == "none") / len(rows)),
            "rows": [{"k": k, "label": labels[k], **t[k]} for k in ("A", "B", "C")]})
    # 🔴n 은 **독립 사건 수가 아니다** -- 상태가 이어지는 동안 같은 읽기가 여러 번 기록된다.
    # 에피소드(방향·1순위가 같은 연속 구간) 수를 같이 내서 검정력을 오해하지 않게 한다.
    # 🔴«상태 서명이 바뀐 횟수»로 세면 1순위가 떨릴 때마다 늘어 실효 표본이 부풀려진다
    #   (09-21 검토 실측: 7.8시간에 182개인데 겹치지 않는 30분 창은 ≤16개). 지평이 겹치는 두 예측은
    #   같은 가격 경로를 공유하므로 **겹치지 않는 창의 수**를 센다 -- 이것이 검정력의 분모다.
    eps, last_ts = 0, None
    for e in sorted(done, key=lambda x: x.get("ts") or 0):
        ts = e.get("ts") or 0
        if last_ts is None or ts - last_ts >= horizon_s:
            eps += 1
            last_ts = ts
    out["episodes"] = eps
    top_hit = sum(1 for e in done if e["outcome"] == max(e["prob"], key=e["prob"].get))
    out["top_hit"] = round(100 * top_hit / len(done))
    out["none"] = round(100 * sum(1 for e in done if e["outcome"] == "none") / len(done))
    # ⭐대칭 라벨 방향 적중 -- 근접 편향이 없어 «실력이 있나»는 이 숫자로만 판단한다(동전 = 50%)
    # ⭐대칭 배리어는 현재가 기준 등거리라 «목표 선점»(scorable=False)과 무관하다 -- Y_geo 가 채점
    # 불가여도 방향 라벨은 유효하므로 여기서는 빼지 않는다.
    sym = [e for e in entries if e.get("outcome_sym") in ("up", "down") and implied_dir(e)]
    # 🔴기준선은 «동전 50%»가 아니라 **실현 기저율**이다 -- 한쪽으로 치우친 창에서는 «항상 상승»이
    #   50%를 크게 넘는다(09-21 8시간 창 실측 62%). 그리고 1순위가 «레인지 유지»면 함의 방향이 없어
    #   이 지표는 **추세 구간만** 잰다.
    out["sym"] = ({"n": len(sym),
                   "hit": round(100 * sum(1 for e in sym if implied_dir(e) == e["outcome_sym"]) / len(sym)),
                   "base_up": round(100 * sum(1 for e in sym if e["outcome_sym"] == "up") / len(sym))}
                  if sym else {"n": 0})
    return out


if __name__ == "__main__":
    # 자체점검 -- 09-21 01:05 화면을 합성해 «그 읽기»가 재현되는지 고정한다
    def bar(t, c, dl, v, wn, rn, oi, ll=0.0, ls=0.0, h=None, l=None):
        return dict(time=t, high=h or c + 3, low=l or c - 3, close=c, delta=dl, vol=v, whale_net=wn, retail_net=rn,
                    oi_delta=oi, liq_long=ll, liq_short=ls)
    bars = [bar(0, 2597, -1900, 8000, 0, 0, 1900), bar(300, 2594, -3400, 9000, 0, 0, 500, ll=150e3),
            bar(600, 2600, 3400, 9500, 0, 0, -400, ls=80e3), bar(900, 2606, 5300, 12000, 0, 0, -600, ls=90e3),
            bar(1200, 2609, 934, 7000, 0, 0, -300, ls=40e3), bar(1500, 2611, 2200, 8000, 0, 0, -500, ls=60e3),
            bar(1800, 2614, 1500, 9000, 0, 0, 300, ls=50e3), bar(2100, 2615, 5900, 20000, 0, 0, -800, ls=120e3, h=2616),
            bar(2400, 2612, -3000, 11000, 0, 0, -400, ls=30e3)]
    inp = dict(bars=bars, mid=2612.5, levels={2606: 15000, 2609: 14000, 2603: 8000, 2612: 6000, 2615: 3000},
               cur=dict(elapsed_s=157, whale_net=-175, retail_net=149, oi_delta=-306, delta=140),
               book=dict(obi=0.60, persist_share=0.20), act_pct=0.98,
               sr=dict(res=2624.74, sup=2562.06, res_bp=47, sup_bp=193), breakout=dict(detect_on=True, prewarn_on=False))
    r = classify(inp)
    assert r["ok"] and r["dir"] == 1, r
    assert r["evidence"]["fuel"] == "스퀴즈" and r["evidence"]["climax"] and r["evidence"]["reject"], r["evidence"]
    assert r["evidence"]["cur_sig"] == "분배" and r["evidence"]["wall"] == 1 and r["evidence"]["persist"] == 0.2
    assert r["evidence"]["va_lo"] == 2606 and r["evidence"]["va_hi"] == 2609 and r["evidence"]["no_cushion"]
    assert r["prob"]["A"] > r["prob"]["B"] and r["prob"]["A"] > r["prob"]["C"], r["prob"]      # 되돌림이 1순위
    assert r["targets"]["A"] == 2606 and r["targets"]["B"] == 2624.74 and r["targets"]["C"] == 2594 - 3   # A = 먼 변
    assert abs((r["sym"]["up"] - inp["mid"]) - (inp["mid"] - r["sym"]["dn"])) < 1e-6 and r["sym"]["k"] == SYM_K
    assert r["scorable"] and r["passed"] == [] and r["dist_bp"]["A"] < 0 < r["dist_bp"]["B"]   # 부호 있는 거리
    # 목표 선점: 상승인데 현재가가 가치영역 **아래**면 A(먼 변)가 위에 있어 첫 봉에서 공짜로 닿는다 → 해당 없음
    rp = classify(dict(inp, mid=2600.0))
    assert rp["targets"]["A"] is None and "A" in rp["passed"] and rp["scorable"] is False, (rp["targets"], rp["passed"])
    # 선점돼도 **가격은 남는다**(표시용). 지우면 «어떻게 지나갔는지»를 화면에서 못 본다.
    assert rp["targets_raw"]["A"] == r["targets"]["A"] == 2606, (rp["targets_raw"], r["targets"])
    assert rp["names"]["C"] == "출발점까지" and rp["names_long"]["C"].startswith("플러시"), rp["names"]
    # 슈미트 트리거: 같은 입력이라도 «이전 레짐»에 따라 답이 달라야 한다.
    _r = abs(r["evidence"]["move_bp"]) / r["evidence"]["range_bp"]
    assert TREND_EXIT < _r, ("자체점검 입력이 EXIT 아래다", _r)
    if _r < TREND_ENTER:      # 경계 안: 중립에서는 못 들어가고, 추세였으면 유지된다
        assert classify(dict(inp, prev_dir=0))["dir"] == 0
        assert classify(dict(inp, prev_dir=1))["dir"] == 1
    else:                     # ENTER 위: 어느 이전 상태에서도 추세다
        assert classify(dict(inp, prev_dir=0))["dir"] == classify(dict(inp, prev_dir=-1))["dir"] == 1
    assert r["regime"]["enter"] == TREND_ENTER and r["regime"]["margin"] is not None
    assert resolve({"ts": 2700, "dir": 1, "mid": 2600.0, "targets": rp["targets"]}, [dict(time=2700 + 300 * i, high=2601, low=2599, close=2600) for i in range(7)]) == "none"
    assert r["dist_bp"]["C"] < r["dist_bp"]["A"] < 0 < r["dist_bp"]["B"]   # 상승: A·C 는 아래, B 는 위
    assert [f["on"] for f in r["flips"]] == [False, False, False, False, False]                # 그 시점엔 다 꺼져 있었다
    assert r["evidence"]["c_target_src"] == "베이스" and "펀딩" not in " ".join(r["labels"])
    # 🔴청산 군집은 플러시 목표로 쓰지 않는다(09-21 실측 도달 0/444). 레벨이 있어도 C 는 베이스 그대로여야 한다
    inp3 = dict(inp); inp3["sr"] = dict(inp["sr"], sup_levels=[{"price": 2589.5, "weight_pct": 30}, {"price": 2587.0, "weight_pct": 45}, {"price": 2562.06, "weight_pct": 60}])
    r3 = classify(inp3)
    assert r3["targets"]["C"] == r["targets"]["C"] == 2591 and "청산 군집" not in r3["names"]["C"], (r3["targets"], r3["names"])
    assert r3["evidence"]["c_target_src"] == "베이스" and r3["prob"] == r["prob"]
    # 펀딩 음수(숏 쏠림) + 상승 = 갇힘 → A 가 오른다 · 프리미엄이 창 동안 임계 이상 벌어지면 «선물 주도»
    inp4 = dict(inp); inp4["deriv"] = dict(funding=-0.0002, basis_bp=3.0, basis_d_bp=2.0, basis_thr_bp=1.0)
    r4 = classify(inp4)
    assert r4["evidence"]["trapped"] and r4["evidence"]["lead"] == 1 and r4["prob"]["A"] > r["prob"]["A"], (r4["evidence"], r4["prob"])
    assert any("갇힘" in x and "선물 주도" in x for x in r4["labels"]), r4["labels"]
    inp5 = dict(inp); inp5["deriv"] = dict(funding=0.00005, basis_bp=0.5, basis_d_bp=-2.0, basis_thr_bp=1.0)   # 중립 펀딩 · 현물 주도
    r5 = classify(inp5)
    assert not r5["evidence"]["trapped"] and r5["evidence"]["lead"] == -1 and r5["prob"]["B"] > r["prob"]["B"]
    assert classify(dict(inp, deriv=dict(funding=-0.0002, basis_bp=3.0, basis_d_bp=2.0, basis_thr_bp=None)))["evidence"]["lead"] == 0   # 임계 없으면 보류
    # 횡보 사전확률은 창 폭의 함수다. 16 에서 BASE_RANGE 를 그대로 재현하고, 단조이며, 합은 100 이다.
    assert base_range(50.0) == {"A": 14, "B": 44, "C": 42} and sum(base_range(50.0).values()) == 100
    _h = [base_range(x)["A"] for x in (10, 25, 35, 50, 60, 120)]
    assert _h == sorted(_h) == [6, 8, 11, 14, 17, 26] and all(sum(base_range(x).values()) == 100 for x in (10, 120))
    #   같은 입력에서 창 폭만 바꾸면 «유지»가 따라 움직여야 한다(전에는 둘 다 16 이었다).
    _chop = lambda h, l: classify(dict(inp, mid=2600.0, bars=[                                   # noqa: E731
        bar(300 * i, 2600 + (1 if i % 2 else -1), 0, 5000, 0, 0, 0, h=h, l=l) for i in range(9)]))
    _w, _n = _chop(2660, 2540), _chop(2605, 2595)
    assert _w["dir"] == _n["dir"] == 0, (_w["dir"], _n["dir"])
    assert _w["prob"]["A"] - _n["prob"]["A"] >= 10, (_w["prob"], _n["prob"])
    # BTC 같은 창: 같이 올랐으면 «동행» B↑ · 창 폭 안에서 못 움직였으면 «ETH 단독» A↑ · 없으면 라벨 없음
    r6 = classify(dict(inp, btc=dict(move_bp=40.0, range_bp=60.0)))
    assert r6["evidence"]["btc_rel"] == "동행" and r6["prob"]["B"] > r["prob"]["B"] and any("시장 전체" in x for x in r6["labels"])
    r7 = classify(dict(inp, btc=dict(move_bp=5.0, range_bp=60.0)))
    assert r7["evidence"]["btc_rel"] == "단독" and r7["prob"]["A"] > r["prob"]["A"] and any("단독" in x for x in r7["labels"])
    assert r["evidence"]["btc_rel"] is None
    # CVD 역행: 상승인데 창 델타 합이 음수면 «흡수» A↑. 원 화면은 창 델타 양수라 꺼져 있다
    assert not r["evidence"]["cvd_div"] and r["evidence"]["cvd"] > 0
    bars8 = bars[:3] + [dict(b, delta=-abs(b["delta"]) if i != 4 else b["delta"]) for i, b in enumerate(bars[3:])]   # 클라이맥스 봉만 양수
    r8 = classify(dict(inp, bars=bars8))
    assert r8["dir"] == 1 and r8["evidence"]["cvd_div"] and r8["prob"]["A"] > r["prob"]["A"] and any("흡수" in x for x in r8["labels"]), (r8["evidence"], r8["labels"])
    # 뒤집기: 마지막 봉 OI↑ 로 바꾸면 «신규 롱» 신호가 켜지고 B 가 오른다
    inp2 = dict(inp); inp2["bars"] = bars[:-1] + [bar(2400, 2616, 2500, 11000, 0, 0, 700, ls=30e3)]
    r2 = classify(inp2)
    assert r2["flips"][0]["on"] and r2["prob"]["B"] > r["prob"]["B"], (r2["flips"][0], r2["prob"])
    # 해결: 되돌림 목표에 먼저 닿는 캔들열 → 'A'
    cs = [dict(time=2700 + 300 * i, high=2613 - i, low=2611 - 2 * i, close=2612 - i) for i in range(7)]
    assert resolve({"ts": 2700, "dir": 1, "mid": 2612.5, "targets": r["targets"]}, cs) == "A"   # 먼 변 2606 관통, C(2591)까진 못 감
    # 🔴같은 쪽 중첩: A(2606)를 지나 C(2591)까지 가면 **C** 다(선착이면 영원히 A 만 나온다)
    deep = [dict(time=2700 + 300 * i, high=2613 - 2 * i, low=2611 - 5 * i, close=2612 - 3 * i) for i in range(7)]
    assert resolve({"ts": 2700, "dir": 1, "mid": 2612.5, "targets": r["targets"]}, deep) == "C", deep[-1]
    # 반대쪽이 먼저 넘겨받으면 거기서 끊는다: B 를 먼저 치면 그 뒤 깊이 빠져도 B
    bfirst = [dict(time=3000, high=2630, low=2611, close=2626)] + [dict(time=2700 + 300 * i, high=2613, low=2590, close=2600) for i in range(2, 7)]
    assert resolve({"ts": 2700, "dir": 1, "mid": 2612.5, "targets": r["targets"]}, [dict(time=2700, high=2613, low=2611, close=2612)] + bfirst) == "B"
    assert resolve({"ts": 2700, "dir": 1, "mid": 2612.5, "targets": r["targets"]}, cs[:3]) is None   # 아직 30분 안 지남
    assert resolve({"ts": 2700, "dir": 1, "mid": 2612.5, "targets": r["targets"]}, cs[2:]) is None   # 🔴창 **머리**가 비면 확정하지 않는다
    assert resolve_sym({"ts": 2700, "dir": 1, "sym": {"k": 0.5, "up": 2620.0, "dn": 2605.0}}, cs[2:]) is None
    cs_up = [dict(time=2700 + 300 * i, high=2612 + 3 * i, low=2611 + 2 * i, close=2612 + 3 * i) for i in range(7)]
    assert resolve({"ts": 2700, "dir": 1, "mid": 2612.5, "targets": r["targets"]}, cs_up) == "B"   # 종가 2627 > 2624.74
    # 🔴고가로만 스치고 종가가 못 넘으면 B 가 아니다(거부). 되돌림도 안 갔으면 미도달.
    wickonly = [dict(time=2700 + 300 * i, high=2630, low=2611, close=2612) for i in range(7)]
    assert resolve({"ts": 2700, "dir": 1, "mid": 2612.5, "targets": r["targets"]}, wickonly) == "none", "꼬리 돌파는 지속이 아니다"
    flat = [dict(time=2700 + 300 * i, high=2613, low=2611, close=2612) for i in range(7)]
    assert resolve({"ts": 2700, "dir": 1, "mid": 2612.5, "targets": r["targets"]}, flat) == "none"
    assert resolve({"ts": 2700, "dir": 0, "mid": 2612.5, "targets": dict(r["targets"], A=None)}, flat) == "A"   # 횡보의 A 는 잔여
    wide = flat[:1] + [dict(time=3000, high=2630, low=2600, close=2626)] + flat[2:]   # 종가가 B 를 넘고 저가가 A 를 찍는다
    assert resolve({"ts": 2700, "dir": 1, "mid": 2612.5, "targets": r["targets"]}, wide) == "amb"   # 한 봉에 **양쪽** = 판정 불가
    assert resolve({"ts": 2700, "dir": 1, "mid": 2612.5, "targets": {"A": [2606, 2609], "B": 2624.74, "C": 2591}}, cs) == "A"   # 옛 장부 호환
    assert resolve({"ts": 2700, "dir": 1, "targets": {"A": 2606, "B": 2624.74, "C": 2591}}, cs) == "A"          # mid 없는 옛 항목 = 선착 규칙
    sp = {"ts": 2700, "dir": 1, "sym": {"k": 0.5, "up": 2620.0, "dn": 2605.0}}
    assert resolve_sym(sp, cs) == "down" and resolve_sym(sp, cs_up) == "up" and resolve_sym(sp, flat) == "none"
    assert resolve_sym({"ts": 2700, "dir": 1}, cs) is None                                   # sym 없는 옛 항목
    assert implied_dir({"dir": 1, "prob": {"A": 50, "B": 30, "C": 20}}) == "down"            # 되돌림 = 이동 반대
    assert implied_dir({"dir": -1, "prob": {"A": 20, "B": 50, "C": 30}}) == "down"           # 지속 = 이동 쪽
    base = {"prob": r["prob"], "dir": 1, "sym": r["sym"], "feat": r["evidence"]}
    cal = calibration([{**base, "outcome": "A", "outcome_sym": "down"},
                       {**base, "outcome": "B", "outcome_sym": "up"},
                       {"prob": r["prob"], "outcome": "A", "dir": 1},                      # 옛 스키마(sym 없음) → 제외
                       {**base, "outcome": "A", "outcome_sym": "down", "scorable": False}])  # 목표 선점 → 제외
    assert cal["n"] == 2 and cal["top_hit"] == 50 and cal["A"]["happened"] == 50
    assert cal["sym"] == {"n": 3, "hit": 67, "base_up": 33}   # 선점 항목도 방향 라벨은 유효 → 3건
    # 에피소드: 방향·1순위가 같은 연속 구간을 하나로 센다(같은 읽기가 여러 번 기록되므로)
    run = [{**base, "ts": 10 * i, "outcome": "A", "outcome_sym": "down"} for i in range(5)]
    assert calibration(run)["episodes"] == 1, "10초 간격 5건은 한 창을 공유한다"
    ep = calibration(run + [{**base, "ts": 2000, "dir": -1, "outcome": "A", "outcome_sym": "down"}])
    assert ep["n"] == 6 and ep["episodes"] == 2, (ep["n"], ep["episodes"])   # 2000초 뒤라야 새 창
    assert ep["pending"] == 0 and ep["amb"] == 0 and ep["span_h"] == 0.6 and ep["samples"] == 6, ep
    warm = calibration([{**base, "ts": 1, "outcome": None}, {**base, "ts": 2, "outcome": "amb"}])
    assert warm["n"] == 0 and warm["samples"] == 2 and warm["pending"] == 1 and warm["amb"] == 50, warm
    assert calibration([{"ts": 1, "prob": {"A": 1}, "outcome": "A", "sym": {}}])["samples"] == 0   # feat 없으면 표본이 아니다
    for k in ("act_pct", "res_bp", "sup_bp", "cur_whale", "cur_retail", "cur_oi", "cur_elapsed_s", "basis_thr_bp"):
        assert k in r["evidence"], f"임계의 원본 {k} 이 표본에 없다"                                # 소급 불가라 자체점검으로 묶는다
    # 경계: 결정 시점이 분 경계면 그 봉을 쓰지 않는다(t 의 체결을 품는다)
    onmin = [dict(time=2700, high=2630, low=2600, close=2612)] + [dict(time=2700 + 60 * i, high=2613, low=2611, close=2612) for i in range(1, 31)]
    assert resolve({"ts": 2700, "dir": 1, "mid": 2612.5, "targets": r["targets"]}, onmin) == "none"   # 첫 봉(2700)을 무시 -> amb 아님
    assert path_targets({"ts": 2700, "mid": 2612.0}, onmin)["up_bp"] < 10
    # 연속 타깃: 창 끝 수익·최대 상승·최대 하락. 커버리지가 모자라면 None
    pt = path_targets({"ts": 2700, "mid": 2612.0}, cs)
    assert pt and pt["dn_bp"] <= pt["ret_bp"] <= pt["up_bp"], pt
    assert abs(pt["ret_bp"] - (2607 - 2612) / 2612 * 1e4) < 0.2, pt
    assert path_targets({"ts": 2700, "mid": 2612.0}, cs[2:]) is None      # 창 머리 없음
    assert path_targets({"ts": 2700, "mid": None}, cs) is None
    # 장부 키: 숫자만 다른 같은 상태는 같은 키, 라벨 종류가 늘면 다른 키
    same = classify(dict(inp, cur=dict(inp["cur"], whale_net=-500, retail_net=90)))
    assert log_key(same) == log_key(r) and same["labels"] != r["labels"], (log_key(same), log_key(r))
    assert log_key(r8) != log_key(r)
    # ── 횡보(d=0) 경로: 기저율·거울상 ──────────────────────────────────────────
    fb = [bar(300 * i, 2600 + (i % 2), 50, 8000, 0, 0, 10) for i in range(13)]   # 이동 없음 = 횡보
    fi = dict(bars=fb, mid=2600.5, levels={2600: 9000, 2601: 8000, 2599: 5000},
              cur=dict(elapsed_s=120, whale_net=0, retail_net=0, oi_delta=0, delta=0),
              book={}, act_pct=0.1, sr={}, breakout={})
    fr = classify(fi)
    assert fr["dir"] == 0 and fr["targets"]["A"] is None, fr["targets"]
    # 규칙이 하나도 안 붙으면 확률은 기하 기저율 그대로여야 한다(34/33/33 이 아니다).
    # 🔴이 창은 고저폭 7px/2600.5 = 27bp 라 «유지»는 평균 16 이 아니라 **8** 이다(HOLD_BY_RANGE).
    assert fr["evidence"]["range_bp"] < 30 and fr["prob"] == base_range(fr["evidence"]["range_bp"]), fr["prob"]
    assert fr["prob"]["A"] == 8 and fr["prob"]["B"] > 45 and fr["prob"]["C"] > 43, fr["prob"]
    assert max(fr["prob"], key=fr["prob"].get) != "A", "횡보의 1순위가 «레인지 유지»면 안 된다(기저율 6~26%)"
    # 지지 근접은 **하단 이탈(C)** 을 깎아야 한다 -- 전에는 상단 이탈(B)을 깎고 있었다
    near_dn = classify(dict(fi, sr=dict(sup=2599.0, res=2650.0, sup_bp=5.0, res_bp=190.0)))
    assert near_dn["prob"]["C"] < fr["prob"]["C"] and near_dn["prob"]["B"] >= fr["prob"]["B"], near_dn["prob"]
    assert near_dn["evidence"]["no_cushion_up"] is True and "위쪽 청산 쿠션 없음" in " ".join(near_dn["labels"])
    # 거울상: 저항이 가까우면 상단 이탈이 깎이고, 지지가 멀면 하단 이탈이 오른다
    near_up = classify(dict(fi, sr=dict(sup=2500.0, res=2601.0, sup_bp=390.0, res_bp=2.0)))
    assert near_up["prob"]["B"] < fr["prob"]["B"] and near_up["evidence"]["no_cushion"] is True
    # ── 벽 중간 지속률: 라벨과 점수가 같은 밴드를 봐야 한다 ──
    mid_wall = classify(dict(inp, book=dict(obi=0.60, persist_share=0.35)))
    assert " · 보통" in " ".join(mid_wall["labels"]) and mid_wall["evidence"]["persist"] == 0.35
    assert any(w["근거"] == "벽_동방향_중간" for w in mid_wall["why"]), mid_wall["why"]
    # ── 보정표: happened 는 «닿은 것» 조건부라 세 값의 합이 100 이다 ──
    cal2 = calibration([{**base, "ts": 3000 * i, "outcome": o, "outcome_sym": "up"}
                        for i, o in enumerate(["A", "B", "C", "none"])])
    assert abs(sum(cal2[k]["happened"] for k in "ABC") - 100) <= 1 and cal2["none"] == 25, cal2   # 반올림 1 허용
    print("situation selftest ok", r["prob"], r["labels"][:3])
