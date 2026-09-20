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
from typing import Any

WINDOW = 6                 # «이동»을 재는 완결 봉 수 (30분)
MOVE_THR_FRAC = 0.35       # |이동| > 이 비율 × 창 고저폭 이면 추세. 절대 bp 가 아니라 창 상대
CLIMAX_RECENT = 2          # 최대 델타 봉이 마지막 몇 봉 안에 있어야 «클라이맥스»
REJECT_FRAC = 0.5          # 거부 봉: 반대 델타가 창 최대 델타의 이 비율 이상
CUR_MIN_ELAPSED_S = 60     # 현재 봉 시그니처는 이만큼 지나야 읽는다(반쪽 봉 방지)
OBI_SIDE = 0.30            # |obi| ≥ 이면 «벽»
PERSIST_THIN, PERSIST_THICK = 0.30, 0.40
NEAR_RES_BP = 60.0         # 저항이 이 안이면 «가깝다»
FAR_SUP_RATIO = 3.0        # 지지 거리 > 저항 거리 × 이 배수면 «아래 쿠션 없음»
ACT_HOT = 0.80             # 활동 분위
FUNDING_NEUTRAL = 0.0001   # 바이낸스 펀딩의 이자 성분(0.01%/8h). 이만큼 더 벗어나야 «쏠림»(음수 = 숏 과밀)
BASIS_PCT = 0.75           # |Δ베이시스(창 동안)| 이 링 분포의 이 분위 이상이면 «선물 주도/현물 주도»
C_DEPTH = 1.0              # 플러시 목표 후보 = 베이스에서 창 고저폭 × 이 배수 안의 청산 군집(가장 두꺼운 것)

# 점수표 -- (근거 라벨 → {시나리오: 점수}). A=되돌림 B=지속 C=플러시(이동 반대쪽 과잉)
SCORES: dict[str, dict[str, int]] = {
    "스퀴즈": {"A": 20, "C": 5}, "신규유입": {"B": 20}, "클라이맥스": {"A": 15, "C": 5},
    "전환탐지": {"A": 5, "C": 5}, "분배": {"A": 10, "C": 10}, "축적": {"B": 10}, "거부봉": {"A": 5, "C": 10},
    "벽_동방향_두꺼움": {"B": 10}, "벽_동방향_얇음": {"B": 3}, "벽_역방향": {"A": 5},
    "가치영역_밖": {"A": 5}, "저항근접": {"A": 5, "B": -5}, "쿠션없음": {"C": 5}, "활발": {"B": 5, "C": 5},
    # 2026-09-21 마크가격 스트림: 펀딩 = «어느 쪽이 갇혔나», 베이시스 = «누가 주도하나»(선물 프리미엄 확장 = 취약)
    "펀딩_반대쏠림": {"A": 10}, "선물주도": {"A": 8}, "현물주도": {"B": 8},
    # BTC 같은 창 이동: 시장 전체가 같이 갔으면 지속, ETH 만 갔으면 되돌림 쪽
    "BTC_동행": {"B": 8}, "BTC_단독": {"A": 8},
    # CVD 역행(흡수): 가격은 이동 방향으로 갔는데 창 누적 델타는 반대 = 공격 체결이 아니라 수동 흡수가 만든 이동.
    # 테이프는 «같은 초» 정보라(09-20 실측) 작게만 준다
    "CVD_역행": {"A": 5},
}
BASE = {"A": 34, "B": 33, "C": 33}


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
    rng_bp = (max(b["high"] for b in w) - min(b["low"] for b in w)) / mid * 1e4
    move_bp = (w[-1]["close"] - bars[-WINDOW - 1]["close"]) / bars[-WINDOW - 1]["close"] * 1e4
    thr = MOVE_THR_FRAC * rng_bp
    d = 1 if move_bp > thr else (-1 if move_bp < -thr else 0)
    ev.update(move_bp=round(move_bp, 1), range_bp=round(rng_bp, 1), dir=d)
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
    ev.update(cvd=round(cvd), cvd_div=cvd_div, climax=climax, reject=reject, breakout_detect=bool(bo.get("detect_on")), breakout_prewarn=bool(bo.get("prewarn_on")))

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
    ev["cur_sig"] = sig

    # ── 호가 질 ──
    book = inp.get("book") or {}
    obi, pers = book.get("obi"), book.get("persist_share")
    wall = 0 if obi is None or abs(obi) < OBI_SIDE else _sign(obi)
    thick = None if pers is None else (pers >= PERSIST_THICK)
    thin = None if pers is None else (pers < PERSIST_THIN)
    if wall:
        labels.append(("매수벽" if wall > 0 else "매도벽") + f" {obi:+.2f}"
                      + (" · 얇음(믿지 말 것)" if thin else " · 두꺼움" if thick else "")
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
    if no_cushion:
        labels.append("이동 반대쪽 청산 쿠션 없음" if d != 0 else "청산맵 비대칭")
    act = inp.get("act_pct")
    hot = act is not None and act >= ACT_HOT
    if hot:
        labels.append(f"활발 (시간대 상위 {100 - int(act * 100)}%) · 큰 움직임 임박")
    ev.update(near_res=near_res, near_sup=near_sup, no_cushion=bool(no_cushion), hot=hot)

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
    ev.update(funding=fr, crowd=crowd, trapped=trapped, basis_bp=bas, basis_d_bp=bd, lead=lead)

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
    sc = dict(BASE)
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
        if wall == d: add("벽_동방향_두꺼움" if thick else "벽_동방향_얇음")
        else: add("벽_역방향")
    if va_lo is not None and (mid > va_hi or mid < va_lo): add("가치영역_밖")
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
        names = {"A": "되돌림 · 가치영역 재방문", "B": "지속 · 저항 테스트", "C": "플러시 · 베이스 재방문"}
    else:
        cont = sup_px if (sup_px and sup_px < mid) else min(b["low"] for b in w)
        names = {"A": "되돌림 · 가치영역 재방문", "B": "지속 · 지지 테스트", "C": "역스퀴즈 · 고점 재방문"}
    if d == 0:
        names = {"A": "레인지 유지", "B": "상단 이탈", "C": "하단 이탈"}
        cont = max(b["high"] for b in w); base_px = min(b["low"] for b in w)   # 횡보는 창 자체가 레인지
    else:
        # 청산맵은 «위치 예보»(레벨 ±15bp 청산 확률 0.52 vs 0.28, 09-20 실측)라 확률이 아니라 목표에 쓴다:
        # 베이스에서 창 고저폭 안에 있는 반대편 청산 군집 중 가장 두꺼운 것이 플러시가 «실제로 멈추는 자리».
        opp = (sr.get("sup_levels") if d > 0 else sr.get("res_levels")) or []
        depth = rng_bp / 1e4 * mid * C_DEPTH
        cands = [lv for lv in opp if lv.get("price") and (base_px - depth <= lv["price"] <= base_px if d > 0 else base_px <= lv["price"] <= base_px + depth)]
        if cands:
            base_px = float(max(cands, key=lambda lv: lv.get("weight_pct") or 0)["price"])
            names["C"] = "플러시 · 청산 군집" if d > 0 else "역스퀴즈 · 청산 군집"
    ev["c_target_src"] = "청산군집" if "청산 군집" in names["C"] else "베이스"
    targets = {"A": [va_lo, va_hi] if va_lo is not None else None, "B": round(float(cont), 2), "C": round(float(base_px), 2)}

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
    return {"ok": True, "dir": d, "labels": labels, "evidence": ev, "prob": prob, "names": names, "targets": targets,
            "why": [{"근거": k, **v} for k, v in why],
            "flips": [{"signal": s, "on": bool(o), "toward": t} for s, o, t in flips]}


def resolve(pred: dict[str, Any], candles: list[dict[str, float]], horizon_s: int = 1800) -> str | None:
    """예측 뒤 horizon 안에 어느 목표가 **먼저** 닿았나. A=가치영역 밴드, B=지속 목표, C=플러시 목표.
    아직 horizon 이 안 지났으면 None. 아무것도 안 닿으면 'none'."""
    t0, t1 = pred["ts"], pred["ts"] + horizon_s
    if not candles or candles[-1]["time"] + 300 < t1:
        return None
    d, tg = pred.get("dir", 0), pred.get("targets") or {}
    first: dict[str, int] = {}
    for i, c in enumerate(candles):
        if c["time"] < t0 or c["time"] >= t1:
            continue
        hi, lo = c["high"], c["low"]
        if tg.get("A") and "A" not in first:
            a_lo, a_hi = tg["A"]
            if (d > 0 and lo <= a_hi) or (d < 0 and hi >= a_lo) or (d == 0 and lo <= a_hi and hi >= a_lo):
                first["A"] = i
        if tg.get("B") is not None and "B" not in first:
            if (d >= 0 and hi >= tg["B"]) or (d < 0 and lo <= tg["B"]):
                first["B"] = i
        if tg.get("C") is not None and "C" not in first:
            if (d >= 0 and lo <= tg["C"]) or (d < 0 and hi >= tg["C"]):
                first["C"] = i
    if not first:
        return "none"
    return min(first, key=lambda k: (first[k], {"C": 0, "B": 1, "A": 2}[k]))


def calibration(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """해결된 예측들의 «말한 확률 vs 실제 빈도»와 1순위 적중률."""
    done = [e for e in entries if e.get("outcome") in ("A", "B", "C", "none")]
    if not done:
        return {"n": 0}
    out: dict[str, Any] = {"n": len(done)}
    for k in ("A", "B", "C"):
        out[k] = {"said": round(sum(e["prob"][k] for e in done) / len(done)),
                  "happened": round(100 * sum(1 for e in done if e["outcome"] == k) / len(done))}
    top_hit = sum(1 for e in done if e["outcome"] == max(e["prob"], key=e["prob"].get))
    out["top_hit"] = round(100 * top_hit / len(done))
    out["none"] = round(100 * sum(1 for e in done if e["outcome"] == "none") / len(done))
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
    assert r["targets"]["A"] == [2606, 2609] and r["targets"]["B"] == 2624.74 and r["targets"]["C"] == 2594 - 3
    assert [f["on"] for f in r["flips"]] == [False, False, False, False, False]                # 그 시점엔 다 꺼져 있었다
    assert r["evidence"]["c_target_src"] == "베이스" and "펀딩" not in " ".join(r["labels"])
    # 청산 군집 목표: 베이스(2591) 아래 창 고저폭 안의 가장 두꺼운 지지 레벨로 바뀐다. 너무 먼 2562 는 안 고른다
    inp3 = dict(inp); inp3["sr"] = dict(inp["sr"], sup_levels=[{"price": 2589.5, "weight_pct": 30}, {"price": 2587.0, "weight_pct": 45}, {"price": 2562.06, "weight_pct": 60}])
    r3 = classify(inp3)
    assert r3["targets"]["C"] == 2587.0 and r3["names"]["C"] == "플러시 · 청산 군집" and r3["prob"] == r["prob"], (r3["targets"], r3["prob"])
    # 펀딩 음수(숏 쏠림) + 상승 = 갇힘 → A 가 오른다 · 프리미엄이 창 동안 임계 이상 벌어지면 «선물 주도»
    inp4 = dict(inp); inp4["deriv"] = dict(funding=-0.0002, basis_bp=3.0, basis_d_bp=2.0, basis_thr_bp=1.0)
    r4 = classify(inp4)
    assert r4["evidence"]["trapped"] and r4["evidence"]["lead"] == 1 and r4["prob"]["A"] > r["prob"]["A"], (r4["evidence"], r4["prob"])
    assert any("갇힘" in x and "선물 주도" in x for x in r4["labels"]), r4["labels"]
    inp5 = dict(inp); inp5["deriv"] = dict(funding=0.00005, basis_bp=0.5, basis_d_bp=-2.0, basis_thr_bp=1.0)   # 중립 펀딩 · 현물 주도
    r5 = classify(inp5)
    assert not r5["evidence"]["trapped"] and r5["evidence"]["lead"] == -1 and r5["prob"]["B"] > r["prob"]["B"]
    assert classify(dict(inp, deriv=dict(funding=-0.0002, basis_bp=3.0, basis_d_bp=2.0, basis_thr_bp=None)))["evidence"]["lead"] == 0   # 임계 없으면 보류
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
    assert resolve({"ts": 2700, "dir": 1, "targets": r["targets"]}, cs) == "A"
    assert resolve({"ts": 2700, "dir": 1, "targets": r["targets"]}, cs[:3]) is None          # 아직 30분 안 지남
    cs_up = [dict(time=2700 + 300 * i, high=2612 + 3 * i, low=2611 + 2 * i, close=2612 + 2 * i) for i in range(7)]
    assert resolve({"ts": 2700, "dir": 1, "targets": r["targets"]}, cs_up) == "B"
    cal = calibration([{"prob": r["prob"], "outcome": "A"}, {"prob": r["prob"], "outcome": "B"}])
    assert cal["n"] == 2 and cal["top_hit"] == 50 and cal["A"]["happened"] == 50
    print("situation selftest ok", r["prob"], r["labels"][:3])
