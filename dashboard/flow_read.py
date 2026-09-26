"""«데이터끼리 무슨 말을 하나» — 청산맵 지지/저항 목록 아래 문장 (2026-09-25 사용자 지시).

화면의 실시간 원천(체결·크기별 수급·OKX·OI·청산·호가·청산맵 S/R·베이시스/펀딩·BTC·추세 veto·활동)을
**둘씩 맞대어** 지금 무슨 관계인지 한 줄씩 말한다. 줄마다 [관찰 숫자 → 해석 → 근거 등급].
근거 등급은 **연구 결과가 허락하는 만큼만** 준다(PRODUCT.md «화면 신호는 측정 통과가 원칙»):
  근거  표본외·연도 부호까지 통과          약함  부호 일관·CI 경계 또는 짧은 표본
  설명  예측력을 재서 0 이었다(상태 설명만)  미측정 아직 안 잰 관계
🔴«설명»·«미측정» 줄은 방향(dir)을 갖지 않는다. 맨 위 요약은 근거·약함 줄의 방향만 센다.
ponytail: 문구·임계는 이 파일 한 곳. 부호 규약 매수·상승 = +.
"""
from __future__ import annotations
from typing import Any
import numpy as np

Z_SIDE = 0.5            # 크기별 60분 순매수 z — 연구와 같은 값(사전 고정)
CVD_Z_BIG = 1.0         # 30분 체결이 «뚜렷하다» (자기 24h 분포의 1σ)
FLAT_FRAC = 0.25        # |30분 이동| < 이 × 창 고저폭 이면 «제자리»
DEEP_HI, DEEP_LO = 0.8, 0.2   # 깊은 호가 불균형 자기 6시간 분위 — 연구 점유 각 20%
NEAR_BP = 15.0          # 청산맵 레벨 근접(연구 판정폭, micro_ref.NEAR_BP 와 같다)
LIQ_MIN_USD = 50_000.0  # 30분 청산이 이보다 작으면 «거의 없다»
# ── 융합 신호 (2026-09-25 사용자 «방향 근거 + 30분 시나리오를 융합») ─────────────────────────────
# 독립 원천 4표(고래 한 표 · 1시간 가격↔OI · 12h 추세↔30분 이동 정렬 · 거부 봉) 중 **같은 쪽 2표 이상** + 크기 관문
# (30분 창 고저폭이 하루 분위 상위 1/3 — 카드의 크기 축 최선 단일피쳐). 지평 30분. tmp/fusion/build_eval*.py 로 쟀다.
# 🔴카드의 A/B/C 확률은 **안 쓴다** — 09-25 통일축 재측정에서 «항상 되돌림»과 소수점까지 같았다(변별 0).
#   카드에서 정보가 있는 것은 30분 방향(dir)·정렬·거부 봉·창 폭이고, 그것들이 여기 표와 관문으로 들어간다.
FUSE_MIN_VOTES = 2
GATE_PCT = 2 / 3
# ── 30분 카드 = 융합 3결과 (2026-09-25 사용자 «옛 시나리오 로직을 걷고 하나의 융합 신호로») ──────────
# 카드 축 그대로(다음 30분 안 ±0.5×30분 창폭 중 먼저 닿는 쪽, 1분봉) — 결과 셋: 위 먼저 · 아래 먼저 · 미도달.
# 확률 = 같은 상태의 **실측 빈도**(ETH 3.7년 모든 5분봉 380,586 결정, 라플라스 +1). 상태 = 레짐(카드 30분 방향) ×
#   정렬 점수 a(융합 4표 합; 추세면 ×방향, −1 이하/0/+1/+2 이상) × 30분 창폭 24h 분위 삼분위 g.
# 🔴두 축이 따로 일한다: g 가 «얼마나 먼가»(미도달 9%→48%), 표가 «어느 쪽»(추세·g2: a=0 28:24 → a≥2 33:26).
#   TEST 로그손실은 레짐×g 가 거의 전부(1.0783→1.0223), 표의 증분은 +0.0003(CI 0 포함) — 표는 «방향 기울기»만 준다.
#   셀별 TRAIN↔TEST 첫 칸 차이 ≤ ~2pp(큰 셀). 횡보 a≥+2 는 n 20~93 이라 +1 칸으로 합친다.
#   (tmp/fusion/card_table.py → scripts/research_eth_fused_card_table_20260925.py)
CARD_CELLS = {   # (레짐, a, g): (n, 이동방향 또는 위 %, 반대 또는 아래 %, 미도달 %)
    ("trend", -1, 0): (5654, 43.3, 46.1, 10.6),
    ("trend", -1, 1): (7150, 37.1, 38.2, 24.6),
    ("trend", -1, 2): (9788, 25.8, 26.7, 47.5),
    ("trend", +0, 0): (32646, 44.2, 45.3, 10.4),
    ("trend", +0, 1): (32024, 37.2, 35.9, 26.9),
    ("trend", +0, 2): (33742, 27.9, 23.9, 48.2),
    ("trend", +1, 0): (26093, 45.2, 45.9, 8.9),
    ("trend", +1, 1): (29338, 39.4, 37.2, 23.4),
    ("trend", +1, 2): (35864, 30.1, 24.5, 45.4),
    ("trend", +2, 0): (1135, 49.0, 45.1, 5.9),
    ("trend", +2, 1): (3321, 42.5, 41.0, 16.5),
    ("trend", +2, 2): (11637, 33.4, 25.6, 41.0),
    ("range", -1, 0): (2354, 45.4, 47.0, 7.6),
    ("range", -1, 1): (3892, 41.5, 41.4, 17.2),
    ("range", -1, 2): (6178, 27.1, 32.9, 40.0),
    ("range", +0, 0): (60540, 45.8, 46.0, 8.3),
    ("range", +0, 1): (40873, 39.2, 38.6, 22.2),
    ("range", +0, 2): (24574, 28.1, 30.9, 41.0),
    ("range", +1, 0): (2650, 47.9, 44.1, 8.1),
    ("range", +1, 1): (4269, 41.1, 40.1, 18.8),
    ("range", +1, 2): (6695, 27.9, 31.1, 41.0),
}
SYM_K = 0.5   # 카드 배리어 = ±0.5 × 30분 창폭 (옛 상황 카드 채점축과 같은 값)
# ── 방향 = HGB 모델 (2026-09-26 사용자 «모델부터 바꾸자 · tabpfn/hgb») ────────────────────────────────
# 표의 위·아래 격차는 가중 평균 2.1pp 라 «항상 비슷»했다. 이제 표는 «닿을 확률»(미도달)만 맡고, 닿는다면 어느 쪽인지는
#   5분봉 피쳐 26개 HGB(보정)가 맡는다. 1분봉 선착 라벨 · TEST 2025~ AUC .528(TabPFN 같은 컨텍스트 .523 = HGB 부분표집과 동률).
#   🔴metrics(OI·롱숏비)를 넣으면 오히려 .525 로 준다 → 5분봉 klines 만 쓴다(라이브 배관 없음).
#   scripts/research_eth_card30_direction_hgb_tabpfn_20260926.py (build · fit · report · export · parity).
DIR_COIN_PP = 5.0     # 닿는다면 위:아래 가 50 에서 이만큼 안 벌어지면 «동전»이라고 말한다
DIR_EVID = ("방향 = HGB(5분봉 26피쳐, 1분봉 선착 라벨) · TEST 2025~ AUC .525 · 50:50 에서 5pp 이상 벌어지는 건 결정의 ~7%, "
            "그때 방향 적중 ~56% · 나머지는 동전 — 30분 방향은 대부분 예측이 안 된다")


def card30_features(bars: list[dict[str, float]]) -> dict[str, float] | None:
    """완결 5분봉(시간순, time=봉 시작 초, high/low/close/volume/taker) → 방향 모델 피쳐. 연구 build() 와 같은 정의.
    봉이 293개 미만이거나 5분 간격이 끊기면 None(모름을 추정으로 메우지 않는다)."""
    if len(bars) < 293:
        return None
    b = bars[-293:]
    t = np.array([int(x["time"]) for x in b])
    if np.any(np.diff(t) != 300):
        return None
    h, lo, c, v, tb = (np.array([float(x[k]) for x in b]) for k in ("high", "low", "close", "volume", "taker"))
    f: dict[str, float] = {}
    lr = np.log(c)
    for k in (1, 3, 6, 12, 24, 48, 144, 288):
        f[f"ret{k}"] = (lr[-1] - lr[-1 - k]) * 1e4
    for n in (6, 24, 144, 288):
        H, L = h[-n:].max(), lo[-n:].min()
        f[f"pos{n}"] = (c[-1] - L) / max(H - L, 1e-9)
    atr = (h - lo)[-144:].mean()
    for n in (48, 144, 288):
        f[f"sma{n}"] = (c[-1] - c[-n:].mean()) / atr
    imb = 2 * tb / np.maximum(v, 1e-9) - 1
    for k in (1, 6, 12, 48):
        f[f"imb{k}"] = (imb * v)[-k:].sum() / max(v[-k:].sum(), 1e-9)
    W = np.lib.stride_tricks.sliding_window_view
    rg = (W(h, 6).max(1) - W(lo, 6).min(1)) / c[5:] * 1e4                 # 288개, 마지막 = 지금
    f["rg30"] = rg[-1]
    f["rgq"] = (np.sum(rg < rg[-1]) + (np.sum(rg == rg[-1]) + 1) / 2) / rg.size   # pandas rank(pct) 평균 순위
    f["volz"] = (v[-1] - v[-288:].mean()) / v[-288:].std(ddof=1)
    pc = c[-2]
    f["last_body"] = (c[-1] - pc) / max(h[-1] - lo[-1], 1e-9)
    f["wick"] = ((h[-1] - max(c[-1], pc)) - (min(c[-1], pc) - lo[-1])) / max(h[-1] - lo[-1], 1e-9)
    tt = np.datetime64(int(t[-1]), "s")
    f["hour"] = float(int(t[-1]) // 3600 % 24)
    f["dow"] = float((tt.astype("datetime64[D]").astype(int) + 3) % 7)       # 1970-01-01 = 목(3), 월 = 0
    return f
# 각 결과 열 아래에 거는 «그쪽으로 미는 조건» -- 융합 4표의 각 방향판. (표 이름 접두, 부호, 문구)
PUSH = {
    +1: (("고래", +1, "고래 매수 · 리테일/중형 매도"), ("가격↔OI", +1, "1시간 하락 + OI 감소(롱 이탈 끝물)"),
         ("추세정렬", +1, "12h 상승 추세 안 30분 상승"), ("거부 봉", +1, "하락 중 매수 거부 봉")),
    -1: (("고래", -1, "고래 매도 · 리테일/중형 매수"), ("가격↔OI", -1, "1시간 하락 + OI 증가(새 숏)"),
         ("추세정렬", -1, "12h 하락 추세 안 30분 하락"), ("거부 봉", -1, "상승 중 매도 거부 봉")),
}
FUSE_EVID = ("ETH 3.7년(서버와 같은 24h 정의): 건당 TRAIN +4.1bp[+0.2,+8.2] · TEST +6.6bp[+2.1,+11.4] · 4/4년 · 롱숏 둘 다 + · "
             "회전 귀무 p<0.001 · 적중 48%(자주 조금 지고 가끔 크게 번다, 중앙 −2bp) · BTC 재현 안 됨 · 하루 ~1.7번 · 메이커 진입이어야 남는다")

EVID = {   # 근거 한 줄 — 출처는 메모리/문서 이름(whale_mid_retail_follow · price_oi_quadrant · rt5_* · liq_hunt …)
    "price_flow": "3.7년 실측: 체결↔가격 역행은 봉의 1/4로 흔하고, 뒤따르는 가격 차이 +0.2bp — 방향 신호 아님",
    "whale_retail": "3.7년 실측: 갈리면 60분 고래 쪽 +6bp[+3,+10]·4/4년·BTC +3.5 — 메이커로만 비용을 넘는다",
    "whale_mid": "3.7년 실측: 고래↔중형 갈리면 60분 고래 쪽 +7.6bp[+3,+12]",
    "same_side": "3.7년 실측: 크기별 단독 추종은 셋 다 되돌림(리테일 최악) — 쏠림을 따라가지 말 것",
    "oi_dn_up": "4.7년 1시간: 하락+OI↑ 다음 1시간 −2.3bp(0/5년 반등) — 숏 조기청산 금지",
    "oi_dn_dn": "4.7년 1시간: 하락+OI↓ 다음 1시간 +1.2bp(OI↑ 대비 +3.5·5/5년) — 숏 거두기 순 +0.5bp로 얇다",
    "oi_up": "상승 쪽 사분면은 1시간 틀에서 안 쟀다(30분 카드 감사에선 스퀴즈·신규유입 효과 0)",
    "oi_small": "1초~5분 OI 는 체결의 4초 뒤 그림자 — 큰 1시간 이동에서만 뜻이 있다",
    "liq_result": "청산은 가격을 따라온다(분→다음 분 −0.34, 반대 +0.01) · 버스트 뒤 5~60분 평평 — 꼭지 신호 아님",
    "liq_flush": "69일: 추세 반대편 청산 뒤 추세 방향 60분 +13.7bp[−2.4,+28] — CI 0 포함",
    "deep_wall": "5분 홀드아웃 통과: 깊은 매수벽−매도벽 +7~8bp(t 2.2) — 상승장 두 창·점유 40%라 «상태»에 가깝다",
    "qi": "최우선 호가+호가 증감 동조는 1~15초만 앞선다(+0.4bp/15초) — 수수료를 못 넘는다",
    "wall_sr": "벽은 지지·저항이 아니다(닿은 뒤 반등 0.509)",
    "sr_near": "레벨 ±15bp 는 청산이 날 자리(청산 확률 0.52 vs 0.28) — 뚫림/버팀 방향은 없다",
    "sr_wall": "깊은 벽이 레벨 쪽이면 버팀 후보(n 82/156, 1.5~2.5SE)",
    "basis": "선물·현물 주도 구분의 예측력은 안 쟀다",
    "funding": "펀딩 쏠림 반대 매매: 4.7년 극단 펀딩 0/10·30분 감사 부호 반대",
    "btc": "BTC 동행/단독: 30분 감사 효과 0(±0.2pp)",
    "veto_align": "30분 지속률 정렬 52.8% vs 역행 50.5%(+2.2pp·6/6년) — 작다",
    "act": "활동은 움직임의 크기를 말한다(방향 아님) — 변동성 확대 OOS AUC 0.82",
}


def _s(x: float | None) -> int:
    return 0 if not x else (1 if x > 0 else -1)


def _side(x: float, buy: str = "매수", sell: str = "매도") -> str:
    return buy if x > 0 else sell


def _updn(x: float) -> str:
    return "상승" if x > 0 else "하락"


def _k(v: float) -> str:
    """게이지 값 칸(좁다)용 짧은 수: 35,574 → +36k · 2,086 → +2.1k · 420 → +420."""
    a, sg = abs(v), "+" if v > 0 else ("−" if v < 0 else "")
    return f"{sg}{a / 1e6:.1f}M" if a >= 1e6 else f"{sg}{a / 1e3:.0f}k" if a >= 1e4 else f"{sg}{a / 1e3:.1f}k" if a >= 1e3 else f"{sg}{a:.0f}"


def _cl(v: float) -> float:
    return max(-100.0, min(100.0, v))


def _tilt(a: float, b: float) -> float | None:
    return None if not (a or b) else (a - b) / (abs(a) + abs(b)) * 100


def gauges(ev: dict[str, Any], x: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """관계마다 롱/숏 게이지 하나 — 옛 카드 «현재 상황» 막대와 같은 문법.
    kind d = 0 가운데 발산(−100 숏 쪽 · +100 롱 쪽), m = 0→100 세기. on=False 면 «조건 미달»(0 자리 눈금만).
    🔴게이지는 «데이터가 어느 쪽으로 기울었나»이지 예측이 아니다 — 예측 근거는 줄의 등급(근거·약함)이 말하고,
      화면은 그 등급일 때만 방향색을 칠한다."""
    G: dict[str, dict[str, Any]] = {}
    mv, cvd, cz = ev.get("move_bp"), ev.get("cvd"), x.get("cvd30_z")
    if cvd is not None:
        G["가격↔체결"] = {"kind": "d", "v": _cl(cz * 33) if cz is not None else (50.0 if cvd > 0 else -50.0 if cvd < 0 else 0.0),
                        "txt": _k(cvd), "on": bool(cvd)}
    net, z = x.get("net60") or {}, x.get("z60")
    if net:
        v = _cl(z["whale"] * 50) if z else _tilt(net["whale"], net["retail"])
        G["고래"] = {"kind": "d", "v": v or 0.0, "txt": _k(net["whale"]), "on": v is not None}
    m60, oi60, p75 = x.get("move60"), x.get("oi60"), x.get("move60_p75")
    if m60 is not None and oi60 is not None:
        big = bool(p75) and m60 < 0 and abs(m60) >= p75 and oi60 != 0   # p75=0(24h 평탄·캐시 이상)이면 아래 나눗셈이 0 으로 나눈다
        G["가격↔OI"] = {"kind": "d", "v": (-1 if oi60 > 0 else 1) * min(100.0, 50 * abs(m60) / p75) if big else 0.0,
                       "txt": f"{m60:+.0f}bp", "on": big}
    if "liq_long" in ev:
        ll, ls = ev.get("liq_long") or 0.0, ev.get("liq_short") or 0.0
        v = _tilt(ls, ll)                                   # 숏이 더 청산 = 강제 매수 = 위쪽
        G["청산"] = {"kind": "d", "v": v or 0.0, "txt": "$" + _k(ls - ll).lstrip("+"), "on": ll + ls >= LIQ_MIN_USD}
    ip, imb, obi = x.get("imb40_pct"), x.get("imb40"), ev.get("obi")
    if ip is not None and imb is not None:
        G["호가↔체결"] = {"kind": "d", "v": _cl((ip - 0.5) * 200), "txt": f"{imb:+.2f}", "on": ip >= DEEP_HI or ip <= DEEP_LO}
    elif obi is not None:
        G["호가↔체결"] = {"kind": "d", "v": _cl(obi * 100), "txt": f"{obi:+.2f}", "on": False}
    sr = x.get("sr") or {}
    sb, rb, near = sr.get("sup_bp"), sr.get("res_bp"), sr.get("near") or "없음"
    if sb is not None or rb is not None:
        hold = near != "없음" and sr.get("deep_wall") == "레벨쪽"
        dist = min((d for d in (sb, rb) if d is not None))
        G["지지/저항"] = {"kind": "d", "v": (70.0 if near == "지지근접" else -70.0) if hold else 0.0,
                       "txt": f"{'지지' if dist == sb else '저항'} {dist:.0f}bp", "on": hold}
    bd, thr = ev.get("basis_d_bp"), ev.get("basis_thr_bp")
    if ev.get("basis_bp") is not None:
        G["선물↔현물"] = {"kind": "d", "v": _cl(bd / thr * 100) if (bd is not None and thr) else 0.0,
                       "txt": f"{bd:+.1f}" if bd is not None else "—", "on": bool(ev.get("lead"))}
    bm = ev.get("btc_move_bp")
    if bm is not None:
        G["ETH↔BTC"] = {"kind": "d", "v": _cl(bm / abs(mv) * 100) if mv else 0.0, "txt": f"{bm:+.0f}bp",
                       "on": ev.get("btc_rel") is not None}
    veto, d30 = ev.get("veto") or 0, ev.get("dir") or 0
    if veto and d30:
        G["추세↔이동"] = {"kind": "d", "v": 50.0 * veto + 50.0 * d30, "txt": f"{'↑' if veto > 0 else '↓'}{'↑' if d30 > 0 else '↓'}",
                       "on": veto == d30}
    vp = x.get("vol_pct")
    if vp is not None:
        G["활동"] = {"kind": "m", "v": 100.0 * vp, "txt": f"{100 * vp:.0f}", "on": vp >= 0.8 or vp <= 0.2}
    return G


def read(ev: dict[str, Any], x: dict[str, Any]) -> dict[str, Any]:
    """ev: situation.classify 의 evidence(30분 창). x: 이 줄들 전용 입력(server `_flow_read_ctx`).
    반환 {lines:[{topic,text,note,grade,dir}], summary, up, down}. dir 은 근거/약함 줄에만 ±1."""
    L: list[dict[str, Any]] = []

    def add(topic: str, text: str, note: str, grade: str, d: int = 0) -> None:
        L.append({"topic": topic, "text": text, "note": note, "grade": grade, "dir": d if grade in ("근거", "약함") else 0})

    # ① 가격 ↔ 체결(30분) — 누가 가격을 움직였나
    mv, rg, cvd = ev.get("move_bp"), ev.get("range_bp"), ev.get("cvd")
    okx, cz = x.get("okx30"), x.get("cvd30_z")
    if mv is not None and cvd is not None:
        obs = f"30분 {mv:+.0f}bp · 바이낸스 체결 {cvd:+,.0f} ETH" + (f" · OKX {okx:+,.0f}" if okx is not None else "")
        flat = rg is not None and abs(mv) < FLAT_FRAC * rg
        loud = cz is None or abs(cz) >= CVD_Z_BIG
        if flat and loud and cvd:
            t = f"{obs} — 시장가가 {_side(cvd)}로 몰렸는데 가격은 제자리: 반대편 지정가가 받아내는 중(흡수)"
        elif _s(cvd) == _s(mv) or not cvd:
            t = f"{obs} — 시장가가 가격을 {'올렸다' if mv > 0 else '내렸다'}"
        elif okx is not None and _s(okx) == _s(mv):
            t = f"{obs} — 바이낸스 시장가는 {_side(cvd)}인데 가격은 {_updn(mv)}: OKX 가 {_side(okx)}로 끌었다"
        else:
            t = f"{obs} — 시장가는 {_side(cvd)}인데 가격은 {_updn(mv)}: 지정가 {_side(mv)}가 받아냈다(흡수)"
        add("가격↔체결", t, EVID["price_flow"], "설명")

    # ② 고래 ↔ 중형 ↔ 리테일(60분) — 누구 편인가
    net, z = x.get("net60") or {}, x.get("z60")
    if net:
        obs = f"60분 순매수 고래 {net['whale']:+,.0f} · 중형 {net['mid']:+,.0f} · 리테일 {net['retail']:+,.0f} ETH"
        if not z:
            add("고래↔리테일", obs + " — 기준 분포 수집 중(서버 기동 뒤 6시간)", EVID["same_side"], "설명")
        else:
            zw, zm, zr = z["whale"], z["mid"], z["retail"]
            big = lambda v: abs(v) >= Z_SIDE   # noqa: E731
            m60 = x.get("move60")
            against = m60 is not None and _s(zw) == -_s(m60)
            if big(zw) and big(zr) and _s(zw) != _s(zr):
                add("고래↔리테일", f"{obs} — 고래 {_side(zw)} · 리테일 {_side(zr)}로 갈렸다: 고래 쪽"
                    + (" (고래가 가격과 반대로 섰다 — 이때 더 강했다 +10bp)" if against else ""),
                    EVID["whale_retail"], "약함", _s(zw))
            elif big(zw) and big(zm) and _s(zw) != _s(zm):
                add("고래↔중형", f"{obs} — 고래 {_side(zw)} · 중형 {_side(zm)}로 갈렸다: 고래 쪽", EVID["whale_mid"], "약함", _s(zw))
            elif big(zw) and big(zm) and big(zr) and _s(zw) == _s(zm) == _s(zr):
                add("고래↔리테일", f"{obs} — 셋 다 {_side(zw)}: 한쪽 쏠림", EVID["same_side"], "설명")
            else:
                add("고래↔리테일", f"{obs} — 크기별로 뚜렷한 갈림·쏠림 없음", EVID["same_side"], "설명")

    # ③ 가격 ↔ OI(1시간) — 새 포지션인가 빠지는 포지션인가
    m60, oi60, p75 = x.get("move60"), x.get("oi60"), x.get("move60_p75")
    if m60 is not None and oi60 is not None:
        obs = f"1시간 {m60:+.0f}bp · OI {oi60:+,.0f} ETH"
        if p75 is None or abs(m60) < p75:
            add("가격↔OI", f"{obs} — 이동이 작다(하루 상위 25% 밖): OI 방향만으로는 말할 게 없다", EVID["oi_small"], "설명")
        elif m60 < 0 and oi60 > 0:
            add("가격↔OI", f"{obs} — 하락 + OI 증가: 새 숏이 밀고 있다", EVID["oi_dn_up"], "근거", -1)
        elif m60 < 0:
            add("가격↔OI", f"{obs} — 하락 + OI 감소: 롱이 손절·청산으로 빠진다", EVID["oi_dn_dn"], "근거", +1)
        elif oi60 > 0:
            add("가격↔OI", f"{obs} — 상승 + OI 증가: 새 롱 유입", EVID["oi_up"], "미측정")
        else:
            add("가격↔OI", f"{obs} — 상승 + OI 감소: 숏 커버(스퀴즈)로 오른다", EVID["oi_up"], "미측정")

    # ④ 청산 ↔ 가격·추세(30분 + 최근 60초). 🔴evidence 에 청산 키가 없으면(상황 카드 워밍업) «거의 없다»가 아니라 안 말한다
    ll, ls = ev.get("liq_long") or 0.0, ev.get("liq_short") or 0.0
    s60 = x.get("liq60s") or {}
    tail = ""
    if (s60.get("long") or 0) + (s60.get("short") or 0) > 0:
        tail = f" · 최근 60초 롱 ${s60.get('long', 0):,.0f} / 숏 ${s60.get('short', 0):,.0f}"
    if "liq_long" not in ev:
        pass
    elif ll + ls < LIQ_MIN_USD:
        add("청산", f"30분 청산 롱 ${ll:,.0f} · 숏 ${ls:,.0f}{tail} — 거의 없다", EVID["liq_result"], "설명")
    else:
        dom = 1 if ls > ll else -1            # 숏이 청산 = 강제 매수(+)
        veto = ev.get("veto") or 0
        who = "숏" if dom > 0 else "롱"
        obs = f"30분 청산 롱 ${ll:,.0f} · 숏 ${ls:,.0f}{tail}"
        if veto and dom == -veto:
            add("청산↔추세", f"{obs} — 12시간 추세({_updn(veto)})의 반대편({who})이 털렸다: 눌림 청산", EVID["liq_flush"], "약함", veto)
        elif mv is not None and _s(mv) == dom:
            add("청산↔가격", f"{obs} — 이동이 {who}을 청산시켰다(결과)", EVID["liq_result"], "설명")
        else:
            add("청산↔가격", f"{obs} — {who} 청산이 우세", EVID["liq_result"], "설명")

    # ⑤ 호가 ↔ 체결(지금) — 깊은 벽 · 최우선 동조 · 벽 지속
    ip, imb = x.get("imb40_pct"), x.get("imb40")
    agree, live = x.get("agree"), x.get("trigger_live")
    if ip is not None and imb is not None and (ip >= DEEP_HI or ip <= DEEP_LO):
        d = 1 if ip >= DEEP_HI else -1
        rank = 100 * (1 - ip) if d > 0 else 100 * ip
        t = f"깊은 호가(±0.4%) 불균형 {imb:+.2f}(6시간 {'상위' if d > 0 else '하위'} {rank:.0f}%) — {_side(d, '매수벽', '매도벽')} 우위"
        if cvd and _s(cvd) == -d:
            t += f", 시장가({_side(cvd)})를 받아낼 쪽이 두껍다"
        add("호가↔체결", t, EVID["deep_wall"], "약함", d)
    elif agree in ("동조매수", "동조매도") and live:
        add("호가↔체결", f"최우선 호가 기울기와 호가 증감이 {agree[2:]} 쪽으로 동조(방금)", EVID["qi"], "설명")
    else:
        obi, pers = ev.get("obi"), ev.get("persist")
        t = "호가 불균형 자료 없음"
        if obi is not None:
            t = f"호가 불균형 {obi:+.2f}" + (f" · 벽 지속 {pers:.0%}" if pers is not None else "") + " — 깊은 벽 쏠림 없음"
        add("호가↔체결", t, EVID["wall_sr"], "설명")

    # ⑥ 청산맵 지지/저항 ↔ 현재가·깊은 벽
    sr = x.get("sr") or {}
    sb, rb = sr.get("sup_bp"), sr.get("res_bp")
    if sb is not None or rb is not None:
        parts = []
        if rb is not None:
            parts.append(f"저항 {sr.get('res'):,.1f}(+{rb:.0f}bp)")
        if sb is not None:
            parts.append(f"지지 {sr.get('sup'):,.1f}(−{sb:.0f}bp)")
        near = sr.get("near") or "없음"
        if near != "없음":
            t = " · ".join(parts) + f" — {near[:2]}에 붙어 있다: 청산이 날 자리"
            dw = sr.get("deep_wall")
            if dw:
                t += f", 깊은 벽은 {dw}"
                hold = dw == "레벨쪽"
                add("지지/저항", t, EVID["sr_near"] + " · " + EVID["sr_wall"], "약함" if hold else "설명",
                    (1 if near == "지지근접" else -1) if hold else 0)
            else:
                add("지지/저항", t, EVID["sr_near"], "설명")
        else:
            add("지지/저항", " · ".join(parts) + f" — 둘 다 ±{NEAR_BP:.0f}bp 밖", EVID["sr_near"], "설명")

    # ⑦ 선물 ↔ 현물 · 펀딩
    bas, bd, lead, fr, crowd = ev.get("basis_bp"), ev.get("basis_d_bp"), ev.get("lead"), ev.get("funding"), ev.get("crowd")
    if bas is not None:
        t = f"베이시스 {bas:+.1f}bp" + (f"(30분 {bd:+.1f})" if bd is not None else "")
        if bd is None:
            t += " — 30분 변화 수집 중"          # 모르는 것을 «아무도 안 끌었다»로 말하지 않는다
        elif not ev.get("dir"):
            t += " — 30분 추세가 없어 주도 판정 보류"
        else:
            t += {1: " — 선물이 이동을 끌었다", -1: " — 현물이 이동을 끌었다"}.get(lead or 0, " — 선물·현물 한쪽이 끈 흔적 없음")
        if fr is not None:
            t += f" · 펀딩 {fr * 100:+.4f}%" + {1: "(롱 쏠림)", -1: "(숏 쏠림)"}.get(crowd or 0, "")
        add("선물↔현물", t, EVID["basis"] + " · " + EVID["funding"], "미측정" if lead else "설명")

    # ⑧ ETH ↔ BTC
    rel, bm = ev.get("btc_rel"), ev.get("btc_move_bp")
    if bm is not None:
        add("ETH↔BTC", f"BTC 30분 {bm:+.0f}bp — " + {"동행": "같이 움직인다(시장 전체)", "단독": "ETH 혼자 움직인다"}.get(rel or "", "ETH 가 추세가 아니라 비교 보류"),
            EVID["btc"], "설명")

    # ⑨ 추세 veto ↔ 30분 이동
    veto, d30 = ev.get("veto") or 0, ev.get("dir") or 0
    if veto and d30:
        al = veto == d30
        add("추세↔이동", f"12시간 추세 {_updn(veto)} 안의 30분 {_updn(d30)} — " + ("정렬: 더 갈 쪽에 조금 기운다" if al else "추세 안의 짧은 역행: 지속은 동전"),
            EVID["veto_align"], "약함" if al else "설명", d30 if al else 0)

    # ⑩ 활동
    act, vp = x.get("act"), x.get("vol_pct")
    if vp is not None:
        edge = vp >= 0.8 or vp <= 0.2      # 가운데는 할 말이 없다 -- 근거 칩은 양끝에만
        add("활동", f"60초 거래량 같은 시간대 {100 * vp:.0f}분위" + (f"({act})" if act else "") + " — " + ("움직임이 커질 자리" if vp >= 0.8 else "움직임이 작을 자리" if vp <= 0.2 else "움직임 크기 평소"),
            EVID["act"], "근거" if edge else "설명")

    fused = fuse(ev, x)
    G = gauges(ev, x)
    for ln in L:   # 줄 이름 → 게이지 키(고래↔리테일/고래↔중형 → 고래 · 청산/청산↔추세/청산↔가격 → 청산)
        t = ln["topic"]
        g = G.get("고래" if t.startswith("고래") else "청산" if t.startswith("청산") else t)
        # 근거·약함 줄은 막대 부호 = 근거가 가리키는 방향(크기는 데이터). 🔴눌림 청산은 데이터(롱이 털림 = 아래)와
        #   근거(추세 방향 = 위)가 반대라, 이걸 안 하면 방향색 막대와 화살표가 서로 반대를 가리킨다.
        if g and ln["dir"]:
            g = dict(g, v=ln["dir"] * max(abs(g["v"]), 30.0), on=True)
        ln["g"] = g
    dirs = [ln for ln in L if ln["dir"]]
    up, dn = sum(1 for ln in dirs if ln["dir"] > 0), sum(1 for ln in dirs if ln["dir"] < 0)
    if not dirs:
        summary = "방향을 말할 근거가 있는 관계가 지금은 없다 — 아래는 상태 설명"
    else:
        names = " · ".join(f"{ln['topic']}({'↑' if ln['dir'] > 0 else '↓'})" for ln in dirs)
        summary = f"방향 근거 {len(dirs)}개: {names}" + (" — 서로 엇갈린다" if up and dn else "")
    return {"lines": L, "summary": summary, "up": up, "down": dn, "fused": fused}


def fuse(ev: dict[str, Any], x: dict[str, Any]) -> dict[str, Any]:
    """독립 원천 4표 + 크기 관문 → {side ±1/0, votes [(이름, ±1)], score, gate, gate_pct, text}. 연구(build_eval)와 같은 정의."""
    votes: list[tuple[str, int]] = []
    z = x.get("z60")
    if z:   # 고래는 한 표: 리테일과 갈리면 그걸로, 아니면 중형과 갈리면
        zw = z["whale"]
        for other, name in (("retail", "고래↔리테일"), ("mid", "고래↔중형")):
            if abs(zw) >= Z_SIDE and abs(z[other]) >= Z_SIDE and _s(zw) != _s(z[other]):
                votes.append((name, _s(zw)))
                break
    m60, oi60, p75 = x.get("move60"), x.get("oi60"), x.get("move60_p75")
    if m60 is not None and oi60 and p75 is not None and m60 < 0 and abs(m60) >= p75:
        votes.append(("가격↔OI", -1 if oi60 > 0 else 1))
    veto, d30 = ev.get("veto") or 0, ev.get("dir") or 0
    if veto and d30 and veto == d30:
        votes.append(("추세정렬", d30))
    if ev.get("reject") and d30:
        votes.append(("거부 봉", -d30))
    score = sum(v for _, v in votes)
    gp = x.get("range30_pct")
    gate = gp is not None and gp >= GATE_PCT
    side = _s(score) if (abs(score) >= FUSE_MIN_VOTES and gate) else 0
    tag = " · ".join(f"{n}{'↑' if v > 0 else '↓'}" for n, v in votes) or "켜진 표 없음"
    gtxt = "관문 ?" if gp is None else f"30분 폭 하루 {100 * gp:.0f}분위{'(통과)' if gate else '(미달)'}"
    if side:
        text = f"{'롱' if side > 0 else '숏'} · 다음 30분 — {tag} · {gtxt}"
    else:
        need = "표가 한쪽으로 2개 필요" if abs(score) < FUSE_MIN_VOTES else "크기 관문 미달"
        text = f"대기 — {tag} (합 {score:+d}) · {gtxt} · {need}"
    return {"side": side, "votes": votes, "score": score, "gate": gate, "gate_pct": gp, "text": text, "note": FUSE_EVID,
            "outcome": outcome3(ev, gp, votes, score, x.get("dir_p"))}


def outcome3(ev: dict[str, Any], gp: float | None, votes: list[tuple[str, int]], score: int,
             pdir: float | None = None) -> dict[str, Any] | None:
    """카드 3결과 — 위 먼저 · 아래 먼저 · 미도달의 실측 확률 + 목표가 + 그쪽으로 미는 조건. 입력이 모자라면 None.
    pdir(닿는다면 위 먼저일 확률, 방향 모델)가 있으면 닿을 확률(표)을 그 비율로 가른다. 없으면 표 그대로."""
    d = ev.get("dir")
    if d is None or gp is None:
        return None
    reg = "trend" if d else "range"
    a = max(-1, min(2 if d else 1, score * d if d else score))
    g = 0 if gp < 1 / 3 else (1 if gp < 2 / 3 else 2)
    n, p1, p2, pn = CARD_CELLS[(reg, a, g)]
    p_up, p_dn = (p1, p2) if (not d or d > 0) else (p2, p1)
    reach = round(100.0 - pn, 1)
    if pdir is not None and 0.0 < pdir < 1.0:
        p_up = round(reach * pdir, 1); p_dn = round(reach - p_up, 1)
    share = 100.0 * p_up / max(p_up + p_dn, 1e-9)
    mid, rg = ev.get("mid"), ev.get("range_bp")
    hi = lo = None
    if mid and rg:
        hi, lo = mid * (1 + SYM_K * rg / 1e4), mid * (1 - SYM_K * rg / 1e4)
    on = {("고래" if n_.startswith("고래") else n_, v) for n_, v in votes}   # 고래↔리테일/고래↔중형 → «고래» 한 표
    push = lambda side: [{"t": t, "on": (key, sg) in on} for key, sg, t in PUSH[side]]   # noqa: E731
    wide = {"t": f"30분 폭 하루 {100 * gp:.0f}분위 — 넓을수록 목표가 멀다", "on": g == 2}
    cols = [
        {"key": "up", "p": p_up, "target": hi, "dist_bp": (SYM_K * rg) if rg else None, "push": push(+1)},
        {"key": "dn", "p": p_dn, "target": lo, "dist_bp": (-SYM_K * rg) if rg else None, "push": push(-1)},
        {"key": "none", "p": pn, "band": [lo, hi] if hi else None, "push": [wide]},
    ]
    return {"reg": reg, "dir": d, "a": a, "g": g, "n": n, "cols": cols, "reach": reach, "up_share": round(float(share), 1),
            "dir_src": "model" if (pdir is not None and 0.0 < pdir < 1.0) else "table",
            "coin": bool(abs(share - 50.0) < DIR_COIN_PP), "dir_note": DIR_EVID,
            "note": f"닿을 확률 = 3.7년 실측 · 같은 상태 {n:,}번 · 다음 30분 안 ±{SYM_K:g}×30분 폭 중 하나에 닿는가"}


if __name__ == "__main__":   # 자체점검 — 관계마다 한 경우씩, 등급·방향이 연구가 허락한 만큼인지
    import json
    ev = dict(mid=2600.0, move_bp=-40.0, range_bp=60.0, cvd=900.0, liq_long=300_000.0, liq_short=20_000.0, veto=1, dir=-1,
              obi=0.1, persist=0.5, basis_bp=2.0, basis_d_bp=-1.0, lead=0, funding=0.0001, crowd=0, btc_rel="단독", btc_move_bp=3.0)
    x = dict(okx30=-500.0, cvd30_z=1.5, net60=dict(whale=800.0, mid=-100.0, retail=-600.0), z60=dict(whale=1.2, mid=-0.2, retail=-0.9),
             move60=-80.0, move60_p75=50.0, oi60=-3000.0, liq60s={}, imb40=0.4, imb40_pct=0.9, agree="중립", trigger_live=False,
             sr=dict(sup=2600.0, sup_bp=10.0, res=2700.0, res_bp=300.0, near="지지근접", deep_wall="레벨쪽"), act="활발", vol_pct=0.9)
    r = read(ev, x)
    T = {ln["topic"]: ln for ln in r["lines"]}
    assert "OKX 가 매도로 끌었다" in T["가격↔체결"]["text"] and T["가격↔체결"]["dir"] == 0           # 역행 + OKX 가 가격 편
    assert T["고래↔리테일"]["grade"] == "약함" and T["고래↔리테일"]["dir"] == 1 and "가격과 반대" in T["고래↔리테일"]["text"]
    assert T["가격↔OI"]["grade"] == "근거" and T["가격↔OI"]["dir"] == 1                              # 하락+OI↓ = 되돌림 쪽
    assert T["청산↔추세"]["dir"] == 1 and "눌림 청산" in T["청산↔추세"]["text"]                      # 상승추세에서 롱 청산
    assert T["호가↔체결"]["dir"] == 1 and "받아낼 쪽" not in T["호가↔체결"]["text"]                   # 매수벽·체결 매수 = 같은 쪽
    assert T["지지/저항"]["grade"] == "약함" and T["지지/저항"]["dir"] == 1
    assert T["활동"]["grade"] == "근거" and T["활동"]["dir"] == 0                                    # 크기 근거는 방향이 아니다
    assert {ln["topic"]: ln for ln in read(ev, dict(x, vol_pct=0.5))["lines"]}["활동"]["grade"] == "설명"
    assert "수집 중" in {ln["topic"]: ln for ln in read(dict(ev, basis_d_bp=None), x)["lines"]}["선물↔현물"]["text"]
    assert "추세↔이동" in T and T["추세↔이동"]["dir"] == 0                                           # 역행 = 동전
    assert r["up"] >= 4 and r["down"] == 0 and "방향 근거" in r["summary"]
    # 흡수: 제자리 + 뚜렷한 체결 / OKX 없음
    r2 = read(dict(ev, move_bp=5.0, cvd=-1200.0), dict(x, okx30=None, cvd30_z=-2.0))
    assert "흡수" in {ln["topic"]: ln for ln in r2["lines"]}["가격↔체결"]["text"]
    # 작은 1시간 이동은 OI 를 설명으로만 / 하락+OI↑ 는 아래
    T3 = {ln["topic"]: ln for ln in read(ev, dict(x, move60=-20.0))["lines"]}
    assert T3["가격↔OI"]["grade"] == "설명" and T3["가격↔OI"]["dir"] == 0
    assert {ln["topic"]: ln for ln in read(ev, dict(x, oi60=500.0))["lines"]}["가격↔OI"]["dir"] == -1
    # 기준 분포가 없으면 방향을 안 준다 · 아무 근거도 없으면 요약이 그렇게 말한다
    quiet = read(dict(ev, liq_long=0.0, liq_short=0.0, veto=0),
                 dict(x, z60=None, move60=-5.0, imb40_pct=0.5, vol_pct=0.5, sr=dict(sup=2600.0, sup_bp=80.0, near="없음")))
    assert quiet["up"] == quiet["down"] == 0 and quiet["summary"].startswith("방향을 말할 근거가")
    # 설명·미측정 등급은 절대 방향을 갖지 않는다
    assert all(ln["dir"] == 0 for rr in (r, r2, quiet) for ln in rr["lines"] if ln["grade"] in ("설명", "미측정"))
    # 빈 입력에서도 죽지 않고, 모르는 것은 «없다»로 말하지 않는다(서버 기동 직후 실측에서 «청산 거의 없다»가 나왔다)
    empty = read({}, {})
    assert all(ln["topic"] != "청산" for ln in empty["lines"]) and empty["up"] == empty["down"] == 0
    # 융합: 고래 한 표(리테일·중형 둘 다 갈려도 1) + OI(하락+OI↓ = +1) = +2, 관문 통과 → 롱
    f = fuse(dict(ev, reject=False), dict(x, range30_pct=0.8))
    assert f["side"] == 1 and f["score"] == 2 and [n for n, _ in f["votes"]] == ["고래↔리테일", "가격↔OI"]
    assert fuse(dict(ev, reject=False), dict(x, range30_pct=0.5))["side"] == 0                       # 관문 미달
    assert fuse(dict(ev, reject=False), dict(x, range30_pct=0.8, z60=None))["side"] == 0            # 한 표뿐
    # 정렬·거부 봉은 카드 방향에서 나온다: 상승 추세 + 30분 상승 = +1, 거부 봉 = −1 → 상쇄
    f2 = fuse(dict(ev, veto=1, dir=1, reject=True), dict(x, z60=None, move60=10.0, range30_pct=0.9))
    assert f2["score"] == 0 and f2["side"] == 0
    f3 = fuse(dict(ev, veto=-1, dir=-1, reject=False), dict(x, oi60=900.0, z60=None, range30_pct=0.9))
    assert f3["side"] == -1 and "숏" in f3["text"]                                                   # 하락+OI↑ ↓ + 정렬 ↓
    assert read(ev, dict(x, range30_pct=0.8))["fused"]["side"] in (-1, 0, 1)
    # 3결과: 하락 추세 + 숏 두 표(OI↓·정렬↓) + 넓은 창 → a=+2(이동 방향 정렬) g=2 → 아래 33.4 · 위 25.6 · 미도달 41.0
    o3 = f3["outcome"]; P = {c["key"]: c["p"] for c in o3["cols"]}
    assert (o3["reg"], o3["a"], o3["g"]) == ("trend", 2, 2) and P == {"up": 25.6, "dn": 33.4, "none": 41.0}
    assert abs(sum(P.values()) - 100) < 0.2
    assert [q["on"] for q in o3["cols"][1]["push"]] == [False, True, True, False]                  # 아래 열: OI·정렬 켜짐
    assert o3["cols"][0]["target"] > ev["mid"] > o3["cols"][1]["target"]
    # 횡보 + 두 표는 +1 칸으로 합친다(n 부족) · 입력 없으면 None
    o4 = outcome3(dict(ev, dir=0), 0.5, [("가격↔OI", 1), ("고래↔리테일", 1)], 2)
    assert (o4["reg"], o4["a"], o4["g"]) == ("range", 1, 1)
    assert outcome3({}, 0.5, [], 0) is None and outcome3(ev, None, [], 0) is None
    # 방향 모델: 닿을 확률(표 100 − 미도달)은 그대로, 위:아래 만 모델 비율로 가른다
    o5 = outcome3(dict(ev, veto=-1, dir=-1), 0.9, [], 0, 0.60); P5 = {c["key"]: c["p"] for c in o5["cols"]}
    assert o5["dir_src"] == "model" and o5["reach"] == 51.8 and P5["none"] == 48.2 and P5["up"] == 31.1 and P5["dn"] == 20.7
    json.dumps(o5); json.dumps(outcome3(dict(ev, veto=-1, dir=-1), 0.9, [], 0, np.float64(0.52)))   # np.bool_ 는 JSON 이 못 쓴다
    assert o5["up_share"] == 60.0 and not o5["coin"] and abs(sum(P5.values()) - 100) < 0.2
    o6 = outcome3(dict(ev, veto=-1, dir=-1), 0.9, [], 0, 0.52)
    assert o6["coin"] and outcome3(dict(ev, dir=-1), 0.9, [], 0, None)["dir_src"] == "table"
    # 피쳐: 봉 부족·간격 끊김 = None, 정상 = 26개 유한값
    rng = np.random.default_rng(0); px = 2600 * np.exp(np.cumsum(rng.normal(0, 8e-4, 299)))
    bars = [dict(time=1_790_000_100 + 300 * i, high=p * 1.001, low=p * 0.999, close=p, volume=100.0 + i, taker=55.0)
            for i, p in enumerate(px)]
    fz = card30_features(bars)
    assert fz is not None and len(fz) == 26 and all(np.isfinite(v) for v in fz.values())
    assert card30_features(bars[:292]) is None and card30_features(bars[:150] + bars[151:]) is None
    assert fz["dow"] == float(np.datetime64(bars[-1]["time"], "s").astype("datetime64[D]").item().weekday())
    # 게이지: 관계마다 붙고, 방향 부호가 줄의 해석과 같은 쪽이다
    Tg = {ln["topic"]: ln["g"] for ln in r["lines"]}
    assert all(g is not None for g in Tg.values()), [k for k, g in Tg.items() if g is None]
    assert Tg["고래↔리테일"]["v"] > 0 and Tg["가격↔OI"]["v"] > 0 and Tg["가격↔OI"]["on"]       # 고래 매수 · 하락+OI↓ = 위
    assert Tg["청산↔추세"]["v"] > 0 and Tg["호가↔체결"]["v"] > 0 and Tg["지지/저항"]["v"] == 70.0   # 눌림 청산 = 근거 쪽(위)
    assert all((ln["g"]["v"] > 0) == (ln["dir"] > 0) for ln in r["lines"] if ln["dir"])                # 색 막대 = 화살표 방향
    assert Tg["활동"]["kind"] == "m" and Tg["추세↔이동"]["v"] == 0 and Tg["추세↔이동"]["on"] is False  # 역행 = 0, 켜지지 않음
    assert all(-100 <= g["v"] <= 100 for g in Tg.values() if g["kind"] == "d")
    assert _k(35574) == "+36k" and _k(-2086) == "−2.1k" and _k(420) == "+420" and _k(0) == "0"
    assert gauges({}, {}) == {}
    assert gauges(ev, dict(x, move60_p75=0.0))["가격↔OI"]["on"] is False                           # 0 으로 나누지 않는다(퍼징이 잡음)
    print("flow_read selfcheck ok")
