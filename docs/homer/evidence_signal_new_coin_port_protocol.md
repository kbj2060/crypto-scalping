# 증거신호 **새 코인 포팅** 프로토콜

**언제 읽나**: ETH 증거신호를 새 자산(BTC/SOL/XRP/HYPE/…)으로 옮길 때, 또는 이미 옮긴 자산의
라이브/섀도우 코드를 손볼 때. 2026-09-02~03에 BTC 포팅에서 **5건의 결함**이 실제로 터졌고,
그 전부가 "포팅"이라는 행위 자체에서 나왔다.

> ## ⭐⭐한 줄 요약
> **포팅한 코드는 "원본과 같다"가 아니라 "어디가 달라졌나"부터 확인한다.**
> 라벨은 자산별로 재스크리닝되는데 서빙 코드는 원본을 따라가려는 관성이 있다.
> 그 틈에서 정확히 5건이 났다.

---

## 1. ⭐가장 중요한 표 — 같은 이름의 신호가 자산 간에 **전부 다르다**

라벨 스크립트에서 직접 확인한 값이다(2026-09-03). **이름만 보고 같다고 가정하면 안 된다.**

| 신호 | ETH HIT / H / K | BTC HIT / H / K | 같은가 |
|---|---|---|---|
| `taker_delta_climax` | **touch** / 24 / 2.00 | **close_at_h** / 6 / 2.0 | ❌ 모드·H 전부 다름 |
| `liquidity_sweep` | **touch** / 30 / 4.00 | **touch_giveback_sustained** / 20 / 2.0 (해상 **40봉**) | ❌ |
| `short_term_return_z` | **touch** / 12 / 1.75 | **touch_mae_capped** / 6 / 2.0 | ❌ |
| `fib_extension_exhaustion` | **touch_and_mae** / 20 / 2.35 | **close_at_h** / 10 / 2.75 | ❌ |
| `orthogonal_combo` | touch(K_hi) / 24 / 3.571 | touch / 8 / 2.0 | 모드만 같음 |
| `demarker_extreme` | touch / 8 / 0.70 | touch / 8 / 0.70 | ✅ 유일하게 동일 |
| `kalman_deviation_meanrev` | touch / 12 / 2.5 | touch / 10 / 3.5 | 모드만 같음 |
| `smt_divergence` | touch / 72 / 4.20 | (BTC 제외 — 교차자산 파트너 미해결) | — |

**8종 중 완전히 같은 것은 `demarker_extreme` 하나뿐이다.** BTC는 2026-09-01 그리드스크린에서
HIT_TYPE 자체를 3번째 축으로 재탐색했다(그 축 자체가 사용자 지적으로 추가됐다).

## 2. HIT_TYPE 4종 — 각각 "언제 결과가 확정되는가"가 다르다

이게 서빙 코드의 거의 모든 결정을 좌우한다.

| 모드 | 정의 | 확정 시점 | 조기 확정 |
|---|---|---|---|
| `touch` | MFE ≥ K×ATR (고가/저가) | **터치 봉** | ✅ 가능 (hit=1) |
| `touch_mae_capped` | 터치 + `MAE(터치 봉까지) ≤ K_LOSS×ATR` | **터치 봉** | ✅ 가능 (hit=0/1 둘 다) |
| `touch_and_mae` | `MFE≥K AND MAE<K_LOSS×K`, **둘 다 전 구간** | 호라이즌 끝 | ❌ (MAE 돌파 시 hit=0만 조기확정) |
| `close_at_h` | `close[i+H]`만 비교 | **H봉 종가** | ❌ 중간 터치 무의미 |
| `touch_giveback_sustained` | `close`기준 fast_move ≥ K×ATR **AND** `giveback ≤ ceiling` | **i+FULL_WINDOW** | ❌ |

⚠️`touch_giveback_sustained`(BTC `liquidity_sweep`)는 특히 함정이 많다:
- `fast_move`가 **종가** max/min이다(고가/저가 아님)
- `peak`은 고가/저가지만 **더 긴 창**(`FULL_WINDOW = 2×H = 40봉`)에서 잡는다
- `giveback = (peak − close[i+40]) / (peak − entry)`, `hit = fast≥K AND giveback ≤ 0.20`
- ⇒ **해상에 40봉이 필요하다.** 호라이즌(20)으로 착각하면 절반 시점에 잘못 확정한다.

## 3. 서빙 코드가 지켜야 할 계약 5개

### C1. 이벤트 봉의 피쳐를 쓴다 — `.iloc[-1]`(최신 봉) 금지
```python
# ❌ 틀림: 봉별 캐시처럼 보이지만 값은 항상 최신 봉
atr = pd.Series(tr).rolling(14).mean().iloc[-1]; atr_map[bar] = atr
# ✅ 맞음: 신호 봉 위치로 인덱싱 (TR이 1봉 밀리므로 NaN 패딩으로 정렬)
atr = _atr_series(bars).iloc[fire_pos]
```
이 ATR은 HIT 문턱(`k×atr`)에 직접 들어간다. 실측 오차 **12.3%**.
⭐기존 ETH 배포 코드는 이미 이 규율이 있었다 — `_tp_price` docstring:
"the **fire bar's own** close … the **fire bar's own** atr_pct", 재시작 복구도
"inference AT THAT bar's own features … **not 'now'**".

### C2. 저장된 행 인덱스(`pos`/`bar_idx`)를 다른 파일의 인덱스로 쓰지 않는다
빌더의 `pos`는 **빌더가 받은 프레임**의 행 인덱스다. BTC 후보 CSV는 2024-01-01 시작,
raw klines는 2023-12-31 15:00 시작 ⇒ **오프셋 108봉(9시간)**.
**타임스탬프로 매핑하고 정확 일치를 검증**한다(불일치 시 예외).

### C3. 표시·해상 규칙 = 그 신호 **자신의 라벨 확정 규칙**
칩 점등, votes/net_score, pending 해상이 전부 같은 규칙을 따라야 한다.
한 화면 안에서 두 규칙이 공존하면 가장 헷갈리는 종류의 불일치가 생긴다.
ETH에서는 `_active`(고정 봉수)와 `_fill`(터치 종료)이 실제로 갈라져 있었다.

### C4. 조기 확정은 **결과가 그 시점에 완전히 결정되는 모드에서만**
§2 표의 "조기 확정" 열을 그대로 따른다. `touch`에서 유효한 최적화를
`touch_giveback_sustained`에 적용하면 **hit률이 부풀려진다**(실측 2.6배).

### C5. 영구 미해소 상태를 만들지 않는다
조회 창(`limit`)보다 오래 러너가 멈추면 신호 봉이 밀려나 pending이 **영원히** 남는다.
창을 넉넉히 잡고(BTC는 60→**500봉≈41시간**), 그래도 밖이면 `expired`로 **분리 기록**한다.
⚠️원장이 아니라 별도 리스트에 넣는다 — hit률 집계를 오염시키면 안 된다.

## 4. 실제 사고 5건 (증상 → 원인 → 영향)

| # | 증상 | 원인 | 영향 |
|---|---|---|---|
| 1 | 경제성게이트 **0/672 전패** | `pos`를 raw klines 인덱스로 사용, 오프셋 108봉 | 결론이 **6/7 통과**로 뒤집힘 |
| 2 | 섀도우 원장 9건 **전부 양수 +69bp**(HOLDOUT의 10배) | 배리어를 **폴링 마크가격 한 점**으로 판정 → wick 놓침 | 손실 트레이드가 통째로 사라짐 |
| 3 | 보유한도가 조용히 5배 | `ticks >= MAX_HOLD_BARS * 5`가 "1틱=1분" 가정 | 루프 주기 변경 시 발현 |
| 4 | 목표 달성 후에도 칩 **최대 6시간 활성** | `_active`가 고정 봉수, 라벨 확정 무시 | 늦은 진입 유도 |
| 5 | 라이브 hit률 **2.6배 과대평가** | `HIT_SPEC` 모드 2건 오기(touch로 뭉갬) | 러너의 존재 이유(hit률 대조)가 무효 |

⭐**2번의 진단이 옳았다는 독립 증거**: 무작위 진입 귀무분포 평균이 −6.16~−7.15bp였는데,
버그 있던 실행의 격자 최선이 −5.76~−6.56bp였다. 두 분포가 겹친다 = 버그가 정확히 무작위 진입.

⭐**5번이 옳았다는 방증**: 정정 후 `liquidity_sweep` 라이브 hit률이 0.375 → **0.143**으로,
학습 hit률 **0.1022**에 훨씬 가까워졌다.

## 5. 새 코인 포팅 체크리스트 (이 순서대로)

- [ ] **1) 라벨 스크립트를 직접 읽는다.** 문서 표나 다른 자산의 값을 믿지 않는다.
      신호별로 `HIT_TYPE / HORIZON / K / GAP / 추가 파라미터(K_LOSS_MULT, FULL_WINDOW,
      GIVEBACK_CEILING)`를 표로 만든다. §1 같은 대조표를 남긴다.
- [ ] **2) 해상에 필요한 봉 수**를 신호별로 계산한다 (`full_window` 있으면 그 값, 아니면 H).
      조회 창(`limit`)이 그보다 충분히 큰지 확인한다.
- [ ] **3) 조기 확정 가능 여부**를 §2 표로 정한다. 모드별로 다르다.
- [ ] **4) 라벨 빌더를 재구현하지 말고 import**한다. 재구현하면 자산별 파라미터가 조용히 어긋난다.
      (⚠️각 스크립트의 prep 함수 시그니처가 다르다 — 첫 prep은 무인자 로더, 이후는 프레임을 받는다)
- [ ] **5) 인덱스는 타임스탬프 매핑**(C2). 후보 CSV와 klines의 시작 시각·행 수를 먼저 대조한다.
- [ ] **6) ATR/피쳐는 이벤트 봉 기준**(C1).
- [ ] **7) 표시/해상 규칙을 라벨 확정 규칙에 맞춘다**(C3).
- [ ] **8) ⭐연구 구현과 직접 대조 검증**을 붙인다 — 무작위 400지점 × 양측 = 신호당 800건을
      그 신호의 **연구 스크립트 원본 함수**(`hit_touch_mae_capped` 등)와 비교해 **불일치 0**을 확인.
      이게 이번에 유일하게 확실했던 검증이다.
- [ ] **9) 결함 스캐너를 돌린다**:
      `python scripts/audit_live_shadow_paths_defect_classes_20260903.py`
      (P1 최신봉 피쳐 / P2 폴링 배리어 / P3 영구 미해소 / P4 주기 의존 시간)
      ⚠️정적 히트는 **후보 목록**이지 판정이 아니다.
- [ ] **10) 대조군이 덮는 경로를 명시**한다. ETH로 대조군을 돌려 통과해도 **BTC 경로는 안 지나간다**
      (실제로 그래서 사고 1번을 못 잡았다).
- [ ] **11) 배포 계약**(CLAUDE.md): md5 대조 → `ast.parse` → 재시작 → **서빙 바이트 재확인** →
      `check_deploy_drift.sh` → 커밋 → 푸시.

## 6. 이번에 만든 재사용 자산

| 파일 | 용도 |
|---|---|
| `scripts/audit_live_shadow_paths_defect_classes_20260903.py` | 결함 4종 정적 스캐너(14경로) |
| `scripts/live_btc_evidence_signal_shadow_runner_20260902.py` | `HIT_SPEC` 4모드 + `_resolve_bars()` + `expired` 분리의 **참조 구현** |
| `scripts/live_evidence_signal_dashboard_20260823.py` | `_fill_until_tp_or_horizon(mode=...)` — 라벨 확정 규칙으로 칩/votes 채우기 |
| `docs/experiments/live_shadow_defect_audit_20260903.md` | 4부 구성 전체 감사 기록(증상·원인·검증 수치) |

## 7. 남은 것

- [ ] SOL/XRP/HYPE는 아직 증거신호 포팅 자체를 안 했다. 하게 되면 §5 체크리스트를 그대로 쓴다.
- [ ] BTC 증거신호 패널은 현재 프론트엔드에서 제거된 상태다(서버 엔드포인트
      `/api/btc-evidence-shadow`는 살아 있음). 되살릴 때 §1 표의 BTC 값이 반영됐는지 확인할 것.
- [ ] 이 문서의 §1 표는 **라벨을 다시 스크리닝하면 무효가 된다.** 재스크리닝 시 반드시 갱신한다.

## 관련 문서

- 전체 감사 기록: `docs/experiments/live_shadow_defect_audit_20260903.md`
- BTC 구축/경제성: `docs/experiments/btc_evidence_signal_and_shadow_20260902.md`,
  `docs/experiments/btc_evidence_signal_economics_gate_20260902.md`
- 경제성 재측정 절차: `docs/homer/evidence_signal_economics_tuning_protocol.md`
- 신호 재사용(피더) 절차: `docs/homer/v_rebound_feeder_signal_protocol.md`
