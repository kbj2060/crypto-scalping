# 라이브/섀도우 경로 결함 유형 일괄 점검 (2026-09-03)

사용자: *"고쳐줘. 전반적으로 전체를 한 번 점검해줘. 지금 비트코인 증거신호만 모두 점검한거야?"*

**답: 그렇다. 그때까지 점검한 건 BTC 증거신호 섀도우뿐이었다.** 이 문서는 같은 결함 유형을
서버에서 실제로 돌고 있는 **14개 경로 전부**로 확대한 결과다.

스캐너: `scripts/audit_live_shadow_paths_defect_classes_20260903.py`
⚠️정적 스캔은 **후보 목록**이지 판정이 아니다. 아래 판정은 각 히트를 직접 읽고 사람이 채웠다.

## 점검한 결함 유형 4종 (전부 실제로 발생했던 것)

| | 유형 | 실제 사례 |
|---|---|---|
| **P1** | 이벤트 봉이 아닌 **최신 봉**의 피쳐를 쓴다 | BTC 증거신호 `record()` — HIT 문턱 `k×atr`이 어긋남 |
| **P2** | 배리어/청산을 **폴링 가격 한 점**으로 판정 | ETH V자반등 섀도우 — 원장 9건 전부 양수 +69bp(HOLDOUT의 10배) |
| **P3** | **영구 미해소 상태** — 창보다 오래 멈추면 pending이 안 풀림 | BTC 증거신호 — `limit=60`봉=5시간 |
| **P4** | **루프 주기에 의존하는 시간 계산** | ETH V자반등 — `ticks >= MAX_HOLD_BARS * 5` |

## 결과

| 경로 | P1 | P2 | P3 | P4 | 판정 |
|---|---|---|---|---|---|
| `live_btc_evidence_signal_shadow_runner` | ❌→✅ | 해당없음(포지션 없음) | ❌→✅ | ✅ 봉 인덱스 기준 | **오늘 수정** |
| `live_eth_v_rebound_econ_shadow_runner` | ✅ | ❌→✅ | ✅ | ❌→✅ | **어제·오늘 수정** |
| `live_btc_evidence_signal_metalabel` | ✅ | — | — | — | 이상 없음 |
| `live_eth_v_rebound_econ_autotrade_signal` | ✅ | — | — | — | 이상 없음 |
| ⭐`live_evidence_signal_metalabel_20260829` (**배포 칩**) | ✅ | — | — | — | **이상 없음** |
| ⭐`live_eth_sweep_v_rebound_signal_20260829` (**배포 칩**) | ✅ | — | — | — | **이상 없음** |
| `live_evidence_signal_dashboard_20260823` | ✅ | — | — | — | 이상 없음 |
| `maker_fill_shadow_worker` | ✅ | — | — | — | 이상 없음 |
| `run_btc_multislot_shadow_loop_20260807` | — | — | 주석뿐 | — | 이상 없음 |
| `live_eth_odyssey4_zig075_entry_veto_shadow` ×2 | ✅ | — | — | — | 이상 없음 |
| `l2_anomaly_snapshot_collector` | — | — | — | ✅ | 이상 없음(오탐) |
| `liq_magnet_collector` | ✅ | — | — | — | 이상 없음 |
| `dashboard/server.py` | — | 다른 용도 | 주석뿐 | — | 이상 없음 |

## ⭐가장 중요한 발견 — 결함은 **내가 어제 새로 쓴 두 러너에만** 있었다

기존 배포 경로는 이미 올바른 규율을 갖고 있었다. `live_evidence_signal_metalabel_20260829.py`의
`_tp_price` docstring은 이렇게 명시한다:

> entry (**the fire bar's own close**, exactly as every offline label script uses) moved by
> k*atr_pct (**the fire bar's own atr_pct**, exactly as move_atr_mult was computed at training time)

재시작 복구 경로까지 "**inference AT THAT bar's own features** (matching how every offline label
script defines entry -- the anchor bar's own close/atr_pct, **not 'now'**)"로 못박아 뒀다.
`live_eth_sweep_v_rebound_signal_20260829.py`도 `atr[i]` / `atr[pos-1]`로 봉 인덱스를 쓴다.

⇒ **새 라이브 코드를 쓸 때 기존 배포 코드의 규약을 먼저 읽었어야 했다.**
이 저장소의 CLAUDE.md에 이미 같은 취지의 계약(Position-Feature Train/Inference Parity)이
있는데, 그건 학습/추론 파리티에 관한 것이고 **"이벤트 봉 vs 최신 봉"** 축은 별도로 명시돼
있지 않았다. 그래서 이 문서를 남긴다.

## 오늘 적용한 수정

### (1) BTC 증거신호 — ATR을 신호 봉 기준으로

예전:
```python
atr = float(pd.Series(tr).rolling(14).mean().iloc[-1])   # 항상 최신 봉
atr_map[bar] = atr                                        # 봉별 캐시처럼 보이지만 값은 동일
```
지금: `_atr_series(bars)`가 봉별 ATR을 만들고(TR이 1봉 밀리는 것을 NaN 패딩으로 정렬),
신호 봉 위치 `bpos`로 인덱싱한다.

검증: 합성 60봉에서 수동 계산과 **1e-9 이내 일치**. 같은 데이터에서
**최신봉 ATR 106.59 vs 신호봉(30) ATR 94.91 — 12.3% 차이.** 오차는 실재했다.

### (2) 죽은 인자 제거

`manage(s, bars, px)`의 `px`가 봉 기준 전환 후 함수 안에서 쓰이지 않게 됐다
(내 어제 변경이 만든 orphan). 시그니처에서 제거했다. `px`는 여전히 `enter()`의 진입가와
로깅에 쓰인다 — 즉 **진입은 마크가격, 청산은 봉 고가/저가**로 명확히 갈렸다.

## 남은 사항 (수정하지 않음, 기록만)

- ETH V자반등 섀도우의 **진입가는 마크가격**이고 규격서는 "다음 봉 시가"다. 실행 강건성
  테스트에서 진입 1~2봉 지연에도 +8.93/+7.56bp로 견고했던 것이 근거이고, 봉 평가는
  진입이 일어난 봉부터 시작하므로 정합적이다. 더 엄밀히 하려면 다음 봉 시가로 바꿀 수 있으나
  라이브 거동과는 오히려 멀어진다.


---

# 2부 — ETH 증거신호 전체 재점검 (2026-09-03)

사용자: *"eth 증거신호들도 전체 재점검 진행해줘. 익절을 했음에도 자신이 신호를 갖고있다고
착각하는 문제도 같이 점검해줘"*

## ⭐결론: BTC와 **같은 문제가 ETH 배포 칩에도 있었다**. 구조는 다르지만 증상은 같다.

ETH 증거신호는 pending 원장이 아니라 **afterglow(발동 후 `horizon_bars` 유지)** 방식이다.
그런데 그 유지 창 동안 **익절가 도달 여부를 전혀 확인하지 않았다.**

| 신호 | afterglow | 목표 달성 후에도 "활성"으로 남는 최대 시간 |
|---|---|---|
| `smt_divergence` | 72봉 | **6시간** |
| `liquidity_sweep` | 30봉 | 150분 |
| `taker_delta_z_climax` / `orthogonal_combo` | 24봉 | **2시간** |
| `fib_extension_exhaustion` | 20봉 | 100분 |
| `short_term_return_z` / `kalman_deviation_meanrev` | 12봉 | 60분 |
| `demarker_extreme` | 8봉 | 40분 |

BTC보다 **더 나쁜 상황이었다** — 이건 사용자가 실제로 보는 **배포된 대시보드**이고,
칩이 **익절가를 함께 표시**하기 때문이다. 이미 달성된 목표를 활성 목표처럼 띄우면
**보고 뒤늦게 진입하게 된다.**

## 수정

`live_evidence_signal_metalabel_20260829.py`에 `_tp_touched()` 신설:
발동봉 **다음 봉부터** 현재까지 고가/저가가 `tp_price`에 닿았는지 본다(라벨 컨벤션과 동일).

⚠️**표시 전용 사실**이다. 각 신호의 학습 라벨 HIT 정의와 반드시 같지 않다(일부는 종가 기준).
여기서는 사람이 화면을 보고 판단할 때 의미 있는 "가격이 목표에 닿았나"를 고가/저가로 답한다 --
닿음을 **더 이르게** 잡는 쪽이라 늦은 진입을 막는 안전한 방향이다.
**모델 확률도 발동 여부도 바꾸지 않는다.**

출력 경로가 6개(신규발동/캐시재사용/afterglow/재시작복구/미발동/워밍업)라 각 경로에 넣지 않고
**함수 끝에서 한 번에 후처리**한다 — 경로가 늘어도 빠뜨리지 않는다.

대시보드: `model_tp_touched` / `model_bars_since_fire`를 페이로드에 추가하고,
칩 제목줄에 **🎯 목표 도달 · N봉 전 발동** 배지를 띄운다(저ATR 경고와 같은 pill 규격).
익절 텍스트도 `익절 2393.01` → `익절 2393.01 도달`로 바뀐다.

## 검증

| 검정 | 결과 |
|---|---|
| ① 롱 도달 / ② 롱 미달 | ✅ |
| ⭐③ **발동 이전** 봉의 도달은 세지 않음 | ✅ |
| ④ 숏 도달 | ✅ |
| ⑤ **발동봉 자신**의 고가는 세지 않음 | ✅ |
| ⑥ `tp_price=None` → None | ✅ |
| ⑦ 발동봉이 마지막 봉 → False | ✅ |
| ⑧ 후처리가 6개 반환 경로 전부 커버 | ✅ |

**배포 직후 실제 사례를 즉시 잡았다:**

```
taker_delta_z_climax   fired=True  tp=2393.01  touched=True   bars_since=14
orthogonal_combo       fired=True  tp=2402.13  touched=False  bars_since=14
smt_divergence         fired=True  tp=2403.27  touched=False  bars_since=10
liquidity_sweep        fired=True  tp=2401.97  touched=False  bars_since=10
```

`taker_delta_z_climax`는 14봉(70분) 전 발동했고 **익절가에 이미 도달**했다.
구버전이면 24봉(2시간) 내내 활성 목표로 계속 띄웠을 상황이다.

## 3부 — 내부 로직(라벨)에 맞추기

사용자: *"내부 로직에 맞춰줘. 내부 로직은 목표 도달되면 어떻게 하지?"*

### ⚠️먼저 2부의 진단을 정정한다

같은 화면 안에 **서로 다른 규칙 두 개**가 이미 공존하고 있었다:

| 컬럼 | 쓰이는 곳 | 규칙 |
|---|---|---|
| `_active` | **칩 점등**(bottom_fired/top_fired) · votes · net_score | 고정 봉수 `rolling(n).max()` — **목표 도달 무시** |
| `_fill` | 히스토리 스트립 | `_fill_until_tp_or_horizon` — **터치에서 종료**(2026-09-01 신설) |

2부에서 "ETH 칩은 목표 도달을 안 본다"고 쓴 것은 `_active`에 대해서는 맞지만,
**저장소에 이미 올바른 로직(`_fill`)이 있었다**는 사실을 빠뜨렸다. 문제는 "없다"가 아니라
**"같은 화면의 두 요소가 서로 다른 규칙을 쓴다"**였다.

### 내부 로직의 답 — 신호마다 다르다 (라벨 스크립트 직접 확인)

| 신호 | 라벨 HIT 정의 | 목표 도달 시 |
|---|---|---|
| `taker_delta_z_climax` | `hit = touched` (MFE≥2.0×ATR) ⚠️v5의 "touched AND end>0"은 AUC 하락으로 **기각**됨 | **즉시 확정** |
| `short_term_return_z` | touched (intrabar MFE) | **즉시 확정** |
| `liquidity_sweep` | `high[i+1:i+H+1].max()` MFE | **즉시 확정** |
| `orthogonal_combo` | move_atr_mult ≥ K_hi (MFE 기반, exclude-middle은 학습 전용) | **즉시 확정** |
| `smt_divergence` | `high[...].max()` MFE | **즉시 확정** |
| `demarker_extreme` · `kalman_deviation_meanrev` | plain 터치(`peak >= K`) — README:173 "plain으로 원위치" | **즉시 확정** |
| ⚠️`fib_extension_exhaustion` | **MFE≥K AND MAE<2.0×K, 둘 다 전 구간**("regardless of which happens first") | **확정 안 됨** — 이후 되돌림이 크면 hit=0으로 뒤집힌다 |

⇒ **7종은 터치 = 사건 종료.** `fib`만 예외다. 다만 MAE는 단조증가하므로 **fib는 MAE가
2.0×K를 넘는 순간 hit=0이 확정**된다 — 그 시점엔 종료할 수 있다.

### 수정

- `_active`를 `_fill`에서 유도하도록 통일했다(고정 봉수 `rolling.max()` 제거).
  ⇒ 칩 점등 · votes · net_score가 전부 라벨 확정 규칙을 따른다.
- `_fill_until_tp_or_horizon`에 `mode`를 추가: `touch`(7종) / `touch_and_mae`(fib).
  fib는 터치로 끝나지 않고 **MAE 돌파에서** 끝난다.
- 메타라벨의 `fired`를 `_active` 컬럼에 **종속**시켰다. 독립 판단하면 칩 점등과 확률 표시가
  어긋날 수 있는데, 그건 화면에서 가장 헷갈리는 종류의 불일치다.
  ⚠️`fired=False`가 되어도 `tp_price`/`tp_touched`/`bars_since_fire`는 남긴다 —
  칩이 **왜** 꺼졌는지("🎯 목표 도달 · N봉 전 발동") 설명할 수 있어야 하기 때문이다.

### 블래스트 반경 (명시)

`_active` 변경은 **votes / net_score에도 영향**을 준다. 의도한 것이다 — 라벨이 확정된 사건은
더 이상 "지금 유효한 증거"가 아니므로 종합 점수에 계속 표를 던지면 안 된다.
같은 버그의 집계 레벨 판본이었다.

### 검증 (9종 전부 통과)

`_active`가 `_fill`에서 유도됨 / 고정 봉수 제거 / HIT_RESOLUTION 8종 / FIB_K_LOSS_MULT=2.0 /
mode 전달 · 알고리즘 동등구현으로 **touch는 도달봉에서 종료**, **touch_and_mae는 도달해도 계속**,
**touch_and_mae는 MAE 돌파에서 종료** 확인.

**배포 전후 실측:**

```
전:  taker_delta_z_climax  fired=True   tp=2393.01  touched=True   bars_since=14
후:  taker_delta_z_climax  fired=False  tp=2393.01  touched=True   bars_since=15   ✅ 종료됨
     orthogonal_combo      fired=True   touched=False  (미확정 -- 계속 활성)
     smt_divergence        fired=True   touched=False  (미확정 -- 계속 활성)
```

## 남은 사항 (수정하지 않음, 기록만)

- `scripts/live_btc_evidence_signal_metalabel_20260902.py`에도 `_fill_until_tp_or_horizon`
  복제본이 있고 **터치 전용**이다. BTC 신호들의 라벨 모드는 ETH와 다르게 재스크리닝됐으므로
  (`taker`/`fib`는 `close_at_h`) 같은 정합화가 필요하다. 다만 BTC 패널은 현재 프론트엔드에서
  제거된 상태라 표시 영향이 없어 이번 범위에서 뺐다.


---

# 4부 — BTC 증거신호 재점검 (2026-09-03)

사용자: *"비트코인은 이더리움에서 파생됐을텐데 내부 로직이 굉장히 유사할거야. 비트코인 증거신호
재점검해줘"*

**직관은 맞았지만 결론은 반대다 — 파생됐기 때문에 오히려 어긋났다.**
BTC는 ETH에서 포팅한 뒤 **자산별로 HIT_TYPE을 재스크리닝**했는데, 섀도우 러너의 `HIT_SPEC`은
그 재스크리닝 결과를 **일부만** 반영하고 있었다.

## ⭐발견: 7종 중 2종의 라벨 모드가 틀렸다

각 신호의 라벨 스크립트를 직접 읽어 대조했다.

| 신호 | 실제 라벨 정의 | 러너 `HIT_SPEC` | |
|---|---|---|---|
| `taker_delta_climax` | `close[i+6] >= entry + 2.0×atr` | close_at_h | ✅ |
| **`liquidity_sweep`** | **touch_giveback_sustained** | **touch** | ❌ |
| `kalman_deviation_meanrev` | touch MFE | touch | ✅ |
| **`short_term_return_z`** | **touch_mae_capped** | **touch** | ❌ |
| `orthogonal_combo` | touch MFE | touch | ✅ |
| `demarker_extreme` | touch MFE | touch | ✅ |
| `fib_extension_exhaustion` | `close[i+10] >= entry + 2.75×atr` | close_at_h | ✅ |

### `liquidity_sweep` — 가장 복잡했다

```
fast_move = close[i+1 : i+20+1].max() - entry      ← **종가** 기준(고가 아님)
peak      = high[i+1 : i+40+1].max()
giveback  = (peak - close[i+40]) / (peak - entry)
hit = fast_move/atr >= 2.0  AND  giveback <= 0.20
```

⇒ 해상에 **40봉(200분)이 필요**하다(호라이즌 20이 아니라). 그리고 **조기 확정이 불가능**하다.
러너는 고가 터치 하나로 hit=1을 찍고 있었다 -- 게다가 어제 넣은 조기 확정이 이걸 더 악화시켰다.

### `short_term_return_z` — 조기 확정은 오히려 valid

```
touch_bar = [i+1, i+6]에서 처음 목표에 닿은 봉
MAE = entry - low[i+1 : touch_bar+1].min()          ← **터치 봉까지만** 측정
hit = 1 iff MAE <= 2.0 × atr
```

MAE를 터치 시점까지만 재므로 **그 봉에서 결과가 완전히 확정**된다. 조기 확정이 정당하다 --
단 MAE 조건을 함께 봐야 한다.

## 영향 — 러너의 존재 이유가 무효였다

이 러너의 목적은 **라이브 hit률이 학습 hit률을 재현하는지** 관측하는 것이다.
틀린 규칙은 그 비교를 통째로 망친다:

| `liquidity_sweep` | hit | n | 라이브 hit률 | 학습 hit률 | 격차 |
|---|---|---|---|---|---|
| 정정 전(터치) | 3 | 8 | **0.375** | 0.1022 | +0.273 (엄청난 초과달성처럼 보임) |
| **정정 후** | **1** | **7** | **0.143** | 0.1022 | **+0.041** |

**2.6배 과대평가**였다. 정정 후 값이 학습 hit률에 훨씬 가깝다는 것 자체가 수정이 옳다는 방증이다.

## 수정

- `HIT_SPEC`에 4개 모드 정의 + 근거 주석(각 라벨 스크립트의 파일:줄 표기)
- `resolve()`를 4모드로 재구현. **조기 확정은 결과가 그 시점에 완전히 결정되는 모드에서만**:

| 모드 | 조기 확정 | 이유 |
|---|---|---|
| `touch` | ✅ 터치 봉 | hit=1 확정, 더 볼 것 없음 |
| `touch_mae_capped` | ✅ 터치 봉 | MAE를 터치 시점까지만 재므로 그 봉에서 완전 확정 |
| `close_at_h` | ❌ | H봉 **종가**로만 판정 |
| `touch_giveback_sustained` | ❌ | `close[i+FULL_WINDOW]`가 필요 |

- 해상에 필요한 봉 수를 `_resolve_bars()`로 분리(giveback만 `full_window`)
- BTC 패널(`live_btc_evidence_signal_metalabel_20260902.py`)의 `_fill_until_tp_or_horizon`도
  같은 `mode`를 받도록 정합화. `FILL_SPEC`에 mode/full_window 추가.
- 틀린 규칙으로 판정됐던 **원장 10건을 pending으로 되돌려 재판정**했다
  (상태 백업: `btc_evidence_signal_shadow_state.json.bak_pre_hitmode_fix_20260903`).

## 검증 — 연구 구현과 직접 대조

무작위 400 지점 × 양측 = **신호당 800건**을, 각 신호의 **연구 스크립트 원본 구현**
(`hit_touch_mae_capped` / `hit_touch_giveback_sustained` / close_at_h / touch)과 대조:

```
✅ short_term_return_z        n=800  불일치 0
✅ liquidity_sweep            n=800  불일치 0
✅ taker_delta_climax         n=800  불일치 0
✅ fib_extension_exhaustion   n=800  불일치 0
✅ kalman_deviation_meanrev   n=800  불일치 0
✅ demarker_extreme           n=800  불일치 0
✅ orthogonal_combo           n=800  불일치 0
⇒ 7모드 전부 연구 구현과 일치
```

재기동 후 실제 재판정 로그에서 `liquidity_sweep`이 **40봉**에 확정되는 것을 확인했다.

## ⚠️교훈

**"ETH에서 파생됐으니 로직이 같을 것"이 정확히 함정이었다.** BTC는 포팅 후
HIT_TYPE을 자산별로 재스크리닝했고(그 재스크리닝 자체가 사용자 지적으로 추가된 축이다),
서빙 코드는 그 결과를 부분적으로만 따라갔다.
**포팅한 코드는 "원본과 같다"가 아니라 "어디가 달라졌나"부터 확인해야 한다.**
