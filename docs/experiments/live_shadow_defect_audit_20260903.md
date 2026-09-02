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

## 남은 사항 (수정하지 않음, 기록만)

- 목표 도달 시 **칩을 끌지**는 결정하지 않았다. 지금은 `fired=True`를 유지하되 배지로 구분한다.
  칩을 끄면 이력이 사라지고 칩 깜빡임이 늘어난다. 사용자가 원하면 afterglow를 도달 시점에
  끝내는 것도 한 줄이다.
