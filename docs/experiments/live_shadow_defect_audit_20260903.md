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
