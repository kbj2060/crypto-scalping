# XRP 증거신호 + 레짐모델 구축 (2026-09-03)

사용자 goal: *"xrp 코인도 증거신호와 레짐모델 개발해줘. 어제 오늘 이더리움에서 진행한 것처럼
같은 작업을 진행해줘. 대신 **비트코인에서 문제가 있었던 부분을 잘 참고해서** xrp에서는 그런
문제가 발생하지 않도록 주의해줘."*

절차 기준: `docs/homer/evidence_signal_new_coin_port_protocol.md` (§5 체크리스트 11단계)

---

## 1단계 · 데이터 실사 (착수 전) ✅

BTC에서 데이터를 가정했다가 파이프라인 3건이 터졌으므로 먼저 실사했다.

| 항목 | 상태 |
|---|---|
| klines | `binance_data/klines/XRPUSDT/XRPUSDT-5m-api.csv` **272,490행**, 2024-01-01 ~ 2026-08-04 |
| 5분 갭 | **0개** ✅ |
| 필수 컬럼 | `taker_buy_base` 포함 전부 존재 ✅ |
| 펀딩 | `funding_rate_other/XRPUSDT-fundingRate-*.zip` **31개월**(2024-01~2026-07), BTC와 같은 디렉토리 ✅ |
| 인덱스 오프셋 | `START`(2024-01-01) == klines 시작 ⇒ **오프셋 0** (그래도 하류는 타임스탬프 매핑 강제) |

⚠️**klines가 ETH(08-28)보다 24일 뒤처져 있다**(마지막 2026-08-04).
HOLDOUT(2026-04-01~)은 4개월분이 확보되어 연구 파이프라인엔 충분하다.
라이브 스코어러/섀도우는 API에서 직접 받으므로 영향 없다.
⚠️펀딩이 2026-07까지라 마지막 ~4일은 `funding_z`가 NaN이다.

## 2단계 · 후보 Tier0 CSV ✅

`scripts/build_xrp_5m_evidence_signal_candidates_tier0_20260903.py`
— BTC 빌더의 **자산 상수만** 바꾼 포팅(로직 재구현 0줄).

결과: **272,490행**, 2024-01-01 ~ 2026-08-04. NaN은 전부 워밍업 구간.

| 트리거 | bottom | top |
|---|---|---|
| `liquidity_sweep` | 7,500 | 7,051 |
| `taker_delta_z_climax` | 7,840 | 5,648 |
| `short_term_return_z` | 3,843 | 3,981 |
| `orthogonal_combo` | 2,076 | 1,015 |
| `fib_extension_exhaustion` | 905 | 839 |
| `local_extreme` | 15,749 | 15,947 |
| **any** | **28,120** | **26,834** |

⚠️**포팅 중 실제로 터진 것 1건**: 펀딩 로더가 `.dt.as_unit("us")`인데 klines CSV 파싱은 `[ns]`라
`merge_asof`가 `MergeError`로 죽었다(BTC 빌더 작성 당시엔 안 터졌다 — 환경 차이).
XRP 빌더에서 klines 단위에 맞춰 캐스팅해 해결. **BTC의 "파이프라인 3건"과 같은 계열이다.**

## 3단계 · ⭐HIT_TYPE 그리드스크린 (자산별 재탐색) ✅

**이 단계를 건너뛰면 BTC의 최대 사고가 재현된다** — 같은 이름의 신호가 자산 간에 HIT_TYPE·H·K가
전부 다른데, 서빙 코드가 원본을 따라가 라이브 hit률이 2.6배 과대평가됐다.

5신호 각각에 BTC 그리드스크린(4 HIT_TYPE × 6 HORIZON × 5 K = 120셀)을 포팅해 재탐색했다.
`scripts/research_xrp_<signal>_gridscreen_hittype_20260903.py`

### ⚠️자동 선택을 그대로 쓰지 않았다 — 표본 두께 감사

XRP는 **5신호 중 4개가 `touch_giveback_sustained`를 골랐다.** 그런데 이 HIT_TYPE은
BTC에서 이미 명시적으로 거부된 전례가 있다("mechanical global argmax … explicitly distrusted
for having only 2-5 OOS hits"). 조건이 둘(fast_mult AND giveback)이라 hit이 희소해져
lift는 커 보이지만 **몇 건 위에 서 있는 lift인지**를 봐야 한다.

`scripts/audit_xrp_gridscreen_selection_thickness_20260903.py`로 family별 최선을 hit 수와 함께 비교:

| 신호 | 자동 선택 | hits(b/t) | 두꺼운 대안 | hits(b/t) | **최종** |
|---|---|---|---|---|---|
| `taker_delta_climax` | giveback H=9 K=1.5, lift 1.45 | **481/233** | close_at_h lift 1.11 | 534/344 | ✅**자동 유지** |
| `liquidity_sweep` | giveback H=15 K=2.0, lift 1.36 | **385/311** | touch_mfe lift 1.07 | 2477/2613 | ✅**자동 유지** |
| `short_term_return_z` | touch_mae_capped H=12 K=1.5 | **746/652** | — | — | ✅**자동 유지** |
| `orthogonal_combo` | giveback H=12 K=2.5, lift 2.03 | **79/34** ⚠️ | touch_mfe H=8 K=2.0, lift 1.56 | **348/211** | ⭐**기각 → touch_mfe** |
| `fib_extension_exhaustion` | giveback H=16 K=3.25, lift 2.06 | **39/33** ⚠️ | touch_mfe H=10 K=1.5, lift 1.38 | **327/328** | ⭐**기각 → touch_mfe** |

판정 기준(BTC 전례 그대로): 측면별 hit이 한 자릿수~수십 건이면 그 lift를 믿지 않는다.
`short_term_return_z`는 스크립트 자신이 이미 얇은 argmax(giveback H=3, 40/21)를 거부하고
`touch_mae_capped`로 폴백했다 — 같은 판단이 자동으로 적용된 사례다.

⚠️`taker`/`liquidity_sweep`의 giveback은 **표본이 두꺼워서**(481/233, 385/311) 유지했다.
"giveback = 무조건 기각"이 아니라 **두께로 판단**한다.

### XRP 확정 라벨 (2026-09-03)

| 신호 | HIT_TYPE | H | K | 해상 필요 봉수 |
|---|---|---|---|---|
| `taker_delta_climax` | `touch_giveback_sustained` | 9 | 1.5 | **18** (2×H) |
| `liquidity_sweep` | `touch_giveback_sustained` | 15 | 2.0 | **30** (2×H) |
| `short_term_return_z` | `touch_mae_capped` | 12 | 1.5 | 12 |
| `orthogonal_combo` | `touch_mfe` | 8 | 2.0 | 8 |
| `fib_extension_exhaustion` | `touch_mfe` | 10 | 1.5 | 10 |
| `demarker_extreme` | plain touch (격자 진행 중) | — | — | — |
| `kalman_deviation_meanrev` | plain touch (격자 진행 중) | — | — | — |

⭐**해상 필요 봉수가 H와 다른 신호가 2개**다(giveback 계열). BTC에서 이걸 놓쳐
`liquidity_sweep`을 절반 시점에 잘못 확정했다. 서빙 코드는 반드시 `full_window`를 쓴다.

### 자산 3종 대조 (포팅 프로토콜 §1 갱신용)

| 신호 | ETH | BTC | **XRP** |
|---|---|---|---|
| `taker_delta_climax` | touch / 24 / 2.00 | close_at_h / 6 / 2.0 | **giveback / 9 / 1.5** |
| `liquidity_sweep` | touch / 30 / 4.00 | giveback / 20 / 2.0 | **giveback / 15 / 2.0** |
| `short_term_return_z` | touch / 12 / 1.75 | touch_mae_capped / 6 / 2.0 | **touch_mae_capped / 12 / 1.5** |
| `orthogonal_combo` | touch / 24 / 3.571 | touch / 8 / 2.0 | **touch_mfe / 8 / 2.0** |
| `fib_extension_exhaustion` | touch_and_mae / 20 / 2.35 | close_at_h / 10 / 2.75 | **touch_mfe / 10 / 1.5** |

**3자산이 모두 일치하는 신호는 하나도 없다.** 재스크리닝이 필수임을 다시 확인했다.

## 남은 단계

- [ ] 3b) `demarker_extreme` / `kalman_deviation_meanrev` 격자 (실행 중)
- [ ] 4) 신호별 TabPFN 메타라벨 → VAL/OOS/HOLDOUT
- [ ] 5) 동결 컨텍스트 + 라이브 스코어러 + 섀도우 러너 (HIT_SPEC에 `full_window` 반영)
- [ ] 6) 연구 구현 대조 검증 (무작위 800건, 불일치 0)
- [ ] 7) 결함 스캐너 (`audit_live_shadow_paths_defect_classes_20260903.py`)
- [ ] 8) 레짐 모델 (BTC S24_K3 절차 — ETH 승자 S12_K3이 BTC에서 3/10 최하위였으므로 재스크리닝 필수)
- [ ] 9) 대시보드 배선
