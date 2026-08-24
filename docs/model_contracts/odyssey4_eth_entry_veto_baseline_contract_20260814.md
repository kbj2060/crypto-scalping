# Odyssey4 — ETH zig075 진입거부 베이스라인 계약 문서 (2026-08-14)

## 상태

| 컴포넌트 | 상태 |
|---|---|
| **Odyssey4 베이스라인 확정** | `locked (연구 확정, 섀도우 미배포)` — Odyssey3 베이스라인(`asymmetric_tabm_liveatr` + h48qual 레짐인지형 exit 가드) **+ zig075 지속상승장 SHORT 진입거부**(실행 로그 #2, Odyssey3의 지속상승장 탐지기를 신규 자유변수 없이 entry-side veto로 재사용). 이 시점부터 Odyssey4는 이 상태를 새 비교 기준(reference)으로 삼는다. **주의**: 이 진입거부 계층은 아직 **어떤 프로세스에도 배포되지 않았다** — h48qual 레짐 가드처럼 서버 섀도우로 상시 관찰되고 있지 않고, 순수 fresh-forward replay 검증(연구 확정)만 완료된 상태다. 실거래 경로(`trading_bot.py`)는 물론, 서버 섀도우 프로세스도 미변경. "Odyssey4 베이스라인"은 연구 비교용 기준이지, 실거래·섀도우 배포 상태를 뜻하지 않는다. |

## 범위

- 목적: Odyssey3가 확정한 베이스라인(h48qual 레짐 가드) 위에, 사용자가 2026-08-14 세션에서 명시적으로 해제한 entry-side 개입을 공식 결합해 새 비교 기준으로 삼는다.
- **Odyssey1·Odyssey2의 실험 44건(entry-side 실패 29건 포함) 및 Odyssey3 실행 로그 #1(zig075 exit-side 개입 불가 진단) 전부 상속한다 — 재검증 불필요.**
- **유일한 신규 계층은 zig075 SHORT 진입거부뿐이다** — Odyssey3 베이스라인의 h48qual 레짐 가드·zig075 원본 로직·모든 모델 헤드·TP/SL·사이징·priority는 무변경.
- Odyssey(1)·Odyssey2·Odyssey3의 미해결 이슈(VAL 구간 신뢰성, exit_head 섀도우 승격기준 미정, `quality_threshold` 정렬버그 잔여 6개 스크립트, ATR TP/SL floor 버그 여부, 레짐 가드 forward 미검증)는 전부 그대로 상속된다.
- 라이브 파일(`trading_bot.py`/`trading_bot_modules/omega4_6_1_live.py`/`runtime_config.py`/`.env`) 미변경 원칙은 Odyssey(1)·Odyssey2·Odyssey3와 동일하게 유지. 서버 섀도우 프로세스(`live_eth_regime_aware_exit_guard_shadow_20260814.py`)도 이번 세션에서 미변경.
- 리소스 레지스트리: `docs/model_contracts/odyssey4_eth_entry_veto_baseline_data_resources_20260814.md`.
- **전체 레이어 요약·다이어그램**: `docs/model_contracts/odyssey4_eth_full_stack_architecture_20260814.md` — 피처부터 청산·렛저까지 전체 의사결정 파이프라인을 계층별로 정리하고 Odyssey1~4가 각각 어느 계층에 무엇을 추가했는지 시각화한 문서.
- 선행 계약: `docs/model_contracts/odyssey3_eth_regime_guard_baseline_contract_20260814.md`(Odyssey3), `docs/model_contracts/odyssey2_eth_live_injection_contract_20260813.md`(Odyssey2), `docs/model_contracts/odyssey_eth_h48qual_corrected_tabm_20260811_contract.md`(Odyssey1 — 최상위 서사).

## G0 참조값 (Odyssey4 베이스라인, 실행 로그 #2 `report.json`에서 그대로 재사용 — 재계산 불필요)

| 창 | 티어 | Odyssey3 베이스라인 (no_gate / with_gate) | Odyssey4 (진입거부 p90 적용, no_gate / with_gate) | veto 발동 |
|---|---|---|---|---|
| 2025-Q1(참고) | context | 97.70%/−20.62%/28 · 44.98%/−20.62%/20 | 동일 | 0 bar |
| 2025-Q2(참고) | context | 106.45%/−13.23%/31 · 31.49%/−15.85%/19 | 65.83%/−14.17%/31 · 5.62%/−23.59%/19 | 10 bar |
| **2025-Q3(참고)** | context | −37.43%/−51.25%/27 · **−15.86%/−44.37%/21** | −10.63%/−29.66%/23 · **+20.17%/−19.72%/17** | 19 bar |
| VAL | val | 46.59%/−21.70%/35 · 77.31%/−21.76%/26 | 41.13%/−21.70%/35 · 77.31%/−21.76%/26(동일) | 12 bar |
| OOS-Q1 | oos_confirm | 93.27%/−15.48%/24 · 67.25%/−15.48%/19 | 동일(렛저 자체 동일) | 0 bar |
| OOS-Q2 | oos_confirm | −9.55%/−20.76%/13 · −12.69%/−20.76%/10 | 동일(렛저 자체 동일) | 0 bar |

판정: VAL 게이트 strict 통과, OOS-Q1+OOS-Q2 단일터치 strict/relaxed 모두 `CONFIRMED`. 근거·전체 과정: `docs/experiments/eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814.md`.

향후 신규 후보는 이 표를 G0 기준으로 삼는다 — 진입거부 로직은
`scripts/research_eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814.py`의
`greedy_replay_entry_veto`(가드 모듈의 `greedy_replay_regime_aware_exit_guard` 복사본에
`short_entry_veto_mask` 한 줄만 추가)로 재사용 가능.

## 신규 계층 정의 (Odyssey4가 추가한 유일한 것)

flat 상태 진입 루프에서 `component == zig075 && side == SHORT && 지속상승장 탐지기 ON(신호 bar)`이면
그 진입만 스킵한다. 탐지기는 Odyssey3 베이스라인이 이미 잠근 것을 그대로 재사용
(`dual_momentum>0`의 2016-bar rolling 비율, threshold=2025-Q1+Q2 전용 표본 p90=0.8025793650793651,
Q3/VAL/OOS 미참조) — **신규 자유변수 0개**. zig075 LONG·h48qual(레짐 가드 포함)·모든 모델
헤드·threshold·TP/SL·사이징·priority·exit-side는 전부 무변경.

## 다음 점검 대상

| # | 항목 | 근거 |
|---|---|---|
| 1 | ~~진입거부 섀도우 관찰 로깅 추가~~ — **cutover 실행 완료(2026-08-15)**: 사용자가 서버에서 `sudo bash scripts/ops/systemd/install_and_cutover_odyssey4_shadow_20260814.sh` 직접 실행. 코딩 에이전트가 사후 검증(read-only SSH): `eth-jmlam4-shadow`/`eth-exithead-shadow` 유닛 완전 제거(목록에서 사라짐, 비활성), `eth-odyssey4-shadow`는 active+enabled로 08-14 23:40부터 재시작 없이 무중단 실행(초기화 로그의 SHORT veto 임계값 0.802579·`duration_threshold=0.005417`이 계약 수치와 일치), `trading_bot.py`/`omega4_6_1_live.py`/`.env` 등 실거래 경로는 `git status` 무변경 확인. **부가 발견 및 정리**: cutover 스크립트는 systemd 유닛만 대상으로 해서, 범위 밖의 별도 프로세스 — Odyssey3 베이스라인의 h48qual 가드-only 섀도우(`live_eth_regime_aware_exit_guard_shadow_20260814.py`, `handoff.sh` 기반 raw nohup job, [[odyssey_eth_h48qual_subproject]] 계열)가 병행 실행 중인 것을 발견했다. 오디세이4 섀도우가 이 가드 로직을 byte-for-byte 포함하는 상위호환이라 중복 판단, 사용자가 서버에서 직접 `kill`로 종료(2026-08-15) — 코딩 에이전트가 프로세스 소멸 재확인. **버그 기록 및 수정**: `handoff.sh stop`은 pidfile에 기록된 wrapper PID만 종료하고 `setsid`로 분리된 실제 워커 프로세스는 살려두는 버그가 있었다(이번엔 수동 kill로 우회) — 원인은 `do_launch`가 바깥쪽 SSH 셸의 `$!`로 pidfile을 채우는데 그 셸 자체가 백그라운드 이후에도 종종 남아있어(기존 코드 주석에 이미 기재된 증상) `$!`이 신뢰할 수 없었기 때문. 같은 날 `run.sh`(생성되는 conda 활성화 wrapper)가 `conda activate`/`cd`/`exec` 전에 자기 자신의 `$$`를 pidfile에 먼저 쓰도록 수정 완료 — 이후 단계는 전부 `exec`라 PID가 안 바뀌므로 항상 정확한 PID가 기록됨. `handoff.sh launch server handoff_selftest -- sleep 600`으로 실제 검증(pidfile PID가 실제 sleep 프로세스와 일치, stop이 정상 종료, 고아 프로세스 없음) 후 정리. | 사용자 지시(2026-08-14/15) |
| 2 | 레짐인지형 exit 가드(h48qual)·진입거부(zig075) 둘 다 forward 관찰 누적 — 섀도우가 실제 지속상승장을 한 번이라도 겪을 때까지는 두 계층 모두 진짜 검증 불가 | Odyssey2 #11·Odyssey4 실행 로그 #2 정직한 한계 |
| 3 | ~~h48qual SHORT에 동일 진입거부 확장~~ — **실행 완료(2026-08-15, 실행 로그 #4)**: CONFIRMED(약함/한계 있음), 참고 3분기 순효과 음수라 배포 권장 안 함 | 실행 로그 #4 |
| 4 | (낮은 우선순위, 상속) VAL 구간 신뢰성 근본원인, exit_head 승격기준 확정 | Odyssey1 미해결 이슈 12·13 |
| 5 | zig075 LONG 지속하락장 진입거부(실행 로그 #5, CONFIRMED)의 forward 관찰 누적 — 판정 근거가 OOS-Q2 거래 1건에 의존해 섀도우 배포 없이는 확신 상향 불가 | 실행 로그 #5 정직한 한계 |
| 6 | ~~zig075 LONG/하락장 손실의 bar-level 메커니즘 진단~~ — **실행 완료(2026-08-15, 실행 로그 #6)**: exit_head 0/33건 관여(SHORT 0/53건과 합쳐 방향 불문 0/86건), 방향/품질 확신도도 승패 미분리 — entry veto가 유일하게 유효했던 개입임을 사후 확인. 다만 실제 표본은 겹침 거래 4건뿐이라 판정 강도 자체는 못 올림 | 실행 로그 #6 |

## 실행 로그

Odyssey3까지의 실행 로그는 `docs/model_contracts/odyssey3_eth_regime_guard_baseline_contract_20260814.md`
참고. #1은 이 계약이 흡수한 규칙 veto(위 G0 표).

| # | 항목 | 결과 | 문서 |
|---|---|---|---|
| 1 | zig075 SHORT 진입거부 (규칙, 탐지기 재사용) | **CONFIRMED** — Odyssey4 베이스라인으로 흡수(위 G0 표) | `docs/experiments/eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814.md` |
| 2 | **학습형 진입거부** (사용자 지시: RL/딥러닝) — TCN이 반사실 barrier 라벨(SL-first)로 거부 게이트를 학습, 2024만 학습·HP탐색 0·임계값은 손익분기 유도(p\*=0.6435)·5시드 | **부정 결과, REJECTED** — VAL strict 통과·시드 5/5 일치에도 OOS-Q1 반전(with_gate 67.25%→−12.70%, 승리 숏 4건 역선택). 메커니즘: 연도 밖 AUC 무작위 이하(2025-Q3 0.498, 2026 0.477), mask 발동 36~58%의 광역 필터로 퇴행 — 2025 내 개선은 판별이 아닌 기저율 효과. **규칙 veto가 유효한 해로 유지되며, "정보 부족" 가설이 모델 비개입 라벨·밀집 샘플 조건에서도 재확인됨(30번째 entry-side 학습 실패).** | `docs/experiments/eth_omega461_zig075_learned_short_veto_tcn_20260814.md` |
| 3 | **증거신호 사이징** (macro veto 위 계층, 별도 연구 라인의 외부 OHLCV+taker_buy 신호를 사이징 신호로 재사용 — entry 거부 아님) — TOP 증거신호 미확인 시 zig075 SHORT margin_fraction×0.5, 확인 시 그대로. v1(신호 8개)과 v2(순위안정성 문서 마스터랭킹 상위5개로 정제, 배율 불변) 둘 다 시도 | **부정 결과, REJECTED (양쪽 다)** — v1: VAL strict 탈락(77.31%→64.07%)으로 OOS 미개봉. v2(신호 목록만 정제, 배율은 그대로 0.5 유지): **더 악화**(VAL 77.31%→43.71%, 목표였던 2025-Q3조차 베이스라인 20.17%보다 낮은 17.68%로 반전). 원인: 신호를 좁힐수록 "OR 확인" 조건이 더 드물어져(확인율 2~13%) sized_down 거래가 늘어나는 구조적 문제 — 신호 품질을 아무리 높여도 이분법적 사이징 규칙 자체가 문제라 개선 안 됨. 배율 재조정은 사후선택이라 하지 않음. **오디세이4 macro veto는 이 결과와 무관하게 CONFIRMED 유지.** 이 축은 종결. | `docs/experiments/eth_omega461_zig075_short_evidence_sizing_20260814.md` |
| 4 | **h48qual SHORT 지속상승장 진입거부 확장** (다음 점검 대상 #3, zig075판과 동일 탐지기·임계값을 h48qual SHORT에 재사용, 자유변수 0개) | **CONFIRMED (약함/한계 있음)** — 사전등록 게이트는 형식상 통과하지만 판정 3창 중 VAL·OOS-Q2는 veto 발동 0건(무해성뿐), OOS-Q1은 거래 1건 교체(+1.35pp, 표본 1건). 참고 3분기 순효과가 오히려 음수(Q3 개선 +3.24pp < Q2 비용 −9.09pp, 순 −5.85pp) — zig075판(Q3 부호반전 +36pp급)과 대조적으로 존재 이유가 약함. **배포/섀도우 후보로 권장하지 않음.** 라이브 무변경. | `docs/experiments/eth_omega461_h48qual_short_entry_veto_sustained_uptrend_20260815.md` |
| 5 | **zig075 LONG 지속하락장 진입거부** (Odyssey4 베이스라인 위에 얹는 신규 후보, 상승장 탐지기의 거울상 — `dual_momentum<0` rolling 비율, 동일 레시피로 새로 계산한 p90=0.9712301587301587, 신규 자유변수 0개이나 산출 상수는 신규) | **CONFIRMED** — VAL·OOS-Q1은 veto 0건(무해성)이지만 **OOS-Q2는 실제 개입**: 37회 발동, 거래 1건 교체(2026-05-23 LONG 손절 −8.38% 제거 → 2026-05-24 SHORT 익절 +13.64%로 대체)로 `with_gate` PnL −12.69%→**+8.30%**(부호반전), MDD −20.76%→−13.72%(개선). 참고 2025-Q1도 큰 개선(+54pp). 다만 강건성이 zig075 SHORT판만큼 깨끗하지 않음(p75로 완화 시 Q3 참고창이 무변화→악화로 전환)과, 판정 근거가 사실상 판정 3창 중 1창·거래 1건에 의존한다는 한계가 있어 **섀도우 배포 최우선순위로 격상하기엔 이름**. LONG/하락장 손실의 bar-level 메커니즘 진단은 실행 로그 #6에서 사후 완료. 라이브 무변경. | `docs/experiments/eth_omega461_zig075_long_entry_veto_sustained_downtrend_20260815.md` |
| 6 | **zig075 LONG/하락장 손실 메커니즘 진단** (실행 로그 #1(SHORT/상승장)과 동형: exit_head 관여 여부 + bar-by-bar MFE/확률 재구성 + 반사실 threshold 스윕, 신규로 방향/품질 확신도 승패분리 검사 추가) | **`diagnosed`** — exit_head는 지속하락장 탐지기와 겹친 4건(전 6창 합산, 실제 체결까지 이어진 표본)에서 단 한 번도 관여하지 않음(0/33, LONG 전체 기준) — SHORT판 0/53건과 합쳐 zig075 exit_head는 방향 불문 0/86건으로 확정. 원칙 검증 범위(threshold≥0.80)의 반사실 exit-threshold는 전부 무반응, 그 아래(사후선택)로 내리면 손실거래 일부는 개선되지만 유일한 승리거래(2025-Q2)를 항상 해치는 SHORT판과 동일한 혼재 패턴. 방향/품질 확신도(dir_p_long/quality_for_action)도 1승3패를 구분 못함(승자값이 패자값 구간 한복판). **entry veto가 유일하게 유효했던 개입이라는 실행 로그 #5의 설계를 사후 지지**하지만, 표본 자체가 4건뿐이라 실행 로그 #5의 판정 강도는 이 진단으로 올라가지도 내려가지도 않음. | `docs/experiments/eth_omega461_zig075_long_downtrend_loss_mechanism_diagnosis_20260815.md` |
| 7 | **랜덤 방향 + 리스크스택 어블레이션** (사용자 제안 검증: 방향 헤드를 50/50 동전던지기로 교체, quality_threshold 게이트·zig075 베토(SHORT/상승장만, LONG/하락장 베토는 범위 밖)·h48qual 가드·TP/SL·사이징·priority는 전부 무수정 재사용) + **exit사유 분포** + **레인지장 재검정 2라운드** + **N=30 대규모 재검정** + **베토 개입 빈도 가설 검증(상관)** | **레짐 의존적 방향 편향, N=30으로 통계 확정 + 비단조 패턴 부분 설명**. exit사유: exit_head 발동률(22~28%)은 방향 품질과 무관 — 리스크관리는 "손실 캡핑"이지 "회피"가 아님. N=30 결과 **4/6 윈도우가 \|t\|>2로 유의**: 저스프레드(진짜 레인지) 2구간은 강하게 음의 방향(t=−3.36, −5.05, 실제모델이 random보다 유의하게 나쁨), 20pp·VAL(73pp)은 양의 방향(t=+4.72, +2.61), 최고스프레드 OOS-Q1/Q2(76~124pp)는 오히려 무유의(t=0.33, −0.74) — 스프레드 단일변수로 설명 안 되는 비단조 패턴. 베토(SHORT/상승장) 개입빈도 상관분석(N=15)으로 OOS-Q2 하나만 부분 설명(real_g0 베토0회 vs random평균6.5회). 이 상관분석은 방법론적으로 교란(시드의 SHORT비중과 공변) 있어 실행 로그 #8의 페어드 설계로 대체 확인. | `docs/experiments/eth_odyssey4_random_direction_risk_management_ablation_20260817.md` |
| 8 | **베토 on/off 페어드 재검증(최종, 실행 로그 #7의 상관분석을 인과설계로 교체)** — 같은 방향 draw를 고정한 채 zig075 SHORT 베토만 켜고 끄는 페어드 비교(N=20 새 독립시드, TabM 추론은 draw당 1회만) | **6윈도우 중 유의한 양의 효과 0곳, 유의한 음의 효과 2곳.** VAL·OOS-Q1·레인지2-A/2-B 4곳은 real_g0·random 전부 무영향(VAL은 베토 12회 발동해도 최종 PnL 0.00pp 변화 — 계약 수치와 정합). **OOS-Q2는 random 평균 −5.31pp(t=−2.64, 유의)로 해로움.** **레인지1(2025-05-12~07-07)은 real_g0 자신이 −20.18pp 손해**(베토 없었으면 −17.59%가 아니라 +2.59%) — random 20/20 전부 같은 방향(평균−19.22pp, 표준편차 3.23pp, t=−26.58, 사실상 결정론적). 탐지기가 2025 Q1+Q2 추세데이터로만 보정돼 레인지 구간에서 국지적 상승스윙을 지속상승장으로 오인, 실제로는 평균회귀로 수익났을 SHORT를 막은 것으로 해석. **정밀한 재확인: 이 결과는 CONFIRMED 판정(사전등록 3개 판정창 VAL/OOS-Q1/OOS-Q2에서 non-worse) 자체를 뒤집지 않는다** — 그 3창에서 real_g0의 WITH/WITHOUT delta가 정확히 0·0·0이라 원 게이트를 그대로 재현한다. 다만 **원 게이트가 커버하지 않는 레짐(진짜 레인지)에서 베토가 유의하게 해로울 수 있다는 신규 증거**이며, 섀도우가 forward에서 이런 레짐을 만나면 이번에 측정된 규모(−20pp급)의 손실이 가능하다. | `docs/experiments/eth_odyssey4_random_direction_risk_management_ablation_20260817.md` |
| 9 | **레인지 오작동 수정 후보** (사용자 지시: 실행 로그 #8이 찾은 손해를 직접 고칠 것. 원 탐지기와 동일 규율 — 신규 자유변수 0개 — 적용해 시도) | **v2(정확히 신고점 조건, `close>=rolling_max(2016)`) REJECTED**: 레인지1 손해는 없앴지만 활성화율이 전 윈도우 60~1000배 붕괴, Q3 이득도 함께 소멸(+20.17%→−15.86%=no-veto와 동일) — 베토를 고친 게 아니라 꺼버린 것. **v3(주간평균 초과 조건, `close>=rolling_mean(2016)`, 여전히 신규 자유변수 0개) 부분 성공**: 레인지1 인과효과 t=−26.58→**t=−0.10**(사실상 완전 해결, real_g0 −17.59%→+0.97%), Q3 이득 보존·소폭개선(+20.17%→+21.41%), VAL/OOS-Q1/레인지2-A/2-B 무해 유지. **단 OOS-Q2는 전혀 못 고침**(random delta·t-stat이 v1과 소수점까지 완전 동일 −5.31pp/t=−2.64) — 하락추세 내 국지반등은 순수레인지와 다른 메커니즘일 가능성, 원인 미해명. **연구 결과일 뿐 배포 결정 아님** — 승격 절차·Red Team 미실행, 라이브/섀도우 무변경. | `docs/experiments/eth_zig075_veto_ranging_misfire_fix_candidate_20260817.md` |
| 10 | **h48qual/zig075 position-feature 학습-추론 불일치 버그 3건 발견·수정·재학습·Fresh-Forward 평가** — exit_head가 조회하는 `pos_unrealized`/`pos_mfe`/`pos_mae`가 학습 시 notional로 스케일압축(라이브는 비압축), `pos_notional`/`pos_leverage`/`pos_exposure`가 학습 시 고정상수(무분산), `pos_tp`/`pos_sl`이 학습 시 BASE_TEMPLATE 고정값(라이브는 ATR-adaptive) — 리포 전수조사로 같은 패턴 10개 파일(+별도 9개 파일 축)에서 재확인, 감사스크립트(`scripts/audit_position_feature_train_inference_parity_20260818.py`) 신설. CLAUDE.md "Position-Feature Train/Inference Parity Contract" 정책 신설의 근거가 된 발견. | **버그 자체는 CONFIRMED·수정 완료**(감사스크립트로 잔존 0건 재확인). 수정 반영 재학습(canonical 데이터, base_cols 원본과 동일 102개 고정, 전용 risk sidecar 신규학습, VAL-best quality_threshold 재선택 h48qual=0.40/zig075=0.80)까지 마친 뒤 이 G0(위 표)와 동일 6창 Fresh-Forward 파이프라인으로 정식 비교한 결과는 **REJECTED_SIGN_MISMATCH**(OOS-Q1 PnL −5.42pp·MDD −4.50pp로 non-worse 기준 미달, OOS-Q2는 +26.36pp/+5.83pp 개선 — single-touch 양쪽동시통과 미충족, 사전등록기준 그대로 적용). 남은 유일 교란변수는 단일시드(N=1) — N≥5 시드 없이는 이 결과가 버그수정의 진짜 효과인지 시드노이즈인지 확정 불가. 이 축은 2026-08-18~08-21 사이 일리아스(Ilias) 서브프로젝트에서 "일리아스 1"로 명명돼 진행됐으나(N=5/N=6 시드검증까지 CONFIRMED REJECTED_SIGN_MISMATCH), **2026-08-21 사용자 지시로 오디세이4 자체 계승판이라는 taxonomy 판단 하에 신규 독립 계약문서로 재이관**됨 — `docs/model_contracts/odyssey5_eth_position_feature_parity_fix_contract_20260821.md`("Odyssey5") 참고, 내용 무변경. 일리아스 계약문서의 "일리아스 1"이라는 이름은 이후 완전히 다른 축(154피쳐 라벨로직 후보)으로 재사용됐으니 혼동 주의. G0 참조값(위 표) 자체는 이 버그를 그대로 포함한 상태로 변경 없이 유지 — 이 실행 로그는 정보 기록이지 G0 교체가 아니다. | `docs/experiments/eth_odyssey4_exit_head_liveatr_barrier_and_label_reaudit_20260818.md` |

## 미해결 이슈

Odyssey(1)·Odyssey2·Odyssey3에서 상속(전부 유효):

- VAL 구간(2025-10~12) 자체의 신뢰성 문제 — Odyssey(1) 미해결 이슈 12.
- exit_head 섀도우 관찰기간·승격 판단기준 미정 — Odyssey(1) 미해결 이슈 13.
- `quality_threshold` 정렬버그, 동일 코드가 있는 미수정 6개 스크립트 — Odyssey(1) 미해결 이슈 14.
- ATR TP/SL floor가 버그인지 의도인지 — Odyssey(1) 미해결 이슈 15.
- h48qual 레짐 가드·zig075 진입거부 둘 다 **forward에서 진짜 지속 상승장을 한 번도 겪지 않았다**
  (OOS 데이터는 2026-07-12까지, 유일한 상승 구간은 12일로 표본 부족) — 두 계층 모두 "관찰 대기"
  지위.
- **신규(실행 로그 #8, 2026-08-17) — 베토가 레인지 구간에서 유의하게 해로울 수 있음, 섀도우
  관찰로 확인 필요**: 사전등록 3개 판정창(VAL/OOS-Q1/OOS-Q2)에서의 CONFIRMED 판정 자체는
  안 흔들렸지만, 그 3창이 커버 못 하는 진짜 레인지 레짐에서 베토 on/off 페어드 비교 결과
  −20pp급(사후선택 1개 구간)·−5.3pp급(OOS-Q2, N=20 무작위 방향)의 유의한 손해가 측정됐다.
  탐지기가 추세 데이터로만 보정돼 레인지에서 오작동할 가능성. **섀도우가 실제 레인지
  구간을 forward로 겪을 때 이 손실 규모를 관찰로 재확인해야 한다** — 현재로선 사후선택된
  과거 구간의 시뮬레이션 증거뿐, forward 관찰 없이 배포 상태를 바꿀 근거는 아니다.
  **후속(실행 로그 #9)**: 레인지1의 −20pp급 손해는 신규 자유변수 0개 수정 후보(v3, 주간평균
  초과 조건)로 Q3 이득을 보존하며 사실상 해결(t=−26.58→−0.10)했으나, **OOS-Q2의 −5.3pp급
  손해는 동일 수정으로 전혀 안 고쳐졌다**(v1과 소수점까지 동일) — 순수 레인지 오작동과
  하락추세 내 국지반등 오작동이 다른 메커니즘일 가능성, OOS-Q2 쪽은 여전히 미해결. v3 자체도
  단일 레인지 구간으로 설계·확인해 과적합 위험이 있고, 정식 승격 절차를 거치지 않은 연구
  결과일 뿐이다.

## 승격 게이트

Odyssey(1)·Odyssey2·Odyssey3와 동일하게 적용:

- VAL 단독 승리는 승격 근거 아님 — 저비용 사전필터로만.
- 신규 post-entry/entry 후보는 VAL 자체 게이트 통과 후, 공식 OOS 확인을 OOS-Q1+OOS-Q2를 한 실행에서
  함께 여는 단일터치로 심사한다(`scripts/eth_omega461_multiwindow_confirmation_gate_20260814.py`,
  순차/반복 확인 금지).
- exit_head/entry 로직 자체를 바꾸는 실험에는 컴포넌트 가드레일(50% 상대악화·부호반전 금지) 적용.
- 재학습 모델은 N≥5개 진짜 다양한 시드 없이 신호/노이즈 판정하지 않는다(결정론적 룰 기반 개입은
  해당 없음 — 시드 축 자체가 존재하지 않음).
- 라이브 파일 무변경, 섀도우 배포 ≠ 승격. **섀도우 배포 자체도 아직 이뤄지지 않았음에 유의**
  (Odyssey3의 h48qual 가드와 달리 zig075 진입거부는 관찰 로깅조차 없는 순수 연구 결과).
