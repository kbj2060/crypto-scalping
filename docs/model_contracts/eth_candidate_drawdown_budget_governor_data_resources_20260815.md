# ETH 드로다운 예산 거버너 후보 — 데이터 및 리소스 관리 (2026-08-15)

**서브 프로젝트 CLOSED 2026-08-16** — 계약서 상태 참고. 아래 표는 종결 시점의 리소스 스냅샷으로
보존한다.

이 문서는 ETH 드로다운 거버너 서브 프로젝트(`docs/model_contracts/eth_candidate_drawdown_budget_governor_contract_20260815.md`)에서 실제로 만지거나 검토한 모든 데이터 소스/리소스를 한 곳에 모은 목록이다. 계약 문서는 모델/아키텍처 상태(결과 + 간단한 근거 요약)만 다루고, 리소스의 위치·커버리지·상태·함정은 여기서 관리한다.

**새 리소스를 만질 때마다 그 턴에 행을 추가/갱신할 것** — 나중으로 미루지 않는다. 상태 값 컨벤션: `활성`, `인프라 확인됨-미착수`, `인프라 차단`, `검증 완료 — 긍정 결과`, `검증 완료 — 부정 결과`.

## 이식 원본 (BTC clean_base, 읽기 전용 참고)

| 리소스 | 위치 | 커버리지 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|---|
| BTC 원본 계약 | `docs/model_contracts/clean_base_deep_gated_drawdown_budget_v5_contract.md` | - | 이식 대상 메커니즘 요약(terse) | 인프라 확인됨-미착수 | 이 계약 문서 자체는 상세 파라미터가 없음 — 실제 파라미터는 스크립트에서 직접 확인 필요(아래 행) |
| BTC 원본 구현(거버너 로직 소스) | `scripts/train_eval_clean_base_deep_drawdown_min_v4.py` | `DrawdownMinConfig`(40행), `backtest_drawdown_min`(200행) | 이식할 실제 게이팅 로직·파라미터 그리드 원본 | 인프라 확인됨-미착수 | ETH 드로다운 거버너 계약이 이미 전체 로직을 인용해 옮겨적었음 — 재조사 시 이 파일이 1차 출처 |
| BTC v5 그리드/셀렉터 래퍼 | `scripts/train_eval_clean_base_deep_gated_drawdown_budget_v5.py` | 그리드서치·score 함수·red team 판정 로직 | 셀렉션 기준(10%-range MDD band, cost2/3 게이트) 참고 | 인프라 확인됨-미착수 | Deep GRU 상태 인코더 학습 코드가 같이 섞여 있음 — ETH 드로다운 거버너는 이 부분(148~157행 학습 블록)은 이식하지 않음, 그리드/셀렉터 로직(176~272행)만 참고 |
| BTC 레드팀 판정 | `docs/experiments/clean_base_deep_gated_drawdown_budget_v5_redteam.md` | verdict `APPROVED_AS_SHADOW_FRONTIER` | 승격 게이트 설계 참고(accounting/causality/notional invariant 체크리스트) | 검증 완료 — 긍정 결과(BTC 기준) | Odyssey 자체 재검증 없이는 이 verdict를 Odyssey 근거로 인용 불가 — 계약 Red Team Gates 절에 별도 체크리스트 명시함 |

## Odyssey4 상속 자산 (그대로 재사용, 미변경)

| 리소스 | 위치 | 커버리지 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|---|
| Odyssey4 causal replay 하네스 | `scripts/research_eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814.py` (`greedy_replay_entry_veto`, 153행) | 6창(Q1/Q2/Q3 참고 + VAL/OOS-Q1/OOS-Q2 판정) | ETH 드로다운 거버너 신규 스크립트가 복사해서 훅을 추가할 베이스 | 활성 | 이 파일 자체는 수정하지 않는다(Odyssey4 계약 자산) — 복사본에만 훅 추가 |
| VAL/OOS 윈도우 로더 + 단일터치 확인 게이트 | `scripts/eth_omega461_multiwindow_confirmation_gate_20260814.py` (`load_all_windows`) | 동일 6창 | 데이터 분할·단일터치 프로토콜 재사용 | 활성 | ETH 드로다운 거버너도 이 게이트로 VAL 선택 후 OOS-Q1+OOS-Q2를 한 실행에서만 개봉 |
| PRIORITY/SCALE_MAP/캡 상수 | `scripts/replay_omega4_6_1_greedy_router_20260706.py` | - | `LEVERAGE_CAP=5.0`, `NOTIONAL_CAP=1.8`, `PRIORITY=(h48qual,zig075)` | 활성 | ETH 드로다운 거버너는 이 상한 자체를 바꾸지 않음 — 그 아래로 추가 캡만 검토(cheap_gate 1번은 예외적으로 이 상수 자체를 낮춰보는 실험) |
| 렛저 지표 계산 | `scripts/research_eth_omega461_exit_head_portfolio_asymmetric_20260813.py` (`portfolio._ledger_metrics`) | - | PnL/MDD/trades 등 표준 지표 산출 재사용 | 활성 | - |
| 리스크 사이징 sidecar | `scripts/train_eval_omega4_2_risk_sidecar_20260622.py` | - | `margin_fraction`/`leverage` 산출원, `selection_objective=log_risk` | 활성 | ETH 드로다운 거버너는 이 sidecar의 출력값 자체를 재계산하지 않음 — 산출된 notional 위에 상한만 추가 |
| Odyssey4 G0 기준선 수치 | `docs/model_contracts/odyssey4_eth_entry_veto_baseline_contract_20260814.md` (G0 표) | 6창 PnL/MDD/trades | 거버너 on/off 비교의 기준값 | 활성 | ETH 드로다운 거버너는 이 표를 재계산하지 않고 그대로 인용(계약 본문에 이미 복사) — 만약 재계산해서 불일치하면 즉시 드리프트로 보고 |

## ETH 드로다운 거버너 신규 산출물

| 리소스 | 위치 | 커버리지 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|---|
| cheap_gate 스크립트 | `scripts/research_eth_candidate_drawdown_governor_cheap_gate_20260816.py` | VAL만 | NOTIONAL_CAP 스윕 + 일일 손실정지 스윕, G0 재현 확인 포함 | 검증 완료 — 부정 결과(둘 다 기각) | OOS 미개봉. `greedy_replay_entry_veto_daily_halt`는 이 스크립트에만 존재하는 일회성 실험 함수 — 전체 거버너 구현 시 그대로 재사용하지 말고 L9 훅으로 재작성 |
| cheap_gate 리포트 | `tmp/causal_regen_20260516/eth_candidate_drawdown_budget_governor_cheap_gate_20260816/report.json` | VAL, NOTIONAL_CAP 5종 + halt 3종 | 원 수치 근거 | 검증 완료 — 부정 결과 | `docs/experiments/eth_candidate_drawdown_budget_governor_cheap_gate_20260816.md`에 표로 요약됨 |
| cheap_gate 결과 문서 | `docs/experiments/eth_candidate_drawdown_budget_governor_cheap_gate_20260816.md` | - | 전체 과정·판단 근거 | 활성 | 계약의 "필수 저비용 게이트"/"다음 단계" 절이 이 문서를 인용 |

| L9.1/L9.2 순환청산 스크립트 | `scripts/research_eth_candidate_drawdown_governor_intrabar_stops_20260816.py` | 6창 G0 회귀 + VAL 단독-메커니즘 ablation 3종 | equity_mdd_budget_stop/hard_loss_stop/profit_trailing_lock 구현·검증 | 검증 완료 — 부정 결과(3종 전부 기각, 재진입 처닝 근본원인 진단) | `greedy_replay_entry_veto_intrabar_governor`는 이 스크립트 전용 — 재진입 쿨다운 없이는 재사용 금지(등급 A 발동 시 hold=0 처닝으로 자멸 확인됨) |
| L9.1/L9.2 리포트 | `tmp/causal_regen_20260516/eth_candidate_drawdown_governor_intrabar_stops_20260816/report.json` | G0 6창 + ablation A/B/C 각 grid | 원 수치 근거 | 검증 완료 — 부정 결과 | `docs/experiments/eth_candidate_drawdown_governor_intrabar_stops_20260816.md`에 표로 요약됨 |
| L9.1/L9.2 결과 문서 | `docs/experiments/eth_candidate_drawdown_governor_intrabar_stops_20260816.md` | - | 전체 과정·근본원인 진단·갈림길(A/B) 제시 | 활성 | 사용자 결정 대기 — (A) 재진입 쿨다운 추가 재시도 vs (B) 축 종결 |

## 미검증 후보 / 보류

- **재진입 쿨다운 메커니즘(안 A)**: 미구현 — 사용자가 선택할 경우 착수. BTC 원본에 없는 신규 요소이므로 별도 설계 필요(쿨다운 길이 자체가 새 하이퍼파라미터, 그리드 최소화 원칙 유지).
- **진입 전 노셔널 스로틀(L4.9)**: cheap_gate 결과로 후순위 확정, 순환청산 축의 갈림길 결정 전까지 미착수.
- **daily_dd의 timezone 컨벤션 확인**: 계약 미해결 이슈 2 — cheap_gate/intrabar 스크립트 둘 다 BTC 원본과 동일하게 `pd.Timestamp(...).date().isoformat()`을 그대로 썼다(별도 timezone 변환 없음) — 라이브 피처 프레임과의 정합성은 아직 명시적으로 검증 안 됨. daily_dd 자체는 이번 순환청산 ablation에서는 쓰이지 않았음(진입 전 스로틀 전용).
