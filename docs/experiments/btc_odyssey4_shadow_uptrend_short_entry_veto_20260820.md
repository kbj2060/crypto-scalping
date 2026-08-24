# BTC h48qual+swingtransition SHORT entry veto(오디세이4 이식) — 정식 N=5 인과적 재현 (2026-08-20)

## 배경

어제 저비용 진단(`eth_odyssey4_veto_guard_btc_sol_transfer_diagnostic_20260820.md`)에서 BTC는
"혼재" 판정(baseline flip 3창 중 2025q2만 대조군 대비 특정성 있게 해소, 나머지 2창은
미해소/악화)이 나왔다. 진단을 만든 에이전트와 이 문서 작성자 둘 다 "전체 메커니즘 구축은
시기상조"라고 판단했으나, 이 사실을 그대로 전달받은 사용자가 명시적으로 "그래도 BTC 전체
메커니즘 구축"을 선택해 정식 구현+G0검증+N=5 인과적 재현 테스트를 진행했다.

## 범위: entry-veto만, exit-guard는 제외

ETH 오디세이4의 exit-guard는 포지션 보유 중 h48qual exit_head를 원본↔liveATR-relabel 두
학습된 변형 사이에서 전환한다. BTC는 이 두 번째 exit_head 변형이 아예 존재하지 않는다
(h48qual+swingtransition 단일 번들뿐) — 새로 학습하려면 라벨링+학습 전체 파이프라인이
필요해 "기존 조각 재조합" 범위를 벗어난다. 따라서 이번 구현은 **entry-veto만** 이식했고
(신규학습 0건), exit-guard는 별도 승인이 필요한, 훨씬 큰 후속과제로 남겨뒀다.

## 방법

- **탐지기(detector)**: ETH `SustainedUptrendDetector`의 수식·구조적 상수를 그대로 재사용
  (`WEEK_BARS=2016`, `DETECTOR_PERCENTILE=0.90` — bar 해상도에 종속된 상수라 자산별로
  다시 고르지 않음, ETH 원 스크립트의 원칙 그대로). BTC 자신의 `dual_momentum`
  (`data/splits/year_oos/btc_features_{2025,2026}_swingtransition.csv`, 어제 진단이 읽은
  것과 동일 파일, 연간 무결측 확인됨)에 적용하고, **캘리브레이션은 ETH와 동일한 원칙
  (2025 Q1+Q2만 사용, Q3/val/oos는 절대 사용하지 않음)으로 새로 계산** — 어제 진단의
  placeholder(0.604167, 2025년 전체 p90)를 재사용하지 않았다. 새로 계산된
  threshold = **0.740476**(BTC 자체 Q1+Q2 p90, 어제 값과 다름 — 더 엄격/선택적).
- **베토 적용**: BTC 리플레이 엔진(`train_eval_omega4_2_risk_sidecar_btc_20260708.py::
  _replay_with_risk`, 무수정)은 `active[i]=(action≠CASH)&(side≠0)&(notional_exposure>0)`로만
  flat상태 진입을 게이트한다. BTC는 단일 컴포넌트라 ETH처럼 리플레이 루프를 통째로 복사할
  필요 없이, `dec.loc[(side==-1)&mask,"side"]=0` 전처리 한 줄만으로 완전히 동일한 효과를
  낸다 — 기존 검증된 리플레이 함수를 한 줄도 건드리지 않았다(ETH의 베토보다 오히려 더
  단순한 통합).
- **신규 학습 없음**: 기존 5개 시드 번들+각자의 예측 CSV(`btc_live_promotion_seed_
  robustness_eval_5seed_20260819.py`가 이미 만들어둔 것) 그대로 재사용. risk sidecar도
  5시드 공통 고정(원 스크립트와 동일 단순화, 명시적 caveat).
- **G0 충실도 게이트**: 베토를 끈 상태로 이 스크립트의 리플레이 복사본이 기존
  `btc_live_promotion_seed_robustness_20260819_eval/report.json`을 bit-for-bit 재현하는지
  먼저 검증(5시드×6창=30칸 전부, tolerance 0.05pp) — **PASS**. 이 검증을 통과해야만
  candidate_run(veto on) 수치를 신뢰할 근거가 생긴다는 원칙(ETH G0b와 동일)을 그대로 따름.

## 결과 — 부호일치 변화 없음 (0/3 해소, 0/3 신규악화)

시드 순서: 260620_original / 750703416 / 160125165 / 626578270 / 179796523 (baseline과 동일 5시드)

| 창 | 부호 | baseline PnL(%) | veto PnL(%) | veto_bars(원시count) |
|---|---|---|---|---|
| 2025q1 | 일치 | +55.67 / +30.18 / +21.03 / +23.30 / +33.42 | +51.92 / +19.27 / +28.86 / +10.75 / +26.22 | 169 / 203 / 97 / 83 / 64 |
| **2025q2** | **불일치** | -13.39 / -31.30 / -27.02 / **+7.91** / -19.74 | -13.12 / -27.15 / -20.05 / **+7.91** / -25.69 | 150 / 249 / 241 / 30 / 129 |
| **2025q3** | **불일치** | +0.02 / -10.12 / -11.24 / -16.57 / -1.87 | +0.02 / -7.76 / -11.24 / -16.57 / -1.87 | 18 / 36 / 21 / 12 / 29 |
| val | 일치 | +18.03 / +7.80 / +15.86 / +26.00 / +12.73 | (변화없음, 전부 동일) | **0 / 0 / 0 / 0 / 0** |
| **oos_q1** | **불일치** | +10.95 / -22.11 / +17.42 / +17.28 / +31.45 | +10.95 / **-22.25** / +17.42 / +17.28 / +31.45 | 2 / 75 / 45 / 58 / 54 |
| oos_q2 | 일치 | -1.20 / -20.14 / -2.90 / -5.04 / -7.84 | (변화없음, 전부 동일) | 171 / **1012** / **1331** / 315 / 834 |

**`baseline_sign_flip_windows == veto_sign_flip_windows == [2025q2, 2025q3, oos_q1]`.
`resolved_windows=[]`, `newly_broken_windows=[]`.**

세 flip창 모두 부호패턴이 베토 적용 전후로 **완전히 동일**하다. 2025q2의 유일한 양수 시드
(626578270)는 베토 적용 후에도 값까지 소수점 둘째자리까지 동일(+7.91→+7.91, veto_bars=30
이지만 실제 거래에는 전혀 영향 없음) — 어제 진단이 시사했던 "2025q2는 대조군 대비 특정성
있게 해소된다"는 신호가, 정식 N=5 인과적 재현에서는 재현되지 않았다.

## 왜 베토가 거의 효과가 없었나 — veto_bars는 실제 억제 건수의 상한선일 뿐

`veto_bars`(원시 count)는 "side==SHORT AND 탐지기 활성"인 bar 수를 셀 뿐, 그 bar에서 실제로
포지션이 flat이었는지는 반영하지 않는다(`active[i]`는 `pos==0`일 때만 읽힌다 — 포지션 보유
중이면 베토로 `side=0`을 넣어도 그 bar의 진입판단 자체가 애초에 발생하지 않는다). oos_q2에서
시드 160125165는 1331개 bar가 베토 대상이었는데도 거래수·PnL이 baseline과 **완전히
동일**하다 — 즉 그 1331개 bar 거의 전부가 이미 포지션 보유 중이라 애초에 진입 후보조차
아니었다. 반대로 2025q1은 베토가 실제로 거래수를 바꿨다(예: 260620_original 27→31건, 한
진입을 막으면 뒤이은 다른 bar의 신호가 같은 슬롯을 채우는 경로의존 효과) — 하지만 이 창은
애초부터 baseline에서도 부호가 일치했던 창이라 최종 결과(부호일치 여부)에는 영향이 없었다.

## 참고: 시드간 베토 발동횟수가 seed-identical이 아닌 이유 (버그 아님)

ETH의 이전 축(h48qual exit_head만 N=5 시드, zig075는 라이브 원본으로 고정)에서는 베토
발동횟수가 5시드 전부 bit-identical했다 — zig075(베토 대상 컴포넌트)가 그 실험에서 애초에
고정이었기 때문이다. BTC는 컴포넌트가 하나뿐이고 그 **하나가 곧 재시드 대상**이므로, 시드마다
SHORT 신호가 나는 bar 자체가 다르다 — 탐지기 마스크(순수 규칙기반, `dual_momentum`+
timestamp만의 함수)는 시드와 무관하게 동일하지만, `veto_bars = SHORT신호 ∩ 탐지기활성`은
시드마다 다른 게 정상이다. 스크립트 실행 로그에 "should be seed-identical"이라는 주석을
ETH 축에서 그대로 가져와 남겼는데, 이는 이 실험 구조에는 맞지 않는 기대였음을 여기 기록한다
— 탐지기 자체의 결함이 아니다.

## 결론

BTC h48qual+swingtransition에 ETH식 "상승추세중 SHORT 베토"를 **정식 이식+G0검증+N=5
인과적 재현**까지 밀어붙였으나, 부호일치 개선 효과는 **0/3창**이다. 어제 저비용진단의
2025q2 특정성 신호는 (a) 정식 Q1+Q2전용 재캘리브레이션 threshold(0.740476, 어제
placeholder 0.604167과 다름)와 (b) 진짜 인과적 리플레이(포지션 상태 반영) 두 조건을 모두
갖춘 이 테스트에서 재현되지 않았다. ETH 오디세이4의 "6/6 전면 안정화"와는 질적으로 완전히
다른 결과다 — **이 설계로는 BTC로의 메커니즘 이식을 기각**한다.

## 한계

1. exit-guard는 미구현(위 범위 설명 참고) — entry-veto 단독 효과만 측정됐다.
2. risk sidecar는 5시드 공통 고정(시드별 재학습은 범위 밖, 기존 N=5 축과 동일 단순화).
3. N=5는 인덱스순 1세트 페어링이다 — CLAUDE.md의 N≥5 게이트 형식 요건은 충족하지만,
   "메커니즘 자체가 무효"라는 결론을 뒤집을 만큼 다른 threshold/percentile을 탐색하지는
   않았다(ETH의 0.90도 스윕 없이 고정 채택된 값이라 동일 원칙을 적용했을 뿐).
4. veto_bars 카운터는 포지션 상태를 반영하지 않는 원시 상한값이다(위 설명 참고) — "몇 건이
   실제로 억제됐는지"의 정확한 수치가 필요하면 `pos==0` 조건을 추가한 재계측이 필요하나,
   headline 결론(0/3 해소)에는 영향 없어 이번 범위에서는 추가하지 않았다.

## 산출물

- 스크립트: `scripts/research_btc_odyssey4_shadow_uptrend_short_entry_veto_20260820.py`
  (신규, 기존 리플레이 함수 무수정)
- 결과: `tmp/causal_regen_20260516/btc_odyssey4_shadow_uptrend_short_entry_veto_20260820/
  report.json`, 시드별 ledger 30개(veto-on, gitignored)
- 서버 job: `btc_uptrend_short_veto_20260820`(handoff.sh 경유 — mamba_ssm 의존성 체인 때문에
  dev에서 직접 실행 불가, `train_regime3_hmm_mamba_20260529.py` 임포트 체인이 원인)

신규 학습 없음(기존 N=5 번들 재사용). 추적 파일 수정/커밋/푸시 없음(신규 스크립트 1개
추가만). `trading_bot.py`/`.env`/`runtime_config.py` 미접촉.

**fresh_forward_bar_by_bar=true**(각 창 진입=해당 시드 번들 자신의 causal 예측, 청산=
`_replay_with_risk`의 순방향 bar-by-bar 루프, 베토는 신호 bar에서만 이미 확정된 과거
데이터 기반 마스크를 읽음 — lookahead 없음), **trade_ledgers_used_as_input=false**,
**saved_parent_exit_timestamps_used=false**, **future_rows_used_for_entry=false**. 단,
CLAUDE.md 기준 이 결과는 N=5 research/candidate 근거이며 그 자체로 promotion 주장은
아니다(애초에 결과가 부정적이라 promotion 논의 자체가 성립하지 않는다).
