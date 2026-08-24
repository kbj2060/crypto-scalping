# ETH Omega4.6.1 quality_threshold 클린 재선택 — fresh-forward 연장 시도, 코드 복구불가로 차단 (2026-08-15)

## 배경

`eth_omega461_oos_selection_bias_scope_and_resolution_20260813.md`(4단계, "향후 원칙" 5번)가
남긴 미실행 항목: `quality_threshold`(h48qual/zig075) 선택에 쓰인 진짜 alpha6/7-lineage 피쳐
파이프라인을 2026-03월 이후로 **순방향 연장**(재학습 아님, frozen 번들 추론만)해서, 08-13
진단이 겪은 8~13pp 자체정합성 노이즈 바닥 문제 자체를 없앤 뒤 VAL-정렬 클린 재선택을
다시 하라는 권고다. 이 문서는 그 실행 여부를 확인하고, 가능하면 실제로 연장한다.

## 1단계 — 중복 작업 여부 확인

`docs/experiments/*20260814*.md`, `*20260815*.md`, `docs/model_contracts/*20260814*.md`,
`*20260815*.md` 전체를 `quality_threshold`/`clean reselect`/`threshold reselect` 키워드로
전수 grep, 파일 목록 확인. **일치하는 문서/스크립트 없음** — 08-14/08-15 사이 다른 세션이
이 항목을 실행한 흔적이 없다. `scripts/`에도 `clean_reselection` 관련 신규 스크립트가
2026-08-13 이후 추가되지 않았다(BTC 버전 `research_btc_omega461_quality_threshold_clean_reselection_20260813.py`만
존재, ETH 신규 없음). → 중복 아님, 계속 진행.

## 2단계 — alpha6/7-lineage 파이프라인 연장 가능성 조사

`trade_candidates_2026_alpha6_current_tail111_exact.csv`(원 선택 스크립트의 "oos" 프레임,
2026-01-01~02-28)의 생성 계보를 역추적했다. 직계 상위는
`tmp/causal_regen_20260516/alpha7_1_01965_v2only_tp_sl_action_score_20260528/`이고, 궁극적으로
`features/elite.py`의 피쳐 계산 로직에 뿌리를 둔다.

**차단 원인을 직접 확인**: `scripts/retest_omega4_6_1_extended_oos_20260706.py`가 만든
`tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/report.json`의
`known_limitations.feature_drift` 필드 원문:

```
"ou_halflife/kel/evt_excess_z/btc_corr_60/dual_momentum differ from original alpha6/7-lineage
scoring; ou_halflife re-selected and confirmed robust, others (parent inputs) unresolved without
a full parent retrain (declined by user 2026-07-06, this project's retrain attempts on this
family have a history of failing validation gates)"
```

`git log --follow -- features/elite.py`로 직접 재확인: 전체 히스토리가 9개 커밋뿐이고 가장
최근이 2026-08-11/12 대규모 스쿼시 커밋이라, 2026-05-29 이전 시점의 정확한 피쳐 계산식을
복구할 방법이 없다. 즉 **alpha6/7-lineage를 진짜로 연장하려면 이미 유실된 옛 코드가
필요하다** — 이건 데이터 문제가 아니라 코드 문제다.

## 3단계 — 원시 데이터 실제 최신 시점 확인 (참고용)

직접 pandas로 연 결과:

| 파일 | 최신 timestamp |
|---|---:|
| `data/eth_5m_1year.csv` / `data/btc_5m_1year.csv`(롤링 1년 캐시) | 2026-02-17 15:00 |
| `data/TOTAL_ETHFIUSDT_fundingRate.csv` | 2026-04-15 |
| `data/ETHUSDT_FR_History.csv` | 2025-12-18 |
| `data/TOTAL_ETHUSDT_metrics.csv`(온체인) | 2026-01-20 |
| `data/splits/year_oos/training_features_2026_rebuilt.csv`(08-13 진단이 쓴 "rebuilt"=drift 있는 대체 파이프라인) | 2026-07-20 |
| 08-13 진단의 `oos_predictions_q050.csv`/`q075.csv` (실제 사용된 예측) | 2026-07-12 09:00 |

`training_features_2026_rebuilt.csv`가 예측 파일보다 8일 더 최신(07-20 vs 07-12)이지만, 이건
**이미 drift가 확인된 "rebuilt" 대체 파이프라인**이지 진짜 alpha6/7-lineage가 아니다 —
08-13 문서가 이미 이 대체 파이프라인으로 재추론해 8~13pp 노이즈 바닥을 확인했고, 향후 원칙
5번은 명시적으로 "그 대체가 아니라 진짜 lineage를 연장하라"고 요구했다. rebuilt 파이프라인으로
07-12→07-20까지 8일만 더 연장하는 건 노이즈 바닥 문제를 전혀 해소하지 못하므로 무의미하다.

## 결론 — 이 스레드는 데이터가 아니라 코드 복구불가로 차단됨

**연장 실행하지 않았다.** 사용자 작업지시가 명시한 분기("추가 fresh 데이터가 없으면 코드가
아니라 데이터 가용성 문제로 차단됐다고 보고")와 정확히 반대의 사실관계를 발견했다: 원시
데이터 자체는 (funding/on-chain 등 일부는 stale하지만) 문제의 핵심이 아니고, **진짜 병목은
alpha6/7-lineage 피쳐 계산 코드(`features/elite.py`의 2026-05-29 이전 버전)가 저장소에서
복구 불가능하다는 것**이다. 이는 08-13 진단이 이미 스스로 기록해둔 한계(`known_limitations`)와
정확히 일치하며, 그 코드를 되살리는 유일한 길인 "parent 전체 재학습"은 2026-07-06에 이미
사용자가 거절했고 이 계열의 과거 재학습 시도는 검증 게이트를 통과하지 못한 이력이 있다.

**실질적 함의**: 08-13 문서의 fresh-window 결과(h48qual 배포값 0.50이 그리드 최고,
zig075는 배포값·VAL-최적 둘 다 마이너스이며 노이즈 바닥 이내 무승부)가 이 프로젝트가 현재
갖고 있는 가장 깨끗한 답이며, 더 깨끗하게 만들 실행 가능한 경로가 지금은 없다. 이 병목을
풀려면 (a) `features/elite.py`의 2026-05-29 이전 버전을 다른 브랜치/백업/외부에서 복구하거나,
(b) parent 모델(h48qual/zig075)을 현재 피쳐 정의로 처음부터 재학습(사용자가 이미 거절, 이력상
실패율 높음) 중 하나가 필요하다 — 둘 다 이번 작업 범위를 벗어난다.

## 준수 확인 / 실행 사항

- 새로 실행한 백테스트/추론 없음 — 이 문서는 조사 결과 문서화만 수행했다(grep, git log, pandas
  로 파일 타임스탬프 직접 확인).
- 라이브 파일(`trading_bot_modules/omega4_6_1_live.py`, `trading_bot.py`, `runtime_config.py`,
  배포된 sidecar/bundle) 무변경 — `git status`로 확인, 이 세션은 이 문서 파일 하나만 신규
  작성했다.
- 승격/배포 판단 아님 — 이 문서는 후속 실험(진짜 lineage 연장을 통한 클린 재선택)이 지금
  실행 가능한지 여부만 확인했고, 답은 "현재는 불가능"이다.

Fresh-Forward 공시: `fresh_forward_bar_by_bar=false`(이번 세션은 조사만 수행, 신규 순방향
추론을 실행하지 않았다), `trade_ledgers_used_as_input=false`, `saved_parent_exit_timestamps_used=false`,
`future_rows_used_for_entry=false`.
