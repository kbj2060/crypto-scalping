# SOL zig075-동형 direction_head 스킬 formal N=5 진짜 다양시드 재검증 (2026-08-24)

## 배경

2026-08-23 세션에서 사용자가 "솔라나 모델이 왜 성과가 좋았는지"(라이브 트레이드 5건, 5/5
승, +22.27% vs 같은구간 always-long +27.25%)를 물었고, 라이브 5건 자체는 실측/Binance
교차검증까지 마쳤지만, 그 성과를 "zig075의 검증된 방향 스킬" 탓으로 돌리기엔 근거가
부족하다는 게 드러났다: `eth_omega461_zig075_direction_head_skill_formal_nseed_20260815.md`가
**ETH 데이터**로 zig075 direction_head를 N=5 진짜 다양시드로 formal 검증해 **REJECTED**
(10/10 칸이 always_short에 패배) 판정을 이미 내려놓은 상태였다. 사용자가 "우연이라기엔
딱 맞았다"고 재반박했고, 그 반박에 답하려면 **SOL 자체 데이터**로 같은 formal 절차를
독립적으로 돌려야 한다는 게 명확해져 사용자가 "시드 검증 진행해줘"로 승인했다.

이 문서는 ETH판과 동일한 방법론(quality_head 완전 무시, direction_head 원본 argmax만으로
매 bar 거래 시뮬레이션 → always_short/always_long과 대조, N≥5 진짜 다양시드)을 SOL의 실제
라이브 배포 계보(`FINAL_GOVERNOR_OMEGA4_6_1_SOL_BUNDLE_PATH` →
`sol_omega4_3head_parent72_loose_entry_quality_20260707_adaptive_squeeze_20260720`,
`omega4_6_1_live.py`에서 SOL 라이브 에이전트가 실제로 로드하는 그 번들)에 그대로 적용한
결과다.

## 레시피 식별

- 라이브 SOL 번들: `trading_bot_modules/runtime_config.py`의
  `FINAL_GOVERNOR_OMEGA4_6_1_SOL_BUNDLE_PATH` (기본값) →
  `tmp/causal_regen_20260516/sol_omega4_3head_parent72_loose_entry_quality_20260707_adaptive_squeeze_20260720/true_3head_tabm_bundle.pt`.
  `report.json`: `model_id=sol_omega4_3head_parent72_loose_entry_quality_20260707`,
  `label_contract.direction_label_dir=sol_zigzag_action_labels_20260707`,
  `quality_mode=same_as_direction`, `base_feature_count=147`(ETH의 102와 다름 — SOL 전용
  피쳐셋, [[sol_adaptive_squeeze_v2_20260720]]의 `adaptive_squeeze` funding 정규화 수정
  포함), `exit_label.mode=entry_label_terminal_giveback`(terminal_window=3,
  adverse_unreal=-0.01, min_mfe_for_giveback=0.006, giveback_min=0.65 — ETH와 동일 스크립트
  기본값), 학습 rows=78,624.
- 학습기: `scripts/train_eval_omega4_3head_parent72_loose_entry_quality_sol_adaptive_squeeze_20260720.py`
  (`--out-suffix`/`--quality-mode=same_as_direction`/`--quality-thresholds=0.40..0.75` 기본
  주입 wrapper) → `scripts/train_eval_omega4_3head_parent72_loose_entry_quality_sol_20260707.py`
  (`--seed` 기본 260620, 배포판 관례와 일치).
- **피쳐 드리프트 방지**: ETH formal 테스트가 발견한 것과 동일한 함정(`_numeric_feature_cols`가
  런타임에 candidate CSV의 신규 컬럼까지 자동 포함해버림)이 SOL 학습기에도 구조적으로
  존재. `--base-feature-contract-bundle`에 배포 번들 자체의 `.pt`를 넘겨 `base_cols`를
  147개로 고정(코드 자체가 이미 이 옵션을 지원 — ETH처럼 별도 pinned 래퍼를 새로 만들 필요
  없었음). 5개 시드 전부 재학습 후 `base_feature_count=147` 재확인 완료(드리프트 0건).
- 리스크/PnL 시뮬레이션은 SOL 전용 모듈(`train_eval_omega1_2_tabm_diffusion_risk_sol_20260707`,
  `omega4_6_1_live.py`가 `_omega_sol`로 import하는 바로 그 모듈) 사용 — SOL 자체
  BASE_TEMPLATE(notional 0.45/leverage 2.0/TP 2.6%/SL 1.4%, `max_hold`/`cooldown`은 ETH판과
  동일하게 0으로 override)와 fee/slip. ETH 상수를 SOL에 재사용하지 않는다는 확립 규칙
  ([[sol_adaptive_squeeze_v2_20260720]]) 그대로 적용.
- 결정(포지션/사이드) 구성은 `train_eval_omega1_2_tabm_3head_20260603.py`의
  `parent._to_decisions`를 통해 호출 — 이 함수는 내부적으로 ETH 모듈의 `_to_fixed_decisions`를
  그대로 쓰는 **공유 아키텍처 레이어**(action 코드 → 포지션 상태 변환, 자산별 fee/TP/SL과
  무관)이고, SOL 학습 스크립트 자신이 실제로 이 경로를 타므로 다르게 구현하면 오히려 실제
  학습/평가 경로와 불일치하게 된다.

## 예상치 못한 블로커: SOL/BTC regime3_current 오버레이 파일 소실

재학습 1차 시도가 `FileNotFoundError`로 즉시 실패: 학습 파이프라인이 요구하는
`data/ensemble/supervised/sol_regime3_current_hmm_sensitive_wide24_20260707/sol_features_2025_regime3_current_sensitive_hmm_wide24.csv`가
**dev와 서버 양쪽 다 없었다**(SSH로 서버도 직접 대조 확인). 같은 디렉토리엔 2024년 학습된
frozen HMM `.joblib`과 `sol_features_2026_..._wide24.csv.bak_pre_extend_20260721`(과거 백업)만
남아있었고, **BTC의 동일 계열 오버레이도 똑같이 소실**돼 있었다(SOL만의 문제가 아님). 원인은
이번 세션에서 조사하지 않음(범위 밖) — 언제·어떻게 사라졌는지는 미상.

**복구**: `scripts/extend_regime3_wide24_sol_btc_20260721.py`(2026-07-21에 이 정확한 파일들을
생성했던 그 스크립트, frozen 2024 joblib을 causal `_transform`만 적용해 재생성 — 재학습
아님)를 재실행해 SOL+BTC 2025/2026 오버레이를 전부 복구. 이 스크립트가 import 체인에서
`mamba_ssm`(GPU 전용 패키지, dev엔 미설치)을 무관하게 끌어오는 기존 알려진 문제
([[dev_machine_amd_gpu_no_cuda_20260818]] 패턴과 동일)가 있어, 저장소에 이미 존재하는
동일 패턴의 stub 우회(`apply_regime3_wide24_sidecar_extended_20260820.py`가 쓰는 것과
동일한 `sys.modules["mamba_ssm"]` 런타임 스텁, 소스 파일 미수정)를 외부에서 적용해 실행.
2025년분은 원래 백업이 없었던 것으로 봐서 무조건 새로 씀, 2026년분은 기존 파일이 없어
재현성 대조(스크립트 내장 fail-fast 안전장치)가 스킵되고 그냥 새로 씀 — 둘 다 정상 동작,
에러 없음. 복구된 SOL 2026 파일은 2026-07-21 11:45까지 커버해 이 문서의 OOS-Q1/Q2 판정
구간(~06-30)을 넉넉히 포함.

**이 소실은 이번 문서의 판정과 무관한 별도 데이터 위생 이슈로 취급한다** — 재현
방법(공식 스크립트 재실행)이 명확하고 causal/재현가능(retraining 아님)해서 아래 판정에
영향 없음. 다만 SOL·BTC 양쪽에서 동시에 사라진 패턴은 후속 세션이 원인을 조사할 가치가
있다.

## 시드 5개 생성

`random.SystemRandom().sample(range(1, 1_000_000_000), 5)`(OS 엔트로피, 고정증분 아님) —
ETH formal 테스트와 동일 방식.

**648645464, 944028689, 822967396, 643442609, 295430784**

(참고용 6번째 지점: 배포 번들 자체의 시드로 추정되는 260620 — 아래 "N=1(배포판, 참고)" 행,
Seed-Diversity Gate N=5 카운트에는 미포함.)

## 재학습 실측 소요

5개 시드 순차 실행(dev, 12코어 CPU, conda env `quant_ai`): 각 시드 약 4분 30초~4분 45초,
총 약 22분 30초(00:36:13~00:59:52 KST). 5개 전부 exit=0, `base_feature_count=147` 재확인
완료(드리프트 0건).

## 평가 방법

신규 스크립트 `scripts/diagnose_sol_zig075_ungated_direction_vs_always_short_20260824.py`
(ETH판 `diagnose_eth_zig075_ungated_direction_vs_always_short_20260815.py`를 SOL 전용
모듈/가격소스로 이식, 로직은 동형). `quality_threshold` 완전 무시, `dir_action`(direction_head
원본 argmax)만 사용. `cost_mult=3.0`(SOL 스크립트 자체 기본값과 동일), `max_hold=0`/
`cooldown=0`.

**구간**: 배포 번들의 저장 예측 CSV가 실제로 덮는 범위 그대로 — VAL 2025-10-01~12-31(ETH와
동일 관례), **주 판정은 VAL + OOS-Q1(2026-01-01~03-31)**으로 ETH formal 테스트와 정확히
동일한 2-구간×5시드=10칸 포맷을 유지. **OOS-Q2(2026-04-01~06-30)는 참고용으로 추가
보고**(SOL 번들 예측 CSV가 이미 07-12까지 존재해 추가 재학습·재추론 비용 없이 보고 가능 —
사후선택 아니라 실행 전 스크립트 docstring에 사전 명시). 가격 소스:
`data/splits/year_oos_adaptive_squeeze_sol_20260720/sol_features_{2025,2026}.csv`
(timestamp/open/high/low/close).

## 결과 (5 시드 × VAL/OOS-Q1/OOS-Q2, cherry-picking 없음)

| seed | split | ungated pnl | always_short pnl | always_long pnl | ungated이 always_short 이김? |
|---|---|---:|---:|---:|---|
| 648645464 | VAL | −20.34 | +22.37 | −28.81 | **NO** |
| 648645464 | OOS-Q1 | −11.77 | −1.24 | −18.70 | **NO** |
| 648645464 | OOS-Q2(참고) | −5.00 | −7.75 | −10.57 | YES |
| 944028689 | VAL | +9.65 | +13.15 | −23.74 | **NO** |
| 944028689 | OOS-Q1 | −6.56 | −0.76 | −19.43 | **NO** |
| 944028689 | OOS-Q2(참고) | +0.57 | −9.59 | −14.72 | YES |
| 822967396 | VAL | +11.50 | +17.02 | −25.89 | **NO** |
| 822967396 | OOS-Q1 | +1.51 | +1.80 | −15.34 | **NO** |
| 822967396 | OOS-Q2(참고) | −3.24 | −4.82 | −10.38 | YES |
| 643442609 | VAL | +11.22 | +23.85 | −29.13 | **NO** |
| 643442609 | OOS-Q1 | +3.13 | +2.91 | −18.72 | **YES** |
| 643442609 | OOS-Q2(참고) | −15.67 | −7.38 | −6.12 | **NO**(최악 칸) |
| 295430784 | VAL | +7.25 | +14.12 | −25.22 | **NO** |
| 295430784 | OOS-Q1 | +2.86 | −2.12 | −22.25 | **YES** |
| 295430784 | OOS-Q2(참고) | −4.49 | −7.30 | −12.97 | YES |

(참고, N=5 카운트 외: 배포 번들 추정 시드 260620 — VAL −20.34/OOS-Q1 −11.77/OOS-Q2 −5.00,
셋 다 always_short에 패배. 재학습 5개 시드 중 648645464와 사실상 동일 패턴.)

**주 판정(VAL+OOS-Q1, 10칸)**: ungated이 always_short를 이긴 칸 **2/10**(둘 다 OOS-Q1:
643442609, 295430784). **VAL은 5/5 전부 패배**(클린 스윕) — ETH formal 테스트의 VAL 결과보다
오히려 더 일관되게 진다. **OOS-Q1은 2승 3패**로 시드 간 부호 자체가 갈린다.

**참고(OOS-Q2, 5칸)**: 4/5가 "이김"으로 표시되지만, 실제 값을 보면 대부분 **둘 다 마이너스인
상태에서 덜 잃은 것**(진짜 플러스 수익은 944028689 한 건, +0.57%뿐)이라 이걸 스킬로 읽기
어렵다. 그리고 그 4/5 중 하나(643442609)는 전체 15칸 중 **가장 나쁜 칸**(−15.67% vs
always_short −7.38%)으로 반전된다 — "4/5는 일관돼 보이지만 5번째가 최악의 손실을 낸다"는
패턴은 정확히 Seed-Diversity Gate가 잡아내려는 것(소수 시드로는 이런 시드-분산 지뢰를
못 본다)의 실례다.

## 해석 및 formal 검증

CLAUDE.md Seed-Diversity Ensemble Promotion Gate: "N≥5개의 진짜 다양한 시드... OOS 부호
일치를 보여야 한다." 주 판정 구간(OOS-Q1)에서 **부호 일치 없음**(2승 3패, 시드 절반이
반대 결론) — 게이트 요건을 명시적으로 충족하지 못한다. VAL은 오히려 더 결정적으로
반대방향(5/5 패배)이라 "VAL 단독 승리는 승격 근거 아님"이라는 기존 원칙을 적용할 필요도
없이 애초에 VAL 자체가 이기지 못한다.

ETH formal 테스트(0/10, 10칸 전부 패배)와 비교하면 SOL이 근소하게 덜 나쁘다(2/10) — 하지만
"완패"와 "부호 불일치로 무의미"는 둘 다 REJECTED 사유이지 CONFIRMED 사유가 아니다. ETH와
SOL 두 자산 모두 독립적으로 같은 결론(zig075 direction_head는 confirmed skill 없음)에
도달했다는 게 이 결과의 핵심 가치다.

## 최종 판정: **REJECTED**

**SOL의 zig075-동형 direction_head도 entry-side 스킬이 confirmed되지 않는다.** ETH와 동일한
결론이 SOL 자체 데이터·SOL 자체 학습 파이프라인·SOL 자체 리스크 모듈로 독립 재현됐다.
2026-08-23 라이브 트레이드 5건 전승(+22.27% vs 벤치 +27.25%)은 이 formal 결과와 완전히
양립 가능한 크기의 표본(19일 단일 강세추세 구간)이며, 이 결과가 그 라이브 성과의 원인이라는
가설을 지지하지 않는다 — 가장 근접한 설명은 여전히 "REJECTED 판정을 받은, 스킬 없는(혹은
VAL 기준으로는 always_short보다 못한) 예측기가 짧고 깨끗한 단일 추세 구간에서 우연히
5번 연속 맞았다"이다.

## Fresh-Forward 체크리스트

`fresh_forward_bar_by_bar=true`(고정 VAL/OOS 구간을 5분봉 단위 causal 예측 → TP/SL/time-exit
시뮬레이션), `trade_ledgers_used_as_input=false`(전부 이번에 새로 재학습한 모델의 자체
출력), `saved_parent_exit_timestamps_used=false`, `future_rows_used_for_entry=false`. 라이브
파일(`trading_bot.py`, `trading_bot_modules/omega4_6_1_live.py`, `runtime_config.py`, `.env`)은
전혀 건드리지 않음 — 이번 세션이 수정/생성한 것은 연구 스크립트 1개(신규,
`diagnose_sol_zig075_ungated_direction_vs_always_short_20260824.py`), regime3 오버레이 재생성
4개 파일(공식 스크립트로 복구, 로직 무변경), 재학습 산출물(`tmp/`, gitignored), 그리고 이
문서뿐. 서버에는 어떤 파일도 push/수정하지 않음(SSH는 상태 조회에만 사용).

## 산출물

- 신규 평가 스크립트: `scripts/diagnose_sol_zig075_ungated_direction_vs_always_short_20260824.py`
- 5개 재학습 번들: `tmp/causal_regen_20260516/sol_omega4_3head_parent72_loose_entry_quality_20260707_formal5seed_20260824_seed{648645464,944028689,822967396,643442609,295430784}/`
- 학습 로그: `tmp/sol_zig075_direction_head_formal_nseed_20260824/pilot_seed<SEED>.log`
- 시드별 진단 CSV: `tmp/sol_zig075_direction_head_formal_nseed_20260824/diag_out/ungated_vs_always_short_seed<SEED>.csv`
- 통합 결과: `tmp/sol_zig075_direction_head_formal_nseed_20260824/combined_5seed_results.csv`
- regime3 오버레이 복구 실행: 공식 스크립트 `scripts/extend_regime3_wide24_sol_btc_20260721.py`
  재실행(로직 미변경, mamba_ssm 런타임 스텁만 외부 적용) → 4개 파일 재생성
  (`data/ensemble/supervised/{sol,btc}_regime3_current_hmm_sensitive_wide24_2026070{7,8}/*_regime3_current_sensitive_hmm_wide24.csv`)

## 다음 단계

이 문서는 verdict만 확정한다 — `docs/model_contracts/`의 SOL/Omega 관련 계약 문서 갱신은
필요 시 사용자 판단으로 별도 처리. regime3 오버레이 소실의 근본원인 조사는 이 세션 범위
밖으로 남겨둠(다음 세션이 필요시 착수).
