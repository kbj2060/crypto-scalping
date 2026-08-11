# ETH h48qual — FINAL12 피쳐 선택 (2026-08-11)

## 배경

`direction_head`(`zigzag_action`)와 `quality_head`(`h48_conservative`/384bar — see
[eth_h48qual_quality_horizon_sweep_20260811.md](eth_h48qual_quality_horizon_sweep_20260811.md))
각각에 대해 독립적으로 relevance를 재검증하고, 공통 윈도우에서 충돌하는 피쳐 쌍을 정리해 최종
입력 셋(FINAL12)을 확정하는 작업.

## 방법

1. **Knockoff 게이트**: `knockpy.KnockoffFilter`(Model-X Knockoff, Candès/Fan/Janson/Lv 2018,
   JRSS-B) — FDR 상한을 통계적으로 보장하면서 비선형 관계도 포착, Ledoit-Wolf 공분산 추정으로 진짜
   피쳐간 상관구조를 보존.
2. **mRMR 압축**(Peng-Long-Ding, mutual-information 기반)으로 관련성 순위화.
3. 공통 윈도우(2025 상반기)에서 `|r|>0.5` 충돌쌍을, **각자 원래 타겟 기준 relevance**로 재판정해서
   병합(풀링된 단일 스코어로 재판정하지 않음).

## 관련 산출물 — 재현 가능 여부

- `scripts/knockoff_h48qual_only63_vs_zigzag_20260811.py` (레포에 커밋됨): `direction_head`
  (`zigzag_action`) 대상, h48qual 후보풀(145)에는 있지만 zig075 후보풀(138)에는 없는 63개 후보
  (raw 거래량/거래횟수 7개 + `ai_*`/`m7_*`/`patchtst_*` 메타피쳐 약 56개)를 knockoff으로 검증.
  TRAIN=2025-01-01~2025-06-30 (zig075의 2024-06~2025-06과 h48qual 패널 존재구간의 겹치는 부분,
  전체 매칭 아님).

**주의 — 재현성 갭**: 실제 mRMR 압축·최종 dedup을 수행한 스크립트(`knockoff_comparison.py`,
`knockoff_resume.py`, `knockoff_h48qual_names.py`, `h48qual_knockoff_mrmr.py`)는 **레포
`scripts/`에 커밋되지 않고 세션 scratchpad에만 존재**한다 —
`docs/experiments/eth_knockoff_feature_comparison_h48qual_vs_zig075_20260811.md`의 "관련 산출물"
절 참고. 그 세션이 끝나면 정확한 재실행이 불가능할 수 있다. 아래 dedup 근거 수치는 그 세션의
출력(위 문서 및 설계 아티팩트)에서 전사한 것이며, **이 문서를 쓰는 시점에 독립적으로 재현하지
않았다**.

## 교차비교 문서와의 정합성

`docs/experiments/eth_knockoff_feature_comparison_h48qual_vs_zig075_20260811.md`가 h48qual/zig075
양쪽 knockoff 결과를 이미 정리해뒀다. 거기서 확인된 것:

- FDR 0.10/0.20 게이트에서 h48qual-direction 38~46/145, h48qual-tradeability 32~35/145 생존.
- `funding_pressure_diff1`/`funding_roc_288`이 `r=0.996`로 사실상 동일 피쳐 — 그 교차비교에서
  처음 발견.
- raw 상태로 오염 확인된 9개(`funding_pressure`, `m7_vae_error`, `last_funding_rate`,
  `squeeze_power`, `long_squeeze_risk`, `funding_abs`, `whale_retail_ratio`,
  `count_long_short_ratio`, `sum_toptrader_long_short_ratio`) — `corr(close)` 0.25~0.62.
  `funding_pressure`→diff1, 나머지는 rolling(288) 평균제거(dt288)로 오염 해소.
- zig075 쪽 별도 mRMR(7개 생존) 중 4개(`cvp_regime`, `funding_roc_288`, `ou_halflife`,
  `breakout_strength`)가 h48qual 쪽과 겹침 — 완전히 다른 라벨·게이트·TRAIN 윈도우에서도
  수렴한다는 교차검증 근거로 인용.

**불일치 플래그(미해결)**: 위 교차비교 문서는 "h48qual FINAL13"이라고 표기한다(13개). 계약 문서/
설계 아티팩트의 FINAL12(12개)와 개수가 다르다. FINAL13이 이후 한 번 더 dedup되어 FINAL12가 됐을
가능성이 있으나, 그 마지막 단계를 보여주는 산출물을 찾지 못했다 — scratchpad 재현성 갭과 같은
원인일 가능성이 높다. 확인 필요.

## 최종 dedup 근거 (설계 아티팩트에서 전사, 위 재현성 캐비어트 적용)

- `funding_pressure_diff1` vs `funding_roc_288`: quality-MI `0.286` vs direction-MI `0.227`로
  diff1 채택(교차비교 문서가 `r=0.996` 동일피쳐임을 별도 확인).
- `regime3_current_sensitive_wide24_chop_prob` vs `parkinson_vol`: `0.056` vs `0.007`(8배) —
  chop_prob 채택.
- `sig_whale_dt288`: raw 형태는 `corr(close)=+0.561`로 오염, detrend(dt288) 후
  `corr=-0.010`으로 오염 해소되며 relevance가 11배 상승(`0.006→0.070`). 이후
  `whale_retail_ratio_dt288`과 비교해 `0.070` vs `0.011`(6.5배)로 승리, 둘 사이 `r=+0.598`
  공선성 확인 후 `whale_retail_ratio_dt288` 탈락 처리.

## 프로덕션 패널 갭 (미해결)

이 relevance 분석은 h48qual 자체 리서치 패널(`fa_features.parquet`, 145컬럼) 기준. 실제 학습에
쓰는 프로덕션 패널(`alpha6_current` 계열, 201/220컬럼)은 같은 이름의 컬럼이라도 생성 계보가 다를
수 있다 — `vwap_dist_24`/`funding_roc_48` 두 개는 프로덕션 패널에 아예 없어서 별도 브릿지
패널(연구용 원본 CSV)에서 조인해 온다. **라이브 경로 통합 전에 반드시 해소해야 함.**

## 결과 (계약 문서 반영용)

FINAL12 확정(아래 12개). 방법론: 헤드별 독립 mRMR+knockoff, `|r|>0.5` 충돌쌍은 각자 타겟 기준
relevance로 재판정.

```text
cvp_regime
funding_pressure_diff1
ou_halflife
m7_vae_error_dt288
realized_skewness
mta_funding
sig_whale_dt288
sum_toptrader_long_short_ratio_dt288
vwap_dist_24
funding_roc_48
breakout_strength
regime3_current_sensitive_wide24_chop_prob
```

미해결 항목 2개: FINAL12/FINAL13 개수 불일치, 프로덕션 패널 브릿지.
