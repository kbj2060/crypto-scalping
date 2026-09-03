# BTC 확장소진(fib_extension_exhaustion) TabPFN 메타라벨 확인 (2026-09-01)

## 요약

라운드2가 찾은 (VAL서 살아남은 안전픽) **종가기준(close_at_h) / H=10 / K=2.75** 라벨로
TabPFN(4시드) 학습·검증. 표본이 작아(TRAIN 1,071건) 클러스터 중복제거는 "클러스터 내 마지막
봉 유지"라는 단순 규칙 사용(이 신호는 스윙방향+피보나치존 터치라 delta_z 같은 단일 연속
강도지표가 없어서).

| 구간 | AUC (mean±std) | n_train | n_eval |
|---|---|---|---|
| VAL | 0.6239 ± 0.0120 | 1,071 | 210 |
| OOS | **0.5521 ± 0.0117** | 1,071 | 170 |
| HOLDOUT (1회성) | **0.5657 ± 0.0094** | 1,071 | 251 |

**여전히 6개 신호 중 가장 약합니다.** VAL은 그럭저럭(0.62)이지만 OOS/HOLDOUT이 무작위(0.50)에
가깝게 떨어져 — ETH 자체 결과(VAL 0.605/OOS 0.620/HOLDOUT 0.621, VAL-OOS 격차 거의 없음)와
달리 BTC는 VAL 과적합 패턴이 뚜렷합니다. 라운드1/2에서 이미 "표본최소·최약체"로 나온 판정이
TabPFN으로도 재확인됐습니다.

## 다음 단계

TabPFN까지는 완료했으나 결과가 약해 경제성 게이트 진행은 권장하지 않음 — 이 신호를 BTC에
배포할 근거는 현재 약합니다. 표본이 더 쌓이길 기다리거나(원본 데이터 자체가 희소 신호), 여기서
접는 게 합리적입니다.

## 산출물
- `scripts/research_btc_fib_extension_exhaustion_metalabel_tabpfn_20260901.py` (서버 실행)
- `data/labels/btc_5m_evidence_signal_candidates_20260901/fib_extension_exhaustion_tabpfn_report.json`
