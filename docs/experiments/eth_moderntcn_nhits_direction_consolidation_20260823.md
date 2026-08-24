# ETH — ModernTCN/N-HiTS 방향예측 결과 통합 (2026-08-23 소급 문서화)

## 왜 이 문서가 필요한가

2026-08-16 아키텍처축 종결 메모가 ModernTCN/N-HiTS를 "미착수, 정당한 다음 단계"로
보존했는데, **실제로는 08-16~19에 다른 세션들이 이미 실행을 완료**했다(서버 GPU 잡
`eth_nhits_moderntcn_direction_quality` 점유 기록이 당시 다른 메모리 3곳에 남아있음).
그러나 ModernTCN 쪽은 전용 결과 문서가 없어 tmp 산출물로만 존재했고, 그 탓에 2026-08-23
세션이 이 축을 "기록상 열려있는 유일한 아키텍처 항목"으로 사용자에게 잘못 재제안했다
(사용자 착수 승인까지 받은 뒤 재확인 과정에서 기실행 사실 발견). 재발 방지를 위해 소급
통합 문서화한다.

## 실행된 것 (tmp 산출물 실측)

### ModernTCN — regime-hardsplit 계열 (2026-08-18~19)

- 설정(report.json 실측): arch=moderntcn, **레짐 하드스플릿**(bull/bear/chop 레짐별 개별
  모델), window=48봉, 레짐별 피쳐 8개, 30 epochs. TRAIN 2025-01~2026-02 / VAL 2026-03~04 /
  OOS 2026-05~06.
- 절차: pilot(seed 314159265) → **HP 탐색**(`hpsearch_eth_moderntcn_regime_hardsplit_20260818`)
  → final 멀티시드(503468/587472/839864/954073) → 3-seed PnL 스윕 + unified 변형
  (`eval_eth_moderntcn_unified_val_oos_20260819.py`).
- **3-seed PnL 임계값 스윕 결과**
  (`tmp/causal_regen_20260516/eth_moderntcn_regime_hardsplit_eval_20260818_final_3seed/`):

| threshold | VAL mean±std (%) | OOS mean±std (%) |
|---:|---:|---:|
| 0.45 | −5.8±2.0 | −5.4±1.0 |
| 0.50 | −9.3±1.8 | −7.9±5.5 |
| 0.55 | −8.7±2.0 | −5.8±1.9 |
| 0.60 | −9.1±2.5 | −13.0±4.8 |
| 0.65 | −8.1±4.5 | −17.6±4.0 |
| 0.70 | −4.7±3.0 | −15.5±6.3 |
| 0.75 | −1.3±3.9 | −12.9±9.1 |
| 0.80 | −12.7±2.1 | −1.2±7.9 |
| 0.85 | −9.8±3.5 | +6.4±3.1 |

**VAL은 9개 임계값 전부 음수. OOS는 8/9 음수**, 유일 양수(0.85, +6.4%)는 같은 임계값의
VAL이 −9.8%인 고립 셀 — 이 저장소 상설 규칙("VAL 패배는 OOS와 무관하게 강한 반대증거",
Candidate C 전례)에 따라 **기각**. 판정: **REJECTED**.

### N-HiTS (2026-08-19~20, 기문서화 — 요약만)

`eth_directional_change_tabm_nhits_training_20260819.md`에 완결 기록: N=1 예비가 OOS에서
always_short에 패배 → 조건부 방향정확도 6런 전부 chance(48~51%) → 안정화 기법 4종
(HP재조정/ASWA/배깅/딥앙상블) 전부 구현·실행에도 불변 → **"최적화 부재가 아니라 신호
부재"로 종결**. DC/CUSUM 라벨, 158→133 피쳐정리, 8,778 전체쌍 상호작용까지 후속 전부
CLOSED.

## 결론

918실험 논문의 point-error 승자 두 아키텍처(ModernTCN/N-HiTS)를 **HP 탐색+멀티시드+
안정화 기법까지 실제 최적화 노력을 들여**([[feedback_dl_needs_optimization_before_failure_verdict]]
기준 충족) 이 저장소 데이터에 적용한 결과, 둘 다 방향예측에서 실패했다. 그 논문 자체도
이 아키텍처들의 방향정확도는 ~50%(동전던지기)라고 보고했었다 — point-error 우승이
방향스킬로 이전되지 않는다는 외부 결과가 내부에서 그대로 재현된 것. 이로써 2026-08-16
종결 메모가 남겨뒀던 마지막 아키텍처 공백 2개가 모두 닫혔고, **아키텍처 축에는 더 이상
미시도 항목이 없다**(남은 DL 축은 아키텍처가 아니라 데이터가 다른 raw L2/OFI, 09-14 게이트).

같은 날(08-23) 별도 실측인 용량 그래디언트(ridge > 300파라미터 MLP > 2.7k MLP ~ GBDT,
4단계 단조 — `eth_candidate_trend_dl_multivariate_probe_20260823.md`)와 정합: 이 데이터
유니버스에서 모델 용량은 정보를 만들지 못하고 파괴만 한다.

## 산출물 위치

- ModernTCN: `tmp/causal_regen_20260516/eth_moderntcn_*` (pilot/hpsearch/final 4시드/eval),
  스크립트 `scripts/{hpsearch,eval,run}_eth_moderntcn_*_2026081{8,9}.*`
- N-HiTS: `docs/experiments/eth_directional_change_tabm_nhits_training_20260819.md`
- registry: `eth_moderntcn_nhits_direction_backbones_20260818` (2026-08-23 소급 등록)
