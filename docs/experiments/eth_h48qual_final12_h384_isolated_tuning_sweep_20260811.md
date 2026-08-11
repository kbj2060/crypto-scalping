# ETH h48qual — FINAL12 + h384 격리 튜닝 스윕 (2026-08-11, 완료 — 엣지 없음으로 결론)

## 배경 및 스코프

[FINAL12 피쳐](eth_h48qual_final12_feature_selection_20260811.md) +
[h48_conservative(384bar) 라벨](eth_h48qual_quality_horizon_sweep_20260811.md)을 기존 프로덕션
하네스에 꽂아서 실제로 학습을 돌려본 격리검증.

**여기 쓴 백본은 구버전 `ThreeHeadTabM`이다.** `ThreeHeadTabMCorrected`는 라이브 전체가 공유하는
파일에 아직 연결하지 않았다 — 효과를 먼저 격리검증한 뒤에 공유 파일을 건드린다는 원칙 때문이다.
따라서 이 문서의 결과는 "새 피쳐셋 + 새 quality horizon"만 검증한 것이고, 백본 교체 효과는 완전히
별도의 열린 질문으로 남는다.

하네스: `train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py` 계열 / zig075의
`train_eval_omega4_3head_parent72_eth_zig075_regime_jmredesign_final15_20260811.py`를 포크.
direction/quality/exit 3-head를 bull/bear/chop regime별 독립 인스턴스 3개로 학습한 뒤 regime3
확률로 라우팅(Omega4가 Omega1.2의 regime-expert 라우팅을 재사용). "공유 인코더"는
`direction_head`와 `quality_head`가 인코더 하나를 같이 쓴다는 뜻이지, 3개 regime 인스턴스가
하나로 합쳐진다는 뜻이 아니다.

## 빌드한 스크립트

| Script | Role |
|---|---|
| `scripts/build_eth_h384_conservative_triple_barrier_label_20260811.py` | `h48_conservative` 배리어 공식 그대로, horizon만 384로 — 캐노니컬 빌더에 새 `BarrierConfig` monkeypatch |
| `scripts/pad_eth_h384_conservative_labels_to_zigzag_timestamps_20260811.py` | 위 라벨을 `direction_head`와 같은 zigzag 타임스탬프 그리드에 패딩(미매칭은 CASH) |
| `scripts/train_eval_omega4_3head_parent72_eth_h48qual_final12_h384_20260811.py` | zig075 final15 포크 — FINAL12로 피쳐 축소, quality-mode를 `quality_label_action`(자체 라벨)으로, quality-label-dir을 h384 디렉토리로 |
| `scripts/tune_eth_h48qual_final12_h384_20260811.py` | `scripts/tune_btc_parent_regime_jmredesign_20260810.py`와 동일 절차(VAL만으로 선택, OOS는 참고용) |

## 1단계 — 베이스라인 (단일 시드, epochs=4 = 라이브 기본값)

OOS pnl이 q0.40~0.55 전 구간 플러스로 나왔지만(+6.5~+9.0%), VAL은 대체로 마이너스라 방향이 뒤집힌
이례적인 패턴. OOS 거래수도 9~21건뿐이라 표본이 작았다.

**해석**: 파라미터 튜닝 없이 BTC의 라이브 값을 그대로 가져다 쓴 것뿐이라는 지적을 받고
정식 튜닝(2단계)으로 이어짐.

## 2단계 — v1 스윕 (epochs 3|4|6 × rows 30000|45000 × seed 5개 = 30런)

같은 `(rows, seed)`에서 epoch만 다른 3개 런이 `best_validation_loss`가 소수점까지 완전히 동일한
버그 발견. `patience=8`은 8에폭 연속 무개선이어야 발동하는데 epoch cap이 3~6이라 애초에 발동할
공간이 없었던 것 — 결과적으로 4~6에폭 구간은 순수 낭비.

VAL MDD floor(`-8.0`, BTC 기준 차용)도 30개 전부 통과 못 함(최선 `-8.33`).

**해석**: epoch 상한 버그를 고쳐야 다음 판단이 가능. VAL MDD floor는 이 라인에 그대로
못 쓴다.

## 3단계 — epoch 상한 확인런 (epochs=30, 시드 2개)

`patience`는 정상 작동 — 두 시드 다 epoch 9~10에서 실제로 발동. 그런데 "제대로" 학습한 결과가 더
좋아지지 않았다:

- `seed=260620`: 기존과 거의 동일(VAL `+14.0%→+13.2%`, OOS는 오히려 `+14.4%→+7.8%`로 하락).
- `seed=481003`: 전체 quality threshold에서 전부 마이너스(`-15.4%~+1.7%`).

**해석**: epoch 버그는 실재했지만, 그걸 고친다고 결과가 좋아지는 게 아니다 — 시드마다
결과가 이렇게 크게 흔들린다는 게 더 근본적인 문제라는 게 드러났다.

## 4단계 — v2 스윕 (epochs=40 고정 × rows 2개 × seed 15개, 완료)

epoch을 더 이상 스윕하지 않고(patience에 위임) 절약된 예산을 seed 5→15로 돌림 — 지금 열린 질문이
"epoch을 얼마나 돌리나"가 아니라 "신호가 시드 노이즈보다 큰가"이기 때문. `--device cuda` 명시(v1은
GPU를 실제로 썼는지 불확실했음). 런당 ~90초, 선택 기준은 VAL pnl 최고(mdd≥-8 우선, 0개면 전체
fallback).

**결과**: `train_rows=30000`에서 5개 quality threshold 전부 OOS pnl이 일관되게 양수였고, 4/5가
10-config Bonferroni 보정(p<0.005)까지 통과(일부 p<0.001). `train_rows=45000`은 5개 전부
유의미하지 않음(p>0.24). 처음엔 이걸 진짜 신호로 읽었다 — VAL 단독으로는 다중비교를 못
이겼지만(최고 config p=0.0074, Bonferroni 기준 p<0.005 필요), threshold 전체에 걸쳐 반복되는
OOS 패턴이라 시드 노이즈로 설명하기 어려워 보였다.

## 5단계 — always-short 대조: "신호"의 정체 (완료)

4단계 결과를 곧이곧대로 믿기 전에 always-short 기준선과 비교했다(같은 진입시점, 방향만 강제
— `scripts/verify_eth_h48qual_always_short_baseline_h384_v2_20260811.py`, `omega._metrics()`
원본 시뮬 재사용). **OOS에서 always_short이 15/15 시드 전부 모델을 이겼다**(모델 평균 8.96 vs
always_short 18.69, paired t=-5.15, p=0.00015). VAL은 7/15로 거의 반반(p=0.57).

원인 분해: direction_head 원본(게이트 전) 숏비중은 53.7%(VAL)/55.4%(OOS)로 균형 잡혀 있는데,
quality 게이트 통과 후(`final_action`)는 78.1%/75.3%로 치솟는다(게이트 통과율 ~20%). 즉
4단계의 "신호"는 direction_head의 방향판별 능력이 아니라 `quality_head` 게이트가 만든 숏
편향이 학습구간(2025-01~04, ETH -51%)과 검증구간(VAL -28%, OOS -36%, 둘 다 하락장)의 추세가
우연히 일치해서 나온 것이었다. 전체 과정과 라이브 가중치 재현·방향별 승률(OOS 롱 6.7%)·
threshold 스윕까지는 [h48orig 컨트롤 문서](eth_h48qual_quality_trend_bias_h48orig_control_20260811.md)에 정리했다.

## 현재 결론

FINAL12+h384 조합 자체의 "엣지"는 **없다 — 있어 보였던 건 게이트 편향과 추세 일치의 우연이었다.**
epoch 버그는 확인·수정됐고 patience는 정상 작동한다(epoch 9~10에서 발동 확인). 5시드 미만
비교로 신호/노이즈를 판정하지 않는다는 원칙을 지켰기 때문에(v2=15시드) 이 결론에 도달할 수
있었다 — 5시드 미만이었다면 4단계의 겉보기 신호에서 멈췄을 것이다. `quality_head`를 tb_quality
회귀로 바꾸는 수정안도 검증했으나 [실전 피쳐로는 신호가 없어 막다른 길로 결론](eth_h48qual_quality_head_regression_conversion_attempt_20260811.md).

## HP (전 구간 공통, 튜닝된 적 없음)

`k=8, hidden=192, layers=3, dropout=0.08, lr=2e-3, weight_decay=2e-4, batch=2048` — BTC 라이브
앵커에서 차용한 값. epoch만 v2에서 patience에 위임하도록 고쳤다.

## 결과 (계약 문서 반영용)

FINAL12+h384(구버전 백본)는 진짜 엣지 없음으로 결론. v2 15시드에서 OOS가 통계적으로 유의미해
보였으나 always-short 대조로 게이트 편향+추세 우연 일치임을 확인. 원인은 `quality_head`
게이트 구조(상세는 h48orig 컨트롤 문서) — 384bar 재설계 특유의 문제가 아니라 라이브 원본
48bar+102피쳐에서도 재현됨.
