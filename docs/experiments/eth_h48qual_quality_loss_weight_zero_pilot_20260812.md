# ETH h48qual — quality_loss_weight=0 파일럿 (2026-08-12, 단일시드)

## 배경

3-head TabM은 `in_proj`/`blocks`(공유 trunk)를 direction/quality/exit 세 헤드가 공유하고
마지막 linear layer만 분리된다. `quality_head`는 이 세션 4갈래 시도 전부에서 실현수익률과
관계없는 것으로 확인됐는데, 그럼에도 `loss = loss_dir + 0.80*loss_qual + 1.15*loss_exit`로
계속 학습되며 그 gradient가 공유 trunk를 업데이트한다 — "신호 없는 목적함수가 direction_head가
쓰는 표현을 오염시키고 있는 게 아닌가"라는 가설을 검증했다.

## 방법

`scripts/train_eval_omega4_3head_parent72_loose_entry_quality_20260620_fullwindow_qualityweight0_20260812.py` —
`dataclasses.replace(CFG, quality_loss_weight=0.0)`로 quality_head의 loss 기여만 0으로 낮춤
(direction_focal_gamma=0, 나머지 전부 라이브 번들과 동일 설정: 2024-2025 전체구간 소스, seed=260620).
서버 GPU로 실행(`scripts/ops/handoff.sh`) — 필요 데이터 대부분이 이미 서버에 있어 48MB 보조
파일만 추가 전송.

## 결과 (같은 시드 260620, gamma=0 기준선과 직접 대조)

| threshold | 기준선(quality_weight=0.80) VAL | quality_weight=0 VAL | 기준선 OOS | quality_weight=0 OOS |
|---|---:|---:|---:|---:|
| q040 | +11.11 (wr 42.9%) | +15.37 (wr 47.4%) | −11.59 (wr 25.0%) | **−19.86 (wr 18.5%)** |
| q045 | +7.75 | −5.20 | −4.37 | **−22.30 (wr 14.3%)** |
| q050 | +26.74 (wr 51.4%) | −4.42 (wr 35.1%) | −9.03 | **−21.31 (wr 17.2%)** |
| q055 | +19.11 | −6.06 | −7.76 | **−16.58 (wr 21.4%)** |
| q060 | +5.97 | −10.28 | −6.72 | **−14.74 (wr 24.1%)** |

## 해석

**가설과 정반대다.** `quality_loss_weight`를 0으로 낮추니 VAL은 5개 threshold 중 4개가
양수→음수로 악화됐고(q040만 소폭 개선), OOS는 5개 threshold 전부 기존보다 더 나빠졌으며 승률도
25~33%에서 14~24%로 떨어졌다. quality_head를 계속 같이 학습시키는 게 direction_head가 쓰는
공유 표현을 오염시키기는커녕, 오히려 정규화 효과처럼 도움을 주고 있었던 것으로 보인다(멀티태스크
학습에서 보조 태스크가 실제 태스크와 무관해 보여도 공유 표현의 일반화를 돕는 경우가 있다는
문헌과 일치하는 방향).

**단일시드 결과다** — [[tabm_hp_low_signal_pattern]] 기준 N≥5 없이는 확정 못 한다. 다만 5개
threshold·양쪽 스플릿 전부 같은 방향(악화)이라는 점은 순수 시드 노이즈치고는 패턴이 뚜렷하다.

## 결론

`quality_loss_weight=0`으로 direction_head를 "quality_head의 방해 없이" 재학습하면 나아질
것이라는 가설은 이 파일럿에서 지지되지 않았다 — 오히려 반대 방향의 증거가 나왔다. 다중시드
재확인은 아직 하지 않았고, 이 결과의 강도(모든 threshold/스플릿 일관된 악화)를 볼 때 우선순위를
높게 둘 필요는 없어 보인다.
