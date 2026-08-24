# BTC-ETH 상관계수 피쳐 cheap_gate — CryptoGAT 저비용 버전 검증 (2026-08-16)

상태: **cheap_gate 완료. 이 특정 형태(단순 페어와이즈 상관계수)는 부정적. CryptoGAT 전체 그래프
어텐션 메커니즘이나 N-HiTS/ModernTCN 아키텍처 자체는 아직 미착수 — 별도 항목으로 분리.**

## 배경

사용자가 `eth_literature_review_cryptogat_and_918experiments_dl_architecture_20260816.md`의 "권장
안 함" 결론에 대해, 딥러닝/모델링은 최적화 노력에 따라 결과가 크게 달라지므로 문헌 기반 추론만으로
거부 처리하지 말고 실제로 더 연구해보라고 지시(2026-08-16). 이 문서는 그중 **가장 저렴하게 즉시
검증 가능한 조각** — CryptoGAT의 핵심 직관(자산 간 상관구조가 방향 정보를 담는다)을 그래프 전체가
아니라 BTC-ETH 페어 하나로 단순화한 버전 — 을 이 저장소의 cheap_gate 관행대로 실제 IC 계산으로
검증한 결과다.

**먼저 정정할 것 하나**: 지난 리뷰에서 "cross-asset 아이디어가 시도된 적 없다"고 쓴 건 부정확했다.
`docs/model_contracts/research_line_registry.json`의 `btc_rho1_panel_direction` 라인이 정확히 이
계열이다 — BTC 방향을 40개 자산 패널의 동시자산 어텐션(Rho1/Rho2, `scripts/train_eval_rho2_crosssymbol_causal_20260804.py`)으로 예측하는 시도였고, "랭크 점수가 거의 상수, 모든
fresh-forward PnL이 음수"로 닫혔다. 또한 라이브 피쳐 `dual_momentum`/`rel_momentum`이 이미
"ETH 모멘텀 - BTC 모멘텀"이라는 상대적 교차자산 신호를 담고 있다 — CryptoGAT의 직관이 이미 부분적
으로 라이브에 들어가 있는 셈이다. 완전히 새 것은 아니었다.

## 방법

`data/eth_5m_1year.csv` + `data/btc_5m_1year.csv`(2023-12-31~2026-02-17, 5분봉, 224,353행)로
직접 계산. 인과적(causal)으로만 — 상관계수는 bar t까지의 데이터로, forward return은 t→t+h.

- 피쳐: `corr_96` (8시간 롤링 상관, ETH·BTC 로그수익률), `corr_288`(24시간, 기존 레짐 윈도우와
  동일 길이)
- 벤치마크(무료, 이미 라이브): `rel_mom` = ETH 288bar 모멘텀 − BTC 288bar 모멘텀
- Forward return: h=12(1시간), 48(4시간), 288(24시간)
- 지표: Spearman IC + 가격 오염 체크(`spearmanr(feature, eth_close)`) — [[feedback_raw_feature_price_trend_contamination]] 관행
- 구간: VAL(2025-09-01~12-31, n=34,849) / OOS(2026-01-01~02-17, 데이터 종료로 부분치, n=13,537)

## 결과

| 피쳐 | horizon | VAL IC | OOS IC | 부호 일치 |
|---|---|---:|---:|:---:|
| corr_96 | 1h | +0.028 (p<.001) | +0.009 (p=.30) | 일치(약함) |
| corr_96 | 4h | +0.037 (p<.001) | −0.003 (p=.77) | **불일치** |
| corr_96 | 24h | −0.060 (p<.001) | −0.021 (p=.02) | 일치 |
| corr_288 | 1h | −0.006 (p=.23) | +0.013 (p=.13) | 노이즈 |
| corr_288 | 4h | −0.017 (p<.01) | +0.003 (p=.69) | **불일치** |
| corr_288 | 24h | −0.056 (p<.001) | **+0.054** (p<.001) | **정반대로 뒤집힘** |
| rel_mom(기존 라이브 스타일) | 1h | −0.037 | −0.023 | 일치(약함) |
| rel_mom(기존 라이브 스타일) | 4h | −0.067 | +0.013 (p=.14) | 불일치 |
| rel_mom(기존 라이브 스타일) | 24h | −0.012 | **+0.065** | **정반대로 뒤집힘** |

오염 체크: `corr_96`/`corr_288` 모두 `eth_close`와 ρ≈−0.16~−0.18 — [[feedback_raw_feature_price_trend_contamination]] 실격 기준(~0.5-0.6)에는 한참 못 미쳐 오염으로 인한 실격은 아니다.

## 판정

**부정적, 이미 익숙한 패턴.** IC 크기 자체가 전부 0.01~0.07 수준으로 작고, 가장 크게 나온
24시간 horizon에서 VAL/OOS 부호가 정확히 뒤집힌다(corr_288: −0.056→+0.054, rel_mom도 동일하게
뒤집힘). n이 13,000~35,000이라 p-value는 쉽게 유의해지지만, 부호가 구간마다 뒤집히는 신호는
[[repo_label_methodology_meta_finding]]과 [[eth_oscillator_confluence_closed_20260814]] 등에서
이미 반복 확인된, 착취 불가능한 패턴과 동일하다. **중요한 대조점**: 이미 라이브에 있는
`rel_mom` 스타일 벤치마크도 정확히 같은 방식으로 실패한다 — 즉 새 상관계수 피쳐가 기존 피쳐보다
나쁜 게 아니라, 이 교차자산 신호 계열 자체가 이 timeframe·이 자산쌍에서 원래 약하다는 것을 다시
확인한 셈이다.

## 남은 것 — 아직 안 닫힌 두 갈래

1. **CryptoGAT의 진짜 메커니즘(그래프 어텐션, 66개 자산 전체)은 여전히 미검증.** 이건 페어
   하나로 단순화한 버전과 다른 질문이다 — 다만 우리는 단일자산(ETH) 절대방향을 맞추는 문제라
   66개 자산 상대순위 프레임을 그대로 이식할 방법이 명확하지 않다. 이 cheap_gate 결과가 그
   질문 자체를 닫지는 않지만, "가장 싼 버전도 안 됐다"는 사실은 우선순위를 낮추는 근거가 된다.
2. **918실험 논문의 N-HiTS/ModernTCN 아키텍처는 아직 손도 안 댔다.** 저장소 전체에서
   `ModernTCN`은 이번 문헌리뷰 문서 밖에서 전혀 언급된 적이 없고, `N-HiTS`는
   `data_ensemble_cleanup_candidates.md`(2026-04-24, M7/PatchTST/DSAC 시대의 레거시 문서)에
   죽은 체크포인트(`NHITS_0.ckpt`)로만 존재 — `ensemble_router.py`는 그 팩에서 PatchTST만 실제로
   쓰고 NHITS/TiDE/iTransformer는 `NeuralForecast.load`가 통째로 불러오는 부산물일 뿐, 단독
   평가된 적이 없다. 이건 사용자가 지적한 정확히 그 문제 — "제대로 최적화해서 시도한 적 없는"
   아키텍처다. 이 부분이 다음 단계의 진짜 작업이고, 이 문서로는 아직 다루지 않았다.
