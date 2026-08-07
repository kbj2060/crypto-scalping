# WS-F 테스트 설계도 — Kronos 파운데이션 모델 실험 (연구 트랙)

성격: 명시적 연구 트랙. 기대치 낮음 (LOBCAST/TLOB의 일반화 실패 결론 감안).
각 단계에 저비용 kill 게이트를 두고, **단독 알파 주장은 어떤 단계에서도 금지** —
평가 축은 "기존 스택(Sigma6/Omega4.6.1 research fork)에 피처로 기여하는가"뿐이다.
근거: [Kronos (HF 2508.02739, AAAI 2026)](https://huggingface.co/papers/2508.02739),
체크포인트 [NeoQuasar/Kronos-base](https://huggingface.co/NeoQuasar/Kronos-base) (컨텍스트 512바).

## 데이터/오염 방지 규칙 (가장 중요)

1. **Frozen holdout**: 2026-07-14 이후 데이터는 어떤 설계 결정(피처 선택, 임계값,
   파인튜닝 여부 판단)에도 사용 금지 (BTC v3 Stage 0 정책 공유).
2. 설계/탐색 구간: ~2025-08-31. 검증 구간: 2025-09-01→2025-12-31 (표준 split).
   OOS 2026-01-01→2026-03-31은 **단계 F3 최종 1회 평가 전 접근 금지.**
3. **사전학습 오염 점검 (F0에서 필수)**: Kronos는 45개 거래소 12B 캔들로 사전학습됨 —
   우리의 "OOS" 구간이 Kronos 사전학습 기간에 포함될 가능성이 높다.
   모델 카드/논문에서 사전학습 데이터 컷오프를 확인하고, 컷오프가 우리 OOS와 겹치면
   해당 구간 결과에 `pretrain_overlap=true` 라벨을 달고 성과 주장 강도를 한 단계 낮춘다
   (겹침 없는 2026-04 이후 fresh 구간 평가를 최종 판정으로 승격).
4. Fresh-Forward 규칙 전면 적용: 추론은 bar-by-bar, 각 시점 입력은 그 시점까지의
   캔들 512개만. 배치 추론 시에도 윈도 우측 경계가 미래를 물지 않는지 전수 검증.

## 환경 격리

- **별도 venv 필수**: 기존 venv는 numba(numpy<2.3) vs Omega risk-sidecar(numpy>=2.3)
  충돌이 이미 존재. Kronos 의존성(torch, transformers)을 기존 venv에 넣지 말 것.
  `venv_kronos/` 신설, GPU 가용성 확인 후 없으면 Kronos-small로 축소.
- 추론 결과는 재사용 가능하도록 캐시: `data/research/kronos/predictions_{tf}_{ver}.parquet`
  (timestamp, 예측 분포 요약). 캐시에 생성 시각·모델 해시·윈도 정의 기록.

## 단계별 테스트

### F0. 준비 게이트 (반나절)
1. 체크포인트 로드 + 5m/1h ETH 캔들 100윈도 추론 스모크 테스트 (NaN/발산 0건).
2. 사전학습 컷오프 조사 → 오염 라벨 정책 확정 (위 규칙 3).
3. 입력 정규화 확인: Kronos 토크나이저가 기대하는 OHLCV 스케일/포맷과 우리 캔들
   (KST tz, 5m/1h) 정합 — 왕복 변환 후 원본 대비 왜곡 없는지 확인.

### F1. Zero-shot sanity (저비용 kill 게이트, 1일)
1. 대상: ETH 1h (Sigma6 시간프레임) + ETH 5m (Omega 시간프레임), 탐색 구간만.
2. 산출 피처 후보: 다음 k바 방향 확률 `kronos_dir_prob`, 예측 분위수 폭
   `kronos_q10_q90_width` (불확실성), 예측 중앙값 vs 현재가 괴리.
3. 벤치마크 비교 (동일 구간, 동일 평가):
   - 방향 정확도 vs (a) 동전 던지기 50%, (b) 단순 모멘텀 부호 (직전 k바 수익률 부호).
   - `q10_q90_width` vs 실현 변동성의 스피어만 상관 (불확실성 캘리브레이션).
4. **Kill 판정**: 방향 정확도가 모멘텀 벤치마크 이하 **이고** 불확실성 상관 |ρ| < 0.2
   → 전체 트랙 종료, 결과 기록. (한쪽이라도 신호 있으면 F2 진행 —
   방향은 못 맞혀도 불확실성 피처는 가치 가능.)

### F2. 피처 기여 테스트 (2~3일)
1. 기존 research 모델(Sigma6 재현 파이프라인 우선 — 1h라 Kronos 컨텍스트 512바=21일로
   적절)에 Kronos 피처 1~2개 추가.
2. 평가: 탐색 구간 내 walk-forward fold에서 (a) 피처 중요도(gain) 순위,
   (b) 추가 전/후 fold별 성과 delta.
3. 통계: fold별 delta의 부호 일관성 (전 fold 양수 요구는 과하고, ≥70% fold 양수 +
   평균 delta > 0).
4. **Kill 판정**: 중요도 하위 20% ∧ 성과 delta ≤ 0 → 종료.
   통과 → F3. 파인튜닝(F2.5)은 zero-shot이 경계선일 때만 검토하고,
   탐색 구간 데이터만으로 수행 (val/OOS 접촉 금지).

### F3. 표준 split 최종 평가 (1회성)
1. 검증 구간(2025-09→12) walk-forward로 피처 추가 전/후 비교 — 여기까지는 반복 가능.
2. 검증 통과 시에만 OOS(2026-01→03) **단 1회** 평가. 결과가 나쁘면 재시도/재조정 금지
   (OOS 재사용 = 오염). `pretrain_overlap` 라벨 적용.
3. 겹침 문제가 있으면 fresh 구간(2026-04→07-13) bar-by-bar 평가를 최종 판정으로.
4. **성공 기준**: 검증+OOS(또는 fresh 구간) 모두에서 기존 스택 대비 성과 delta > 0,
   day-block bootstrap t > 2 (연구 단계 기준; 라이브 후보 주장 시 t > 3 + shadow 별도).

### F4 (조건부). 후속
- F3 통과 시: Omega 5m 스택 반복 (5m×512 = 42h 컨텍스트 — 짧아서 기대 더 낮음),
  BTC/SOL 확장은 2026-10-14 해금 후.
- 라이브 반영 논의는 이 문서 범위 밖 — 별도 승격 절차 + Artifact Integrity Gate.

## 리포트 의무 필드

모든 결과 JSON에 명시: `fresh_forward_bar_by_bar`, `trade_ledgers_used_as_input=false`,
`future_rows_used_for_entry=false`, `pretrain_overlap`, `holdout_boundary=2026-07-14`,
`kronos_model_hash`, `context_bars=512`.

## Kill 기준 요약

| 단계 | 중단 조건 |
|---|---|
| F0 | 로드 실패 / 포맷 왜곡 해결 불가 |
| F1 | 방향·불확실성 둘 다 벤치마크 이하 |
| F2 | 피처 중요도 하위 20% ∧ delta ≤ 0 |
| F3 | 검증 또는 OOS/fresh에서 delta ≤ 0 (OOS 재시도 금지) |

## 산출물

- `results/ws_f_f1_zeroshot_YYYYMMDD.json/md`, `results/ws_f_f2_feature_contrib_YYYYMMDD.md`
- 예측 캐시 parquet + 재현 스크립트 `scripts/run_kronos_inference_YYYYMMDD.py`
- 종료 시(성공/실패 무관): 한 페이지 결론 문서 — 무엇이 신호였고 무엇이 아니었는지
