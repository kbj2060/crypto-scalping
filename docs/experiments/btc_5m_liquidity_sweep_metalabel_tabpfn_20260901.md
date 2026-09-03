# BTC 유동성 스윕(liquidity_sweep) TabPFN 메타라벨 확인 (2026-09-01)

## 요약

라운드2가 찾은 **터치+되돌림지속(touch_giveback_sustained) / H=20 / K=2.0** 라벨로
TabPFN(4시드) 학습·검증. 클러스터 중복제거(6봉이내, 스윕 침투깊이 최댓값 기준) 적용.

| 구간 | AUC (mean±std) | n_train | n_eval |
|---|---|---|---|
| VAL | 0.5467 ± 0.0036 | 5,261 | 1,097 |
| OOS | 0.5262 ± 0.0030 | 5,261 | 733 |
| HOLDOUT (1회성) | 0.5214 ± 0.0012 | 5,261 | 1,173 |

**라운드2의 lift 개선(VAL 1.33~1.46x)과 달리, TabPFN AUC는 무작위(0.50)에 가깝습니다.**
라운드1/2가 이미 경고했던 패턴("트리거/HIT 자체는 개선됐지만 그 위 피쳐분석은 오히려
약해짐")이 TabPFN으로 확정됐습니다 — 즉 이 라벨은 "발동하면 무작위보다 조금 더 잘 맞는다"는
lift 수준의 정보는 있지만, Tier0 피쳐들이 어떤 발동이 성공/실패할지 구분하는 데는 거의
도움을 못 준다는 뜻입니다. ETH 자체 결과(VAL 0.659/OOS 0.637/HOLDOUT 0.661)에 크게 못 미칩니다.

## 다음 단계

이 라벨 정의(터치+되돌림지속)로는 경제성 게이트를 진행할 근거가 약합니다. 다른 HIT정의
조합을 더 찾거나, 이 신호는 BTC에서 "약함"으로 결론짓는 게 합리적입니다.

## 산출물
- `scripts/research_btc_liquidity_sweep_metalabel_tabpfn_20260901.py` (서버 실행)
- `data/labels/btc_5m_evidence_signal_candidates_20260901/liquidity_sweep_tabpfn_report.json`
