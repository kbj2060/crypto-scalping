# BTC V자급등락 TabPFN 메타라벨 확인 (2026-09-01)

## 요약

ETH의 확정 라벨 공식(FAST_BARS=6/FULL_BARS=12/ATR_MULT=1.5/T_SUSTAIN=0.20)을 그대로 재사용,
후보풀은 6트리거(유동성스윕/체결쏠림/15분급변/오실레이터/확장소진/local_extreme — smt_divergence·
demarker_extreme·kalman_deviation_meanrev는 이번 라운드 스코프 밖이라 제외, ETH 현재 라이브는
9트리거)로 TabPFN(4시드) 학습·검증.

| 구간 | AUC (mean±std) | n_train | n_eval | 라벨률(labeled중) |
|---|---|---|---|---|
| VAL | **0.8351 ± 0.0003** | 13,185 | 2,691 | 43.8% |
| OOS | **0.8202 ± 0.0011** | 13,185 | 1,844 | 46.5% |
| HOLDOUT (1회성) | **0.8277 ± 0.0005** | 13,185 | 2,989 | 46.8% |

**ETH의 9트리거 라이브 버전(VAL 0.8292/OOS 0.8127/HOLDOUT 0.8465)과 거의 동일하거나 VAL·OOS는
근소 우위** — 6트리거만으로도 ETH급 성능이 나온다는 뜻입니다. 6개 신호 중 압도적으로 가장
강한 결과입니다.

## 피쳐 중요도 (VAL, 순열중요도)

`sweep_penetration_atr`(0.118) > `is_bottom`(0.091) > `flow_aligned_delta_z`(0.089) >
`p_fast`(0.078) > `ret3_z`(0.069) — 방향/침투깊이 계열이 지배, 원시 가격레벨(`atr`)이나
세션타이밍(`hour_utc`)은 거의 무의미. ETH와 같은 패턴.

## 다음 단계

TabPFN까지 완료, 결과 우수. smt_divergence·demarker_extreme·kalman_deviation_meanrev를 BTC로
포팅한 뒤 9트리거로 넓힐지는 별도 결정 필요(사용자 확인 대기 중). 경제성 게이트는 미착수.

## 산출물
- `scripts/research_btc_v_rebound_metalabel_tabpfn_20260901.py` (서버 실행)
- `data/labels/btc_5m_evidence_signal_candidates_20260901/v_rebound_tabpfn_report.json`
