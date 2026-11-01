# 배포 대기 항목 (deploy_queue.md 의 «A. 통과» 표에 병합할 행) — 횡보/추세 분별 (2026-11-02)

`docs/deploy_queue.md` 는 이 브랜치에 없고 브랜치 `claude/chart-indicators-tuning-ecfa25` 에 있다. 병합 시 아래 행을 거기 «A. 통과» 표에 넣는다(사용자 10-26: 통과 항목은 누적, 개별 배포 질문 안 함).

| # | 항목 | 근거(사전등록 문서) | 코드·브랜치 | 배포에 필요한 것 | 주의 |
|---|---|---|---|---|---|
| A8 | **횡보/추세 분별(앞으로 4h 지지~저항 띠 이탈 확률)** — 4코인(BTC·ETH·SOL·XRP) HGB | `docs/prereg/regime_acc_20261102.md` — 시험(2026-06~10, 13,908건) 1회: 정확도 **68.7%**·AUC .747, 기하 로지스틱 61.8% 대비 **+6.9%p [+5.6,+8.1]**, 항상 횡보 54.3% 대비 +14.4%p · 확신도 0.8↑ 95%·0.6~0.8 86%(전체의 16%·11%) · 위약 50.9% · 라벨 재구성 0/400 불일치 · **제로샷 5코인(ADA·AVAX·BNB·DOGE·LINK, 2026-06~08, 10,395건) 66.6% vs 기하 63.6% (+3.1%p [+1.9,+4.3], 5/5 코인)·확신 0.8↑ 95.0%** (`docs/experiments/regime_zeroshot_20261102.md`) · 결과 `docs/experiments/regime_acc_20261102.md` | `scripts/live_regime_band_20261102.py`(`predict_latest`) · 학습 `scripts/research_regime_acc_20261102.py`(build·final) · 패리티 테스트 `test/test_live_regime_band_20261102.py`(24건, 피쳐 최대차 1.4e-12·확률 차 0) · 브랜치 claude/xrp-ledger-trading-analysis-a85423 | ① 모델 아티팩트 `tmp/regime_acc_20261102/regime_band_h48_model.joblib` 는 미커밋: `python scripts/research_regime_acc_20261102.py build` 후 `final` 로 재생성(결정적, 인라인본과 예측 차 0) 또는 서버에 복사 ② 라이브 입력 = 코인별 5분봉 2,500봉 이상(마지막 봉 정시 마감, 시가 55분) + 대시보드 `compute_spliced_levels`(24h) ③ 화면: 확률·«횡보/추세»와 확신도 표시(확신 0.4↑만 강조 권장) ④ `bash scripts/ops/check_deploy_drift.sh` · 서버 md5 대조 | 정확도의 큰 몫은 기하(띠 폭÷변동성; 기하 로지스틱만으로 61.8%). 라벨은 «가장 가까운 S/R 띠 4h 이탈»이지 눈으로 본 횡보와 같지 않을 수 있음. 시험 1회 사용 — 개선하려면 새 사전등록 + 2026-10-24 이후 전진 데이터. 1시간·12시간 라벨은 비율 편향으로 다수결을 거의 못 넘음. 수익 판정 아님(정확도 판정) |

## 전진 기록기 (A8 의 일부) — 서버에 등록하지 않음
- 코드: `scripts/live_regime_band_recorder_20261102.py` (`record` = 매 시 정각 4코인 예측을 `data/live/regime_band_forward.jsonl` 에 추가, 같은 (시각, 코인) 중복 무시 · `score` = 4시간 지난 행을 **기록해 둔 지지·저항**으로 채점해 정확도·확신도별·코인별 요약).
- 검증: `test/test_live_regime_band_recorder_20261102.py` — 최근 53시간을 «실시간처럼» 매 시 잘라 record→score: 212건에서 학습 데이터셋의 같은 시각 확률과 차이 0, 라벨 불일치 0, 중복 기록 없음. (모델이 이 구간을 학습에 포함하므로 이때 정확도는 전진 증거가 아니다.)
- 서버에 올릴 때 남은 일: ① 라이브 5분봉 → 패널 변환(`live_regime_band_20261102.from_klines`) 호출부와 코인당 2,500봉 이상 확보 ② 매 시 정각 +20초 이후 실행(마감 봉만 사용; 형성 중 봉 금지 -- `breakout_detector_prewarn` 교훈) ③ 모델 아티팩트 배치 ④ crontab/supervisor 등록 -- 전부 사용자 결정(배포 일괄) 대기.
- 전진 판정 제안(사전등록 필요): 전진 4주, 정확도 ≥ 기하 기준선(61.8%) + 3%p 이고 확신도 0.6↑ 정확도 ≥ 80%.
