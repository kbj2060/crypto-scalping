# 파이프라인 무결성 & 리서치 방향 재설계 — 2026-07-30

## 0. 요약

지금까지 축적된 조사 결과를 한 줄로 정리하면 이렇다.

> 병목은 "더 좋은 모델을 못 찾는 것"이 아니라 **측정을 신뢰할 수 없다는 것**이다.

근거:
- 리서치 축은 실제로 고갈됐다 (지표탐색, 크로스섹셔널 모멘텀 3/25코인, 볼타게팅, 펀딩 스파이크 역추세,
  베이시스 캐리, 매크로/TradFi, h48qual label×model 2×2, 학습 하이퍼파라미터, exit 로직 21회,
  TP/SL·quality·time-exit·slot 스윕, 1m 스캘핑 4회 — 전부 VAL 또는 OOS에서 탈락).
- 그런데 그 "탈락" 판정을 내린 백테스트의 **입력 데이터가 버전 관리되지 않고 in-place로 덮어써지고 있다**.
  이미 두 건의 frozen baseline이 재현 불가로 확인됐다 (Sigma6, Omega4.6.1).
- 동시에 **라이브에 실제로 걸린 리스크가 리포트보다 나쁘다**는 것이 확인됐는데
  (bar-level MDD ETH OOS −28.3%, SOL −25.7%/−24.1% vs 프로젝트 게이트 −25%),
  현재 promotion gate는 이 지표를 아예 보지 않는다.

즉 **"리서치 축이 다 막혔다"는 결론 자체가 신뢰도가 낮은 측정 위에 서 있다.**
따라서 새 알파를 찾기 전에 측정 계층을 먼저 고쳐야 한다.

---

## 1. 문제 진단

### A. 재현성 붕괴 — **P0, 다른 모든 것의 전제**

| # | 문제 | 근거 |
|---|---|---|
| A1 | feature 파일을 in-place 덮어쓰기 | `scripts/update_features.py:507` `result.to_csv(FEATURES_CSV, index=False)`, `FEATURES_CSV`는 고정 경로 (`:64`). `data/splits/year_oos/*.csv`가 전부 동일 시각(2026-07-21 20:5x)에 통째로 재생성됨 |
| A2 | frozen baseline 2건 재현 실패 | Sigma6 tape: OOS +45.9%/+16.6% → −22.0%/−9.7%. Omega4.6.1 greedy router: +145.34%/−10.13%/24trades → +82.53%/−15.48%/31trades (동일 코드/설정, 데이터만 3주 경과) |
| A3 | prediction CSV가 문서화 없이 연장됨 | `oos_predictions_q050.csv` 등이 원래 종료일 06-30을 넘어 07-12까지 늘어남. 겹치는 구간은 byte-identical, 길이만 변함 → 정렬이 조용히 깨짐 |
| A3b | extend 이력이 ETH 파일 1개에만, 수동으로 존재 | `training_features_2026_rebuilt.csv.bak_pre_extend_{20260704,20260713,20260720}` (38MB→112MB→120MB→125MB 단조 증가). SOL/BTC split에는 대응 백업 없음. 내용 해시가 아닌 날짜 이름 기반이라 어떤 리포트가 어느 버전을 썼는지 역추적 불가 |
| A4 | promotion gate가 **입력 데이터 아이덴티티를 검사하지 않음** | `scripts/audit_omega_artifact_integrity_20260630.py`는 `sha256_file`로 아티팩트만 해싱. `dataset` / `lineage` / `features_csv` 개념이 스크립트 전체에 없음 |
| A5 | 라이브 코드 버전 식별 불가 | 마지막 커밋 `bea61a2` = **2026-04-28**. 5~7월 리서치 전체(Omega4.6.1 라이브 스택 포함)가 미커밋. tracked 144 changed / untracked 3382. `CURRENT_LIVE_MANIFEST.json`이 스스로 `dirty_worktree`를 promotion blocker로 기록 중 |

A5의 함의가 특히 크다. **현재 게이트는 구조적으로 통과 불가능하다** — `promotion_eligible`이
`dirty_worktree` 때문에 영원히 false다. 게이트를 만들어 놓고 게이트를 통과할 수 없는 상태.

### B. 리스크 측정 오류 — **P1, 라이브에 실제 영향**

| # | 문제 | 근거 |
|---|---|---|
| B1 | 모든 report.json이 trade-ledger MDD 사용 | bar-level(mark-to-market) 대비 2~4pp 과소평가. 6개 셀 전부에서 일관 |
| B2 | 전체 사이징 체인을 함께 모델링한 리포트가 0건 | component cap → asset NOTIONAL_MULTIPLIER → portfolio share cap → min_notional. 기존 리포트는 component cap만 반영 |
| B3 | 그 결과 라이브 스택이 게이트 위반 가능성 | ETH OOS bar-level MDD −28.3%, SOL VAL −25.7% / OOS −24.1% (게이트 −25%). SOL은 **VAL PnL이 −7.6%로 마이너스** |
| B4 | SOL sidecar가 OOS 오염 선택 | `selection_scope: "validation_oos_guard"` — 매니페스트에 blocker로 기록되어 있으나 라이브에서 계속 사용 중 |
| B5 | 백테스트와 라이브의 사이징 코드가 분리 | 라이브는 `trading_bot.py` + `trading_bot_modules/portfolio_risk.py`, 백테스트는 별도 경로. parity 테스트 없음. 실제로 "no-op today"라는 stale 주석이 라이브 동작과 반대였던 사례 있음 |

### C. 통계적 규율 — **P1**

| # | 문제 | 근거 |
|---|---|---|
| C1 | 다중검정 보정이 2026-07-26 이전엔 전무 | `core/selection_stats.py`가 그날 처음 생김. 그 이전 모든 승격 판단은 DSR/PBO 없이 이뤄짐. 현재 라이브 스택도 마찬가지 |
| C2 | 홀드아웃 윈도우 소진 | 2026-01..06 OOS를 하루에 4개 스윕이 훑음 → "statistically spent". 신규 후보를 평가할 깨끗한 구간이 부족 |
| C3 | standalone 결과를 승격 근거로 사용 | TP/SL floor 사례: 컴포넌트 단독으로는 win이었으나 실제 greedy 단일 슬롯 라우터에서 +82.5% → **−14.6%**, MDD −15.5% → −29.8%로 반전 |
| C4 | pre-registration이 관행이지 규칙이 아님 | 07-26 세션은 훌륭하게 수행했으나(pre-reg 커밋 후 실행), 게이트가 이를 강제하지 않음 |

### D. 리서치 축 고갈 — **P2**

닫힌 축(재개 금지):
지표 탐색 · 크로스섹셔널 모멘텀(3코인/25코인) · 볼 타게팅 · 펀딩 스파이크 역추세 ·
베이시스 캐리 + 펀딩 플로어 · 매크로/TradFi(유료 데이터 포함) · h48qual 방향 모델
(zigzag/triple-barrier × TabM/HGB 2×2 전부) · 학습 하이퍼파라미터 · exit 로직(21라운드) ·
TP/SL 폭 · quality threshold · time exit · 2번째 슬롯 · 1m 스캘핑(4회) · BTC 튜닝(~2026-10까지).

살아남은 구조적 결론: **저빈도 + 추세추종 + 레짐 필터 + 트레일링 exit + 적은 거래수**.

미해결로 남은 구조적 사실 하나: ETH는 슬롯 점유율 84%로 **기회의 97~98%를 버리고 있다**.
이걸 ETH 안에서 푸는 시도(time exit, 2nd slot)는 전부 MDD에서 탈락했다.
구조적 해법은 ETH의 슬롯이 아니라 **비상관 자산 추가**다.

### E. 코드베이스 — **P2**

- `trading_bot.py` 단일 파일에 +15,888 라인 diff. 라이브 로직이 거대 모놀리스.
- `pytest` 미설치 → `test/` 62개 파일이 있는데 전체 스위트를 돌린 적이 없음
  (07-29 repair도 `unittest`로 focused test만 실행했다고 명시).

---

## 2. 설계

### Phase P0 — 데이터 아이덴티티 고정 (선행 필수)

P0가 끝나기 전의 모든 백테스트 숫자는 재측정 대상으로 본다.

**P0-1. content-addressed 데이터셋 스냅샷 — 원본 데이터 계층까지 포함 [구현 완료 2026-07-30]**

- **원본 zip 계층 (`ensure_metrics()`)**: `scripts/update_features.py`에 `RAW_SOURCE_MANIFEST`
  해시 매니페스트(`binance_data/RAW_SOURCE_MANIFEST.json`)를 배선. 신규/기존 파일 모두 처음 보면
  등록, 등록된 파일 내용이 바뀌면 `RuntimeError`로 즉시 실패. 네트워크 호출 없이 격리 테스트로
  검증(변조 → 실패 → 원복 → 통과), ETH metrics zip 931개(2024-01-01~2026-07-19, 공백 없음)
  전체를 기준선으로 등록 완료. 범위: `ensure_metrics()`만 수정, `ensure_funding()`/
  `ensure_klines()`는 그대로 둠(요청 범위 밖).
- **파생 feature CSV 계층 (`scripts/dataset_snapshot.py`, 신규)**: `training_features_2026_rebuilt.csv`의
  실제 writer를 추적한 결과 이 저장소는 그 파일을 만드는 단일 정식 스크립트가 없고(수십 개의
  1회성 빌드 스크립트가 후보), 매 빌드 스크립트를 개별 수정하는 대신 **독립 실행형
  register/verify 도구**를 만들어 `data/splits/DATASET_MANIFEST.json`에 등록하는 방식을 택함.
  `data/splits/` 하위 CSV 31개(연도별 split, RL 데이터셋, adaptive-squeeze 변형,
  `.bak_pre_extend_*` 백업 포함) 전부를 현재 상태로 기준선 등록 완료. 격리 테스트로 검증
  (파일 변조 → `verify` exit 1 + 정확한 해시 불일치 메시지 → 원복 → exit 0).
  `register`는 이미 등록된 경로를 절대 조용히 덮어쓰지 않고(`--adopt-drift` 명시 시에만,
  `previous_sha256`을 남기고 교체), `verify`는 CI/승격 게이트에서 재사용 가능한 형태
  (`exit 0`/`exit 1`).
- ~~원래 계획은 `scripts/update_features.py` 자체를 append-only 스냅샷 저장 방식
  (`<asset>_features_<year>__<sha8>.csv` + `LATEST` 포인터)으로 바꾸는 것이었으나, 실제
  구현 과정에서 `training_features_2026_rebuilt.csv`를 만드는 단일 정식 writer가
  없다는 게 드러나 위 register/verify 도구 방식으로 대체했다 (더 적은 코드로 동일한
  보장을 제공하고, 어떤 1회성 빌드 스크립트가 출력하든 동작한다).~~
- **원본 데이터도 동일하게 고정 (실측으로 필요성 확인됨)**: `binance_data/metrics/*.zip` 등
  원본 다운로드 파일도 최초 획득 시점에 sha256을 `binance_data/RAW_SOURCE_MANIFEST.json`에
  기록한다. `ensure_metrics()`가 같은 파일명을 다시 받았을 때 해시가 달라지면 **재수집이 아니라
  경고 + 별도 버전으로 보존**(덮어쓰지 않음). Binance의 OI/long-short-ratio 메트릭은 사후에
  소급 수정된다는 것이 실측으로 확인됐으므로(아래 발견 참고), 재현성은 파생 feature CSV가
  아니라 원본 zip 계층에서부터 고정해야 한다.
- **검증**: 같은 입력으로 2회 실행 → 동일 sha256. 다르면 비결정성이 있는 것이므로 그 자체가 버그.

**P0-2. 리포트에 입력 데이터 계보 기록 [구현 완료 2026-07-30]**
- `scripts/audit_omega_artifact_integrity_20260630.py`에 `dataset_lineage_checks()` 추가,
  `audit_component()`가 risk sidecar report와 parent bundle report 양쪽에 각각
  (`risk_*`/`parent_*` 접두사로) 적용. 4단계 순차 검사: (1) `dataset_lineage` 필드 존재 →
  (2) `features_path`/`features_sha256` 필드 완전성 → (3) `features_path`가
  `data/splits/DATASET_MANIFEST.json`에 등록돼 있는지 → (4) 등록된 해시가 리포트가 주장하는
  해시, 그리고 **현재 디스크상의 실제 파일 해시**와 모두 일치하는지. 어느 하나라도 실패하면
  fail-closed(등록 안 됨/일치 안 함=fail, 조용한 통과 없음).
- 앞으로 새 report.json을 쓰는 스크립트는 `scripts/dataset_snapshot.py`의
  `build_lineage_record(path)`를 호출해 `dataset_lineage` 필드를 채우면 된다(내부적으로
  register까지 자동 수행, 이미 등록돼 있으면 그대로 재사용).
- **검증 (4가지 케이스 전부 격리 테스트)**:
  1. 실제 존재하는 07-06 이전 frozen report(`.../report.json`, 이 게이트 이전에 작성됨)로
     감사 실행 → `risk_dataset_lineage_present`/`parent_dataset_lineage_present` 둘 다
     `fail`, 전체 exit=2. **계보 없는 기존 리포트는 의도대로 통과 못 함.**
  2. 합성 리포트에 실제 등록된 매니페스트 해시를 정확히 넣음 → 4개 체크 전부 `pass`.
  3. 리포트가 주장하는 해시를 조작 → `dataset_lineage_report_matches_manifest`만 `fail`.
  4. **핵심 재현**: 리포트/매니페스트 해시는 서로 일치하지만, 파일을 등록 이후 실제로 변조 →
     `dataset_lineage_matches_current_file`만 `fail` — 이게 정확히 Omega4.6.1 07-06 baseline이
     겪은 실패 유형이고, 이제 게이트가 이 케이스를 정확히 잡아낸다.
  기존 `test/test_omega_artifact_risk_selection_gate.py`(4개 테스트, `unittest`)도 회귀 없이
  전부 통과 확인.

**P0-3. frozen baseline 재현 검증 잡 [구현 완료 2026-07-30]**
- `scripts/verify_frozen_baselines.py` + 레지스트리 `docs/model_contracts/FROZEN_BASELINE_REGISTRY.json`
  생성. 각 baseline 항목은 `reference`(pnl/mdd/trades), `tolerance`, `features_path`,
  `features_sha256_at_freeze`(있으면 현재 매니페스트 해시와 대조해 어느 입력이 바뀌었는지
  보고, 07-06 baseline처럼 매니페스트 이전 시점이라 기록이 없으면 그 사실 자체를 정직하게
  보고), `wired`(실제 재실행 가능 여부) 필드를 가진다. `wired: false` 항목은 SKIP으로 보고되고
  전체 exit code에 영향 주지 않음(조용한 통과 아님, 명시적 backlog).
- Omega4.6.1 ETH greedy router를 실제 runner로 완전히 배선(`_run_omega461_eth_greedy_router()`,
  기존 replay 로직 재사용, prediction CSV가 06-30 이후로 조용히 연장된 것에 대한 truncate 처리는
  `research_eth_omega461_tpsl_floor_portfolio_check_20260728.py`가 쓴 것과 동일한 패턴 적용).
  Sigma6은 레지스트리에 항목은 등록했지만 별도의 1h trend-scan 파이프라인이라 runner 배선은
  이번 세션 범위 밖으로 미루고 `wired: false`로 명시(스코프 확장 방지).
- **검증**: 실제 실행 결과 —
  ```
  [FAIL] omega461_eth_greedy_router_20260706
         actual={pnl:82.53, mdd:-15.48, trades:31} reference={pnl:145.34, mdd:-10.13, trades:24}
  [SKIP] sigma6_1h_regime_trend_lev4_oos  (not wired, documented backlog)
  overall: FAIL (exit 1)
  ```
  설계 문서가 미리 정한 판정 기준("지금 실행하면 Sigma6과 Omega4.6.1이 실패로 잡혀야 한다")을
  정확히 충족 — Omega4.6.1은 실제로 재현 실패가 잡혔고, Sigma6은 조용히 통과 처리되지 않고
  명시적 SKIP으로 남았다.
- **이분 탐색 실행 완료 (2026-07-30, `tmp/research_20260730/bisect_omega461_baseline_reproduction.py`)**:
  예측 CSV(고정, 재계산 안 함)와 각 후보 프레임의 공통 타임스탬프 교집합으로 정렬해 동일한
  greedy router를 4개 버전에 재실행.

  | 버전 | PnL | MDD | trades |
  |---|---|---|---|
  | current (07-21) | 82.53% | −15.48% | 31 |
  | bak_pre_extend_20260720 | 82.53% | −15.48% | 31 (07-20 이전과 완전 동일) |
  | bak_pre_extend_20260713 (07-06에 가장 가까운 스냅샷) | 105.68% | −15.48% | 28 |
  | bak_pre_extend_20260704 (2월까지만, 창 너무 짧아 참고용) | 40.97% | −15.48% | 11 |
  | **frozen target (07-06)** | **145.34%** | **−10.13%** | **24** |

  **결론: 재현 불가.** 07-06에 가장 가까운 복구 가능 스냅샷(07-13 백업)조차 frozen 수치의
  절반 정도 갭을 이미 안고 있다 — drift가 07-13/07-20/07-21 extend 이벤트만으로 설명되지 않고,
  **07-06(모델 빌드)과 07-13(첫 백업) 사이에 이미 조용한 재생성이 있었으며 그 시점 파일은
  디스크에 남아있지 않다.** 수치가 시간을 거슬러 82.5%→105.7%→145.3%로 단조적으로 개선되는
  패턴은 단발 사고가 아니라 **연속적 점진적 drift**임을 시사한다.
  **현재 보유 파일로는 07-06 frozen baseline을 바이트 단위로 복구할 수 없다** — 이는 손실로
  확정하고, P0-1(스냅샷 매니페스트)이 있었다면 이 손실 자체가 발생하지 않았을 것이라는 점을
  재현성 정책의 근거로 남긴다.
  (부수 관찰: 4개 버전 모두 MDD가 소수점 14자리까지 −15.48%로 동일 — 최악 drawdown을 만드는
  단일 트레이드가 매 버전에서 동일 entry/exit·notional cap을 맞아 우연히 고정된 것으로 추정되며
  결론에 영향 없으나 감사 스크립트 작성 시 참고할 이상 신호로 기록.)

- **근본 원인 특정 완료**: current와 bak13 간 95행 차이(A3b)를 행 단위로 대조한 결과, 공통
  타임스탬프 자체는 95행만 다르지만 **그 공통 구간의 컬럼값은 51,746행 중 46,268행에서 서로
  다르다** — `sum_open_interest_value`, `sum_toptrader_long_short_ratio`,
  `count_long_short_ratio`, `whale_retail_ratio` 4개 컬럼(141개 중 125개 컬럼에 영향).
  반면 raw OHLC(`close`/`volume` 등)는 딱 1행만 다르다. 즉 **feature 엔지니어링 로직이나
  `update_features.py`의 concat 순서(07-13 fix가 다룬 문제) 문제가 아니라, 그 위쪽 원본
  데이터 자체가 통째로 바뀐 것.**
  `binance_data/metrics/ETHUSDT-metrics-*.zip` mtime을 확인하니 Jan-Jun 2026 구간 파일 중
  **78개(약 43%)가 2026-07-02에 재다운로드**돼 있었다(그 외 03-11/03-16/04-14에도 각각
  68/6/28개씩 재다운로드 이력 — 일회성이 아니라 **반복되는 패턴**). `TOTAL_ETHUSDT_metrics.csv`
  캐시 파일 자체는 2월 이후 변경 없음(코드의 cache-first 로직은 정상 작동) — **drift의
  실제 출처는 Binance의 open-interest/long-short-ratio/고래비율 메트릭이 사후에 소급
  수정되어 재수집되는 것**이다.
  **함의**: P0-1의 content-addressing 대상은 최종 feature CSV만으로는 불충분하다 — 원본
  `binance_data/metrics/*.zip`(그리고 잠재적으로 klines/funding zip도) 자체를 최초 다운로드
  시점에 해시 고정(pin)하고, 이후 재다운로드로 내용이 달라지면 그 사실 자체를 감지·기록해야
  한다. Binance API가 과거 값을 "최종"으로 보장하지 않는다는 뜻이므로, 재현성은 원본 데이터
  계층에서부터 고정해야 한다.

**P0-4. Git 상태 정상화**
- 라이브 스택(코드 + config + 계약 문서)을 커밋. 대용량 아티팩트는 커밋하지 말고
  DATASET_MANIFEST와 동일한 해시 매니페스트만 커밋.
- **규모 확인: `tmp/` 75GB, `data/` 32GB, `tmp/causal_regen_20260516/` 하위 디렉터리 2151개.**
  LFS도 현실적이지 않다 — 해시 매니페스트만 커밋하는 방식이 유일한 선택지다.
  untracked 3382의 대부분이 이 두 디렉터리이므로, `.gitignore` 정리만으로 대부분 해소된다.
- 부수 효과로 `tmp/causal_regen_20260516/`의 2151개 디렉터리 중 어느 것이 라이브 계보에 속하는지가
  매니페스트로 명시된다(현재는 `CURRENT_LIVE_MANIFEST.json`의 6개 경로 외에는 구분 불가).
- **검증**: `git status --short`가 의미 있는 길이로 줄고, `CURRENT_LIVE_MANIFEST.json`의
  `dirty_worktree` blocker가 해소.

### Phase P1 — 리스크 측정 교정 + 라이브 노출 결정

**P1-1. bar-level MDD를 표준 지표로 승격 [구현 완료 2026-07-30]**
- `core/backtest_metrics.py::bar_level_performance()`로 승격 완료.
  `test/test_backtest_metrics.py`(unittest 6건)가 07-28 리서치 스크립트가 저장해둔 ETH/SOL/BTC
  VAL/OOS 6개 셀의 equity curve(.npy)+ledger(.csv)를 그대로 읽어 정확히 재현하는지 검증 —
  전부 통과, 재구현이 아니라 충실한 이관임을 확인.

**P1-2. 사이징 체인 단일화 + parity 테스트 [구현 완료 2026-07-30]**
- 조사 결과 라이브 쪽(`trading_bot.py`)은 이미 다른 세션이 `finalize_sizing()`
  (`trading_bot_modules/omega4_6_1_runtime_contract.py`)과
  `PortfolioRiskManager.scale_to_budget()`(`trading_bot_modules/portfolio_risk.py`)로
  component cap → multiplier → portfolio cap → final cap 체인을 단일 함수로 통일해둔 상태였음.
  반면 07-28 리서치 스크립트는 이 함수들이 생기기 전에 작성돼서 **같은 로직을 손으로
  재구현**하고 있었다 — 이게 실제 parity 갭.
- `test/test_sizing_chain_parity.py`(unittest 2건): (1) 현재 라이브 상수 하에서는 손으로 만든
  공식과 진짜 공유 함수가 정확히 일치함을 확인, (2) 포트폴리오 캡을 컴포넌트 캡보다 느슨하게
  만드는 스트레스 케이스에서는 둘이 갈라짐을 확인 — 오늘의 일치가 우연이지 구조적 보장이
  아님을 증명.
- **검증**: 두 테스트 파일 모두 통과.

**P1-3. 현재 라이브 스택 재판정 [완료 2026-07-30, 한 차례 정정 포함]**
- `tmp/research_20260730/three_asset_bar_level_mdd_parity_verified.py`: 07-28 스크립트를
  실제 공유 함수를 쓰도록 바꾼 버전으로 3자산을 현재(등록된) 데이터로 재측정.
- **1차 실행 결과가 틀렸음이 드러남**: ETH의 실제 라이브 전용 chop soft-sizing 오버레이
  (`trading_bot.py` ~9248-9259줄, `.env FINAL_GOVERNOR_OMEGA4_6_1_ETH_CHOP_SOFT_SIZE_ENABLE=True`,
  `THRESHOLD=0.3`)가 재사용한 07-28 컴포넌트 준비 모듈에 없다는 걸 사용자 질문으로 발견.
  정확한 공식(`regime3_current_sensitive_wide24_chop_prob` 컬럼 기반, threshold 미만은
  full size, 이상은 0까지 선형 감소)을 라이브 코드 그대로 이식해 재실행.
- **최종(정정된) 결과**:

  | 자산 | Split | PnL (07-28→07-30 chop 미반영→chop 반영) | Bar MDD (동일 순서) | −25% 게이트 |
  |---|---|---|---|---|
  | ETH | VAL | 84.73%→80.45%→**66.61%** | −20.28%→−17.58%→**−15.46%** | 통과 |
  | ETH | OOS | 58.67%→60.80%→**58.35%** | −28.28%→−25.09%→**−21.00%** | **여유있게 통과** (chop 미반영시 나왔던 "경계선 위반"은 오류였음, 정정) |
  | SOL | VAL/OOS | 세 번 측정 모두 완전 동일 | 동일 | VAL 위반 지속 |
  | BTC | VAL/OOS | 세 번 측정 모두 완전 동일 | 동일 | 통과 |

- **판정**: (a) ETH/BTC는 게이트 통과 — 노출 축소 불필요. (c) SOL은 VAL PnL 마이너스(−7.58%)
  + MDD 위반(−25.69%)이 3회 독립 측정(다른 날짜, chop 유무 무관 — SOL엔 chop 로직 자체가
  없음)에서 완전히 동일하게 재현 — 구조적 문제로 확정. **SOL 실계좌 실행 비활성화를 권고**
  (`.env`의 `SOL_BTC_REAL_EXECUTION_ENABLE=True`는 건드리지 않음, 권고만).
  SOL sidecar의 `validation_oos_guard` 오염은 여전히 별도 미해결 이슈.
- **이 정정이 남기는 교훈**: 라이브 동작을 재현한다고 주장하는 리플레이는 `.env`의 모든
  `FINAL_GOVERNOR_OMEGA4_6_1_*` 플래그를 하나씩 확인해서 반영 여부를 명시해야 한다 —
  스크립트가 에러 없이 돌아간다고 해서 완전하다고 가정하면 안 된다.

### Phase P2 — 통계 예산 관리

**P2-1. 홀드아웃 예산 레지스터**
- `docs/holdout_budget.json`: 윈도우별 평가 횟수와 소진 표시.
  2026-01..06은 이미 `spent`로 등록.
- 신규 후보는 미개봉 구간(2026-07 이후)에서만 최종 판정. 시간이 지나야 예산이 생기므로
  **평가 횟수를 자원으로 취급**한다.

**P2-2. pre-registration 강제**
- `docs/preregistration/<id>.md`에 그리드·게이트·코스트 모델을 먼저 커밋.
- 결과 리포트가 유효한 pre-reg id와 커밋 해시를 참조하지 않으면 promotion gate 거부.
- 07-26 세션이 자발적으로 한 절차를 규칙으로 승격하는 것뿐 — 새 프로세스가 아니다.

**P2-3. router-level 검증 의무화**
- standalone 컴포넌트 결과는 승격 근거로 인정하지 않음.
- greedy 단일 슬롯 라우터 리플레이 통과를 필수 조건으로 게이트에 추가.

### Phase P3 — 리서치 방향

방향 예측 개선은 접는다. 남은 갈래는 셋:

**(a) 리스크/사이징 레이어 — 최우선.**
유일하게 *실제 미해결 문제가 남아 있는* 영역이고(§B),
**새 알파를 요구하지 않는다**. 같은 신호로 MDD를 −28%에서 게이트 안쪽으로 되돌리는 것은
PnL을 늘리는 것보다 성공 확률이 훨씬 높다.

**(b) 비상관 자산 추가 — ETH 84% 슬롯 점유의 구조적 해법.**
SOL/BTC 스택이 이미 재사용 가능한 템플릿(`project-sol-pilot-20260707`).
단, ETH 튜닝 상수는 자산별로 재도출해야 하고 audit 스크립트가 ETH import를 하드코딩 중이라
그 부분 일반화가 선행 작업이다. **P1-3에서 SOL이 (c)로 판정되면 이 갈래는 보류** —
상관 낮은 자산을 더 붙이기 전에 기존 자산이 게이트를 지키는지가 먼저다.

**(c) alt-data — 2026-10 이후.**
Fear&Greed, Binance-OKX funding spread, 청산 캐스케이드(`tail_risk_1m`).
전부 히스토리 부족으로 대기 중. **지금 할 일은 실험이 아니라 수집기가 끊기지 않게 하는 것**
(`run_live_collectors.py` 가용성 모니터링). 데이터 공백이 생기면 10월에도 못 한다.

**재개 금지 목록**: §D의 닫힌 축 전부. 유료 인트라데이 주식/선물 데이터 구매도 포함
(무료 캘린더 신호가 이미 VAL에서 탈락 → 유료 데이터의 증분 가치 미입증).

---

## 3. 실행 순서와 판정 기준

| 순서 | 작업 | 완료 판정 | 상태 |
|---|---|---|---|
| 1 | P0-1 데이터셋 스냅샷 + 매니페스트 | 2회 실행 sha 동일 | ✅ 완료 |
| 2 | P0-3 재현 검증기 | Sigma6/Omega4.6.1을 실패로 검출 | ✅ 완료 (Omega4.6.1만 배선, Sigma6 backlog) |
| 3 | P0-2 게이트에 계보 검사 추가 | 계보 없는 리포트 exit≠0 | ✅ 완료 |
| 4 | P0-4 git 정리 | `dirty_worktree` blocker 해소 | 미착수 |
| 5 | P1-1 bar-level MDD 표준화 | 기존 6셀 수치 재현 | ✅ 완료 |
| 6 | P1-2 사이징 단일화 + parity | parity 테스트 통과 | ✅ 완료 |
| 7 | P1-3 라이브 재판정 | (a)/(b)/(c) 중 하나로 결론 + 노출 조정 실행 | ✅ 완료 — ETH/BTC 통과(a), SOL 비활성화 권고(c). 노출 조정은 권고만, `.env` 미변경 |
| 8 | P2-1~3 예산·pre-reg·router 게이트 | 게이트가 위반 케이스를 거부 | 미착수 |
| 9 | P3 (a) 리스크 레이어 리서치 착수 | pre-reg 후 실행 | 미착수 |

1~4가 끝나기 전에는 새 백테스트 숫자를 승격 근거로 쓰지 않는다 (4는 아직 미착수 — git 정리는
남아있음).

---

## 4. 이 설계가 명시적으로 하지 않는 것

- 새 모델·새 피처·새 지표를 제안하지 않는다. 그 축은 §D에서 닫혔다.
- 라이브 실행을 활성화하지 않는다. `BINANCE_ACCOUNT_ENABLED`는 독립 안전 게이트로 유지.
- 기존 frozen 아티팩트를 삭제하지 않는다. 재현 실패는 **기록**하고, 덮어쓰지 않는다.
