# ETH spot-perp 베이스 — 방향(direction) 바이어스 cheap gate (2026-08-20)

## 판정: **기각 (CLOSED)**

## 목적

`docs/eth_direction_alpha_non_microstructure_research_20260817.md` 로드맵에 새로 식별된
후보(spot-perp 고빈도 동학, 로드맵 §3.4/§5 5순위 — cheap-gate 미실행 상태였음). perp(선물)
가격과 spot 가격의 순간 베이스가 ETH 5분봉 방향에 정보를 갖는지 검정한다.

## 착수 전 문헌 재검증(중요)

최초 제안 시 인용된 Makarov & Schoar(2019/2020, *JFE*)와 Alexander & Heck(2020, *J. Financial
Stability*)를 원문 확인한 결과, 둘 다 이 가설과 정확히 맞지 않음이 드러났다 — 전자는
spot-vs-spot **거래소간** 차익거래(김치프리미엄류), 후자는 베이스 수준이 아니라 **가격발견**
(어느 시장이 먼저 움직이는가) 연구다. 실제로 부합하는 문헌(Schmeling/Schrimpf/Todorov "Crypto
Carry", BIS WP1087/Management Science DOI:10.1287/mnsc.2024.05069; He/Manela/Ross/von Wachter
"Fundamentals of Perpetual Futures", arXiv:2212.06888)을 원문 확인한 결과:
- 인과관계는 **과거추세→레버리지 롱 수요→베이스**(베이스가 추세의 결과이지 원인이 아님).
- 일별 자기상관 0.86~1.0 — 느리게 움직이며 빠르게 평균회귀하지 않음.
- 표준화 캐리 10%↑ → 향후 1개월 **숏 청산** 22%↑, 미래 **변동성**을 유의하게 예측.
- 즉 문헌은 베이스를 **방향이 아닌 변동성/청산크라우딩 신호**로 지목한다.

사용자에게 이 사실을 알리고 그래도 방향 가설로 직접 검정할지 확인 → **방향 가설로 진행하라는
명시적 지시**를 받아 아래를 실행. 즉 이 게이트는 "문헌 예측이 실제로 맞는지"를 검증하는
성격도 겸한다.

## 데이터

- **perp**: 기존 canonical `data/splits/year_oos/training_features_{2024,2025,2026_rebuilt}.csv`
  의 `close`(Binance `fapi.binance.com` 유래, `scripts/extend_klines_20260713.py` 확인 —
  이 저장소는 지금까지 선물만 수집해왔고 spot 컬럼은 전무했음).
- **spot**: 이번에 신규 수집(`scripts/eth_fetch_spot_klines_20260820.py`, `api.binance.com`
  공개 REST, 계정 자격증명 미사용). ETHUSDT spot 5분봉 2024-01-01~2026-08-20,
  **277,105행, 5분간격 커버리지 100%(gap 없음)**.
- 병합: timestamp 기준 inner join, **268,082/268,082행(100%) 성공**(perp 전체가 spot과
  매칭됨 — spot이 perp보다 늦게 시작한 것도 아니고 gap도 없어 손실 없음).

## 방법

- **신호 구성**(causal, 매 5분bar 종가 기준): `basis_raw = (perp_close - spot_close) /
  spot_close`. 문헌 권고에 따라 두 파생신호 추가: `basis_z48`(48bar=4시간 롤링 z-score,
  funding 정산주기 96bar=8h의 절반 — 추세/레짐 성분 제거용), `basis_roc12`(12bar=1시간
  변화량 — funding 정산 간격보다 짧은 정보 격리용, house convention `cvd_12`/`funding_roc_12`
  와 동일 창).
- **오염체크**(신호 vs 동시점 perp 종가 Spearman, `feedback_raw_feature_price_trend_
  contamination` 규칙): `basis_raw` rho=+0.366(임계 0.5 미만이라 통과하나 다른 둘보다 확연히
  높음 — 미차분 원시신호라 레짐 드리프트 일부 반영), `basis_z48`/`basis_roc12`는 rho≈0(설계대로
  깨끗).
- **IC 스캔**: 3-split(**TRAIN 2024-01-01~2025-08-31 / VAL 2025-09-01~12-31 / OOS
  2026-01-01~03-31**, 캐노니컬 Fresh-Forward 경계와 동일) × 3신호 × 4호라이즌(1/3/12/48bar=
  5분/15분/1시간/4시간 — 일별해상도였던 기존 ETF/스테이블코인 cheap gate의 1/3/7일 대신,
  funding 정산주기보다 짧은 구간에서 문헌이 지목한 가설을 직접 검정하도록 조정) × 순열귀무
  (라벨 2000회 셔플, 벡터화 랭크-행렬곱 방식).
- **벤치마크 백테스트**: 방향규칙은 **TRAIN IC 부호로만 결정**(VAL/OOS는 방향선택에 전혀
  사용 안 함 — look-ahead 없는 사전등록 원칙), 매 h-bar 겹침없는 주기적 재진입, 왕복비용
  10bp(이 저장소 반복 인용 관행) 고정 차감, `max(always_long, always_short)` 대비 증분 비교.

## 결과

### 오염체크 — 통과(단 basis_raw는 다른 둘보다 높음)

| 신호 | rho vs 가격 |
|---|---|
| basis_raw | +0.366 |
| basis_z48 | +0.001 |
| basis_roc12 | −0.001 |

### IC 스캔 — 순열귀무 \|z\|≥2 36칸 중 19칸 통과(다른 3개 후보보다 훨씬 높은 비율)이나 부호가 split마다 뒤집힘

| 신호 | TRAIN 부호(4개 호라이즌) | VAL 부호 | OOS 부호 |
|---|---|---|---|
| basis_raw | 혼재(3음1양) | 전부 양(2개 유의) | 전부 음(1개 유의) |
| basis_z48 | **전부 음(4개 다 유의)** | **전부 양(2개 유의)** | 대부분 음(2개 유의, h48만 양) |
| basis_roc12 | **전부 음(3개 유의)** | 혼재(h12/48만 양, 유의) | 대부분 음(2개 유의) |

`basis_z48`/`basis_roc12`는 TRAIN에서 4개 호라이즌 전부 일관되게 음수(유의)였다가 VAL에서
양수로 뒤집히고 OOS에서 다시 음수로 돌아오는 **3-way 부호교대** 패턴 — 문헌이 예측한 "베이스는
레짐 종속적"이라는 진단과 정확히 부합한다(고정된 방향 관계가 아니라 그때그때 시장 레짐에 따라
상관 부호 자체가 바뀜).

### 벤치마크 백테스트 — **0/12 조합 3-split 전부 양수**

방향규칙을 TRAIN IC 부호로만 고정(look-ahead 없음)한 뒤 VAL/OOS에 그대로 적용:

| 신호 | h | TRAIN 증분(bp) | VAL 증분(bp) | OOS 증분(bp) |
|---|---|---:|---:|---:|
| basis_raw | 1 | −0.0 | −0.3 | −0.3 |
| basis_raw | 48 | −1.3 | −10.6 | −13.0 |
| basis_z48 | 1 | −0.0 | −0.2 | +0.1 |
| basis_z48 | 48 | −0.5 | −3.3 | −9.3 |
| basis_roc12 | 12 | −0.6 | −2.4 | +1.0 |
| basis_roc12 | 48 | +1.0 | −4.9 | +0.9 |

(전체 12칸 상세는 `tmp/eth_spot_perp_basis_backtest_20260820.json`) **12개 조합 중 어느
하나도 TRAIN·VAL·OOS 전부 양수가 아니다.** 더 근본적으로: **TRAIN 자체 증분조차 대부분
0 근처이거나 음수**(방향규칙을 TRAIN에서 뽑았는데도) — IC스캔에서 "유의"했던 상관이 비용
반영 실거래 증분으로는 거의 전환되지 않는다는 뜻이다. 호라이즌이 길수록(48bar) 손실이
커지는 경향(basis_raw h48: VAL−10.6bp/OOS−13.0bp)도 관찰된다.

## 판정 근거

1. IC 스캔의 표면적 통과율(19/36)은 이 로드맵에서 가장 높지만, split별 부호를 보면
   TRAIN↔VAL↔OOS가 반복적으로 뒤집힌다 — "우연히 자주 유의선을 넘는 잡음"의 전형적 서명이지
   안정된 관계의 서명이 아니다.
2. TRAIN IC 부호로 look-ahead 없이 고정한 방향규칙조차 TRAIN 자체에서 벤치마크를 못 이긴다 —
   스캔 단계 유의성이 애초에 방향 판별력으로 이어지지 않는다.
3. 12개 (신호,호라이즌) 조합 전부 3-split 동시 양수 실패 — 완전 기각.
4. 결과 패턴이 착수 전 문헌 예측(베이스는 과거추세의 결과물이라 방향 선행지표가 아님)과
   정확히 부합 — 문헌을 무시하고 직접 검정했지만 문헌이 옳았다.

## 한계

- 호라이즌 4종(1/3/12/48bar)·랙 없음(당bar 종가)만 스윕 — 더 넓은 그리드는 다중비교만
  키운다는 이 로드맵의 기존 원칙(ETF cheap gate 한계절 참고)을 그대로 따름.
- perp/spot 둘 다 kline **종가**를 씀(진짜 mark price/index price 아님) — 문헌의
  이론적 정의(perp mark vs index)에 대한 근사치.
- basis_raw는 오염체크상 완전히 깨끗하진 않음(rho=0.366, 임계 미만이나 z48/roc12보다 높음) —
  단, 백테스트 결과 자체가 이미 명확히 기각이라 이 한계가 결론을 바꾸지 않음.

## 재개 조건

동일 신호(perp-spot kline종가 베이스, 원시/z-score/변화량)로 방향 가설 재제안 금지. 문헌이
실제로 지목하는 역할(변동성 예측/청산·크라우딩 리스크 게이트)로 재프레이밍하는 것은 질적으로
다른 가설이라 재개 가능 — 단 이는 direction_head가 아니라 리스크사이징/청산타이밍 레이어용
피쳐이므로 `eth_tabm_label_logic_retest_initiative_20260819`(방향라벨 서브프로젝트)의 범위
밖이다.

## 참고

- `docs/eth_direction_alpha_non_microstructure_research_20260817.md` — 로드맵 원문
- `scripts/eth_fetch_spot_klines_20260820.py` — spot kline 신규 수집
- `scripts/research_eth_spot_perp_basis_direction_cheap_gate_20260820.py` — IC스캔
- `scripts/research_eth_spot_perp_basis_backtest_20260820.py` — 벤치마크 백테스트
- 메모리: `eth_gex_status_and_next_direction_candidates_20260820`
