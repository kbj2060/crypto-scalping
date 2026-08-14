# h48qual exit_head 레짐인지형 지속상승장 가드 — 검증 (2026-08-14)

## 배경

`eth_omega461_exit_head_liveatr_sustained_uptrend_vulnerability_20260814.md`가 발견한 리스크:
현재 섀도우 배포된 baseline(`asymmetric_tabm_liveatr` = h48qual만 exit_head liveATR 재라벨,
zig075는 원본 `exit_threshold=0.95` 정적 판단 그대로)이 2025-Q3(2025년 유일한 강한 지속
상승장, 가격드리프트 +66.63%, 세 분기 중 최저 변동성 60.75%)에서 재라벨 전 원본보다 4.7배
악화(no_gate PnL -9.73%→-46.26%)됐다. 메커니즘: exit_head 재라벨이 h48qual 평균 보유기간을
2~3배 단축시키는 "회전 가속기"인데, Q1(거친 하락장)에서는 거래수가 안 늘지만(8건→8건) Q3(노이즈
적은 지속 상승)에서는 풀린 슬롯이 "이미 하락장 베타에만 의존하는 무편향 숏 신호"를 반복
재점화시켜 거래수가 8건→18건(전부 SHORT)으로 폭증한다.

그 문서 6절이 미검증 제안으로만 남긴 완화 방향 — "지속 상승 레짐에서만 h48qual을 원본 exit로
되돌리는 조건부 정책" — 을 이 문서가 실제로 구현·검증한다.

이 실험은 **Odyssey2 #11**이다(계약 문서 실행 로그 참고). 스크립트:
`scripts/research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814.py`. 실행 로그:
`tmp/causal_regen_20260516/eth_omega461_regime_aware_exit_head_uptrend_guard_20260814/report.json`
(아래 모든 수치는 이 파일을 직접 읽어 대조했다).

## 이 실험의 게이트가 이전 9건과 다른 이유

이전 Odyssey2 후보들(#7 대기압력, #8 risk-controlled, #9 conformal kelly 등)은 전부 "VAL에서
개선 → OOS 확인"이었다. 이건 이번엔 안 통한다 — VAL(2025-10~12)은 하락 구간(드리프트 -28.41%)
이라 고칠 대상인 실패 모드(지속 상승장에서의 거래량 폭증) 자체가 VAL에 나타나지 않는다. 따라서:

1. **2025 Q1/Q2/Q3(참고용 context tier)가 이번엔 핵심 증거**다 — Q3 손상이 완화되는지, Q1/Q2의
   기존 이득이 보존되는지를 최우선으로 본다.
2. VAL·OOS-Q1·OOS-Q2는 **비악화(non-regression) 확인 전용**이다 — 이 구간들엔 지속상승 레짐이
   없으므로 "개선"을 요구하지 않는다.
3. 최종판정은 PASS/REJECT 이분법이 아니라 3단 서술이다(아래 "최종 판정" 절).

## 탐지기 설계: causal "지속 상승" 레짐 탐지 (순서가 핵심 — 사후 끼워맞춤 방지)

### 하지 않은 것

라이브에 이미 있는 HMM `regime3`(bull/bear/chop) 분류기를 재사용하지 않았다 — 어제 조사
(`eth_val_oos_regime_mismatch_investigation_20260813.md` §2)가 이미 Q3의 regime3 bull 비중이
27.50%로 Q1(25.57%)·Q2(27.64%)와 거의 차이가 없음을 보였다(가격 드리프트는 Q3가 압도적으로
강한데도).

### 1단계 — 탐색: 기존 피처 후보들의 quarter 구분력 (임계값 확정 전, 순수 서술)

사용자가 지정한 목록(`regime_persistence`, `chop_index`, `hurst_48`, `regime_trending`,
`sig_trend_health`, `hma_slope`, `mtf_trend_1h`, `mtf_trend_4h`, `dual_momentum`,
`breakout_strength`, `cvd_slope_12/48`, `funding_roc_12/48/288`) 전부를 `data/splits/year_oos/
training_features_{2025,2026_rebuilt}.csv`에서 로드해(6개 창 공통 로더
`eth_omega461_multiwindow_confirmation_gate_20260814.load_all_windows()` 재사용) 각 피처의
per-bar 순간값이 2025 Q1/Q2/Q3를 얼마나 구분하는지 조사했다.

**발견: 순간값(≤288bar 윈도우) 기반 피처는 전부 quarter 구분에 실패했다.** 예시(`regime_persistence`,
2025 Q1+Q2로만 캘리브레이션한 90th percentile=0.5212 기준 활성화율):

| 구간 | 2025q1 | 2025q2 | 2025q3 | val | oos_q1 | oos_q2 |
|---|---:|---:|---:|---:|---:|---:|
| `regime_persistence`>P90 활성화율 | 8.7% | 11.3% | 10.6% | 9.2% | 10.1% | 9.0% |
| `chop_index`(<38.2, 코드베이스 기존 관례) AND `regime_persistence`>0 | 5.3% | 6.4% | 7.0% | 6.1% | 6.5% | 7.2% |

Q3가 전혀 튀지 않는다(Q2가 오히려 더 높기도 함) — **어제 발견한 regime3 HMM bull 비중의 동일
패턴(분기 간 거의 무차이)이 여기서도 재현됐다.** `chop_index`(표준 Choppiness Index 공식,
`_calc_chop`)·`hurst_48`(`regime_trending`의 원 정의, `>0.5`가 텍스트북 기준)·`sig_trend_health`
등 나머지 후보도 동일하게 확인했다 — 전부 5m봉 국지적 기술 레짐 상태가 분기 스케일의 지속
드리프트를 못 잡는다는 결론으로 수렴했다.

**예외: `dual_momentum`**(`features/engineering.py _dual_momentum` — 이미 1주일 lookback
[`close.shift(2016)`]로 설계됨, ETH 자체 1주 수익률과 BTC 대비 상대 1주 수익률이 **둘 다** 양수일
때만 +1) 만은 평균값으로 quarter 드리프트 방향을 깨끗이 추적했다: Q1 평균 -0.331, Q2 -0.014,
**Q3 +0.248(최고)**, val -0.165, oos_q1 -0.158, oos_q2 -0.257.

### 2단계 — 단순 조합 1개만 시도: rolling 1주 fracpos

`dual_momentum` 자체는 bar 단위로 노이즈가 있어(1주 수익률이 0을 넘나드는 근처에서 흔들림),
**딱 하나의 단순 조합만** 시도했다: `(dual_momentum>0)`의 `rolling(2016, min_periods=2016).mean()`
— 2016bar(=1주)는 `dual_momentum` 자신이 이미 쓰는 lookback을 재사용한 것이지 새로 고른 숫자가
아니다. 같은 window를 순간값 후보들(`regime_persistence` 등)에 적용해도, 또는 더 짧은 288bar(1일)
window를 `dual_momentum`에 적용해도(둘 다 확인함, 후자는 p90/p95가 1.0/0.0%로 퇴화) Q3 분리에
실패했다 — "롤링 집계"가 아니라 "이미 1주 스케일인 입력을 1주 스케일로 다시 집계"하는 조합
자체가 핵심임을 확인했다.

### 3단계 — 임계값 고정 (Q3를 보기 전에)

이 rolling 시리즈의 **90th percentile**("피처의 과거 분포 상위 X%"라는, 과제 자체가 예시로 든
원칙적 규칙), 계산 구간은 **2025-01-01~2025-06-30(Q1+Q2만, Q3는 캘리브레이션 표본에서 제외)**.
90%는 표준 십분위 관례(탐색적으로 스윕하지 않음). 결과: **threshold=0.802579**
(75th=0.561012, 95th=0.877331도 강건성 참고용으로 함께 계산).

### 4단계 — 확인 (임계값을 고정한 뒤에만 Q3 수치를 확인)

| 구간 | n_bars | NaN(warm-up) | 활성화율(P75=0.561) | **활성화율(P90=0.8026, 주 사용)** | 활성화율(P95=0.877) |
|---|---:|---:|---:|---:|---:|
| 2025q1 | 25,918 | 2,015 | 18.3% | 7.6% | 3.0% |
| 2025q2 | 26,204 | 0 | 29.7% | 11.6% | 6.6% |
| **2025q3** | 26,483 | 0 | **58.4%** | **43.0%** | **38.4%** |
| val | 26,209 | 0 | 23.4% | 7.6% | 5.6% |
| oos_q1 | 25,538 | 2,015 | 17.8% | 5.4% | 4.3% |
| oos_q2 | 25,921 | 0 | 24.8% | 8.2% | 3.9% |

세 percentile 전부에서 Q3가 나머지 5개 창보다 4~10배 높다 — 정확히 90th percentile 하나만의
우연이 아니라는 강건성 근거. NaN(rolling 미충족 warm-up)은 각 base CSV의 연초 최초 1주에만
발생(causal, window 경계 아티팩트 아님 — VAL/oos_q2/2025q2/q3는 연중 시작이라 NaN=0).

## 개입 지점과 설계 원칙

- 바뀌는 것은 **h48qual의 "held position 매 bar 청산확률을 어느 학습된 exit_head 모델로
  물어볼지"** 뿐이다. 탐지기 활성 → 원본 번들(`research_eth_omega461_exit_head_portfolio_
  asymmetric_20260813._component_cfg("h48qual")`, override 없음)의 `base_np`/`exit_runtime`/
  `pos_idx`. 비활성 → 현재 섀도우 기본값인 liveATR 재라벨 번들
  (`bundle_override=portfolio.NEW_H48QUAL_BUNDLE`)의 것. `exit_threshold`는 둘 다 0.95로 동일
  (다른 건 모델 가중치뿐, 숫자 임계값이 아니다).
- **entry/사이징은 가드 상태와 무관하게 항상 liveATR 준비값만 사용**(설계상 그렇게 구성 — 원본
  번들의 `dec`/`margin`/`leverage`는 아예 읽지 않음). VAL에서 직접 대조한 결과 두 번들의
  `dec[side]`/`take_profit`/`stop_loss`/`margin`/`leverage` 배열은 **완전히 동일**했다
  (`entry_side_diagnostic_val` = 전부 `true`) — 가정이 아니라 확인된 사실.
- zig075는 모든 창에서 완전 동결(`portfolio._component_cfg("zig075")`, override 없음).
- 재학습 없음 — `replay_omega4_6_1_greedy_router_20260706.greedy_replay`의 이름 바꾼 복사본
  (`greedy_replay_regime_aware_exit_guard`)이 h48qual 보유 중 exit_head 확률 조회 블록 하나만
  조건부로 교체한다. 원본 모듈은 import·읽기 전용.

## G0 (task 지정 필수 기준)

| 항목 | val no_gate | val with_gate | oos_q1 no_gate | oos_q1 with_gate |
|---|---:|---:|---:|---:|
| 요구값 | 46.59%/-21.70%/35 | 77.31%/-21.76%/26 | 93.27%/-15.48%/24 | 67.25%/-15.48%/19 |
| **G0a**(기존 게이트 모듈 `run_portfolio_variant` 재사용) | 46.59%/-21.70%/35 ✓ | 77.31%/-21.76%/26 ✓ | 93.27%/-15.48%/24 ✓ | 67.25%/-15.48%/19 ✓ |
| **G0b**(이 스크립트의 신규 렌임드카피, 가드/mask 완전 미부착) | 46.59%/-21.70%/35 ✓ | 77.31%/-21.76%/26 ✓ | 93.27%/-15.48%/24 ✓ | 67.25%/-15.48%/19 ✓ |

G0a·G0b 둘 다 정확히 일치(`gate_pass_g0=true`). G0b는 탐지기 로직 자체의 무결성 확인용 — mask를
아예 부착하지 않은 상태(구조적으로 탐지기가 "미발동 고정"인 경우)에서 신규 렌임드카피가
`greedy_replay`와 바이트 단위로 같은 결과를 내는지 검증한다(`guard_active_bars=0` 둘 다 확인).

## 6창 비교 (원본 / 재라벨(현 섀도우 기본값) / 조건부 정책)

no_gate PnL / MDD / 거래수 · with_gate PnL / MDD / 거래수, `report.json`의 `comparison`에서
직접 대조:

| 창(tier) | 원본(baseline_both_original) | 재라벨(asymmetric_tabm_liveatr) | **조건부 정책(regime_aware_guard)** | 가드 발동 bar / 실제 결정변경 bar |
|---|---|---|---|---|
| 2025q1(context) | NG 82.96/-20.62/27 · WG 28.54/-20.62/19 | NG 97.70/-20.62/28 · WG 44.98/-20.62/20 | NG 97.70/-20.62/28 · WG 44.98/-20.62/20 (재라벨과 byte-identical) | 0 / 0 |
| 2025q2(context) | NG 92.47/-16.41/24 · WG 39.99/-10.82/15 | NG 106.45/-13.23/31 · WG 31.49/-15.85/19 | NG 106.45/-13.23/31 · WG 31.49/-15.85/19 (재라벨과 byte-identical) | 1,340 / **0** |
| **2025q3(context)** | NG **-35.54**/-49.79/25 · WG **-9.73**/-44.37/19 | NG **-46.26**/-56.94/38 · WG **-18.87**/-43.49/30 | NG **-37.43**/-51.25/27 · WG **-15.86**/-44.37/21 | 6,053 / **28** |
| val(비악화 전용) | NG 36.82/-24.34/29 · WG 54.88/-31.11/22 | NG 46.59/-21.70/35 · WG 77.31/-21.76/26 | NG 46.59/-21.70/35 · WG 77.31/-21.76/26 (재라벨과 byte-identical) | 0 / 0 |
| oos_q1(비악화 전용) | NG 49.32/-16.20/24 · WG 44.48/-15.48/20 | NG 93.27/-15.48/24 · WG 67.25/-15.48/19 | NG 93.27/-15.48/24 · WG 67.25/-15.48/19 (재라벨과 byte-identical) | 1,029 / **0** |
| oos_q2(비악화 전용) | NG 3.13/-15.00/12 · WG 9.85/-15.00/10 | NG -9.55/-20.76/13 · WG -12.69/-20.76/10 | NG -9.55/-20.76/13 · WG -12.69/-20.76/10 (재라벨과 byte-identical) | 0 / 0 |

"가드 발동 bar"는 탐지기가 True였던 h48qual 보유-bar 수, "실제 결정변경 bar"는 그 중 원본
모델의 exit-or-hold 이진 결정이 재라벨 모델의 결정과 실제로 달랐던 수(진단 전용 카운터, 어느
쪽 실행 경로에도 영향 없음 — `greedy_replay_regime_aware_exit_guard`가 가드 활성 시에도 항상
default 경로 확률을 추가로 한 번 더 계산해 비교만 한다). Q1/val/oos_q2는 가드가 아예 발동하지
않았고, Q2·oos_q1은 수백~천 회 발동했지만 단 한 번도 결정을 바꾸지 않았다(0.95라는 높은
문턱을 두 모델이 그 특정 bar들에서 한 번도 서로 다르게 넘지 않음) — 그 결과 5개 창 전부 재라벨판과
**완전히 동일한 원장**이 나왔다. Q3만 6,053회 발동 중 28회 실제로 결정이 바뀌어 거래수가
38건→27건으로, 원본(25건)에 가깝게 줄었다.

## 비악화 확인 (val/oos_q1/oos_q2, `summarize_multiwindow` 재사용)

| 기준 | val | oos_q1 | oos_q2 | 최종판정 |
|---|---|---|---|---|
| 원기준(mdd_slack_pp=0) | PASS | PASS | PASS | **CONFIRMED** |
| 완화기준(mdd_slack_pp=3) | PASS | PASS | PASS | **CONFIRMED** |

세 창 모두 "비악화" 수준이 아니라 **재라벨판과 원장이 완전히 동일**(위 표 참고) — 조건부 정책이
이 세 창에서 재라벨판의 이득을 조금도 깎아먹지 않는다는 것을 가장 강한 형태로 확인한 결과다.

## 최종 판정 (3단 서술 — 이분법 아님)

**(a) Q3 손상이 얼마나 완화됐는가**: no_gate 기준 재라벨판 -46.26% → 조건부정책 -37.43%로,
원본 -35.54%까지의 격차(10.72pp) 중 **82.4%를 회복**했다. with_gate 기준은 -18.87%→-15.86%로
원본 -9.73%까지 격차(9.14pp)의 **32.9%를 회복**했고, MDD는 -44.37%로 **원본과 정확히 일치**했다.
손상이 완전히 사라지지는 않았다(no_gate·with_gate 둘 다 여전히 원본보다 소폭 나쁨) — 부분적,
그러나 뚜렷한 완화다.

**(b) Q1/Q2의 기존 이득이 보존됐는가**: 완전히 보존됐다. Q1은 가드가 단 한 번도 발동하지 않았고
(구조적으로 재라벨판과 동일), Q2는 1,340회 발동했지만 실제 결정을 한 번도 바꾸지 않아
(경험적으로 재라벨판과 동일) 두 창 모두 PnL·MDD·거래수가 재라벨판과 소수점까지 일치한다.

**(c) VAL/OOS-Q1/OOS-Q2에서 비악화인가**: 그렇다 — 원기준·완화기준 둘 다 세 창 전부 통과
(`CONFIRMED`). 게다가 세 창 모두 재라벨판과 원장이 완전히 동일해, "비악화"를 요구조건보다
훨씬 강하게 만족한다.

**세 조건이 다 성립하지만, 이것이 "승격 가능"을 뜻하지 않는다.** 이 결과는 전부 2025년(TRAIN
기간, in-sample) 또는 재라벨과 무관하게 이미 여러 실험이 재확인한 하락/혼조 OOS 구간(OOS-Q1/
OOS-Q2 2026)에서 나온 것이다. `eth_omega461_exit_head_liveatr_sustained_uptrend_vulnerability_
20260814.md` §4가 이미 명시했듯, 확보된 OOS 구간(~2026-07-12까지) 중 "지속 상승" 레짐은 아직
한 번도 없었다 — 이 조건부 정책이 **실제 forward 지속상승 레짐**에서도 같은 완화 효과를 낼지는
전혀 검증되지 않았다. 이 실험이 정직하게 말할 수 있는 최대치는: **"섀도우 관찰 대상으로 추가할
가치가 있는 후보"** — Q1/Q2/VAL/OOS-Q1/OOS-Q2에서 부작용이 전혀 없고(대부분 원장이 완전
불변) Q3(in-sample)에서 손상을 상당 부분 되돌린다는 근거가 있으니, 다음에 실제 지속상승장이
오면(라이브 또는 새 OOS 구간) 이 정책의 forward 성과를 관찰할 가치가 있다는 뜻이다. 승격
판단기준(Odyssey1 미해결 이슈 13)은 여전히 미정이며, 이 결과가 그 기준을 대신하지 않는다.

## 생성 파일

- `scripts/research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814.py` (신규)
- `tmp/causal_regen_20260516/eth_omega461_regime_aware_exit_head_uptrend_guard_20260814/report.json`
  — G0a/G0b, 탐지기 캘리브레이션·활성화율, entry-side 진단, 6창 비교, 비악화 판정 전부 포함.
- 같은 디렉터리의 `portfolio_ledger_{2025q1,2025q2,2025q3,val,oos_q1,oos_q2}_{baseline_both_
  original,asymmetric_tabm_liveatr,regime_aware_guard}.csv`, `_aligned_{train,validation,oos}_
  {h48qual,zig075}_predictions.csv` — 거래 원장(diagnostic, 참고용) + 정렬된 예측 CSV.

## 준수 확인

`fresh_forward_bar_by_bar=true`(신규 렌임드카피는 단일 causal forward pass, `i` 증가 순서로만
진행, 탐지기 rolling window도 순수 backward-looking), `trade_ledgers_used_as_input=false`(렛저는
전부 output 전용), `saved_parent_exit_timestamps_used=false`, `future_rows_used_for_entry=false`
— report.json에 그대로 기록됨. 재학습 없음, GPU 불필요(conda env `quant_ai`, CPU). 라이브 파일
(`trading_bot.py`/`trading_bot_modules/omega4_6_1_live.py`/`trading_bot_modules/runtime_config.py`/
`.env`) 무수정 — `git diff`로 직접 확인(0줄). 기존 공유 모듈
(`eth_omega461_multiwindow_confirmation_gate_20260814.py`,
`research_eth_omega461_exit_head_portfolio_asymmetric_20260813.py`,
`replay_omega4_6_1_greedy_router_20260706.py`,
`research_eth_omega461_exit_sweep_20260721.py`,
`research_eth_omega461_live_sltp_mfe_width_20260813.py`,
`train_eval_omega4_2_risk_sidecar_20260622.py`) 전부 import·읽기 전용, 수정 없음.
