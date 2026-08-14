# ETH Omega4.6.1 라이브 모델 — 심층 해부 리스크 평가 (2026-08-12)

## 배경

Odyssey 서브 프로젝트(2026-08-11~12)에서 h48qual `quality_head` 게이트의 구조적 숏 편향을
5단계까지 root-cause 분석했고, TabM 대체 모델 6종을 전부 시도했으나 확정된 대체 엣지를 찾지
못했다. 사용자 질문: "지금 라이브에서 쓰고 있는 이더리움 오메가4.6.1 전체 모델을 심층 해부
분석해봐. 괜찮다고 하면 이 문제는 안고 계속 진행해야겠어." — 이 문서는 그 결정을 위한 종합
평가다. Odyssey 계약 문서(`odyssey_eth_h48qual_corrected_tabm_20260811_contract.md`)가
"신규 백본 개발"에 초점을 둔 반면, 이 문서는 **현재 실제로 라이브에 연결된 구성**(구버전
`ThreeHeadTabM` 기반) 전체를 다룬다 — 별도 문서로 분리한 이유.

## 1. 실제 아키텍처 (검증됨)

`FINAL_GOVERNOR_OMEGA4_6_1_ENABLE=true`일 때 ETH에서 실행되는 전체 구성 요소
(`trading_bot_modules/omega4_6_1_live.py`):

- **두 개의 parent 모델**: `h48qual`(우선순위 1, `quality_threshold=0.50`)과
  `zig075`(우선순위 2, `quality_threshold=0.75`). 각각 독립적으로 bull/bear/chop 3개 전문가
  서브네트워크를 내장(공유 라우팅 레이어가 아니라 컴포넌트별로 따로 있음).
- **Regime3-Current HMM**: causal `filter_proba`로 bull/bear/chop 확률 산출 → 6개 입력
  피쳐로도 쓰이고, 어느 전문가 서브넷이 답할지 argmax로 라우팅도 함.
- **ATR 적응형 TP/SL**: 리서치 스크립트들이 쓰는 `BASE_TEMPLATE`(take_profit=0.026,
  stop_loss=0.014)의 값 자체는 라이브에서 안 씀 — ATR 기반 배리어로 대체.
- **컴포넌트별 리스크사이징 사이드카**(HistGradientBoostingRegressor) → sigmoid →
  `SCALE_MAP` → `leverage≤5.0`/`notional≤1.8` 캡. **레포의 Futures Risk Sizing Contract가
  요구하는 margin_fraction 예측 방식과 부합** — 리서치 스크립트들의 고정 sizing(notional=0.45,
  leverage=2.0)과는 다르다는 점 주의: 이 세션의 모든 always-short/always-long 트레이드시뮬은
  고정 sizing template을 쓰므로, **방향성(선택) 스킬은 정확히 검증하지만 실제 라이브 사이징
  로직까지 재현하진 않는다.**
- **exit_head는 실제로 라이브에 연결돼 있고 동작 중**(정적 threshold=0.95). Odyssey 계약서의
  "exit_head 파이프라인 설계만 존재, 미구현"은 **차세대 `ThreeHeadTabMCorrected` 백본용
  이야기지, 현재 라이브 번들의 것이 아니다** — 혼동하지 말 것. 라이브 번들 report.json에
  `exit_label: entry_label_terminal_giveback, rows=30000` 확인.
- **cmamba/risk overlay 컬럼**: 미사용 확인(그레핑 결과 0건) — 이전 세션 메모(dead code)가
  Omega4.6.1에도 그대로 적용됨.
- **Duration gate**(`ou_halflife<=0.005417`)와 **portfolio notional cap**(옵트인, 기본
  꺼짐)이 추가 veto 레이어.

**h48qual/zig075 결합 방식**: `PRIORITY=("h48qual","zig075")` — greedy 우선순위 라우터.
매 bar마다 h48qual이 CASH가 아닌 side를 내면 그걸 쓰고, h48qual이 CASH면 zig075로 넘어감.
동시에 한 포지션만 유지(2026-07-06에 수정된 버그: 이전엔 두 컴포넌트가 독립적으로 100%
자본을 시뮬레이션하는 오류가 있었음).

## 2. 알려진 결함 — Odyssey 서브 프로젝트의 핵심 발견 요약

**h48qual `quality_head` 게이트: 구조적 숏 편향, 실제 선택 스킬 없음 확인.**

- `direction_head`의 원본 픽은 균형(학습구간 short 49.94%, p=0.66 동전던지기).
- 게이트 통과 후 short 비중이 75~91%로 급등. VAL/OOS(2025-10~2026-02, ETH -32% 하락구간)
  둘 다 always-short가 게이트 통과 모델을 이김.
- **short-only 격리 테스트(가장 결정적)**: 모델이 실제로 숏을 고른 시점만 always-short와
  비교(승률 기준) — h48orig(실제 라이브 레시피) **0/5 시드가 always-short를 이김**, OOS
  승률 격차 -10pp. 즉 게이트가 이미 발동한 상태에서도 "어느 숏이 더 좋은 숏인지" 판별하는
  능력이 없다 — 오히려 더 나쁜 숏을 고른다.
- **root cause 5단계**: 게이트 ≈ direction_head 자체 confidence 필터(confidence-only
  top-K가 게이트 결과를 4.3~6.5pp 내로 재현) → 그 confidence 비대칭(숏이 3~5pp 높음)은
  **학습 자체에 내재**(레짐 무관, 학습구간에서 오히려 더 큼) → 부분적으로 실제 시장 기하
  (숏 스윙이 완성 전 되돌림이 적음, MAE/magnitude 유의하게 낮음, 7분기·4반기 전부 안정)로
  설명 가능하나 완전한 인과관계는 미확정 → temperature scaling/focal loss 재보정 시도 전부
  실패(일반화 안 됨, focal loss는 calibration을 오히려 악화시킴).
- **zig075도 같은 방향의 편향**(pre→post gate short 57~54%→70~66%, h48qual의 절반 수준)이지만
  **short-only 격리 테스트는 zig075에 대해 아직 실행된 적 없음** — h48qual만큼 확정적으로
  "무스킬"이라 말할 근거가 아직 zig075엔 부족하다. **미해결 갭.**

**Update 2026-08-12: zig075 short-only 격리 테스트 실행 — h48qual과 정반대 방향, 다만 N=1
인스턴스라 미확정.** `scripts/diagnose_eth_zig075_short_only_vs_always_short_20260812.py`(h48qual
격리 스크립트를 zig075 라이브 번들 데이터 로딩에 맞춰 최소 수정, 재학습 없이 기존 예측
재사용). `model_short_only`(모델이 실제로 숏을 고른 bar만) vs `always_short`(같은 active
set 전체 강제숏) 비교:

| | VAL | OOS |
|---|---:|---:|
| model_short_only pnl | **+16.48** | **+17.27** |
| always_short pnl | +15.57 | +15.63 |
| model_short_only 승률 | **46.9%**(32건) | **66.7%**(12건) |
| always_short 승률 | 45.7% | 61.5% |

**VAL·OOS 둘 다 pnl·승률 기준 short_only가 always_short을 이긴다** — h48qual이 0/5 시드
전부 완패했던 것과 정반대 방향. 다만 (a) 재학습 없이 실제 라이브 가중치 **단일 인스턴스(N=1)**만
본 것이라 h48qual의 5~15시드 통계 검정과 같은 신뢰 수준이 아니고, (b) 거래수도 12~32건으로
작다. **레포의 Seed-Diversity Ensemble Promotion Gate 기준(N≥5 진짜 무작위 시드) 미달 —
"zig075는 진짜 숏 선별 스킬이 있다"고 확정할 근거가 아니라 "h48qual과 달리 조사해볼 가치가
있는 긍정적 단서"로만 취급.** 확정하려면 zig075의 학습 레시피로 N≥5 독립 시드 재학습이
필요(GPU 필요, 서버 디스패치 대상). 산출물:
`tmp/eth_zig075_short_only_vs_always_short_20260812/short_only_vs_always_short.csv`.

**Update 2026-08-12(같은 날, 이어서): N=5 독립 시드 재학습 완료 — N=1의 "승리"는 재현 안 됨,
always_short과 통계적으로 구분 불가.** zig075 실제 라이브 레시피(direction/quality 모두
zigzag_action_labels_20260531, quality_mode=same_as_direction, quality_threshold=0.75)를
`scripts/train_eval_omega4_3head_parent72_eth_zig075_liverecipe_20260812.py`로 그대로
재현, 완전히 새로운 진짜 무작위 N=5시드(913588538/702006280/238746861/689517735/605384781)를
서버 GPU에 디스패치해 재학습(`scripts/diagnose_eth_zig075_liverecipe_5seed_short_only_vs_
always_short_20260812.py`로 분석):

| | VAL(5시드) | OOS(5시드) |
|---|---:|---:|
| short_only가 always_short 이긴 시드(pnl) | **2/5** | **3/5** |
| short_only가 always_short 이긴 시드(승률) | 3/5 | 4/5 |
| wilcoxon p (pnl) | **0.8125** | **0.6250** |
| model_short_only pnl | +11.88±6.78 | +23.88±2.61 |
| always_short pnl | +12.19±5.35 | +22.67±1.90 |

**N=1(라이브 가중치)에서 봤던 "확실히 이긴다"는 결과가 재현되지 않는다** — 두 구간 모두
완전히 무의미한 수준(p>0.6)으로, 시드에 따라 이기고 지고가 거의 동전던지기다. 결론:
**zig075는 h48qual만큼 확실히 "무스킬"은 아니지만(h48qual은 0/5 완패), "진짜 숏 선별
스킬이 있다"고도 할 수 없다 — always_short과 통계적으로 구분 불가능.** 롱 선택은 h48qual과
동일 패턴으로 나쁨(VAL -7.73±5.51 wr 28.5%, OOS -7.35±5.11 wr 28.3%) — 전체 모델 pnl이
always_short보다 낮은 건 이 나쁜 롱 트레이드가 끌어내리기 때문. **미해결 이슈 1 완전 해소.**
산출물: `tmp/eth_zig075_liverecipe_5seed_short_only_vs_always_short_20260812/short_only_vs_always_short.csv`.

**인프라 부산물**: 이 재학습 준비 중 `data/ensemble/supervised/`의 정식 overlay 학습 파일
3종(regime3_current/cmamba/risk)이 dev·서버 양쪽 모두에서 사라져 있는 걸 발견(cmamba는
[[omega_cmamba_risk_overlay_dead_code]]로 이미 알려진 문제였지만 risk/regime3_current도
동일하게 사라진 건 신규 발견). 처음엔 `scripts/train_eval_omega4_3head_parent72_eth_zig075_
liverecipe_20260812.py`에 로컬 스코프 우회로만 처리했으나(공유 모듈 안 건드림), 사용자
지시로 **정식 경로 자체를 dev·서버 양쪽에서 복구 완료**(아래 미해결 이슈 5 참고) — 이제
공유 모듈 오버라이드 없이도 이 로더에 의존하는 모든 스크립트가 정상 동작한다.

**대체 모델 탐색(6종 전부 시도, 전부 부정 수렴)**: TabM HP서치(0/40+), GBDT(0/48),
오토인코더(0/18), TCN(OOS 0/75, 최대 표본 결정적 부정 — post-OOS 최초 양성은 재검증에서
역전·철회), CNN 캔들차트(애초 무의미), one-vs-rest 독립 3모델(POST_OOS 신호는 재현됐지만
메커니즘 진단 결과 always-short 기준선의 레짐별 취약성 때문이지 진짜 스킬이 아님).
**현재 시점 direction_head를 대체할 확정된 개선안 없음.**

## 3. 프로모션/무결성 감사 상태

**`audit_omega_artifact_integrity_20260630.py`가 두 번(2026-07-06) `promotion_pass=true`로
통과**(`tmp/causal_regen_20260516/omega4_6_1_gate2_fix_20260706/`,
`tmp/causal_regen_20260516/omega4_6_1_duration_ou_halflife_risk_gate_20260630/`) — h48qual/
zig075 둘 다 threshold-exact artifact 무결성 통과.

**주의**: 이 감사는 **아티팩트 정합성**(정확한 threshold의 prediction 파일 존재, report.json과
sidecar 태그 일치 등)을 검증하는 것이지, **방향 예측에 진짜 스킬이 있는지는 검증하지 않는다.**
Odyssey의 always-short 대조/short-only 격리 테스트가 답하는 질문과 완전히 다른 축이다.
"프로모션 통과 = 스킬 있음"으로 오독하면 안 됨.

## 4. 현재 실제 운영 상태 (중요 — 실자본 리스크 없음)

- **실계좌 실행 꺼짐**: `data/live/dashboard_state_governor.json`의 `account.enabled=false,
  testnet=true` — 모든 거래가 `exchange_execution_dry_run=true`. **현재 실자본이 걸려있지
  않다.**
- **거버너 상태**: `agents.omega4_6_1.enabled=true`(ETH), 감사 통과 모델
  `omega4_6_1_duration_ou_halflife_risk_gate_20260630`가 연결돼 있음. 최근 결정(2026-08-11
  12:30 UTC)은 `hold`; 프로세스가 현재 실행 중인지는 정적 파일만으로 완전히 확인 안 됨.
- **페이퍼 거래 이력**: ETH 전체 통틀어 **2건**뿐. ① 2026-07-07 LONG 진입→07-15 청산,
  +9.53% 가격변동, 누적 자기자본 +15.14%(레버리지 반영). ② 2026-07-16 SHORT 진입, 마지막
  스냅샷(08-11) 기준 **26.3일째 보유 중**(`hold_bars=7569`), 미실현 +2.10%.

  **정정(2026-08-12)**: 처음엔 이 보유기간이 "아키텍처 blueprint 문서가 명시한 백테스트 최대
  보유(약 11.75일/282h)를 2배 이상 초과"하는 운영상 이상 신호라고 플래그했으나, **이건 blueprint
  원문을 잘못 읽은 것이었다.** 직접 조사한 결과(`docs/model_contracts/
  omega4_6_1_full_architecture_blueprint_20260706.md:183`) 원문은 "...이게 몇 시간~2주 정도의
  보유를 허용하는 이유다(2026년 1~6월 OOS에서 관측된 최대치가 282시간) — **고정된 최대보유는
  없다**; 포지션은 배리어나 학습된 exit 판단으로만 닫힌다"로, **같은 문장 안에서 이미 282h가
  상한이 아니라 관측치일 뿐임을 명시**하고 있었다. 코드 확인 결과도 일치: `evaluate_exit`
  (`trading_bot.py:9189-9203`)은 SL→TP→학습된 exit_head(threshold=0.95, `omega4_6_1_live.py:68`)
  순서로만 청산하며 보유기간을 상한으로 쓰는 로직 자체가 없음. `hold_bars`/`mfe`/`mae`는
  재시작 시에도 유지되는 실시간 카운터로 정상 공급되고 있어 코드 경로 자체의 결함도 안 보임.
  **결론: 설계대로 동작 중, 버그 아님.** 유일하게 정적 분석만으로 못 닫는 질문은 "이 특정
  거래에 대해 exit_head가 실제로 계속 낮은 확률을 내는 게 맞는 판단인지"뿐이며, 이건 실제
  라이브 `exit_prob` 로그가 있어야 확인 가능(아직 미확인, 코드 결함을 시사하는 근거는 없음).

## 5. 종합 판단

**"안고 가도 괜찮은가"에 대한 답은 두 갈래다:**

**지금 당장(현재 실자본 없음, 페이퍼 전용)**: 안고 가도 실질적 손실 리스크는 없다. 급하게
끄거나 고칠 필요는 없다.

**실자본 연결을 고려하는 시점부터는 다르다**: h48qual의 무스킬 결론은 short-only 격리
테스트까지 거친 가장 강한 증거 수준이고, 그 원인(direction_head confidence 비대칭)이
레짐과 무관하게 학습 자체에 박혀있다는 것도 확인됐다 — 즉 "지금까지 우연히 하락장이라
숏 편향이 맞아떨어졌을 뿐"이라는 가설이 유력하며, 이는 **레짐이 바뀌면(상승장/횡보장)
같은 편향이 반대로 작용해 손실 요인이 될 수 있다**는 뜻이다(TCN/one-vs-rest 실험에서
반복 확인된 "매끈한 추세 vs 거친 추세에 따라 승패가 갈린다"는 패턴과 정확히 같은 구조적
위험). 게다가 6종의 대체 모델 시도가 전부 실패해서 **당장 교체할 대안이 없다.**

**권고**:
1. 지금 상태(페이퍼 전용) 유지는 문제없음 — 급한 조치 불필요.
2. ~~실자본 전환 전에는 최소한 (a) zig075도 h48qual과 같은 short-only 격리 테스트를 돌려
   편향의 심각도를 확정하고, (b) 26일 보유 중인 오픈 포지션의 exit 로직이 왜 백테스트 상한을
   초과했는지 확인해야 함~~ **둘 다 완료(2026-08-12)** — (a) N=5 시드로 확정(무스킬은 아니되
   유스킬도 아님), (b) 버그 아님으로 확인(아래 미해결 이슈 2 참고, "백테스트 상한 초과"라는
   전제 자체가 blueprint 오독이었음).
3. 구조적 편향 자체("이 문제")는 인지한 채로 페이퍼 운영을 계속하는 것과, 그 편향을 안은 채
   **실자본**을 넣는 것은 리스크 크기가 다른 결정이다 — 이 구분을 유지할 것.

## 미해결 이슈

1. ~~zig075 short-only 격리 테스트~~ **완전 해소(2026-08-12)** — N=5 독립 시드로 확정:
   always_short과 통계적으로 구분 불가(p=0.81 VAL, p=0.63 OOS). N=1 라이브 가중치의 "승리"는
   재현 안 됨.
2. ~~26일 보유 SHORT 포지션의 exit_head/duration-gate 라이브 동작 이상~~ **완전 해소, 이상
   아님으로 정정(2026-08-12)** — "백테스트 최대보유 282h 초과"라는 최초 플래그 자체가 blueprint
   원문("고정된 최대보유는 없다, 282h는 관측치일 뿐")을 잘못 읽은 것이었음. `max_hold_bars=0`은
   코드베이스 전체가 "시간 기반 청산 없음"으로 쓰는 관례이고, omega4_6_1의 `evaluate_exit`
   (`trading_bot.py:9189-9203`)은 애초에 보유기간을 상한으로 쓰지 않음(SL→TP→학습된
   exit_head(threshold=0.95)만 사용). 코드 경로 자체에 결함 없음, 설계대로 동작 중. 유일한
   잔여 불확실성(이 거래에 대한 실제 exit_prob이 낮게 나오는 게 타당한 판단인지)은 라이브
   로그 없이는 정적 분석만으로 못 닫지만, 코드 결함을 시사하는 근거는 전혀 없음.

   **추가 발견(2026-08-12, 사용자가 이전에 찾았던 별개 이슈 재조사로 확인)**: 26일 보유
   포지션이 왜 그렇게 오래 걸리는지에 대한 보완 설명이 생겼다 — 아래 신규 이슈 6("ATR
   적응형 TP/SL이 사실상 고정 floor") 참고. 그 포지션의 실제 TP/SL(0.075/0.04)이 정확히
   floor 값과 일치 — "최대보유 없음"이라는 설계 자체는 문제 없지만, TP/SL 자체가 이름과
   달리 거의 항상 고정값이라 5분봉에서 도달하는 데 구조적으로 오래 걸린다.
6. **신규(2026-08-12): "ATR 적응형" TP/SL이 사실상 고정 floor(TP 7.5%/SL 4.0%)로 동작** —
   `min_tp=0.075`/`min_sl=0.040`이 전체 시간의 95~98.5%에서 바인딩되는 유일한 값이고,
   `max_tp=0.22`/`max_sl=0.12`는 2025~2026 전체 데이터에서 단 한 번도 발동한 적 없음(죽은
   파라미터). `tp_mult=12.0`/`sl_mult=6.0`(ATR 스케일링)이 ETH 5분봉 실제 ATR% 규모에
   비해 너무 낮게 캘리브레이션돼 있어 거의 항상 floor 밑에 깔림. 버그인지 의도된 설계인지는
   미확정(사용자 판단 필요). 상세: `docs/experiments/eth_omega4_6_1_atr_tpsl_floor_binding_
   investigation_20260812.md`.
5. ~~`data/ensemble/supervised/`의 정식 overlay 학습 파일 3종(regime3_current/cmamba/risk)이
   dev·서버 양쪽 모두에서 사라져 있음~~ **완전 해소(2026-08-12)** — dev·서버 둘 다 정식 경로
   자체를 복구함. regime3_current는 tmp/의 진짜 사본을 정식 경로로 복사, cmamba/risk는
   0-fill placeholder를 정식 경로에 영구 배치(둘 다 라이브에서 이미 확인된 무사용 컬럼이라
   0-fill이 실제 라이브 동작과 정확히 일치 — 이 CSV들은 `.gitignore`(`data/ensemble/**/*.csv`)
   대상이라 git에는 안 잡힘, 로컬 디스크에만 존재하는 게 원래 의도된 설계와 일치).
   `omega._load_omega_frames()`를 dev·서버 양쪽에서 **아무 오버라이드 없이** 직접 호출해
   정상 동작 확인함(`train rows=105064, eval rows=16897`). 이제 이 로더에 의존하는 ~50개
   스크립트 전체가 다시 정상 동작한다. 상세: [[omega_cmamba_risk_overlay_dead_code]].
3. 라이브 사이징(HGB 리스크사이드카)이 이 세션의 고정-template 트레이드시뮬과 다르다는 점 —
   방향 스킬 결론에는 영향 없지만, 정확한 실제 PnL 재현이 필요하면 별도 시뮬레이터 필요.
4. `trading_bot.py` 프로세스가 현재 실행 중인지 정적 파일만으로 완전 확인 못 함.
