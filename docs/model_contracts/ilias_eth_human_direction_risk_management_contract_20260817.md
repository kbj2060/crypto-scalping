# 일리아스(Ilias) — ETH 사람 방향입력 + 능동적 리스크관리 계약 문서 (2026-08-17)

slug: `ilias_eth_human_direction_risk_management`

## 상태

| 항목 | 상태 |
|---|---|
| 서브프로젝트 부트스트랩(계약+리소스 레지스트리) | `완료` |
| 1차 연구 질문 설계 | `완료` — `docs/experiments/ilias_eth_adaptive_exit_direction_quality_signal_design_20260817.md` |
| exit_head 수동성 근본원인 진단 | `완료` — `docs/experiments/ilias_eth_exit_head_passivity_root_cause_20260817.md` |
| 1차 연구 질문 학습·백테스트(라벨 재정의+베이스라인 분류기+6윈도우 평가) | `완료 — 성공조건1 재확인, 성공조건2는 2차 정정으로 2/5(40%)로 하향` — `docs/experiments/ilias_eth_adaptive_exit_signal_baseline_test_20260817.md` §1~§6(최초 실행, **오염 발견됨**), §7(side-blind 정정), §8(2차 정정). **경위**: 최초 학습 모델의 계수가 `pos_side`/`pos_leverage`/`pos_exposure`/`pos_notional`에서 quasi-separation 수준(\|coef\|=21~27, TRAIN 구간이 SHORT우위 하락추세라 방향 암기 위험)이라 그 4개를 제외한 side-blind 재학습(§7)을 수행 → **성공조건1(방향품질 반응성)은 side-blind에서도 6/6 윈도우 전부 통과 유지**(\|t\|=9.05~42.4, 대부분 원본과 비슷하거나 큼) — **진짜 신호로 재확인**. 성공조건2(fresh-forward MDD/PnL)는 처음 "3/6"으로 보고됐으나(§7), 통과 3곳 중 레인지 2025-03-10~05-05는 always_long 거래 6건 중 신호가 **단 한 번도 발동하지 않은(0%) 트리비얼 통과**임을 사후 확인(§8) — 실제 신호가 개입한 5개 창만 놓으면 **2/5(40%)만 성공**(OOS-Q1, 레인지 2026-02-09~04-06), 하락추세/레인지 구분으로도 안 갈리고 N=5로는 설명변수 특정 불가. 단일 config(로지스틱회귀, threshold=0.5) 결과 — 증거 강도는 실험문서 §7.7/§8.4 참고. |
| 레짐게이팅 하이브리드(Odyssey3 탐지기, ON=h48qual 원본 exit_head 0.95 / OFF=side-blind 신규신호 0.5) 테스트 | `완료 — 관성 아님으로 기각되나, 목표했던 3개 실패창 회복은 1/3만 성공` — `docs/experiments/ilias_eth_adaptive_exit_signal_baseline_test_20260817.md` §9. 다른 세션이 같은 탐지기+같은 ON브랜치를 **기존**(liveATR재라벨) exit_head 대상으로 검증했을 때는 6개 창 전부 완전 관성(real_g0 PnL 마스크무관 소수점동일, `docs/experiments/eth_zig075_veto_ranging_misfire_fix_candidate_20260817.md` "추가" 절) — 이번엔 OFF브랜치를 side-blind 신규신호로 교체해 재검증. G0 identity check(신규 게이팅 함수가 `new_exit_model` 미부착 시 기존 배포 가드와 바이트 동일 재현)로 구현 정확성 확인 후 6개 창 재평가: **관성 아님**(6개 창 중 5개에서 실제 발동, 트리비얼은 레인지② 1개뿐, `reason_counts`로 직접 확인) — 성공조건2 헤드라인 3/6→4/6(트리비얼 제외 진짜개입 기준 2/5→3/5, 60%)로 개선. 그러나 §8이 지적한 3개 실패창(VAL/OOS-Q2/레인지①) 중 **VAL·OOS-Q2는 여전히 실패**(AL arm이 side-blind 단독과 소수점까지 동일 — 두 창 모두 6개 중 탐지기 활성률이 가장 낮은 축, 7.55%/8.19%)하고, **레인지①만 회복**(AS arm 가드레일 위반 190%악화→34%악화로 해소, 단 이 회복은 AL arm이 아니라 AS arm 보호 효과 — 원본 exit_head 가드의 원래 존재이유[Q3 조기회전 방지]와 정성적으로 같은 메커니즘). 이미 통과했던 OOS-Q1/레인지③은 하이브리드로 추가 개선(mdd −47.27%→−43.51%, −20.82%→−15.17%). 배경 탐지기 활성률(창별): VAL 7.55%/OOS-Q1 5.44%(최저)/OOS-Q2 8.19%/레인지① 15.98%(최고)/레인지② 11.53%/레인지③ 8.66% — 단, 활성률이 회복/개선 정도를 단조적으로 예측하지 않음(OOS-Q1은 최저 활성률인데도 최대폭 추가개선). 신규 자유변수 0개(탐지기/양쪽 exit모델/양쪽 threshold 전부 기존값 재사용). 단일 config 결과, N=6 창(진짜개입 5개)로는 회복 조건의 설명변수 특정 불가 — 증거 강도는 실험문서 §9.6 참고. |
| ~~Baseline v1 — 2026-08-17 재정의: 오디세이4 G0과 동일 시스템~~ | **`폐기됨(2026-08-20, 사용자 지시)`** — "사람 입력 방향" 정체성 자체가 아래 정체성 재정의로 폐기되면서 그 정체성 위에서 정의된 이 베이스라인도 함께 무효화됨. **신규 베이스라인 미정** — 다음 세션에서 정체성 재정의(아래) 기준으로 다시 정한다. 아래 원본 내용(사용자 결정 2026-08-17: exit_head 재정의 신호/레짐게이팅은 오디세이4 G0을 판정 3창 중 1곳(OOS-Q1)에서만 확실히 앞섰고 나머지는 못 앞섰다는 이력)은 **삭제하지 않고 역사적 기록으로 보존** — 상세는 아래 Label Contract 절. |
| **일리아스 1 — 2026-08-18 명명, 2026-08-21 오디세이5로 이관** | `이관됨`. 원래 내용(오디세이4 position-feature 버그수정 재학습, Baseline v1 대비 REJECTED_SIGN_MISMATCH, N=5/N=6 시드검증 CONFIRMED)은 무변경 그대로 `docs/model_contracts/odyssey5_eth_position_feature_parity_fix_contract_20260821.md`로 이동 — 오디세이4 자체 계승판이라는 taxonomy 판단. 모델 자산도 물리이동(구경로 심볼릭링크 보존). **"일리아스 1"이라는 이름 자체는 아래 새 정의로 재사용**됨 — 다음 행 참고. |
| **일리아스 1(신정의) = 일리아스 라벨로직 후보축 — 2026-08-21**: zigzag/h48qual/cusum 154피쳐 스크리닝 | `등록됨, N=3 예비 스크리닝 — 승격/확정 근거 아님`. 별도 서브프로젝트(`eth_tabm_label_logic_retest_initiative`)에서 진행되던 154피쳐 엔지니어링셋 + 라벨로직(zigzag/h48qual/dc/cusum/분포적회귀) 비교 작업을 사용자 지시로 이관 — 상세는 아래 "일리아스 라벨로직 후보축" 절. Shared Feature Contract의 "신규 피쳐 도입 안 함" 원칙에 대한 **명시적 예외**로 등록(아래 Shared Feature Contract 절 참고). zigzag(→zig075슬롯)/h48qual(seed133725056) 번들이 새 "일리아스 1" 모델 자산으로 지정됨(Label Contract 절 "일리아스 1 — 2026-08-21 재편" 참고) — cusum은 대응 옛슬롯이 없어 이 지정에서 제외, 별도 후보로 유지. Baseline v1(오디세이4 G0)을 대체하지 않음 — 완전히 별도의 탐색 축. |
| **레짐 분류기 — wide24 HMM 확정, 2026-08-21 → 재확정(앵커드 walk-forward) → 시드검증 CONFIRMED(N=5)** | `재확정 — states=24/sticky=0.90(sticky=0.85와 사실상 동급), 신ADX라벨(balancedish_adx16_slope03_bb006, 불변) 채택. N=5 진짜무작위 시드 전부 OOS 순위 일치(std≈0.0001) — Seed-Diversity Gate CONFIRMED`. 같은 날 네 차례 작업: (1) 최초 확정 TRAIN=2024단독 states=30/sticky=0.85. (2) 앵커드 walk-forward 적용(TRAIN 2026-06-30까지 확장) 재스윕 → **states=24가 VAL·OOS 최상위 클러스터로 재확인**. (3)(4) top-3(states24/0.90·states24/0.85·states30/0.85)에 진짜무작위 시드 N=5(원시드+4개, 2단계로 N=4→N=5 확장) 검증 → **5개 전부 동일 순위** — Open Issue (k) 해소. JM/SJM 대비 wide24 구조우월성 결론은 불변. 상세는 아래 "레짐 분류기 계약" 절. **Open Issue (k)(l)(m)(n) 전부 해소/완료 — (m)은 N=1 재확인(zigzag/h48qual/cusum 전부 VAL→OOS 부호 유지, 아래 절 참고), N≥5 승격게이트는 여전히 미충족(별개 이슈, 반복 없음).** |
| **라벨 퓨전(3라벨 결합모델) 연구 — 2026-08-21 연구 → 2026-08-22 실제 테스트+최신문헌 재구현 후 종결** | `종결 — 예측레벨 결합 탐색 전체 폐기, 근본원인 정보이론적 확증`. 문헌(Bates & Granger 1969, Krogh & Vedelsby 1994)+feasibility체크(vote/consensus)+정식 스태킹 메타모델(로지스틱회귀, 2024학습→2025평가, long_frac 0.979로 붕괴)에 이어, 08-22 최신문헌 2편(Zou 2025, Felici & Sudoso 2023)을 이 저장소 구조에 맞게 실제 재구현·테스트 — zigzag/cusum/h48qual 각자의 방향확률·품질점수가 순방향48bar수익률에 대해 갖는 선형회수정보가 절편전용모델 이론적하한과 사실상 같음(BCE 0.692대 vs 이론하한 0.6928)을 확인, §6의 극단적 long_frac도 "진짜학습"이 아니라 노이즈계수의 threshold 우연 초과로 재해석. 상세는 아래 "라벨 퓨전(3라벨 결합모델) 연구" 절과 `docs/experiments/ilias_eth_label_fusion_combined_model_research_20260821.md` §6~§8. |
| **레짐 직결 노출(RDE) 정책 — 시드-안정 방향원 아키텍처 — 2026-08-22 → OOS 조기실행으로 최종 REJECTED** | `REJECTED(OOS 확정 기각, 사용자 지시로 09-30 대기 override) — 방향전략 자체는 폐기, 이 실험 안에서 검증된 집행(비용) 방법론은 독립 자산으로 아래 별도 절에 보존`. wide24 HMM filtered 확률을 유일 방향원으로 쓰는 Schmitt 트리거 정책, VAL에서 시드간 PnL 편차 0.06%p·net +5.86%까지 확인했으나(펀딩 델타뉴트럴 캐리 대안도 별도로 실측 기각, 2026 펀딩 붕괴). 같은 정책을 OOS(2026-07-01~08-19, 레짐분류기/DC154와 동일 override 3번째 사례)에서 N=5 시드 **단일터치**로 평가 — **5/5 전부 음수, −4.90%±0.19%p**, always-long(+43.08%) 대비 **48%p 열위**. 시드안정성이 손익방향을 보장하지 않음을 실증. 이 정책×이 OOS 창은 소진, 재조회·재시도 금지. **같은 날 데이터결함(BTC오염→−4.38%, 이어서 metrics미래참조→−7.37%, 0/5 유지) 2연속 재검증에도 판정 불변, 오히려 악화** — "데이터 버그 때문" 가설 확정적 기각. 상세: `docs/experiments/eth_ilias_regime_direct_exposure_seed_stable_direction_20260822.md`. |
| **3심볼 데이터 무결성 대수술 — 2026-08-23** | `완료 — metrics 계열 기준 ETH/BTC/SOL 전부 검증·수정됨`. ETH 캐노니컬 2026의 BTC-metrics 병합(2026-01-20~07-12, 07-13 미커밋 재빌드 원인) 발견·제거 + BTC/SOL의 아카이브 vintage 어긋남(**BTC 2024는 24%가 1-bar 미래참조**) 13개 파일 수정 + 오버레이 3종 재생성 + 154셋 패치·2/28 갭 95행 삽입. 사후 재감사 전 파일 exact 99.7~100%. 깨끗한 데이터 위치·보증범위·백업은 아래 "데이터 무결성 현황" 절 참고. 상세: `docs/experiments/eth_binance_metrics_archive_backfill_and_canonical_divergence_20260823.md`. |
| **peg-maker 집행 인프라 — 비용축(RDE와 독립) — 2026-08-22** | `자산으로 확정 — 방향전략의 성패와 무관하게 재사용 가능`. maker 실효비용 3.1~4.0bp/leg 실측(taker 실측 5.0bp/가정 7bp 대비 절감), **OOS(미터치 데이터)에서도 예측 밴드(3.5~4.0bp) 안에 재확인**(RDE의 방향손실과는 별개로 비용모델 자체는 살아남음), 라이브 섀도우 가동 중(9월 중순 체크포인트). 상세는 독립 절 "peg-maker 집행 인프라 계약" 참고(아래, 라벨 퓨전 절 앞). |

## Scope

- **⚠️ 2026-08-20 정체성 재정의(사용자) — "사람 방향 입력" 목표 폐기, 방향도 모델이 결정**:
  이 프로젝트의 정체성을 "사람이 방향을 입력하고 모델은 리스크관리만 담당"에서 **"방향 결정까지
  전부 모델이 직접 담당"**으로 변경한다. 방향은 오디세이4처럼 TabM direction head가 결정하되,
  그 라벨로직(zigzag/h48qual/cusum, 154피쳐)의 승자를 가리는 게 이제 이 프로젝트의 핵심
  연구질문이다 — 아래 "일리아스 라벨로직 후보축" 절이 곧 이 새 정체성의 실제 작업 축이 된다.
  이 재정의로 아래 내용 중 **"사람 입력"을 전제한 문구는 전부 무효화**됐다(위 상태표
  Baseline v1 폐기 포함): 바로 아래 "목적"·"사용자가 확정한 스코프 #1(입력)"·"오디세이4로부터
  상속 vs 대체"의 "대체" 항목, Layer Contracts L2 행의 "direction head 출력 미사용" 서술,
  Output Contract의 "action/side=사람 입력" 서술, Open Issue (c). 이 문구들은 삭제하지 않고
  **2026-08-17 시점 정체성의 역사적 기록**으로 남긴다 — 앞으로 이 문서를 읽을 때 "방향은
  사람이 낸다"는 문장은 전부 폐기된 과거 계획으로 읽을 것. **L3/L4 진입 사전거부(entry veto)의
  스코프 밖 처리는 "사람 권한 유지"가 근거였는데 그 근거 자체가 사라졌으므로 재검토가 필요하나,
  사용자가 이 부분을 명시적으로 지시하지 않아 여기서 임의로 바꾸지 않는다 — Open Issue로
  신규 등록(아래 (j))**. 새 베이스라인은 미정.
- **2026-08-17 되돌림 결정(사용자, 역사적 기록 — 아래 "사람 입력" 관련 문구는 위 재정의로
  무효화됨)**: 일리아스의 실제 시스템은 **오디세이4와 완전히 동일**하다
  (exit_head를 포함해 전부 무변경) — L2~L4 방향결정을 사람 입력으로 대체하는 것, L9 exit_head를
  능동형으로 재설계하는 것은 전부 **이 프로젝트가 추구하는 목표(goal)**이지, 아직 구현된 다른
  아키텍처가 아니다. 오늘 시도한 exit_head 라벨 재정의/레짐게이팅은 그 목표를 향한 탐색적 연구로
  남지만, 판정 3창(VAL/OOS-Q1/OOS-Q2) 중 1곳(OOS-Q1)에서만 오디세이4 G0을 확실히 앞서 아직 "다른
  시스템"이라 주장할 근거가 안 된다 — Baseline v1 절 참고. 아래 "목적"·"오디세이4로부터 상속 vs
  대체" 절은 **목표 서술**로 읽을 것 — 대체는 아직 아무것도 실행/배포되지 않았다.
- **목적**: 오디세이4 파이프라인에서 방향 결정(L2~L4)을 사람이 대체하고, 모델은 **능동적(active)
  리스크관리 및 노출 축소**에 전념하는 새 아키텍처를 설계한다. "능동적"이라는 말의 의미는
  `docs/experiments/eth_odyssey4_random_direction_risk_management_ablation_20260817.md`의 핵심
  발견(아래 인용)과 직접 대비된다.
- **왜 이 서브프로젝트가 시작됐는가**: 사용자가 2026-08-17 제안한 "오디세이4 방향은 못 맞추지만
  리스크관리는 할 줄 안다면, 사람이 방향을 입력하고 모델은 리스크관리만 맡기면 어떤가"를
  코딩 착수 전에 검증한 어블레이션(`docs/experiments/eth_odyssey4_random_direction_risk_management_ablation_20260817.md`)에서:
  1. 오디세이4의 방향 픽은 게이트 통과 후에도 순수 무작위와 거의 구분 안 됨(N=5 시점 관측),
     그러나 N=30 확장 재검정에서는 **레짐에 따라 부호가 바뀌는 실재하는 편향**으로 격상됐다
     (하락추세에서 약한 우위, 저스프레드 레인지에서 유의하게 열위, |t|>2 4/6윈도우).
  2. 리스크관리 스택(quality 게이트+진입베토+duration 게이트+ATR TP/SL+HGB 사이드카+exit_head)은
     하방을 줄이지만 없애지 못한다(always_long MDD −36~−51%).
  3. **결정적 발견**: exit_head 발동률(21.8~27.7%)은 방향 품질(맞았는지 틀렸는지)과 거의
     무관하게 일정하다 — 즉 지금의 "리스크관리"는 방향 품질을 판별해서 나쁜 거래를 피하는
     **능동형(adaptive)** 장치가 아니라, 방향 무관하게 고정 TP/SL 폭을 기계적으로 적용해
     per-trade 손실 상한만 거는 **수동형(passive)** 안전장치다. 방향 품질을 실제로 반영하는
     유일한 신호는 SL 히트율(always_long 58.2% vs always_short 32.7%)뿐이다.
  이 서브프로젝트의 1차 연구 질문(아래 실험문서)은 발견 3번을 직접 이어, exit_head를
  수동형에서 능동형(방향 품질에 실제로 다르게 반응하는 조기경보)으로 바꿀 수 있는지를 검증한다.
- **사용자가 확정한 스코프 (2026-08-17 세션, 재질문 없이 그대로 적용)**:
  1. **입력**: 사람은 **방향만**(LONG/SHORT/청산의도) 결정한다. 사이징, 레버리지, TP/SL은
     전부 모델이 결정한다.
  2. **액션 범위**: 모델은 **노출 축소/청산만** 할 수 있다. 방향 전환·반대포지션 헤지·진입
     자체를 막는 사전거부(entry veto)는 이번 서브프로젝트 범위 밖이다 — 방향은 항상 사람
     권한으로 유지된다.
  3. **주기**: 매 5분봉(bar)마다 재평가 — 오디세이4 L9(보유 중 매 bar 체크) 루프와 동일한 캐던스.
  4. **시작 모델**: 오디세이4(`docs/model_contracts/odyssey4_eth_entry_veto_baseline_contract_20260814.md`,
     `docs/model_contracts/odyssey4_eth_full_stack_architecture_20260814.md`)를 시작점으로 삼는다.
     h48qual/zig075 중 어느 TabM 베이스를 쓸지는 미정(Open Issues 참고).
- **오디세이4로부터 상속 vs 대체** (상세는 Layer Contracts 표):
  - 상속: L0 피처엔진, L6 TP/SL 산출식, L7 HGB 리스크사이드카(사이징), L4.5 duration 리스크게이트,
    L8 포지션 오픈, L10 청산/렛저 기록.
  - 대체: L2~L4의 **방향 결정**(action=argmax(direction))은 사람 입력으로 대체된다. L3/L4의
    **진입 사전거부**(quality_threshold 게이트, zig075 SHORT/LONG 지속추세 진입거부)는 사용자
    스코프 결정 #2에 따라 이 프로젝트 범위 밖 — 사람이 낸 방향은 항상 진입 시도된다(모델이
    사이징으로 사후에 노출을 0에 가깝게 줄이는 것은 허용되지만, 이는 "진입 자체를 막는
    사전거부"와 개념적으로 다르다. Open Issues 참고).
  - 개선 대상(1차 연구 질문): L9의 exit_head — 현재의 방향 품질 무관 수동형 신호를 능동형으로
    재설계할 수 있는지 검증한다.
- **Owner agent**: Model Architect 단독(Sonnet) — `feedback_architect_team_single_agent_sonnet.md`
  정책에 따름.
- **상위 서사 인용**: `docs/model_contracts/odyssey_eth_h48qual_corrected_tabm_20260811_contract.md`
  (Odyssey1, direction_head 무스킬 확정), `docs/model_contracts/odyssey4_eth_entry_veto_baseline_contract_20260814.md`
  (Odyssey4 G0 베이스라인 — 이 프로젝트가 리스크관리 스택을 상속하는 원천).
- **1차 연구 질문 실험문서**: `docs/experiments/ilias_eth_adaptive_exit_direction_quality_signal_design_20260817.md`
- **리소스 레지스트리**: `docs/model_contracts/ilias_eth_human_direction_risk_management_data_resources_20260817.md`
- **라이브 파일 무변경 원칙**: `trading_bot.py`, `trading_bot_modules/*`, `runtime_config.py`, `.env`는
  이 서브프로젝트 전체에서 절대 건드리지 않는다(오디세이 계열과 동일 원칙) — 지금은 순수
  연구/설계 단계다.

## 데이터 무결성 현황 — 2026-08-23 대수술 이후 (새 실험은 이 절 기준으로 데이터 선택)

**보증 범위 주의**: 아래 "검증됨"은 **OI/롱숏비(metrics) 계열 컬럼+파생**에 대한 것이다 —
OHLCV/펀딩/CVD 등 나머지 컬럼은 독립 참조본 대조를 한 적이 없다("문제 발견된 적 없음" ≠
"무결 증명됨"). 전체 경위: `docs/experiments/eth_binance_metrics_archive_backfill_and_canonical_divergence_20260823.md`.

| 분류 | 위치 | 상태 |
|---|---|---|
| **참조 진실값**(아카이브, +5분 종료라벨 보정본) | `data/TOTAL_{ETH,BTC,SOL}USDT_metrics_2024_2026.csv` | 검증됨 — metrics 계열 신규 조인은 반드시 이것 기준. 재생성: `scripts/download_eth_binance_metrics_archive_20260823.py`(METRICS_SYMBOL env) |
| ETH 캐노니컬 2026 | `data/splits/year_oos/training_features_2026_rebuilt.csv` | **완전 수정(2건 순차 발견·수정)**: ①BTC-metrics 병합(01-20~07-12) 제거, ②같은 날 후속 발견 — **07-12 00:05~파일끝(08-19)은 metrics 1버킷 미래참조 조인**(+07-20~08-02 raw OI 스케일결함 ~1/300) 별도 수정(`fix_eth_canonical_2026_oi_futureleak_20260823.py`). 현재 사후 참조본 exact match 99.97~100% |
| ETH 캐노니컬 2024/2025 | `data/splits/year_oos/training_features_{2024,2025}.csv` | 99.8~100%. 잔여 0~0.2%는 **의도적 보존** — 구 수식 vintage 빌드라 현재 수식 재계산 시 kel 등이 2~5% 바뀌어 수정이 결함보다 큰 왜곡(게이트 실측 판정). 차세대 전체 재빌드에서만 균질화 |
| BTC 파일군(11개) | `btc_features_{연도,결합}`, `btc_raw_frame*`, `btc_*_metrics4*`, swingtransition/zigzag/regimeline/1h_full | **완전 수정**(2024의 24% 1-bar 미래참조 제거 포함), 사후 exact 99.7~100% |
| SOL 파일군(5개) | `sol_features_{연도,결합}`, `sol_raw_frame` | **완전 수정**, 사후 exact 99.97~100% |
| wide24 오버레이 3종 | `tmp/ilias_labellogic_recheck_20260821/{train_2024_2026H1,eval_2026H1,oos_20260701_20260819}_regime3_current_states24_sticky090.csv` | 수정된 캐노니컬로 2회 재생성 완료(2차 수정 후 oos 오버레이만 최대확률변화 0.63 재반영, train/eval은 입력 범위 밖이라 무변화) — **레짐 소스는 이걸 쓸 것**(구 balancedish 사이드카 아님) |
| 154피쳐셋 | `tmp/ilias_eth_154feature_dataset_20260821/` (2026=51,841행, combined=262,322행) | 오염 25컬럼 패치 + 2/28 갭 95행 삽입 + 금융ML 12컬럼 연속 재계산. `manifest.json`의 `patched_20260823` 참고 |
| 백업(당시 입력 재현용) | `.bak_pre_btc_metrics_fix_20260823` / `.bak_pre_gap_fix_20260823` / `.bak_pre_metrics_vintage_fix_20260823` | 08-23 이전 실험 재현 시 사용 |
| ⚠️ 비권장/플래그 | `data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/`(구모델 사이드카, 미재생성), `btc_features_1h_metrics4*.csv`/`btc_features_5m1h_*_metrics4at1h_*.parquet`(종결축 산출물, vintage 불확실) | 새 실험의 소스로 쓰지 말 것 |
| 영구 결손(수정 불가) | tail_risk_1m 2026-07-18 이전, orderbook_decision_snapshots 갭 4건(최대 07-14~08-01) | 수집 부재 — 해명·문서화로 종결 |

**함의**: (a) 2026-01-20~07-12 창을 입력으로 쓴 08-23 이전 실험은 재실행 시 수치가 미세하게
달라질 수 있다(판정 소급무효 아님 — 재현은 백업본으로). (b) train/live 패리티는 회복됨.
(c) BTC 2024로 학습한 과거 실험은 미래참조가 제거된 만큼 재실행 성과가 낮아지는 게 정상.

## Dataset Split

**⚠️ 2026-08-22 유일 컨벤션으로 확정 — 아래 "레거시 고정표"는 더 이상 유효한 규약이 아니다.**
이 프로젝트를 포함해 리포지토리 전체에 이제 split 규약은 이것 **하나만** 적용한다(사용자 지시,
여러 서브프로젝트가 서로 다른 날짜범위로 학습해 생기는 문제를 근본적으로 없애기 위함). 방법론
근거·전체 결정 경위는 `docs/eth_recency_weighted_walkforward_data_split_literature_review_20260820.md`
+[[eth_recency_walkforward_data_split_literature_review_20260820]](2026-08-20 문헌조사+실측
교차분석을 거친 "최종 결정") 참고 — CLAUDE.md의 Fresh-Forward 기본값(VAL 2025-09-01~12-31,
OOS 2026-01-01~03-31)은 리포지토리 최상위 문서라 **그대로 유지**하기로 결정(2026-08-22,
사용자 확인)했고, 이 규약 통일은 그 아래 서브프로젝트/모델계약 레벨에서만 적용한다.

**방법명: 분기 앵커드(확장) Walk-Forward + 경계 Purge/Embargo + TRAIN내 Time-Decay 가중**

| 티어 | 규칙 | 2026-08-22 시점 구체값 | 용도 |
|---|---|---|---|
| TRAIN | 2024-01-01 ~ 최근 완결분기 말일 | 2024-01-01 ~ **2026-06-30** | 학습(fit). 구 VAL+OOS-Q1+OOS-Q2 전부 편입 — 이미 10회+ 조회돼 "신선한 미터치 창"으로 더는 취급 안 함 |
| VAL(판정 티어) | TRAIN 내 최근 분기 재사용 | **2026-04-01 ~ 2026-06-30**(2026 Q2) | 조기종료+체크포인트+보고 지표. 튜닝용 반복사용 허용 |
| OOS(단일터치) | TRAIN 바로 다음 분기 전체, **분기 완결 전 조기 부분체크 금지** | **2026-07-01 ~ 2026-09-30**(⚠️ 09-30까지 대기 — 오늘 08-22 시점 아직 미도달, 접촉 금지) | 승격/성능주장 근거. 조기 부분체크 자체가 "조회 1회"로 집계되는 걸 막기 위해 09-30까지 통째로 대기 |

매 분기 재앵커(TRAIN이 한 분기씩 확장, VAL/OOS도 한 분기씩 밀림) — staleness가 최대 9개월에서
최대 1분기로 캡된다. 경계 purge/embargo(라벨 forward-window가 분할선을 못 넘게)와 TRAIN 내
time-decay 가중(오래된 레짐 희석 완화)도 이 방법론의 일부다. Fresh-Forward Rule과 충돌 없음 —
causal bar-by-bar 평가 자체는 불변, 어느 캘린더구간이 train/OOS로 라벨링되는지만 바뀐다.

⚠️ **예외 기록(OOS "09-30까지 대기" override, 개별 승인 필요)**:
- 레짐 분류기(wide24 HMM, [[eth_regime_classifier_wide24_vs_jm_sjm_investigation_20260821]]) —
  N=5 시드검증까지 CONFIRMED 완료된 상태에서 OOS를 2026-07-01~08-19(즉시가용)로 확정. 이미
  완료·확정된 결과라 재실행하지 않는다.
- DC154피쳐 tabular 트랜스포머 스모크테스트(2026-08-22, [[eth_candidate_lob_ofi_pipeline_smoke_test_20260822]]
  계열 문서) — 사용자가 동일한 종류의 override를 재지시. 상세는 아래 "데이터 Split 재설계
  제안" 절 "두 번째 적용 사례" 참고.
- **RDE(레짐 직결 노출) 정책 — 2026-08-22, 3번째 사례** — 사용자가 "현재까지 데이터로 OOS
  진행해줘, 09-30까지 기다리지 말고"라고 직접 지시. 동일 창(2026-07-01~08-19)에서 VAL 확정
  단일정책×N=5시드 단일터치 평가 → **REJECTED**(5/5 음수, always-long 대비 48%p 열위) —
  위 "레짐 직결 노출(RDE) 정책" 상태표 행 참고. 이 창은 이 정책에 대해 소진, 재조회 금지.

세 사례 모두 **개별 사용자 지시로 승인된 예외**다 — "09-30까지 대기"가 기본 규칙이라는 점은
안 바뀐다. **⚠️ 3회 누적 — 아래 "데이터 Split 재설계 제안" 절이 이미 예견한 대로("반복적으로
override되고 있다는 점 자체를 다음 세션이 인지할 것"), 이 override가 일회성이 아니라 사실상
상시 패턴이 되고 있다.** 앞으로 새 작업에서 이 override가 필요하면 매번 명시적으로 확인받을
것(자동 상속 아님) — 다만 이 빈도 자체를 다음 세션이 사용자와 논의할 가치가 있다.

⚠️ **소급 적용 안 함**: 이 문서 이전에 이미 CLOSED된 결과(154피쳐 TabM chance-level,
라벨퓨전 BCE-절편하한 등 — TRAIN=2025/EVAL=2026H1 등 구식 관례로 나온 결과)는 재실행하지
않는다. 두 결과 모두 이론적 하한과 사실상 동일한 "정보량 0" 판정이라 split 창을 바꿔도 결론이
뒤집힐 가능성은 낮다고 판단 — 다만 그 결과들의 split이 지금 이 규약과 다르다는 점은 인용 시
명시할 것.

<details>
<summary>레거시 고정표(2026-08-17 작성, 2026-08-22부로 비활성 — 역사적 기록으로만 보존)</summary>

오디세이4가 쓰는 정확한 날짜 범위(`scripts/eth_omega461_multiwindow_confirmation_gate_20260814.py`
`WINDOW_DEFS`, `scripts/research_eth_omega461_exit_sweep_20260721.py` `VAL_START/END`,
`OOS_START/END`에서 그대로 확인, 추측 없음)를 그대로 상속한 표였다.

| Split | Source | Timestamp range | Rows(예측 CSV 실측) | Use |
|---|---|---|---:|---|
| Train(모델 학습, 상속 — 재학습 없음) | h48qual/zig075 TabM 번들 | 2024-01-01 ~ 2025-09-30 | 183,936(Odyssey1 계약 실측) | 이 프로젝트는 재학습하지 않음(체크포인트 그대로 재사용) |
| 2025 Q1/Q2/Q3(context, 참고 티어) | `omega4_6_1_extended_oos_20260706/{h48qual,zig075}/train_predictions_q0XX.csv` | 2025-01-01 ~ 2025-09-30 | 78,510 | 오늘 어블레이션의 레인지장 재검정이 이미 이 구간 예측을 재사용함 — 1차 연구 질문의 사전점검 후보 |
| VAL(판정 티어) | `.../validation_predictions_q0XX.csv` | 2025-10-01 ~ 2025-12-31 | 26,490 | VAL 게이트(strict) |
| OOS-Q1(판정 티어) | `.../oos_predictions_q0XX.csv`(oos_q1+oos_q2 통합 파일, 날짜 필터로 분리) | 2026-01-01 ~ 2026-03-31 | 55,405(oos_q1+oos_q2 합산) | 단일터치 OOS 확인 |
| OOS-Q2(판정 티어) | 위와 동일 파일 | 2026-04-01 ~ 2026-06-30 | 위와 동일 파일에 포함 | 단일터치 OOS 확인 |

h48qual/zig075 TabM 번들 자체(상속 체크포인트)의 원래 학습 이력은 위 표대로가 맞다 — 이 표가
틀렸던 게 아니라, **새로 만드는 모델에도 그대로 적용되는 살아있는 규약처럼 취급된 것**이 문제였다.

</details>

Audit: 이 프로젝트는 신규 데이터 로딩 코드를 아직 작성하지 않았다(설계 단계) — 타임스탬프
중복/워밍업/OOF 감사는 `scripts/eth_omega461_multiwindow_confirmation_gate_20260814.py`가
이미 수행한 것을 그대로 상속하며, 별도 감사는 실제 구현 시작 시점(다음 세션)에 수행한다.

## Shared Feature Contract

- Canonical feature source: `features/engineering.py`(`FeatureEngineer` 클래스) — 오디세이4와
  동일. 라이브 TabM 번들(`true_3head_tabm_bundle.pt`)의 `bundle["base_cols"]`가 실제 102
  base 피처 계약을 고정한다.
- 이 프로젝트는 **신규 피처를 도입하지 않는다** — L2 quality/exit head의 입력 피처(102 base +
  13 pos = 115차원)는 오디세이4 그대로 상속한다. 1차 연구 질문에서 검토하는 "post-entry 상태
  벡터"(미실현 PnL, MFE-so-far, 사이드카 리스크 재추정치 등)는 파생 신호이지 신규 원시 피처가
  아니다 — 상세는 실험문서.
- **⚠️ 명시적 예외(2026-08-21) — "일리아스 라벨로직 후보축"**: 아래 해당 절의 zigzag/h48qual/
  cusum 라벨로직 비교 작업은 위 원칙과 별개로 **154개 엔지니어링 피쳐셋**(102 base와 다른,
  158 캐노니컬에서 리던던시제거+VIF+조합+금융ML문헌표준으로 재구성한 별도 피쳐공간)을 쓴다.
  이 154개는 Baseline v1/일리아스 1의 102 base 피쳐 계약과 호환되지 않으며, 두 피쳐공간을
  섞어 쓰지 않는다 — 라벨로직 후보축 전용, 다른 절(Baseline v1/일리아스 1)에는 영향 없음.
- Normalization/Missing fallback/Stale handling/Live availability: 오디세이4 계약과 동일(변경 없음).

## Layer Contracts

오디세이4 L0~L10(`docs/model_contracts/odyssey4_eth_full_stack_architecture_20260814.md`) 기준,
이 프로젝트가 무엇을 상속/대체/개선 대상으로 삼는지.

| Layer | 오디세이4 원본 | 이 프로젝트 |
|---|---|---|
| L0 피처엔진 | 102 base + WIDE24 route 확률 + dual_momentum | **상속(무변경)** |
| L1 레짐 라우팅 | regime3 HMM bull/bear/chop argmax, 컴포넌트별 독립 | **상속(무변경), 단 분류기 자체는 2026-08-21 wide24 HMM으로 확정·재튜닝됨** — 어느 TabM 베이스(h48qual/zig075)를 쓰든 그 컴포넌트의 라우팅 그대로 사용. 확정 근거·하이퍼파라미터·비교결과는 아래 "레짐 분류기 계약" 절 참고 |
| L2 3-Head TabM | direction(3)/quality(3)/exit(2) | **부분 대체** — direction head 출력은 사용하지 않음(사람 입력으로 대체). quality head는 사람이 선택한 방향에 대해 그대로 조회(`quality_p_{사람선택방향}`, 오늘 어블레이션의 `prepare_component_direction_override`와 동일 원칙). exit head는 **1차 연구 질문의 개선 대상** |
| L3 진입 게이트(quality_threshold) | action=argmax(direction), quality[action]≥threshold 미달 시 CASH | **범위 밖(스코프 결정 #2)** — 사람이 낸 방향은 항상 진입 시도된다. quality 게이트를 "진입 차단"으로 쓰지 않음. quality 값 자체는 L7 사이징의 context feature로는 계속 살아있을 수 있음(설계 시 결정) |
| L4 진입 사전거부(zig075 SHORT/LONG 지속추세 veto) | 규칙 기반 entry veto | **범위 밖(스코프 결정 #2)** — 방향 전환·헤지·진입거부는 사람 권한 대체 대상이 아님 |
| L4.5 Duration OU-halflife 리스크 게이트 | funding_roc_12 AR(1) half-life 기반, 방향 무관 | **상속(무변경)** — 방향 무관 외부 레짐 신호이므로 이 프로젝트의 "사람 방향 대체" 범위와 무관 |
| L5 우선순위 중재(단일 슬롯) | `PRIORITY=(h48qual, zig075)` | **단순화 예정** — 사람이 단일 방향 커맨드를 내리므로 두 컴포넌트 간 우선순위 중재 개념 자체가 h48qual/zig075 베이스 선택(Open Issues (a))에 종속됨. 두 컴포넌트를 동시에 쓸지 여부도 미정 |
| L6 TP/SL 산출 | ATR 기반, floor 0.075/0.040 | **상속(무변경)** — Futures Risk Sizing Contract(아래) 준수 |
| L7 사이징 사이드카(HGB) | `margin_fraction`/`leverage` 산출, `selection_objective=log_risk` | **상속(무변경)**, 단 사람 방향의 "신뢰도"를 신규 context feature로 넣을지는 오픈(1차 연구 질문 범위 밖, 후속 과제) |
| L8 포지션 오픈 | — | **상속(무변경)** |
| L9 보유 중 체크(TP/SL·레짐 exit가드·exit_head) | TP/SL 항상 우선 → (h48qual만) 레짐 exit 가드 → exit_head≥0.95 | TP/SL·레짐 exit가드는 **상속(무변경)** — 2026-08-17 별도 세션이 이 "무변경" 결정을 zig075 진입베토의 레인지장 오작동([[eth_zig075_veto_ranging_misfire_fix_candidate_20260817]])과 같은 방법론(NONE/V1/V3 마스크 교체, N=20)으로 직접 검증했다: 가설 기각, **exit 가드는 real_g0 실거래 경로에서 6개 창 전부 마스크와 무관하게 PnL 소수점까지 완전동일**(zig075와 정반대 — zig075는 같은 교체로 레인지1이 완전히 뒤집혔음) — 즉 exit 가드는 이 축에서 위험하지도 않지만 사실상 관성적(causally inert)이다(`docs/experiments/eth_zig075_veto_ranging_misfire_fix_candidate_20260817.md` "추가" 섹션). "상속(무변경)" 결정은 이 검증 범위 안에서는 안전하다는 근거를 얻었다. **exit_head 자체(가드가 아니라 exit_head의 라벨/결정로직)는 별도로 1차 연구 질문의 핵심 개선 대상**이며 이건 이 검증과 무관하게 진행됨(§ 위 상태표, `docs/experiments/ilias_eth_adaptive_exit_signal_baseline_test_20260817.md`) |
| L10 청산·렛저 | — | **상속(무변경)** |

## Label Contract

**2026-08-17 구현 완료** (`docs/experiments/ilias_eth_adaptive_exit_signal_baseline_test_20260817.md`
§1): h48qual 신규 exit 신호 라벨 = 반사실적(counterfactual) TP/SL 가격 배리어 재구성 — exit_head의
실제 발동 이력을 완전히 무시하고, 각 진입의 실제 TP/SL 가격 배리어만을 기준으로 가격 경로를 따라가
"SL을 먼저 건드리는가 TP를 먼저 건드리는가"(`label_sl` ∈ {0,1})를 판정한다. max_hold 강제청산은
리플레이 엔진 자체에 없음을 코드로 확인해 엣지케이스에서 제외했고, 윈도우 끝단까지 미해소된
포지션은 드롭한다(라벨 추측 없음). 구현: `scripts/research_ilias_eth_adaptive_exit_signal_common_20260817.py`
(`simulate_private_barrier_trades`). Feature 14개(POS_COLS 13개 + entry 시점 `quality_for_action`
1개 파생) 전부 기존 파이프라인 재사용, 신규 원시 피처 없음(Shared Feature Contract 준수).

### Baseline v1 — 2026-08-17 재정의: 오디세이4 G0

**Baseline v1 = 오디세이4 G0, 완전 무변경**(`docs/model_contracts/odyssey4_eth_entry_veto_baseline_contract_20260814.md`
G0 참조값 그대로 상속). 일리아스의 향후 모든 후보(exit_head 재설계 포함)는 이 숫자 대비 개선을
주장해야 한다.

| 창 | PnL | MDD | 거래수 |
|---|---:|---:|---:|
| VAL(판정) | +77.31% | −21.76% | 26 |
| OOS-Q1(판정) | +67.25% | −15.48% | 19 |
| OOS-Q2(판정) | −12.69% | −20.76% | 10 |

**되돌림 사유(2026-08-17, 사용자 결정)**: 오늘 아래 "탐색적 연구(미채택)"에서 시도한 exit_head
라벨 재정의 신호 및 레짐게이팅 하이브리드를, 사람 방향(always_long-type/always_short-type)이
고정됐다고 가정한 프록시로 오디세이4 G0의 실제 exit_head와 나란히 비교한 결과 — 판정 3창 중
**OOS-Q1 한 곳에서만 명확히 앞섰고**, VAL·OOS-Q2에서는 기존 exit_head 대비 뚜렷한 우위가 없었다
(같거나 악화). 이 증거로 "일리아스는 오디세이4와 다른 시스템"이라 주장하기엔 이르다고 판단해,
Baseline v1을 다시 오디세이4 G0 자체로 정의한다 — **일리아스와 오디세이4는 현재 완전히 동일한
시스템이며, 차이는 오직 이 프로젝트가 추구하는 목표(사람이 방향 입력, 모델은 리스크관리/노출축소
전담)뿐이다.**

### 일리아스 1 — 2026-08-21 재편: 오디세이5로 이관, 새 정의로 교체

**⚠️ 계보 변경(2026-08-21, 사용자 지시)**: 2026-08-18 "일리아스 1"로 명명됐던 h48qual/zig075
position-feature 버그수정 재학습판(REJECTED_SIGN_MISMATCH, N=5/N=6 시드검증까지 CONFIRMED)은
애초부터 일리아스 고유 연구목표(사람 방향입력+능동적 리스크관리)와 무관한 **오디세이4 자체의
버그수정 계승판**이라는 판단 하에 별도 문서로 이관됐다 — 전체 내용(버그수정 3건 정의, G0 대비
6창 비교표, N=5/N=6 시드재현성 검증, zig075 단독 always-benchmark 부수조사) 무변경 그대로:
`docs/model_contracts/odyssey5_eth_position_feature_parity_fix_contract_20260821.md`. 모델
자산(h48qual/zig075 TabM 번들 2개 + risk sidecar 2개)도 물리적으로 그 문서가 관리하는 새
경로(`tmp/causal_regen_20260516/odyssey5_eth_{h48qual,zig075}_{bundle,risk_sidecar}_20260821/`)로
이동했다 — 구 경로는 심볼릭링크로 보존되어 기존 19개 의존 스크립트는 무수정 동작한다.

**"일리아스 1"이라는 이름은 이 시점부터 아래 새 정의를 가리킨다**: "일리아스 라벨로직
후보축"(이 문서 하단 절 참고, 2026-08-21 `eth_tabm_label_logic_retest_initiative`에서 이관된
154피쳐+zigzag/h48qual/cusum 비교)에서 나온 TabM 단일모델(zigzag→zig075슬롯, h48qual, 둘 다
seed=133725056)이 새 "일리아스 1"로 지정됐다.

| 컴포넌트 | 위치 |
|---|---|
| 일리아스1 zig075슬롯 TabM 번들 | `tmp/causal_regen_20260516/ilias1_eth_zig075slot_154feat_unified_single_model_seed133725056_20260821/true_3head_tabm_bundle.pt` |
| 일리아스1 h48qual슬롯 TabM 번들 | `tmp/causal_regen_20260516/ilias1_eth_h48qualslot_154feat_unified_single_model_seed133725056_20260821/true_3head_tabm_bundle.pt` |

⚠️ **cusum은 이 페어 구조에 자연스러운 옛 슬롯이 없어 이 "일리아스 1" 지정에서 제외**했다(zigzag/
h48qual과 달리 오디세이4의 h48qual/zig075 컴포넌트에 대응되는 원본이 없음) — 별도 후보로
"일리아스 라벨로직 후보축" 절에 그대로 남아있다. **이 새 "일리아스 1" 지정은 문서/파일 레벨
재편일 뿐 재검증이 아니다** — 아래 "일리아스 라벨로직 후보축" 절의 N=3 예비 스크리닝 결과가
그대로 이 지정의 유일한 근거이며, N≥5 재확인·risk sidecar 신규학습·quality_threshold 재선택
등 옛 "일리아스 1"이 거쳤던 절차는 아직 전혀 거치지 않았다.

### 탐색적 연구(미채택) — exit_head 라벨 재정의 + 레짐게이팅

Baseline v1으로 채택되지 않았으나 향후 재시도의 출발점으로 보존한다.

- **라벨 재정의**: 반사실적(counterfactual) TP/SL 배리어 재구성(`label_sl`) + h48qual 전용
  로지스틱회귀(side-blind, 방향/사이징 노출 컬럼 4개 제거, 65트레이드 학습). 성공조건1(방향품질
  반응성)은 CONFIRMED(6/6 판정 윈도우, N=30, \|t\|=9.05~42.4)이었으나, 성공조건2(fresh-forward
  MDD/PnL)는 진짜 개입 기준 2/5(40%)에 그쳤다.
- **레짐게이팅 하이브리드**: 같은 Odyssey3 탐지기로 ON=원본 h48qual exit_head(0.95)/OFF=위 신호
  (0.5) — 신규 자유변수 0개. 성공조건2를 3/5(60%)까지 개선했으나 VAL·OOS-Q2는 회복 못함(오디세이4
  G0 대비 우위는 판정 3창 중 OOS-Q1 하나뿐).
- **알려진 한계**: 단일 config(하이퍼파라미터 스윕 없음), 학습 트레이드 65건, 판정 표본이 윈도우
  6개뿐(거래 단위 아님). 다음에 이 축을 재시도한다면 §8.4/§9가 지목한 "거래 단위 표본 확장"이
  선행 조건이다.
- 상세: `docs/experiments/ilias_eth_adaptive_exit_signal_baseline_test_20260817.md` §1~§9,
  `docs/experiments/ilias_eth_exit_head_passivity_root_cause_20260817.md`(근본원인 진단,
  결론 자체는 유효 — 왜 기존 exit_head가 방향 무관인지는 여전히 CONFIRMED).

## Cost/Risk Assumptions

- **Futures Risk Sizing Contract**(`.claude/CLAUDE.md` 인용, 이 계약 하의 모든 실험이 따라야
  함): `notional = margin_fraction * leverage`, `margin_fraction = notional / leverage`,
  `PnL = price_move * notional`. TP/SL 모델 출력은 leverage 적용 전 price-move 타겟으로 해석하고
  (`take_profit = tp_price_move * notional`, `stop_loss = sl_price_move * notional`), notional에
  이미 레버리지가 반영된 뒤 TP/SL price line에 레버리지를 다시 곱하지 않는다(이중계상 금지).
- **Fresh-Forward Validation/OOS/Test Rule**(`.claude/CLAUDE.md` 인용): validation/OOS는 5분봉
  bar 단위 causal walk-forward로 처음부터 끝까지 순차 진행하며, 그 시점까지 확정된 feature/state만
  사용한다. 저장된 trade ledger/candidate-event ledger/과거 원장의 entry-exit 결과를 promotion,
  모델 선택, live 후보 성과 근거로 쓰지 않는다(diagnostic/accounting audit 전용). 리포트는
  `fresh_forward_bar_by_bar=true`, `trade_ledgers_used_as_input=false`,
  `saved_parent_exit_timestamps_used=false`, `future_rows_used_for_entry=false`를 명시해야 한다.
  이 규칙을 어긴 평가는 수치와 무관하게 promotion/test 근거로 무효다 — 이 서브프로젝트의 모든
  향후 실험(1차 연구 질문 포함)에 그대로 적용된다.
- Fee/Slippage/Leverage cap/Notional cap 등 구체 수치는 오디세이4 그대로 상속
  (`LEVERAGE_CAP=5.0`, `NOTIONAL_CAP=1.8`, `SCALE_MAP`) — 이 프로젝트가 변경할 계획 없음.

## Output Contract

아직 확정하지 않았다(설계 단계, 결과 없음). 현재 예상되는 변화만 기록:

- `action`/`side`: 오디세이4에서는 모델 출력이었으나, 이 프로젝트에서는 **사람 입력**(LONG/SHORT/
  청산의도)이 그대로 전달된다 — 모델은 이 필드를 생성하지 않는다.
- `notional_exposure`/`leverage`/`position_fraction`: 오디세이4와 동일하게 모델(L7 사이드카) 출력.
- exit_head 능동화 연구 결과에 따라 신규 필드(예: 조기 노출축소 트리거 사유)가 추가될 수 있으나,
  1차 연구 질문 실행 전이라 확정하지 않는다.

## Red Team Gates

2026-08-17 구현 세션은 연구용 베이스라인 성공/킬 검증(위 상태 표, 실험문서 참고)만 수행했다 —
승격/live 후보 신청이 아니므로 template.md의 전체 표준 게이트는 아직 실행하지 않았다. 실제 승격
시도 시 표준 게이트(train/val/test
타임스탬프 중복 감사, bfill/미래피처 금지, fee/slippage 1x/2x/3x 랭킹, calibration, walk-forward,
라이브 train state parity, funding/liquidation 한계 문서화)를 그대로 적용한다.

## Open Issues

- **(a) h48qual vs zig075 TabM 베이스 — 1차 연구 질문 범위에서는 h48qual로 진행, 결과 긍정적**:
  2026-08-17 구현 세션(`docs/experiments/ilias_eth_adaptive_exit_signal_baseline_test_20260817.md`)이
  h48qual 전용으로 새 exit 신호를 학습·테스트했고 성공조건 1을 6/6 윈도우에서 통과시켰다 — h48qual을
  1차 연구 질문의 베이스로 쓴 선택 자체는 근본원인 진단의 예측대로 유효했음이 확인됐다. zig075는
  exit_head 구조적 무관여(0/86)가 그대로이므로 이번 세션에서도 다루지 않았다 — h48qual/zig075를
  함께 쓸지, 리스크관리 스택 전체를 h48qual 단독으로 재설계할지는 여전히 다음 세션 결정 사항으로
  남지만, "순수 h48qual 단독 축이 작동하는가"는 더 이상 미정이 아니다.
- **(b) exit_head를 방향 무관 배경신호에서 방향 품질 반응형으로 만드는 방법 — 근본원인 진단
  완료(2026-08-17), 탐색적 구현·실증 완료(2026-08-17, 조건부 성공이나 Baseline v1으로 채택 안 됨
  — 위 "Baseline v1 — 2026-08-17 재정의" 절 참고, Baseline v1은 다시 오디세이4 G0 자체)**:
  `docs/experiments/ilias_eth_exit_head_passivity_root_cause_20260817.md`.
  h48qual exit_head가 실제로 조회하는 두 라벨 세트(원본 `entry_label_terminal_giveback`: 양성의
  99.86%가 오라클 트렌드 세그먼트 종료 임박 신호, `tmp/causal_regen_20260516/
  omega4_3head_parent72_loose_entry_quality_20260620_zigzagfix_06_h48_quality_noctx_padded_e2_
  fulltrain_exit30k_20260630/report.json`의 `exit_label` 필드 직접 확인; liveATR 재라벨: 양성의
  75.7~79.8%가 국소 MFE-되돌림 노이즈, 08-15 진단 인용) **둘 다 방향 품질을 라벨 정의에
  반영하지 않는다** — 발동률이 방향 품질과 무관한 건 라벨 설계의 직접적 귀결(CONFIRMED, 강한
  근거). 반면 입력 feature(`pos_unrealized`/`pos_mfe`/`pos_hold_bars` 등 13개, `POS_COLS`)는
  학습·라이브·리플레이 3경로 모두에서 보유-bar마다 실제 값으로 갱신됨을 코드로 확인
  (REFUTED) — exit_head가 "잘 되고 있는지" 판별할 입력 자체는 이미 정상 공급 중이다. zig075의
  구조적 무관여(0/86)도 같은 원본 라벨을 공유하는 같은 뿌리임을 확인(별도 메커니즘 아님).
  함의: 1차 연구 질문의 새 exit 신호는 **라벨을 "거래의 최종 SL/TP 귀결"로 재정의**해야 하며,
  feature 배관은 재구축 불필요(기존 `pos_*` 재사용). 오늘 어블레이션 수치(21.8~27.7%)와
  08-15 진단 수치(82~96%)의 차이는 측정 모집단 차이(포트폴리오 전체+L4.5게이트 vs h48qual
  단독)로 정성적으로 설명되나 산술적으로는 완전히 분해하지 못함(정직한 한계로 기록).
  **2026-08-17 구현·테스트로 실증 확인됨**(`docs/experiments/ilias_eth_adaptive_exit_signal_baseline_test_20260817.md`):
  반사실적 TP/SL 배리어 재구성 라벨(exit_head 실제 발동 이력 완전 배제, 순환논리 회피) +
  기존 `pos_*`/entry quality 피처로 로지스틱회귀 베이스라인을 학습한 결과, 발동률/precision이
  방향 품질(always_long-type vs always_short-type)에 따라 **6/6 평가 윈도우 전부에서 통계적으로
  매우 유의하게(N=30, 대부분 \|t\|>15) 갈렸다** — "exit_head는 라벨 설계상 방향 무관하다"는
  근본원인 진단의 진단과 "라벨만 재정의하면 반응성이 생긴다"는 예측이 둘 다 실증적으로 확인됨.
  단, 이 반응성을 실제 fresh-forward replay의 MDD 완화/PnL 비훼손으로 전환하는 데는 레짐의존성이
  있음(6/6 중 3/6 윈도우만 통과 — OOS-Q1/OOS-Q2/레인지 2026-02-09~04-06, VAL과 순수 레인지
  2구간은 탈락) — Open Issue (d)와 직접 연결. **결론: exit_head를 방향-무관 배경신호에서
  방향-반응형으로 만드는 방법 자체는 더 이상 미정이 아니다**(반사실적 라벨 재정의로 가능함이
  확인됨) — 남은 질문은 "어느 레짐/조건에서 그 반응성이 실제 손익 개선으로 이어지는가"로 좁혀졌다.
  **2026-08-17 정정(같은 날 후속 세션, `docs/experiments/ilias_eth_adaptive_exit_signal_baseline_test_20260817.md`
  §7)**: 위 베이스라인 모델의 표준화 계수를 직접 확인한 결과 `pos_side`/`pos_leverage`/
  `pos_exposure`/`pos_notional` 4개가 절대값 21~27로 quasi-separation 수준이었다 — `pos_side`가
  always_long/always_short arm 정의 상수 자체이고 TRAIN 라벨 구간(2025-01~09)이 SHORT 우위
  하락추세였으므로, 위 "6/6 윈도우 통과"가 부분적으로 "이 TRAIN 구간엔 SHORT가 이겼다"를
  `pos_side`로 암기한 결과일 위험이 있었다(`pos_unrealized`/`pos_dist_to_sl` 등 실제 "잘 되고
  있는지" 피처의 계수는 사실상 0이었음). 그 4개 컬럼을 제외한 10개 피처로 같은 라벨 CSV를 재사용해
  side-blind 재학습(`scripts/research_ilias_eth_adaptive_exit_signal_train_sideblind_20260817.py`)·
  재검증(`scripts/research_ilias_eth_adaptive_exit_signal_arm_eval_sideblind_20260817.py`)을
  수행한 결과: 재학습된 모델의 계수는 전부 정상 범위로 복귀(최대 절대값 0.44)했고, **성공조건 1은
  6/6 윈도우에서 그대로 통과**했으며 t값이 전반적으로 줄지 않았다(4/6 윈도우에서 오히려 원본과
  비슷하거나 더 큼) — pos_side 오염이 실재했음에도 성공조건 1의 결론 자체는 무효화되지 않았고,
  남은 10개 피처(포지션의 실제 미실현손익/MFE/SL까지거리 궤적)만으로도 방향 품질 반응성이 재현됨을
  확인했다. 성공조건 2는 3/6 윈도우로 개수는 동일하나 구성이 바뀌었다(하락추세 OOS-Q2가
  통과→실패, 순수 레인지 2025-03-10~05-05가 실패→통과) — 원본이 시사한 "하락추세에서 더 잘
  통한다"는 레짐 해석은 side-blind 재검증에서 유지되지 않으므로, 아래 (d)의 레짐의존성 질문은
  재개방 상태로 취급한다.
  **2026-08-17 2차 정정(같은 날 3차 세션, §8, 메인스레드 직접 분석, 신규 학습/백테스트 없음)**:
  위 "3/6" 중 레인지 2025-03-10~05-05는 always_long 거래 6건 중 새 신호가 **단 한 번도 발동하지
  않은(0%) 트리비얼 통과**였음을 `reason_counts` 재분석으로 확인 — `mdd_improve` 판정식이 등호
  포함(`new_mdd >= base_mdd`)이라 무개입도 "개선"으로 카운트된 것. 실제로 신호가 개입한 5개 창만
  놓으면 **성공조건 2는 2/5(40%)로 하향**(OOS-Q1, 레인지 2026-02-09~04-06만 순수 통과) — 헤드라인
  "3/6(50%)"보다 나쁘고, 하락추세/레인지 구분으로도 안 갈리며 N=5로는 설명변수를 특정할 수 없다.
  성공조건 1의 결론(진짜 신호)은 이 정정과 무관하게 유지된다.
  **2026-08-17 레짐게이팅 하이브리드 후속 세션(§9, `docs/experiments/
  ilias_eth_adaptive_exit_signal_baseline_test_20260817.md` §9)**: side-blind 단독이 실패한 3개
  창(VAL/OOS-Q2/레인지①)이 Odyssey3의 기존 배포 레짐가드(ON=h48qual 원본 exit_head, OFF를 이번엔
  side-blind 신규신호로 교체)로 회복되는지 검증했다 — 다른 세션이 같은 탐지기를 **기존**
  exit_head 대상으로 검증했을 땐 6개 창 완전 관성이었으나(아래 Layer Contracts L9 행 인용), OFF
  브랜치를 side-blind 신호로 바꾸자 **관성은 아니었다**(6개 창 중 5개 실제 발동, 트리비얼은
  레인지② 1개뿐, G0 identity check로 신규 게이팅 함수가 배포 가드를 바이트 동일 재현함을 확인 후
  측정). 그러나 목표했던 3개 실패창 중 **레인지①만 회복**(AS arm 가드레일 위반 해소,
  190%악화→34%악화)했고 **VAL·OOS-Q2는 그대로 실패**(AL arm 결과가 side-blind 단독과 소수점까지
  동일 — 두 창 다 탐지기 활성률이 6개 중 최저 축, 7.55%/8.19%). 이미 통과했던 OOS-Q1/레인지③은
  추가 개선(mdd −47.27%→−43.51%, −20.82%→−15.17%). 성공조건2 헤드라인 3/6→4/6, 트리비얼 제외
  진짜개입 기준 2/5→3/5(60%)로 개선됐으나 "게이팅이 side-blind 신호의 레짐의존적 약점을 대체로
  고쳐준다"는 주장은 데이터와 맞지 않는다 — 최종 판정: 부분 개선, 관성 아님(실험문서 §9.6).
- **(c) ~~"사람 방향 입력"을 백테스트에서 시뮬레이션하는 방법 미정~~ — 2026-08-20 정체성
  재정의로 무효(moot)**: 방향을 모델이 결정하는 것으로 바뀌어 "사람 결정을 어떻게
  시뮬레이션할까"라는 질문 자체가 더 이상 존재하지 않는다. 아래는 역사적 기록으로 보존.
  (원문: 실제 라이브 사람 결정 로그가 없다. 1차 연구 질문 문서에서 최소 1개의 구체적
  프록시(오늘 어블레이션의 always_long/always_short/random arm 재사용, 및 h48qual/zig075
  argmax를 사람 대리로 취급하는 보조 프록시)를 제시했으나 확정된 것은 아니다 — "사람이
  실제로 무작위보다 나은 방향을 낼 것이다"라는 전제 자체는 이 서브프로젝트에서 검증할 수
  없는 근본적 한계로 남는다.)
- **(d) 레짐의존성**: 오늘 어블레이션의 레인지장 재검정(N=30)에서 방향 편향의 부호가 레짐마다
  뒤집힘이 통계적으로 확정됐다(저스프레드 레인지 t=−3.36/−5.05, 20pp/VAL t=+4.72/+2.61). 이
  프로젝트가 설계할 능동적 리스크관리 신호도 레짐마다 다르게 작동할 가능성이 있다 — 1차 연구
  질문의 성공/킬 기준에 반영했으나, 완전히 해소된 이슈는 아니다. **2026-08-17 실증(위 (b) 인용
  실험문서)이 이 가설을 새 exit 신호에서도 재확인**: 성공조건 2(fresh-forward MDD/PnL) 통과가
  6/6 중 3/6 윈도우에 그쳤고, 특히 "순수 레인지"(스프레드가 가장 작은 2구간)에서는 always_short-type
  PnL 가드레일이 일관되게 깨졌다 — 저스프레드 레인지에서 방향 정보 자체가 빈약해지는 기존 패턴과
  같은 방향. 여전히 완전히 해소된 이슈는 아니다(무엇이 조건2 통과/실패를 가르는지는 미분리).
- **(e) L3 quality 게이트/L5 우선순위 중재의 최종 형태 미정**: "진입 자체는 막지 않는다"는
  스코프 결정 #2를 따르되, quality 값을 L7 사이징의 context feature로 계속 쓸지, 두 컴포넌트를
  동시에 살릴지는 (a)와 함께 다음 설계 단계에서 확정한다. ⚠️ 스코프 결정 #2 자체가 2026-08-20
  정체성 재정의로 근거(사람 권한 유지)를 잃었으므로 (j)와 함께 재검토 필요.
- **(j) L3/L4 진입 사전거부(entry veto)의 스코프 밖 처리 재검토 필요 (2026-08-20 신규,
  정체성 재정의 직결)**: 원래 "방향 전환·헤지·진입거부는 사람 권한 대체 대상이 아니다"가
  근거였는데, 방향 자체가 모델 결정으로 바뀌면서 이 근거가 사라졌다. zig075류 진입 사전거부를
  이제 다시 스코프 안으로 들여올지, 계속 밖에 둘지는 사용자가 아직 명시적으로 지시하지 않아
  임의로 정하지 않는다 — 다음 세션 결정 사항.

## 일리아스 라벨로직 후보축 — 2026-08-21 이관: zigzag/h48qual/cusum 154피쳐 스크리닝

**출처**: 별도 서브프로젝트 `eth_tabm_label_logic_retest_initiative`(2026-08-19 시작, DC/CUSUM/
분포적회귀 라벨 재검토 축)에서 진행되던 작업을 사용자 지시로 이 계약에 이관. Baseline v1/일리아스
1 축과 완전히 별개이며 대체하지 않는다 — 순수 탐색 축.

**⚠️ 전체 판정 요약**: 이 절의 모든 결과는 **N=3(또는 N=1) 예비 스크리닝**이다. CLAUDE.md
Seed-Diversity Ensemble Promotion Gate(N≥5, 진짜 무작위 시드)에 못 미친다 — 승격/확정 근거로
쓰지 않는다. "다음에 N≥5로 볼 가치가 있는가"를 가리는 것이 유일한 목적이다.

### 피쳐 계약 — 154개 확정

158개 캐노니컬 ETH 피쳐 유니버스에서 파생: 완전상수 11개 제거 → 상관클러스터 dedup 14개 →
확률단체 선형종속 1개 제거 → 반복 VIF<10 수렴 20개 제거(112) → RIT식 트리구조 조합 30개
추가 → 금융ML 문헌표준(분수차분/Corwin-Schultz/Roll/Kyle's Lambda/VPIN/실현모멘트/분산비율검정)
12개 추가 = **154개**. Wrapper: `scripts/eth_dc_engineered_features_canonicaldata_20260820.py`.
전체 파생계보: `docs/model_contracts/eth_dc_engineered_feature_set_lineage_20260820.json`.

⚠️ **순수 정보량 자체는 chance로 확인됨**(개별/조합 permutation-null, 실제 TabM N=5×2배치
재학습 — cond_acc 48.2~51.4%, 원본 158피쳐와 사실상 동일). 이 절의 가치는 "154피쳐가 정보량이
있다"는 주장이 아니라 "동일 피쳐·동일 아키텍처 위에서 **라벨로직만 바꿨을 때** 상대적 차이가
있는가"를 보는 데 있다.

`regime3_current_sensitive_wide24_{bull_prob,bear_prob,confidence}` 3개가 154 안에 포함된다
(chop_prob는 확률단체선형종속, entropy/margin은 VIF로 이미 제거). **2024년은 이 3개의 원본
HMM 오버레이(`REGIME3_CURRENT_2024`) 파일이 누락돼 있었으나, 이번 이관 작업 중 재생성해 정식
경로에 저장 완료**(기존 fitted joblib 재사용, 재적합 없음) — 상세는
[[eth_regime3_current_2024_training_data_compatibility_20260821]] 메모리 참고. TRAIN 구간
피쳐가 그 구간 전체 정보로 fit되는 것은 VAL/OOS causal-safety(Fresh-Forward Rule)와 무관하다는
점, 실측(2024 vs 2025 KS effect size=0.032, 무시가능)으로 확인됨.

⚠️ **154→151(위 3개 제거) 변경만으로 실제 재학습 시 zigzag의 OOS 부호가 뒤집힘을 확인**
(동일 seed=133725056: zigzag 154피쳐 OOS+13.76%→151피쳐 OOS−10.11%, h48qual/cusum은 부호
유지하되 h48qual은 VAL+21.65→+0.34/OOS+17.10→+4.63으로 크게 약화). 이 저장소 전역의 "작은
변화에도 OOS 부호가 잘 뒤집힌다"는 패턴과 일치 — **154개 고정 스펙 자체가 이미 이런 취약성의
증거를 내포한다는 뜻으로 읽을 것**, 아래 결과들도 같은 수준의 불안정성 위험을 안고 있다고
가정해야 한다.

### 데이터 계약 — 2024-01-01 ~ 2026-06-30, 154피쳐 전체 완전성 확인

| 파일 | 경로 | 행수 | 구간 |
|---|---|---:|---|
| 2024 | `tmp/ilias_eth_154feature_dataset_20260821/ilias_eth_154feature_2024.csv` | 105,380 | 2024-01-01~12-31 |
| 2025 | `tmp/ilias_eth_154feature_dataset_20260821/ilias_eth_154feature_2025.csv` | 105,101 | 2025-01-01~12-31 |
| 2026 | `tmp/ilias_eth_154feature_dataset_20260821/ilias_eth_154feature_2026.csv` | 51,746 | 2026-01-01~06-30 |
| 통합 | `tmp/ilias_eth_154feature_dataset_20260821/ilias_eth_154feature_2024_2026H1_combined.csv` | 262,227 | 2024-01-01~2026-06-30 |
| 매니페스트(피쳐목록/NaN감사/소스파일 전체) | `tmp/ilias_eth_154feature_dataset_20260821/manifest.json` | — | — |
| 빌드 스크립트 | `scripts/ilias_eth_154feature_dataset_build_20260821.py` | — | — |

**완전성 감사 결과**: 154개 중 **143개는 NaN 0개**(전체 262,227행). 나머지 11개(분수차분
3종, 실현모멘트/분산비율검정 6종, 엔트로피/VPIN/Kyle's Lambda 3종, Corwin-Schultz 1종)는
2024-01-01 첫날 rolling-window 워밍업으로 최대 200행(0.076%)만 NaN — 예상된 정상 범위,
실질적으로 154개 전 구간 완전. 타임스탬프 중복 0건, 연도경계 이어붙임 시 인위적 gap 없음
(combo/financial-ML 피쳐를 2024+2025+2026 연속 시계열 위에서 한 번에 계산, 연도별 재계산
아님).

**⚠️ 독립 재검증(2026-08-20, 별도 세션)**: 위 주장을 4개 파일 전부 직접 재확인 — 행수
(105,380/105,101/51,746/262,227)·구간·타임스탬프 중복(0건)·컬럼목록(manifest `feature_list`와
완전일치, 누락/초과 0)·NaN(143개 0건 확인, 나머지 11개 전부 2024-01-01 워밍업 구간에만 연속
분포 — contiguous_from_row0 확인)·inf값(0건, 신규 체크) 전부 통과. **단, 계약이 보고 안 한
결측 발견**: combined 파일에 5분그리드 gap이 총 11건인데, 10건은 기존에 알려진 거래소단절급
미세gap(2024/2025, [[eth_canonical_data_date_range_verification_20260820]])과 일치하지만
**11번째(2026-02-28 16:00→03-01 00:00, 8시간/약96bar 결측)는 원본
`training_features_2026_rebuilt.csv`에는 없던 gap**(그 파일 2026년분은 내부 gap 0건으로 별도
실측 확인됨) — 154피쳐 데이터셋 구축 과정에서 새로 생긴 것으로 보인다. zigzag/h48qual 라벨소스가
끊기는 바로 그 날짜(2026-02-28, 위 라벨 계약 절)와 일치하는 게 의심스러우나 우연인지 빌드
스크립트가 그 경계에서 뭔가를 참조했는지는 미조사 — 후속 확인 필요(규모는 작음, 전체의
0.037%).

⚠️ **주의**: 위 5-way/구조분석/트레이드원장 결과(아래 절)는 전부 **2025 TRAIN(2025-01~
09-30) + 2026 OOS(01~06)** 조합으로 계산됐다 — 이번에 완성한 2024-01~2026-06 전체 데이터셋과는
아직 **재학습으로 연결되지 않았다**. 일리아스 자체 Train 컨벤션(2024-01~2025-09)으로
재학습하면 아래 수치는 달라질 수 있다(위 154→151 부호반전 사례가 그 위험의 직접 증거) —
2024 확장분을 실제로 반영한 재검증은 이 이관의 범위 밖, 다음 단계 후보다.

### 라벨 계약 — zigzag / h48qual / cusum (5-way 중 3개만 이 축에 남김, dc/분포적회귀 제외)

| 라벨 | direction 소스 | quality-mode | 비고 |
|---|---|---|---|
| zigzag | `tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531` | same_as_direction | zig075 production과 동일 direction 소스. **⚠️ 이 파일 자체가 2026-02-28에서 끊김**(빌드일 20260531 시점 원본데이터 한계로 추정, 근본원인 미조사 — 재빌드 여부 Open Issue로 아래 등록) |
| h48qual | (zigzag와 동일, 동일 제약) | quality_label_action + `sltp_h48_conservative_padded_to_zigzag_timestamps` | direction은 zigzag와 동일 소스 공유 — 두 라벨의 판정을 "독립 확인 2건"으로 세지 말 것 |
| cusum | `tmp/eth_cusum_triple_barrier_labels_dense_cashfill_20260820`(이 축에서 자체 빌드) | same_as_direction | 전체 2024-2026 커버(끊김 없음) |

dc(directional-change)/분포적회귀(Gaussian NLL regression head)는 이 5-way 비교에 포함됐으나
OOS 부호 1/2·0/3로 이 3개보다 약해 이관 대상에서 제외 — 원본기록은
`docs/experiments/eth_tabm_label_logic_5way_comparison_20260820.md`.

### 결과 요약 (154피쳐, seed=133725056 대표, N=3 시드 중 1개)

**5-way 스크리닝(N=3, VAL선택→OOS부호)**: zigzag 2승1패, h48qual 3승0패, cusum 3승0패.

**구조분석**(`scripts/eth_zigzag_h48qual_cusum_structural_similarity_20260821.py`, 공통커버리지
2024-01~2026-02, 227,186행): 방향전환빈도 zigzag 1.93%(연속유지 median 33bar=2.75h) <
h48qual 5.80%(11bar) << cusum 42.17%(2bar=10min) — cusum이 zigzag보다 약 22배 자주
방향전환. 같은-bar 매칭은 circular-shift 순열귀무 대비 우연 수준(p=0.99~1.00, 3쌍 전부).
±15분 허용해도 cusum의 zigzag 대비 방향일치율은 66.1%(h48qual-zigzag 81.7%보다 뚜렷이 낮음)
— cusum은 "더 잘게 쪼갠 같은 신호"가 아니라 판단 자체가 다른 축.

**⚠️ CRITICAL: OOS 기간 불공정 발견 + 정정**: zigzag/h48qual은 direction 소스 제약으로 OOS가
실질 2026-01~02(2개월, 16,897행)뿐인데 cusum은 전체 2026-01~06(6개월, 51,746행)를 그대로
썼다 — 애초 5-way 비교가 라벨마다 다른 길이의 기간을 비교한 것이었음. cusum을 zigzag/h48qual과
**같은 01-01~02-28로 잘라도** cusum(+22.92%/31건/승률51.6%)이 zigzag(+13.76%/23건)·
h48qual(+17.10%/12건) 둘 다보다 우위(거래표본도 더 두터움). 트레이드 원장 전체:
`scripts/eth_zigzag_h48qual_cusum_oos_trade_ledger_20260821.py` 산출,
`/tmp/claude-1000/.../scratchpad/oos_trade_ledgers/{zigzag,h48qual,cusum}_seed133725056_*_oos_trade_ledger.csv`
(세션 로컬 scratch, 세션 종료 시 유실 가능 — 필요시 스크립트 재실행으로 즉시 재현 가능,
report.json과 소수점까지 cross-check 통과 검증됨). 등가곡선 이미지:
`tmp/research_20260821/chart_zigzag_h48qual_cusum_oos_equity_curves.png`.

### 다음 단계 (Open Issue로 등록)

- **(f) zigzag/h48qual direction+quality 라벨의 2026-02-28 절단 — 완전 해소(2026-08-21)**:
  zigzag 원인 확정 = 2026-05-31 빌드 시점의 stale 입력 스냅샷(알고리즘 한계 아님), 원본 생성기를
  최신 입력으로만 재실행해 2026-08-19까지 확장. h48qual의 quality축(h48_conservative barrier)은
  원본 입력(alpha6/7 계열)이 **영구 재현불가**로 확인돼([[eth_omega4_quality_threshold_alpha67_pipeline_irreproducible_20260815]])
  단순 재실행은 불가능했으나, barrier 계산 로직·파라미터 자체는 alpha6/7 고유 피쳐가 불필요함을
  확인해 표준 canonical 가격데이터 위에서 동일 로직으로 재계산(사용자 지시) — 원본의 "연장"이
  아니라 "같은 레시피의 재계산"이라는 점은 구분해서 기록됨(겹치는 과거구간 절대값이 원본과
  정확히 일치한다는 보장 없음). 상세는 아래 "레짐 분류기 계약" 절 "재검증" 하위 항목.
- **(g) N≥5 재확인 미실행**: 이 절의 모든 수치는 N=3 이하 예비. h48qual/cusum이 상대적으로
  나아 보이나 승격 판단 근거 아님.
- **(h) 2024 확장 TRAIN 미반영**: 154피쳐+2024-01~2026-06 데이터셋은 완성됐으나(위 데이터
  계약 절), zigzag/h48qual/cusum 재학습에 아직 연결 안 됨 — 154→151 변경만으로도 부호가
  뒤집힌 전례가 있어, 2024 확장 반영 시에도 재검증 없이 위 수치를 그대로 신뢰하면 안 된다.
- **(i) 154피쳐 vs 일리아스 1의 102 base 피쳐 — 통합 여부 미정**: 이 축이 앞으로도 154피쳐를
  유지할지, 장기적으로 일리아스 본 계약의 102 base로 수렴할지는 다음 세션 결정 사항.

## 레짐 분류기 계약 — wide24 HMM 확정 (2026-08-21)

**결론**: `regime3_current_sensitive_wide24_{bull_prob,bear_prob,confidence}`(위 154피쳐 계약에
포함된 3개 컬럼)의 산출원은 **wide24 HMM**(`scripts/experiment_regime3_current_hmm_wide24_20260529.py`,
24개 hand-crafted 피쳐: STATE7 7개+RAW5 5개+WIDE24_EXTRA 12개)으로 확정한다. 논문충실 Jump
Model(EWM 9피쳐)/Sparse Jump Model(14/151/154피쳐)을 동일 ADX 정답지 기준으로 전부 비교했고,
어떤 대안도 wide24를 앞서지 못했다.

**왜 wide24가 이기는가(원인규명 완료, 우연 아님)**: 두 가지 구조적 이유.
1. **12+ 히든스테이트**(JM/SJM은 항상 정확히 3개로 강제).
2. **상태→클래스 매핑이 지도학습**이다(`_state_class_matrix()`가 TRAIN=2024의 실제 ADX 정답
   라벨을 직접 사용해 학습) — JM/SJM은 라벨을 전혀 안 보는 `sort_by=cumret`(평균수익률 순
   정렬) 휴리스틱뿐이다. 즉 JM/SJM과 wide24 HMM은 "같은 문제를 다른 방법으로 푸는" 비교가
   아니라애초에 지도학습 정보량이 다른 비교였다.
3. 피쳐 자체도 원인: ADX 정답지가 추세/모멘텀/변동성 정의(ADX+slope+BB폭)인데, wide24의 24개는
   정확히 그 개념(rsi/macd_hist/mtf_trend_1h·4h/trend_efficiency 등)에 맞춰 손수 설계됐다.
   151피쳐(CVD/오더플로우 중심)는 **동일한 wide24 HMM 방법론**을 그대로 써도(states=30,
   sticky=0.85, 신라벨 기준 VAL 0.5942 vs wide24 0.7414) 확실히 못 미친다 — 방법론이 아니라
   피쳐 자체의 정보량 차이도 독립적으로 확인됨.

**하이퍼파라미터 — states/sticky 35점 그리드 스윕 확정**: `states∈{6,9,12,16,20,24,30}` ×
`sticky∈{0.85,0.90,0.93,0.96,0.99}`(n_iter=22 고정, seed=7529 고정 — **시드 다양성 스윕 아님**,
아래 한계 참고). 결과 산출물: `tmp/eth_hmm_wide24_sweep_20260821/`(구라벨)·
`tmp/eth_hmm_wide24_sweep_20260821/*_v2/`(신라벨), 대응 151피쳐 비교는
`tmp/eth_hmm_ilias151_sweep_20260821/`. 전용 실험문서는 아직 작성되지 않음(이 세션의 대화기록이
유일한 상세 기록 — 후속 세션에서 `docs/experiments/`로 정리 권장, Open Issue로 아래 등록).

| 설정 | ADX 정답라벨 | VAL(2024Q4) | 2025 | 2026 |
|---|---|---:|---:|---:|
| 기존 기본값(states=12,sticky=0.93) | `balancedish_adx16_slope15_bb012`(구) | 0.7638 | 0.7683 | 0.7843 |
| **신기록(states=30,sticky=0.96)** | `balancedish_adx16_slope15_bb012`(구) | 0.8082 | 0.8091 | 0.8159 |
| **최종 채택(states=30,sticky=0.85)** | `balancedish_adx16_slope03_bb006`(신, 아래) | 0.7414 | 0.7463 | **0.7470** |

마지막 행이 **현재 확정 설정**이다(신라벨 채택으로 절대 정확도는 자연 하락 — 라벨 자체가 어려워진
결과이지 모델 퇴보 아님, 위 두 행과 직접 비교 불가). states=24(sticky 0.85~0.96 전부)도 근소한
차이(2025 0.747~0.748/2026 0.744~0.746)로 states=30보다 sticky 변화에 덜 민감 — 사실상 동급으로
취급 가능.

**ADX 정답라벨 재정의(2026-08-21, 사용자 지시) — `balancedish_adx16_slope03_bb006`**: 기존
`balancedish_adx16_slope15_bb012`(chop 69~78%)이 "너무 chop이 많다"는 사용자 판단으로 대체.
`LABEL_CONFIGS`에 신규 항목으로 등록(기존 항목은 보존, `scripts/experiment_regime3_current_hmm_wide24_20260529.py`
27~42행) — `trend_adx_min=16`(불변), `weak_adx_max=12`(불변, 구조적으로 거의 발동 안 함),
`slope_min: 0.00015→0.00003`, `tight_bb_max: 0.012→0.006`. **원인진단**: chop의 진짜 지렛대는
slope가 아니라 BB폭이었다(slope만 5배 완화해도 chop 75%→72%뿐인데 BB폭만 0.012→0.006으로
완화하면 75%→52~55%) — "평평한 slope"와 "좁은 BB폭"이 서로 크게 겹치는 조건이라 slope를
풀어도 그 바 대부분이 BB필터에 다시 걸리기 때문. 신라벨 최종분포(3개년): bull 22~36%/bear
23~34%/chop 36~55%(연도별 차이 있음, 상세는 세션 대화기록). **부작용**: flip_rate가 대략
2배로 증가(chop을 줄인 대가로 지속성 저하) — 트레이드오프로 명시.

**한계(Open Issue로 등록)**:
- **(k) HMM 시드 다양성 미검증**: 위 35점 그리드는 seed=7529 고정 상태에서 states/sticky만
  바꾼 하이퍼파라미터 스윕이다 — CLAUDE.md Seed-Diversity Ensemble Promotion Gate(N≥5 진짜
  무작위 시드)를 아직 거치지 않았다. states=30/sticky=0.85 확정 전 다른 시드로도 재확인이
  필요한지는 미결정.
- **(l) 154피쳐 데이터셋의 regime3_current_sensitive_wide24_* 컬럼 미갱신**: 위 "피쳐 계약"
  절의 154피쳐 데이터셋(`tmp/ilias_eth_154feature_dataset_20260821/`)은 **이번에 확정한
  states=30/sticky=0.85/신라벨 설정이 아니라 기존(구)설정**으로 만들어진 값을 그대로 쓰고
  있다 — 이 확정을 실제로 반영하려면 154피쳐셋의 그 3개 컬럼을 재생성해야 하며, 아직
  실행되지 않았다. 재생성 시 zigzag/h48qual/cusum 재학습도 함께 재검증 필요(154→151 변경만으로
  부호가 뒤집힌 전례 있음, 위 "일리아스 라벨로직 후보축" 절 참고).
- **(m) 신라벨 도입 시 위 "일리아스 라벨로직 후보축" 절 전체 재검증 필요**: zigzag/h48qual/
  cusum 5-way 비교(N=3)와 트레이드 원장은 전부 구라벨(`balancedish_adx16_slope15_bb012`) 기준
  ADX 정답지로 평가됐다 — 신라벨로 바뀌면 그 절의 수치도 다시 봐야 할 수 있다(단, 그 절의
  "정답지"는 zigzag/h48qual/cusum 자체 라벨의 실제 트레이딩 성과 평가이지 ADX 매칭이 아니므로
  영향 범위는 제한적일 수 있음 — 미확인).

### 재스윕 — 앵커드 Walk-Forward 적용, 같은 날 2차 확정 (2026-08-21)

**배경**: 위 최초 확정(states=30/sticky=0.85)은 TRAIN=2024 단독 fit 기준이었다. 사용자가 "레짐
분류기 학습 데이터를 본 모델 데이터 split에 맞추자"고 제안했고, 논의 중 아래 "데이터 Split
재설계 제안"(앵커드 walk-forward, 당시 상태=제안됨/미실행)이 바로 이 문제를 다루는 이미 결정된
방법론임이 확인되어, **이 축에 한해 즉시 적용**했다. 원안과 2가지 의도적 차이:
1. **VAL을 TRAIN 내부에 둠** — 이건 원안 그대로다. `scripts/experiment_regime3_current_hmm_wide24_20260529.py::_train_one`이
   이미 `--val-start` 이전을 fit, 그 뒤를 VAL 채점에만 쓰고, **최종 저장 모델은 TRAIN 전체(VAL
   포함)로 재fit**하는 구조라(스크립트 수정 불필요) 별도 구현 없이 원안의 "VAL 재사용" 메커니즘과
   정확히 일치한다.
2. **OOS를 09-30까지 기다리지 않음** — 사용자의 명시적 선택으로 원안과 다른 지점이다. 전체 13주
   분기 대신 그 시점 실제 가용 데이터(~7주)로 확정했다. 표본이 작아진 트레이드오프이지 원칙 위반은
   아니다(지켜야 할 핵심은 "한 번 보고 재사용 안 함"이지 "반드시 분기 전체"가 아님) — 아래
   "OOS 사용이력" 참고.

**구간 확정**:

| 구간 | 값 | 행수 | 비고 |
|---|---|---:|---|
| TRAIN | 2024-01-01 ~ 2026-06-30 | 262,609 | `data/splits/year_oos/training_features_{2024,2025,2026_rebuilt}.csv` 병합 |
| VAL(TRAIN 내부, `--val-start`로 분리) | 2026-04-01 ~ 06-30 | 26,208 | 최근분기 재사용, 최종 fit엔 포함 |
| OOS(single-touch) | 2026-07-01 ~ 08-19 | 14,400 | ~7주, 전체분기(13주) 아님 — 사용자 선택 |

**데이터 출처 관련 발견 2건** (둘 다 진행 전 확인, 근거 없이 넘어가지 않음):
- **동명 파일 함정 재발**: wide24 스크립트의 기본 입력(`tmp/causal_regen_20260516/funding_clean_splits_20260528/training_features_2026_rebuilt.csv`)은 2026-06-30에서 멈춰있고, 실제 라이브에 가까운 사본은 `data/splits/year_oos/training_features_2026_rebuilt.csv`(2026-08-19까지)였다. 후자를 사용.
- **미해명 corruption-incident 백업**: `data/splits/year_oos/training_features_2026_rebuilt.csv.bak_pre_extend_20260820_corruption_incident_evidence`라는 백업이 존재 — 8/20 확장작업 중 문제가 있었다는 뜻이나 이를 설명하는 기록을 memory/docs 어디서도 못 찾음(동시세션 작업 가능성). **현재 파일에 대해 수행한 체크**: 중복 timestamp 0건, 2026-01-01~08-19 5분봉 기대행수(66,528)와 정확히 일치, tail 500행 NaN스파이크 없음, corruption-백업과 컬럼수 동일(142) — 기본 무결성은 통과했으나 **인시던트 자체의 원인은 미규명 상태로 남아있음**. 후속 세션에서 이 파일을 다시 쓸 일이 있으면 먼저 이 항목을 확인할 것.

**실행**: dev(WSL2)에서 시작했으나 states=30 그룹 fit 도중 **동일 지점에서 2회 연속 재시작**(기존
[[dev_machine_wsl2_instability_20260816]] 패턴 재현) → 사용자 지시로 전량 서버 이관. 이관 중
자체 오케스트레이션 스크립트(tmp/ 임시 wrapper, 원본 wide24 스크립트 자체는 무수정)에서 버그 2개
발견·수정: (a) wrapper에 로컬 절대경로 하드코딩 → 서버에서 전량 `ModuleNotFoundError`로 즉시
실패, (b) 배치 러너가 xargs 서브셸 안에서 실패를 감지 못 하고 무조건 `[done]`을 찍어 **가짜
성공으로 보였음**(서버 파일시스템을 직접 열어 report.json 부재로 확인 후 발견). 수정 후 전량
재실행, 35/35 성공·`[FAILED]` 0건 확인. seed=7529·n_iter=22는 최초 확정과 동일(불변, 아래
Open Issue (k) 그대로 적용됨).

**결과 — states/sticky 35점, 신라벨(balancedish_adx16_slope03_bb006) 기준**:

| states | sticky | VAL(2026Q2) | OOS(07-01~08-19) |
|---:|---:|---:|---:|
| 24 | 0.99 | 0.7551 | 0.7512 |
| 6 | 0.90 | 0.7548 | 0.7280 |
| 6 | 0.85 | 0.7547 | 0.7286 |
| **24** | **0.90** | **0.7545** | **0.7682** |
| 24 | 0.93 | 0.7543 | 0.7611 |
| 24 | 0.85 | 0.7541 | 0.7677 |
| 30 | 0.96 | 0.7541 | 0.7618 |
| 30 | 0.85 | 0.7539 | 0.7670 |
| 30 | 0.93 | 0.7537 | 0.7668 |
| ... | | (전체 35점은 `tmp/eth_hmm_wide24_resweep_train2026h1_20260821/states*/report.json`) | |

**핵심 발견 — VAL 단독 선정의 위험이 실측으로 드러남**: states=6(5개 sticky 전부)이 VAL에서
1~2위권(0.7545~0.7548, 1위와 0.0003~0.0006차, 사실상 잡음 수준)까지 올라오지만, **OOS에서는
같은 5개가 전부 0.728대로 이 35점 중 최하위권**(states=9/12보다도 낮음)이다. 최초 확정 때는
VAL만으로 순위를 매겼는데, 이번엔 마침 states=6이 VAL 1위를 근소하게 놓쳤을 뿐 — VAL 잡음의
방향이 조금만 달랐어도 OOS에서 가장 나쁜 설정이 뽑혔을 수 있었다는 뜻. 반면 **states=24는 5개
sticky 전부 VAL·OOS 양쪽에서 안정적으로 상위권**을 유지해 가장 견고한 클러스터로 확인됨.

**최종 채택(states=24, sticky=0.90)** — sticky=0.85와 사실상 동급(VAL/OOS 둘 다 0.0005 이내
차이)이라 상호교환 가능. OOS 1위(0.7682)이면서 VAL은 1위(states=24/0.99, 0.7551)와 통계적으로
구분 안 되는 차이(0.0006)라 OOS를 우선 근거로 채택. 최초 확정(states=30/sticky=0.85)은
기각된 게 아니라 근소한 2위군(OOS 3위 0.7670)으로 남음 — TRAIN 구간이 바뀌면서 순위가
재조정된 것이지 이전 결론이 틀렸던 게 아니다.

<details>
<summary>states=24/sticky=0.90 상세 지표</summary>

VAL(2026Q2, n=26,208): accuracy=0.7567, balanced_accuracy=0.7545, recall bull=0.678/bear=0.830/chop=0.756, flip_rate=0.122
OOS(07-01~08-19, n=14,400): accuracy=0.7340, balanced_accuracy=0.7682, recall bull=0.806/bear=0.826/chop=0.673, flip_rate=0.120

</details>

**OOS 사용이력**: `wide24_regime_classifier` 축의 `2026-07-01~08-19` 구간은 위 스윕 1회로
**소진(consumed)** 처리한다 — 이 states/sticky 재선정 목적으로는 재조회 금지. 다음 세대부터 이
구간은 TRAIN으로 편입된다(앵커드 확장). 별도 사용이력 ledger 파일은 아직 없음 — 필요성이
커지면(본 모델 자체도 이 방법론을 실행할 때) 별도 파일로 승격 검토.

### 시드 다양성 검증 — CONFIRMED, N=5 (2026-08-21, 같은 날 3~4차)

**대상**: OOS 상위 3개 — states24/sticky0.90(신pick), states24/sticky0.85, states30/sticky0.85(구pick).
**시드**: 원 seed=7529(baseline) + 진짜 무작위 4개(534964/116595/666940/505456, `random.sample`/
`random.randint` 시스템엔트로피, 고정간격 증가 아님 — CLAUDE.md Seed-Diversity Gate 요건 충족) =
**N=5, 정식 게이트 문턱 충족**. 같은 TRAIN/VAL/OOS split, 동일 데이터 재사용, 15개 fit 전부
서버에서 성공(`[FAILED]` 0건, 3차 N=4 예비검증 후 4차로 5번째 무작위 시드 추가).

**결과 — N=5 전부 순위·수치 완전히 안정적**:

| config | OOS 평균(N=5) | OOS std | OOS min~max |
|---|---:|---:|---|
| **states=24/sticky=0.90** | **0.7684** | 0.0001 | 0.7682~0.7685 |
| states=24/sticky=0.85 | 0.7678 | 0.0001 | 0.7677~0.7679 |
| states=30/sticky=0.85 | 0.7671 | 0.0001 | 0.7670~0.7672 |

**5개 시드 전부 예외 없이** `states24/0.90 > states24/0.85 > states30/0.85` 순서를 재현했고,
절대수치 편차는 std≈0.0001로 사실상 잡음 수준이다 — 이 프로젝트의 다른 시드검증(TabM 계열,
Sigma3-1h 등)에서 흔히 보이는 "시드분산이 HP효과보다 큼" 패턴과 정반대다. 해석(가설, 미확증):
이 wide24 HMM은 EM(Baum-Welch)+지도학습 state-class 캘리브레이션 구조라 신경망 랜덤초기화보다
시드 민감도가 구조적으로 낮을 수 있음.

**판정: CONFIRMED (N≥5 진짜무작위 시드, OOS 순위 5/5 일치)** — states=24/sticky=0.90 최종pick이
CLAUDE.md Seed-Diversity Ensemble Promotion Gate를 충족한다. Open Issue (k) 해소.

**⚠️ 후속(2026-08-23) — BTC-metrics 오염 수정 후 재적합 스팟체크**: [[eth_binance_metrics_
archive_backfill_canonical_divergence_20260823]]에서 wide24의 관측피쳐 `state12_oi_change_rate`
가 2026-01-20~07-12 구간(TRAIN 내부) 오염됐음이 발견됨 — 위 N=5는 전부 이 오염 데이터로 학습된
것이었다. states=24/sticky=0.90/seed=7529(baseline seed)만 클린 데이터로 재적합해 스팟체크:
**OOS balanced_accuracy 0.7691**(기존 N=5 range 0.7682~0.7685) — 시드노이즈(std=0.0001) 대비로는
실차이지만 절대크기(+0.0006~0.0009)는 작다. N=5 전체 재검증은 안 함(우선순위 낮음 판단, 필요시
후속 가능) — CONFIRMED 판정은 유지하되 "완전히 클린한 데이터 기준"은 아니라는 점을 기록.

**신규/갱신 Open Issue**:
- (k) 시드다양성 — **CONFIRMED, 해소**. Top-3 config 전부 N=5 무작위 시드에서 순위·수치 완전
  안정(위 표). 재확인 불필요.
- (l) 154피쳐셋 컬럼 미갱신 — **해소(2026-08-21)**. `tmp/ilias_eth_154feature_dataset_20260821/`의
  4개 CSV(2024/2025/2026/combined) 전부 `regime3_current_sensitive_wide24_{bull_prob,bear_prob,
  confidence}` 값을 states=24/sticky=0.90(신라벨, TRAIN=2024-01~2026-06-30) 모델 출력으로
  timestamp join 교체(0 NaN, row/컬럼구조 불변 확인). **사용자 명시 선택으로 컬럼명은 유지**
  (구 2026-05-30 설정 이름 그대로, 값만 최신화) — 재현성 혼선 위험을 완화하기 위해
  `manifest.json`에 `source_regime_overlay_files_UPDATED_20260821` 필드로 신규 출처(모델
  경로/라벨/states/sticky/변환소스)를 명시적으로 기록. 원본은 `*.bak_pre_states24sticky090_regen_20260821`로
  보존.
- (m) 라벨로직 후보축 재검증 — **N=1 재확인 완료, 데이터 확충 후 결과 갱신됨(2026-08-21)**.
  최초 N=1 재확인(TRAIN 2024+2025, EVAL 2026-01~06 — zigzag/h48qual 라벨소스 한계로 그 이상
  불가능해 우회)에서는 3개 전부 승이었으나, **이후 사용자 지시로 데이터 자체를 확충**해 이 결과가
  갱신됐다:
  - **zigzag 재빌드**: 원본 생성기(`scripts/build_wave3_action_labels_20260531.py`)를 알고리즘
    무변경, 최신 입력파일(`data/splits/year_oos/training_features_2026_rebuilt.csv`, 이제
    2026-08-19까지)로만 재실행 → 2026-02-28 절단이 **"알고리즘 한계"가 아니라 2026-05-31 빌드
    시점의 stale 입력 스냅샷 문제였음이 확인됨**(Open Issue (f) 원인규명+해소). 신규 위치:
    `tmp/ilias_labellogic_recheck_20260821/zigzag_action_labels_rebuilt_20260821/`, 2026년분
    16,897행→66,528행(2026-01-01~08-19 전체).
  - **cusum 재빌드**: 동일 패턴(`scripts/build_eth_cusum_triple_barrier_labels_20260819.py` +
    `scripts/build_eth_directional_change_dense_cashfill_labels_20260819.py --sparse-dir`)으로
    2026-07-20→08-19까지 확장. 도중 numba 캐시버그 재발(`<dynamic>` 모듈 재로드 실패,
    [[eth_tabm_label_logic_retest_initiative_20260819]]에 이미 기록된 것과 동일 증상) —
    `core/__pycache__/event_label_engine.*.nb{i,c}` 삭제로 해결.
  - **h48qual quality축도 확충 완료(사용자 재지시)**: 원본 quality barrier
    (`sltp_triple_barrier_h48_conservative`)의 실제 빌더를 재탐색해 발견
    (`scripts/build_omega1_2_triple_barrier_labels_20260619.py`) — 그런데 그 스크립트의
    입력이 `trade_candidates_*_alpha6_current_tail111_exact.csv`(alpha6/7 계열)였고, 이
    파이프라인은 **2026-08-10 커밋 `4c46d20`에서 생성 스크립트가 의도적으로 삭제돼 영구
    재현불가**임이 확인됨([[eth_omega4_quality_threshold_alpha67_pipeline_irreproducible_20260815]],
    별도로 이미 기록돼있던 경고 — "연장/확장 재제안 금지"). **단, barrier 계산 로직 자체
    (`BarrierConfig("h48_conservative",48,1.2,0.8,0.006,0.004)`+ATR/TP-SL/quality 페널티
    함수)는 alpha6/7 고유 피쳐가 전혀 필요 없고 timestamp/OHLC 4개 컬럼만 요구** — 사용자
    지시로 그 로직·파라미터를 그대로 import(재구현 아님)해 표준 canonical 데이터
    (`data/splits/year_oos/training_features_*.csv`, zigzag/cusum과 동일 소스) 위에서
    재계산(`scripts/build_h48_conservative_barrier_canonicaldata_20260821.py`) → 신규 zigzag
    timestamp에 패딩(`scripts/pad_h48_conservative_canonicaldata_to_zigzag_timestamps_20260821.py`).
    **⚠️ 이건 원본의 "연장"이 아니라 같은 레시피를 다른(표준) 가격소스 위에서 재계산한
    것** — 겹치는 과거구간 절대값이 원본 alpha6/7판과 완전히 일치한다는 보장 없음(가격소스가
    다름), 원본은 대체하지 않고 별도 산출물로 유지. 부수효과로 **2024년도 신규 확보**(원본은
    2025~2026만 있었음). 이제 h48qual도 신규 zigzag(direction)+신규 quality 둘 다 같은 소스
    기반이라 정합성 문제 없음.

  **최종 갱신 결과 — TRAIN=2024+2025, EVAL=2026-07-01~08-19(레짐분류기와 완전 동일 창,
  3개 라벨 전부), seed=133725056(N=1)**:

  | 라벨 | VAL 최고 threshold | OOS 부호 |
  |---|---|---|
  | zigzag | q=0.55(VAL+5.69%) | OOS **+0.82%** (승, 매우근소·거래6건) |
  | h48qual | q=0.60(VAL+16.89%) | OOS **+5.45%** (승, 거래2건뿐) |
  | cusum | q=0.50(VAL+18.96%) | OOS **−12.65%** (**패로 반전**) |

  **⚠️ 핵심 변화**: 3개 라벨 전부 데이터를 확충해 처음으로 완전히 동일한 EVAL 창(진짜
  레짐분류기 OOS창)에서 직접 비교 가능해졌다. 결과: **cusum이 승에서 패로 뒤집혔고,
  zigzag/h48qual의 승도 둘 다 거래건수가 극히 적어(6건/2건) 근소하다.** 이전 N=1(제한된
  EVAL 사용)에서 관측된 "3개 전부 승"은 짧거나 부정확한 EVAL 창의 산물이었음이 확정됨.

  **⚠️⚠️ N=6 시드검증 완료(원시드+진짜무작위 5개) — CLAUDE.md Seed-Diversity Gate 정식 적용,
  판정 대폭 갈림**: TRAIN=2024+2025/EVAL=2026-07-01~08-19 고정, seed만
  {133725056,325805917,775149439,126593178,286919795,310216042}로 6개 재실행(3라벨×6=18회,
  전부 성공 — 병렬실행 중 공유 placeholder 파일 쓰기 경합으로 cusum 2개 시드 최초 실패 후
  순차재시도로 해결, 로직버그 아님).

  | 라벨 | OOS 부호 일치(N=6) | 판정 |
  |---|---|---|
  | **h48qual** | **6/6 승** | 시드 전반에 안정적 — Seed-Diversity Gate 충족 |
  | zigzag | 4/6 승 (2패) | 다수이나 확실치 않음 — 게이트 미충족 |
  | cusum | 3/6 승 (사실상 동전던지기) | **완전 불일치 — 대표시드(133725056)의 "패" 자체가
  신뢰 불가능한 단일관측이었음이 확정** |

  즉 원래 대표시드(133725056)만 봤을 때 "zigzag 근소승/h48qual 승/cusum 패"였던 결론 중,
  **h48qual만 시드에 안정적으로 재현**되고, zigzag는 다수결 정도, cusum은 애초에 부호
  자체에 아무 정보가 없었다(6개 중 3승3패). 이 결과가 이 절 전체의 **최종 판정**이다 — N=1
  대표시드 수치는 더 이상 인용하지 말 것. N≥5 정식 승격게이트 관점에서도 h48qual만 방향
  일치를 보였을 뿐, "승격 근거로 쓸 수 있다"는 뜻은 아니다(거래건수가 여전히 2~7건 수준으로
  얇음, 위 반복 경고 그대로 적용).

  **⚠️⚠️⚠️⚠️⚠️ 트레이드 원장 분석 — N=6 시드차이의 진짜 원인은 라벨 품질이 아니라 우연한
  LONG/SHORT 편향이었음 확정 (같은 날 6차, 결정적 발견)**: 18개 run 전부(`scripts/
  eth_ilias_anchored_oos_trade_ledger_20260821.py`, 기존 검증된 `eth_zigzag_h48qual_cusum_
  oos_trade_ledger_20260821.py` 로직 그대로 이식, report.json과 소수점까지 cross-check 18/18
  OK) 개별 트레이드를 재구성해 분석한 결과:
  - **이 OOS 구간(2026-07-01~08-19) 자체가 ETH $1574→$2252(**+43.08%**)의 거대한 상승장이었다.**
  - **long_frac(LONG비중)과 최종 PnL의 상관계수 = 0.888** — 사실상 "롱을 많이 탔는가"가
    승패를 거의 다 설명한다. cusum의 지는 시드 3개는 전부 long_frac≤0.09(거의 전량 숏),
    이기는 시드 3개는 전부 long_frac≥0.625(대다수 롱) — **완벽하게 갈림.** h48qual 6/6승도
    6개 시드 전부 long_frac≥0.5였을 뿐(SHORT 비중이 큰 시드가 우연히 하나도 안 나온 것).
  - **풀링 사이드별 승률**: zigzag(LONG71%/SHORT15%), h48qual(LONG94%/SHORT25%),
    cusum(LONG87%/SHORT9%) — 3개 라벨 전부 같은 패턴. 즉 "어느 라벨이 나은가"가 아니라
    "그 시드가 얼마나 우연히 롱에 치우쳤는가"가 결과의 대부분을 설명한다.
  - **항상-롱(always-long) 벤치마크와 대조**: 이 구간 단순 매수보유 수익률(+43.08%, 2x
    레버리지 환산 +86.16%)이 **18개 run 중 최고 성적(h48qual +14.41%)보다도 압도적으로
    높다.** [[h48qual_standalone_replay_invalid]]가 이미 경고한 "항상 always 벤치마크 대조
    없인 편향을 스킬로 착각" 패턴이 여기서 다시 확정됨.

  **최종 결론**: 위 N=6 시드검증 판정(h48qual 6/6, zigzag 4/6, cusum 3/6)은 여전히 사실이지만,
  **그 차이의 원인이 "라벨 로직의 방향 정보량"이 아니라 "훈련 노이즈로 우연히 갈린 LONG/SHORT
  편향 × 우연히 이 구간이 강세장이었다는 것"의 조합임이 확정됐다.** 3개 라벨 모두 always-long
  대비 확실한 열위이므로, 이 재검증 전체는 "zigzag/h48qual/cusum 중 어느 것도 이 N=1~6 규모
  실험으로는 방향 alpha를 주장할 근거가 없다"로 종결한다 — 이 저장소의 지배적 패턴
  (40개+ 선행 라벨방법론이 전부 동일 결론에 수렴)과 다시 한번 정합.

  **⚠️⚠️⚠️⚠️⚠️⚠️ TRAIN구간(2025 전체) 원장분석 — 상관관계 붕괴로 "구간효과" 가설 직접 확증,
  단 라벨 자체의 근본 열위는 그대로 (같은 날 7차)**: "OOS를 늘리면 어떤가"라는 질문에서
  발전 — OOS(single-touch)는 안 건드리고, **이미 저장된 예측(train_predictions_qXXX.csv+
  validation_predictions_qXXX.csv, oof 예측)을 그대로 재사용**해 2025 전체(Q1−45%/Q2+36%/
  Q3+67%/Q4−28%, 강세·약세 다 포함)로 같은 원장분석을 반복(`scripts/
  eth_ilias_anchored_train_period_ledger_20260821.py`, 재추론 없음).
  **⚠️ 부수 발견**: `_prepare_frames()`가 `_read_labels(direction_label_dir, 2025, ...)`로
  **연도를 2025로 하드코딩**하고 있어서, 오늘 "TRAIN을 2024까지 확장했다"고 여러 차례 기록한
  것은 **부정확했다** — `omega.TRAIN_CSV`를 2024+2025로 넓혀도 2024는 `_align()`에서 조용히
  버려지고 실제로는 항상 2025(1~9월 train/10~12월 val)만 쓰였다. 위 (m) 절의 "TRAIN=2024+2025"
  표현은 전부 이 정정을 전제로 읽을 것 — 실제로는 시종일관 2025-only였다.

  **결과**: long_frac↔PnL 상관계수가 0.888(OOS, 순수강세 7주) → **0.221(2025 전체, 918건
  풀링)**로 붕괴 — "구간이 하나의 추세로 도배돼서 롱비중이 곧 성적이었다"는 가설이 직접
  확증됨. 단, 라벨별로 온도차가 크다: cusum(상관0.072, 거의무관)은 승6개중2개로 OOS보다도
  나쁨(풀링 승률 LONG37%/SHORT36%, 평균거래손익+0.037%로 사실상 0에 근접) — 장기로 보면
  cusum은 방향편향 문제가 아니라 그냥 못한다. h48qual(상관**0.836**, 여전히 높음)은 5/6승
  유지 — 장기간에도 롱편향-성적 연동이 구조적으로 지속된다는 뜻이라, "라벨이 좋다"보다는
  "이 모델이 구조적으로 롱쪽에 치우치고 그게 이 자산 특성상 대체로 맞아떨어진다"는 해석이
  더 설득력있음(자산 자체의 장기 우상향 편향과 얽혀있을 가능성, 별도 확인 안 됨). zigzag(상관
  0.308, 5/6승) 중간. **최종 종합**: "구간 하나 때문"이라는 가설은 부분적으로만 맞았다 —
  더 다양한 구간에서도 승패패턴이 남아있는 라벨(h48qual)이 있지만, 그게 "방향 alpha"인지
  "자산 자체의 우상향 성향에 우연히 정렬된 구조적 편향"인지는 이 실험 규모로 구분 불가능하다
  (풀링 승률 전부 36~40%로 동전던지기 미만이라는 게 그 방증). 최종 판정(방향 alpha 근거 없음,
  always-long 대비 열위)은 불변.

  **⚠️⚠️⚠️⚠️⚠️⚠️⚠️ 2024 데이터 실제 학습 반영 — 근본원인(구간효과) 재확증 (같은 날 8차, 사용자
  지시 "3개 라벨 모두 24년 학습에 추가")**: 위 하드코딩(`_read_labels(direction_label_dir,
  2025,...)`) 발견 직후 사용자가 실제 수정을 지시. 공유 스크립트(`train_eval_omega4_3head_
  parent72_loose_entry_quality_20260620.py`, BTC/SOL 형제+라이브 quality_threshold 선정에도
  쓰임)는 무수정 — `_read_labels`가 파일명만 보고 내용 연도를 검증 안 한다는 점을 이용해,
  "2025"라는 이름의 파일 안에 2024+2025 병합 라벨을 넣은 신규 디렉토리로 로컬 우회
  (direction: zigzag/cusum 각자 라벨, quality: h48_conservative 재계산본 — 전부 이 세션에서
  이미 검증된 소스 재사용, 신규 재구현 없음). 레짐 오버레이는 이미 2024부터 채워진 v2
  파일(`train_2024_2026H1_regime3_current_states24_sticky090.csv`)을 쓰고 있어 추가 작업
  불필요, 자동 반영됨. 실측 확인: train_predictions 78,605행(2025 1~9월)→**183,985행(2024-01~
  2025-09)**로 확장. VAL(2025 Q4)/OOS(07-01~08-19)는 불변.

  **N=6 재실행 결과(2024 포함 vs 미포함)**: zigzag 4/6→**1/6**(악화), h48qual 6/6→5/6,
  cusum 3/6→3/6(불변). 18개 원장 전부 report.json cross-check OK. long_frac↔PnL 상관계수
  **0.888→0.918로 오히려 강화**(OOS 구간 자체가 불변이므로 당연) — zigzag가 나빠진 이유는
  2024 포함 후 평균 long_frac이 0.28로 더 숏편향돼서(하필 강세장인 이 OOS와 더 어긋남)임을
  확인. **결론: 2024 추가는 "라벨 품질 개선"으로 작동하지 않았다** — 학습데이터량과 무관하게
  이 OOS 구간의 근본 문제(단일방향 7주)가 결과를 지배한다는 게 다시 확증됨(위 2025-전체
  진단, 상관계수 0.221과 종합). 최종판정 불변.

  **⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️ "왜 2024 포함이 zigzag를 숏쪽으로 끌었나" 메커니즘 분석 (같은 날 9차,
  사용자 질문 "24 데이터 포함이 왜 이유가 됐는지")**: 순진한 "2024 라벨 자체가 더
  숏편향"이라는 가설을 실측으로 직접 검증·**기각**. zigzag/cusum/h48_conservative 3개
  라벨소스 전부 2024 vs 2025 LONG:SHORT 클래스비율 차이가 0.3%p 이내(예: zigzag 53.15:46.85
  vs 52.98:47.02) — 스윙/피벗 기반 라벨은 그 해가 순추세로 오르든 내리든 거의 대칭으로
  잡히기 때문. 실제 가격은 2024 +45.71%(강세) vs 2025 −11.05%(약세)로 정반대였지만 5분봉
  상승/하락 비율은 두 해 다 50.2~50.3%로 거의 동일 — "강세장 데이터를 넣었으니 롱 예측이
  늘 것"이라는 반대 방향 가설도 성립 안 함(실제로는 반대로 감). 레짐v2(HMM chop/bull/bear)
  구성도 2024(chop37/bear33/bull30)·2025(bear35/chop33/bull32)·OOS(chop44/bear28/bull28)로
  뒤섞여있어 숏 쏠림 방향을 설명하지 못함.

  **결정적 단서**: zigzag와 h48qual은 direction_label_dir가 **완전히 동일한 zigzag 라벨**인데
  (h48qual=zigzag방향+h48_conservative 품질게이트), 2024 포함 후 OOS 시드별 long_frac을
  재계산하니 zigzag 평균 **0.2835**(6시드중 5개 long_frac≤0.5, 1/6승) vs h48qual 평균
  **0.6942**(6시드중 5개 long_frac≥0.6, 5/6승)로 극단적으로 갈렸다(cusum은 중립 0.4773,
  3/6승 불변). 품질게이트의 LONG/SHORT측 활성화율은 양쪽 다 60~63%로 비슷해 이 차이를
  설명 못함(교차표 확인). 남는 구조적 차이는 h48qual만 h48_conservative를 **별도의
  3-클래스 학습타겟**(`quality_mode=quality_label_action`)으로 공동학습한다는 점뿐
  (zigzag/cusum은 `same_as_direction`이라 사실상 단일 타겟) — 이 보조태스크가 공유 백본의
  그래디언트를 다르게 끌면서 같은 방향라벨이라도 수렴점(=시드별 우연 편향)이 달라졌을
  가능성이 가장 유력한 설명.

  **결론**: 결정론적 인과사슬(라벨 불균형, 가격추세, 레짐구성 전부 기각/불충분)은
  실측으로 뒷받침되지 않는다. 클래스비율이 거의 안 바뀌었는데 학습결과(시드별 방향편향)가
  이렇게 요동친다는 사실 자체가, 오늘의 핵심결론("라벨 승패=학습된 방향스킬 아니라 우연한
  시드편향×교란변수")을 약화시키는 게 아니라 **한 번 더 재확증**한다 — 진짜 스킬이라면
  입력 라벨 분포가 거의 그대로인데 이렇게 극단적으로 뒤집히지 않는다. 최종판정 불변.
- **(n) 신규 — 라벨 설정 git 미커밋** — **해소(2026-08-21)**: `LABEL_CONFIGS`의
  `balancedish_adx16_slope03_bb006` 항목을 커밋함(`313e481`).

## peg-maker 집행 인프라 계약 — 2026-08-22 (RDE와 독립된 재사용 자산)

**⚠️ 이 절이 다루는 것은 "방향을 어떻게 정할지"가 아니라 "정해진 방향을 얼마에 체결할지"다
— 완전히 다른 축이다.** RDE(레짐 직결 노출) 정책은 위 절에서 OOS 확정 REJECTED됐지만, 그
실험 안에서 검증된 **집행(execution) 방법론 자체는 REJECTED되지 않았다**. 레짐 분류기가
위 별도 절로 독립됐던 것과 같은 이유로, 이 절도 독립시킨다 — **어떤 방향 신호를 쓰든(오늘의
RDE든, 미래의 raw LOB/OFI 신호든, 지금 라이브 중인 Omega 4.6.1이든) 그대로 재사용 가능한
인프라**이기 때문이다.

### 핵심 주장

라이브/섀도우 시스템은 현재 모든 진입·청산을 **taker**(즉시 시장가 체결)로 실행하고
있다고 가정하고 비용을 산정해왔다(리포 관례 가정 7bp/leg). 실측 결과 이 가정 자체가
과대했고(**taker 실측 5.0bp/leg** — 이 창의 스프레드가 1틱=0.05bp뿐이라 슬리피지 가정이
컸음), 여기에 **maker(지정가+추격 재호가) 집행으로 전환하면 leg당 추가로 1.9~1.8bp를
더 아낄 수 있다** — 신호가 뭐든, 거래마다 이 절감이 그대로 적용된다.

### 검증 이력 (4단계, 전부 완료)

| 단계 | 방법 | 결과 |
|---|---|---|
| 1. raw L2 시뮬(v1) | WS-E 격리 파일럿(연속 10초, 53h) + aggTrades, 보수적 체결규칙(뚫는체결/큐소진/호가크로스만 인정) | **maker 3.1bp/leg**(peg, 체결률 99.5%+), taker 실측 5.0bp/leg |
| 2. 고변동 재계측(v2) | bookTicker 2023년 아카이브 중단 확인 → aggTrades 단독 재구성(진실L2 대비 89% 정확일치 검증) 신작으로 2026H1 극단일(폭락 −14.8%/반등 +13.9%) 재측정 | 비용 커브 **평평**(3.1→3.7~3.8bp) — 고변동일엔 체결이 빨라져 폴백이 사라지고 peg가 drift를 한 리페그 주기로 캡핑하기 때문 |
| 3. 레이어 통합(VAL) | RDE의 실제 142개 포지션 전환 timestamp마다 개별 체결 시뮬 (평균값 일괄적용이 아니라 전환 하나하나) | **평균 3.20bp/leg**(flat 근사치 3.1bp와 거의 일치 — 근사가 요행이 아니었음 확인), N=5 시드 net PnL 표준편차 0.06%p |
| **4. 레이어 통합(OOS, 미터치 데이터)** | 같은 방법을 09-30 조기실행 override로 확보한 **진짜 아웃오브샘플** 창(2026-07-01~08-19)의 83개 실제 전환에 적용 | **평균 3.47~3.59bp/leg, N=5 시드 전부 이 구간(0.12bp 이내) — 비용모델이 VAL이 아니라 미터치 데이터에서도 예측 밴드(3.5~4.0bp) 안에 정확히 들어맞았다** |

**4번이 이 절에서 가장 중요한 사실이다**: RDE라는 방향전략은 이 OOS에서 죽었지만, 그
안에서 같이 평가된 비용모델은 **미터치 데이터로 재검증된 예측**이라는, 이 리포에서 흔치
않은 지위를 얻었다. 이 리포의 압도적 패턴은 "OOS에서 뭐든 뒤집힌다"인데, 이 비용모델은
그 패턴에서 예외였다.

**⚠️ 2026-08-23 플래그**: 4번 실험이 쓴 83개 전환 timestamp는 이후(같은 날) 발견된 OOS
창 metrics 미래참조 결함(아래 데이터 무결성 절 참고) 수정 전 오버레이로 뽑힌 것이라, 결함
수정 후 재계산하면 구체적 전환 시각 집합이 달라질 수 있다. 비용모델 자체(오더북 물리적
체결 특성)는 레짐 신호와 무관해 결론이 바뀔 가능성은 낮지만, 공식 재확인은 아직 안 함 —
후속 확인 대상으로만 기록.

### 라이브 검증 (진행 중)

`scripts/maker_fill_shadow_worker.py`(실주문 없음, 공개 WS만, trading_bot 무접촉) +
`scripts/ops/supervisor_maker_fill_shadow.sh`(crontab @reboot 등록) — 서버에서 5분마다
peg/static 가상 leg를 실제 라이브 시장에 굴려 `data/live/maker_fill_shadow.duckdb`에 기록
중. 배포 직후 첫 legs 1.98~3.38bp로 시뮬 밴드 정합 확인됨. **체크포인트: 9월 중순**(3주
축적 후) `scripts/analyze_eth_maker_fill_shadow_vs_sim_20260822.py`로 실측 대조.

### 정량 요약 (권장 비용 가정, 어떤 전략이든 적용 가능)

| 집행 방식 | 비용 |
|---|---:|
| taker(기존 리포 가정) | 7.0bp/leg |
| taker(실측) | 5.0bp/leg |
| **maker peg(권장 기본)** | **3.5bp/leg** |
| maker peg(스트레스) | 4.0bp/leg |
| maker peg(tail, p90~p95) | 7.5bp/leg |

### 다음 단계 — 라이브 반영 (별도 승인 필요)

가상 주문에서 실주문으로 넘어가려면 `trading_bot.py`/`trading_bot_modules/*`를 건드려야
하므로 일리아스 서브프로젝트의 "라이브 파일 무변경 원칙" 밖이다 — **명시적 승인 시
착수**. 제안 순서: (1) 섀도우 실측이 9월 중순 시뮬 밴드를 통과하는지 먼저 확인, (2) 통과 시
현재 라이브 중인 전략(Omega 4.6.1 계열 등, RDE 여부와 무관)의 집행 경로에 peg-maker를
이식해 실체결로 최종 검증, (3) 검증되면 배포. 자기 주문의 큐 점유·시장영향은 가상 주문
시뮬로는 끝까지 확인 불가 — 이 지점만 실주문 소액 테스트가 필요하다.

### 한계 (정직성)

- 자기 주문의 시장영향/큐 점유는 가상 leg로 측정 불가(소액 가정으로 무시해왔음).
- OOS 재검증(4단계)의 창은 다른 축(원래 RDE 판정, maker sim pilot)에서 이미 조회된 적
  있는 창이라 "완전 처음 보는 창"은 아니다 — 그럼에도 이 특정 비용모델 자체를 이 창에서
  최초로 미터치 검증했다는 점은 유효하다.
- 라이브 섀도우는 아직 3주 미만 축적 — 9월 중순 체크포인트 전까지는 시뮬 기반 수치가
  가장 신뢰할 수 있는 근거다.

전체 원본 실험 기록: `docs/experiments/eth_maker_fill_simulation_l2_20260822.md`.

### 후속 — 과거 비용-기각 신호 전수 재심 (2026-08-23, 0/6 부활)

이 비용모델(leg당 7→3.5bp)이 과거 "신호 실재하나 비용 미달" 기각들을 되살리는지 전수
재심 → **6축 전부 기각 유지**. 최근접인 20-23 UTC 세션 엣지(5.49bp/거래)조차 maker 왕복
6.2~7.0bp에 미달 + 비용 무관 강건성 결함(TRAIN 음수, 월클러스터 t=1.14). 펀딩 델타뉴트럴
캐리는 비용 산술을 신규 실측으로 확정(2026Q2 상한 69bp/분기 < 퍼프-only 비용 217bp) —
maker와 무관하게 펀딩 레짐 붕괴가 사인. **maker 인프라의 가치는 기존 신호 부활이 아니라
미래 신호의 손익분기 여유 2배 확보로 확정.** 상세:
`docs/experiments/eth_maker_breakeven_rescreen_20260823.md`.

## 라벨 퓨전(3라벨 결합모델) 연구 — 2026-08-21 (연구 완료, 실행 없음)

**질문**: zigzag/h48qual/cusum 3개 라벨을 종합해서 하나의 모델로 만들 수 있는가. 전체 기록:
`docs/experiments/ilias_eth_label_fusion_combined_model_research_20260821.md`.

**한 줄 결론**: 3개를 앙상블/합의로 결합해도 이미 확정된 공통 교란변수("시드별 우연한
롱/숏 편향 × 우연한 구간 추세")는 이론적으로도 실증적으로도 제거되지 않는다. Bates &
Granger(1969)/Krogh & Vedelsby(1994)의 고전 결과가 예측하는 대로, zigzag/h48qual이 동일
direction 소스를 공유(방향일치율 81.7%)해 오류상관이 구조적으로 높은 이 저장소 상황에서는
결합의 분산감소 효과가 무력화된다. López de Prado식 메타라벨링(합의필터)은 "오류평균화"가
아니라 "노출축소"라는 다른 축이라 같은 논리가 그대로 적용되진 않지만, 실증에서도 거래수만
40~77% 줄었을 뿐 PnL 개선은 확인되지 않았다.

**저비용 feasibility 체크**(신규 학습 없음, 저장된 18개 run 예측 재사용, TRAIN+VAL구간
2024-2025만 사용, OOS 미접촉 — `scripts/eth_ilias_label_fusion_train_period_feasibility_20260821.py`):
다수결(vote)은 h48qual 단독의 강한 long_frac↔PnL 상관(0.609)을 −0.108로 부분 해소하는 약한
긍정 신호를 보였으나 표준편차는 줄지 않았다(35.12, solo_cusum 다음으로 큼). 합의필터
(consensus)는 거래수는 줄였으나(−20.1% MDD로 개선) PnL은 최우수 개별 라벨(cusum 29.23%)보다
낮았다(17.10%). vote의 평균 PnL(35.73%)이 표면상 always-long 벤치마크(29.72%)를 넘지만,
N=6의 넓은 분산과 기존에 확정된 N=918 풀링 결론(승률 36~40%, 장기상관 0.221)에 비춰
"결합이 alpha를 만들었다"로 읽지 않는다.

**추천(실행 미승인 → 2026-08-22 실제 테스트 후 철회)**: 옵션1(예측레벨 앙상블, zigzag+cusum만
독립소스로 취급하고 h48qual은 품질축으로 별도 사용)을 저비용·이론상 상한 계산 가능이라는
이유로 낮은 우선순위 후속으로 제안했었다. 옵션2(멀티태스크 3-head)는 공유 트렁크가 교란변수를
3개 head에 동시 주입할 이론적 위험이 있어 비권장(불변).

**⚠️⚠️ 2026-08-22 후속 — 정식 스태킹 메타모델 실제 테스트, 옵션1 추천 철회**: 사용자 승인 후
로지스틱회귀 메타모델(피처: zigzag/cusum 방향신호 + h48qual 품질점수, 타겟: 순방향48bar
수익률 부호, 2024로 학습→2025 held-out 평가, OOS 미접촉)을 실제로 학습·테스트했다
(`scripts/eth_ilias_label_fusion_stacking_meta_model_20260822.py`). **결과: 결합가중치가
단일 학습구간(2024, +45.71% 강세장)의 추세를 흡수해 평가구간(2025)에서 거의 상시-LONG으로
붕괴**(평균 long_frac 0.979, 6시드중 4개가 정확히 1.000) — 표준화계수 분해 결과 zigzag
자체 신호 기여도는 12.5%뿐이고 cusum(39.3%)·h48qual품질(48.1%)에 쏠림. 성능(평균PnL −0.36%)도
6개 태그 중 최하위권(vote+12.17%/consensus+14.86%/h48qual단독+12.87%보다 낮음). 문헌(Bates-
Granger/Krogh-Vedelsby)이 예측한 "오류상관 높으면 결합 무력화"를 hand-picked 규칙(vote/
consensus)보다 더 직접적으로 재확증 — **옵션1 추천을 철회하고, 3라벨 예측레벨 결합 탐색
전체를 종결한다.** 전체 기록:
`docs/experiments/ilias_eth_label_fusion_combined_model_research_20260821.md` §6.
남은 우선순위는 결합모델이 아니라 "데이터 Split 재설계 제안" 절(앵커드 walk-forward+
time-decay)로 근본원인(단일구간 지배)을 직접 다루는 쪽.

**⚠️⚠️⚠️ 2026-08-22 3차 후속 — 최신 문헌 2편(Zou 2025, Felici & Sudoso 2023) 실제 구현·
테스트, 더 근본적인 원인 확인**: 사용자가 "최신 논문 없어?" → 문헌 4편 확인 후 "제대로
해보자"고 재지시해, 적용 가능하다고 본 2편을 이 저장소 구조에 맞게 실제 구현했다(Lee & Lee
2023은 "예측기 수≫표본" 전제라 예측기 2~3개인 우리와 수학적 스코프 자체가 안 맞아 실행
안 함). **Test A**(Zou/NCL 정신 — zigzag/cusum/h48qual을 완전 독립 소형모델 3개로 두고
개별정확도+모델간 예측분산을 공동손실로 최적화, λ=0 대조군 vs λ=0.5): 개별 모델 3개 전부
BCE가 0.6918~0.6926으로 수렴했는데, 이는 **피처 없이 2024 기본상승비율(51.40%)만 예측하는
절편전용 모델의 이론적 최저 BCE(0.6928)와 사실상 동일** — zigzag/cusum/h48qual 각자의
방향확률·신뢰도·품질점수 어느 것도 순방향48bar수익률에 대해 선형회수 가능한 정보를 거의
안 가진다는 뜻. **Test B**(Felici & Sudoso 정신 — 3피처를 12개로 넓히고 정확도+다양성
동시만족 그리디선택, 선택절차는 2024 내부에서만 완결): 평균 1.5개만 선택되고 대부분
h48qual 품질점수 재발견, 성능(−0.34%)도 기존 stack(−0.36%)과 동일. **결론: §6 stack의
극단적 long_frac(0.979)도 진짜 학습이 아니라 거의 0에 가까운 계수가 유한샘플 노이즈로
threshold를 우연히 넘은 것으로 재해석**된다 — 결합방법이 무엇이든(규칙/학습형/다양성규제/
다양성선택) 문제가 안 되는 건 입력신호 자체에 회수 가능한 정보가 없기 때문임이 정보이론적으로
확증됨. 종결 판단 불변, 근거는 더 근본적으로 강화. 전체 기록: 위 문서 §8.

**⚠️ 산출물 신뢰성 경고**: `tmp/ilias_labellogic_recheck_20260821/train_period_trade_ledgers/
summary.csv`는 2024 데이터 학습 반영 이전 시점의 stale 산출물이다 — 같은 날 늦은 단계(2024
포함 재학습)가 같은 출력 경로의 예측 파일을 덮어써서 더 이상 재현되지 않는다(현재
`train_predictions_qXXX.csv`는 210,481행=2024-01~2025-12 전체, 그 summary.csv는 78,605행
시절 산출물). 후속 세션은 이 파일을 인용하지 말 것 — 필요하면 위 실험 문서의 스크립트를
재실행한다.

## 데이터 Split 재설계 제안 — 2026-08-20 리서치 결론 (2026-08-22 확정·적용됨)

**상태**: `확정·적용됨(2026-08-22, 사용자 지시)` — 위 "## Dataset Split" 절이 이제 이 방법론을
**이 프로젝트(본 모델 포함) 유일 규약**으로 명시한다. 2026-08-20 작성 시점엔 "제안됨, 본
모델(TabM/오디세이 계열 최종 결정 모델)엔 미실행" 상태였고 레짐 분류기(wide24 HMM) 축에만
먼저 적용돼 있었으나(2026-08-21, N=5 CONFIRMED — 상세는 "레짐 분류기 계약" 절의 "재스윕 —
앵커드 Walk-Forward 적용" 참고), 2026-08-22에 여러 서브프로젝트가 서로 다른 날짜범위로
학습하는 문제를 근본적으로 없애기 위해 **이 프로젝트 전체(신규 154피쳐 트랜스포머 스모크테스트
포함)의 유일 규약으로 승격**됐다. 재학습·purge구현·ledger신설은 본 모델(h48qual/zig075 TabM
번들) 기준으로는 여전히 미실행 — 그 번들은 상속된 체크포인트라 재학습 계획이 없는 한 이
규약이 적용될 대상 자체가 아직 없다(적용 대상이 생기면 이 규약을 그대로 씀).

**두 번째 적용 사례(2026-08-22) — DC154피쳐 tabular 트랜스포머 스모크테스트**: 레짐 분류기와
동일한 종류의 예외(OOS를 09-30 대기 없이 즉시가용 데이터로 조기실행)를 사용자가 다시 명시적
지시 — `scripts/eth_candidate_dc154feat_tabular_transformer_smoke_test_20260822.py`,
[[eth_candidate_lob_ofi_pipeline_smoke_test_20260822]] 계열 문서 참고. "09-30까지 대기"가
반복적으로 override되고 있다는 점 자체를 다음 세션이 인지할 것 — 매번 개별 예외로 기록되고
있으나, 누적되면 이 원칙을 사실상 상시 규약으로 재정의해야 할 수도 있다.

전체 기록: `docs/eth_recency_weighted_walkforward_data_split_literature_review_20260820.md`.

**배경**: 고정 VAL/OOS-Q1/OOS-Q2 캘린더 스플릿에서는 라이브에 가장 가까운 최근 데이터가 영원히
학습에 안 쓰이고 OOS로만 소비된다는 문제제기(사용자, 2026-08-20) → 데이터완전성감사(Jan2024~
Jun2026 캐노니컬 ETH 피쳐, 결측 47/262,656=0.018%, 사실상 완전) + 외부문헌 6개축 조사(74회
조사, DOI/arXiv-ID 상호검증) 수행.

**핵심 발견 — OOS-Q1/Q2는 이미 다회 소진됨**: 메모리기록만으로도 최소 10회 이상의 독립 시도
(오디세이4 G0/일리아스1/일리아스1 dual N=5/zig075단독 N=5/ETH·BTC·SOL라이브 N=3~5×3/섀도우
풀리시드 N=6/섀도우exit_head N=5/veto+guard이식 N=3/5-way N=3 등, 각 N=3~6시드)가 같은
OOS-Q1/Q2를 반복 조회했다 — Bailey et al.(2014)의 "재시도 횟수가 늘수록 백테스트 증거가치가
준다"는 정확히 이 상황이며, `docs/experiments/`에 기록된 라이브 DSR=0.915(선 0.95 미달) 첫
FAIL이 이 누적 재시도비용이 수치로 드러난 사례로 재해석된다. 즉 OOS-Q1/Q2는 더 이상 "신선한
미터치 창"이 아니다.

**제안 방법: 분기 앵커드(확장) Walk-Forward + 경계 Purge/Embargo + TRAIN 내부 Time-Decay 가중.**

| 구성요소 | 내용 |
|---|---|
| TRAIN | Anchored(확장) — 시작점 고정, 매 세대 끝점만 전진. Rolling(고정 lookback)은 채택 안 함 — lookback 길이라는 임의결정이 늘고, time-decay가 같은 문제(오래된 레짐 희석)를 split 안 건드리고 해결 |
| 경계 처리 | forward-looking 라벨(DC/CUSUM류 triple-barrier)의 라벨윈도우가 VAL/OOS 경계를 넘는 학습샘플 purge + embargo. 이 저장소 라벨빌더가 이미 이걸 하는지는 미확인 |
| TRAIN 내부 가중 | time-decay(López de Prado AFML Ch.4) — 오래된 데이터를 버리지 않되 최근 bar가 그래디언트를 지배. 기존 고정split 위에서 격리 프로토타입 가능(split 자체 안 건드리는 최저위험 레버) |
| OOS 세대교체 | 매 세대 단 하나의, 그 세대 기준 진짜 미터치 분기만 — 사용 즉시 소진 기록, 다음 세대부터 TRAIN 편입, 재사용 금지. 신규 OOS 사용이력 ledger 필요(지금 이게 없어서 위 10회+ 재사용이 발생) |
| CPCV | **프로모션 증거에서 배제**. 룰 조문(원장재사용금지/미래row조인금지)이 아니라 룰의 목적("원장 대신 라이브처럼 매매결정을 평가")과 충돌 — 조합경로 대다수가 test그룹보다 미래인 그룹을 학습에 쓰는데, 이 시스템은 비온라인학습(고정 가중치 배포)이라 그 순간 존재할 수 없었던 모델을 시뮬레이션하는 셈. "라이브처럼 되는 경로만 쓰는 CPCV"는 수학적으로 walk-forward와 같아짐. Research-stage(아키텍처/HP 스크리닝) 전용 옵션으로만 남김, Seed-Diversity 게이트처럼 프로모션 증거와 분리 |

**적용 순서(실행 시)**:
1. (즉시 가능) Q1/Q2를 TRAIN으로 편입 → 다음 세대 TRAIN ~2026-06-30. OOS 사용이력 ledger
   신설, purge/embargo 구현 확인, time-decay 가중 격리 프로토타입.
2. (2026-09-30까지 대기) 완전한 새 OOS 분기 2026-07-01~09-30 확정 후 그 세대의 single-touch
   프로모션 판정 — 3주치 부분데이터 조기체크는 하지 않는다(그 자체가 또 하나의 "조회"로
   집계될 위험).
3. 이후 매 분기 1~2 반복 — staleness가 최대 9개월 → 최대 1분기로 캡됨.

**위 "일리아스 라벨로직 후보축"과의 관계**: 그 축이 구축한 2024-01~2026-06-30 154피쳐
데이터셋(위 데이터 계약 절, 독립 재검증 완료)은 이 제안과 방향은 비슷하나(더 긴 TRAIN 활용)
목적이 다른 별개 축이다 — zigzag/h48qual/cusum 라벨로직 비교가 목적이지 프로덕션 split
재구성이 목적이 아니며, 아직 서로 연결되지 않았다. 두 축을 언제/어떻게 합칠지는 Open Issue
(i)와 함께 다음 세션 결정 사항.

**장기 고려사항(온라인학습, 미착수)**: 이 앵커드 walk-forward 구조는 재학습주기를 점점 줄여가는
축소판으로 볼 수 있어, 나중에 온라인학습으로 전환해도 재작업이 최소화된다. 단 실제 전환 시
고려할 것: (a) 이 시스템의 라벨(TP/SL/time-exit)은 트레이드 청산 시점에야 확정되므로 "매
bar마다 학습"이 아니라 트레이드 청산마다 반영하는 형태가 현실적, (b) 표준 평가 프로토콜
용어는 concept-drift/스트리밍러닝 문헌의 prequential(test-then-train) evaluation(일반지식
수준 인용, 별도 검증 필요), (c) 모델 가중치 업데이트(온라인학습 본연)와 사람이 개입하는
threshold/하이퍼파라미터 재튜닝(반복시도 위험 재발)은 구조적으로 분리해야 함.
