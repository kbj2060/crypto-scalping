# ETH Omega4.6.1 라이브 exit_head — 라이브 ATR 배리어 재라벨 재학습 (2026-08-13)

상태: `research_only_not_live_promoted` — 결론: **비대칭 채택(h48qual만 교체)이 포트폴리오
레벨 VAL·OOS 모두에서 baseline 대비 PnL·MDD 개선으로 확인됨. 단, OOS 절대 수치는
`quality_threshold` 선택편향 오염으로 "깨끗한 검증"이 아니라는 유보가 붙는다(상대비교는 유효).**
컴포넌트 단독 평가에서는 h48qual VAL PnL·MDD 둘 다 개선, zig075는 둘 다 악화하는 혼재된 결과가
나왔다. 오케스트레이터가 "h48qual=새 exit head, zig075=원래(냉동) exit head"로 비대칭 채택을
결정했고, 실제 라이브가 쓰는 단일계좌·우선순위 포트폴리오 결합 로직으로 재검증한 결과 VAL
PnL/MDD 둘 다 개선(+36.82%→+46.59%, -24.34%→-21.70%)을 확인했다. 오케스트레이터 승인 하에
1회 한정 OOS 확인도 같은 방향(+49.32%→+93.27%, -16.20%→-15.48%)을 재현했다 — 다만 baseline과
후보가 공유하는 `quality_threshold` 자체가 이 OOS 구간(특히 2026-01~02월)에 최적화되어
선택됐다는 사실이 별도 조사로 확인돼, 절대 OOS 수치를 과대해석하지 않도록 유보를 명시한다.
상세는 문서 하단 "후속 2·3" 절 참고. 최종 승격 여부는 오케스트레이터가 판단한다.

**선행 문서(반드시 먼저 읽을 것)**: `docs/experiments/eth_omega461_live_exit_head_h48cons_relabel_20260813.md`
(h48_conservative 배리어를 쓴 1차 시도, VAL PnL이 두 컴포넌트 모두 음전환하며 명확히 실패).
이 문서는 그 실패의 근본원인 진단("h48_conservative 배리어가 라이브 실제 ATR 배리어보다
~10~12배 타이트해서 학습-추론 기간 스케일 불일치")을 그대로 이어받아, 배리어 소스만 라이브
실제 ATR-adaptive 공식으로 교체한 후속 실험이다.

라이브 어댑터(`trading_bot_modules/omega4_6_1_live.py`), `trading_bot.py`, `runtime_config.py`,
`.env`, 프로덕션 SLTP/exit_head 번들은 전혀 건드리지 않았다. `fresh_forward_bar_by_bar=true`,
`trade_ledgers_used_as_input=false`, `saved_parent_exit_timestamps_used=false`,
`future_rows_used_for_entry=false`. VAL만 실행했고 OOS(2026-01-01~03-31)는 이번에도 전혀
로딩·평가하지 않았다.

## 방법

### 배리어 소스 교체

새 스크립트 `scripts/research_eth_omega461_exit_head_liveatr_relabel_20260813.py`에 새 함수
`_build_exit_dataset_entry_label_live_atr_barrier`를 작성했다(h48cons 스크립트의 함수도, 원본
`_build_exit_dataset_entry_label_terminal_giveback`도 수정하지 않음). h48cons 시도와 동일한
"매 `zigzag_action` bar를 독립 진입 후보로 놓고, 그 후보 자신의 배리어 해소 지점까지 시뮬레이션"
구조를 그대로 유지하되, 배리어 폭 계산만 h48_conservative CSV 조회 대신 **라이브가 실제로 쓰는
공식**으로 바꿨다:

- `scripts/eval_omega4_1_atr_safety_sltp_20260622.py`의 `_atr_pct`/`_apply_atr_safety_sltp`를
  그대로 import해서 사용(재구현 안 함) — `atr_window=192, tp_mult=12.0, sl_mult=6.0,
  min_tp=0.075, min_sl=0.040, max_tp=0.22, max_sl=0.12`, `trading_bot_modules/omega4_6_1_live.py`의
  `_ComponentConfig` 기본값과 동일(라이브 모듈 자체는 import하지 않고 값만 확인 후 하드코딩 —
  `research_eth_omega461_exit_sweep_20260721.py`의 `COMPONENTS`도 같은 값을 독립적으로 하드코딩해
  두고 있어 상호 대조 확인됨).
- 각 후보의 진입 bar에서 causal `atr_pct`로 tp/sl 가격폭을 계산한 뒤, 그 폭으로 매 bar
  intrabar high/low를 검사해(SL 우선 동시-bar 타이브레이크, 원본 관례와 동일) 배리어 해소
  bar를 직접 찾는다 — h48_conservative처럼 사전계산된 CSV가 없으므로 시뮬레이션 필요.
  horizon cap은 6,000bar(전체 후보군 기준 실제 타임아웃 비율 2.2%로 확인, 대부분 그 전에
  tp/sl로 해소됨).
- 후보 밀도: 라이브 ATR 배리어의 중앙값 해소 기간(~600~850bar)이 h48_conservative(~9~10bar)보다
  훨씬 길어서, 전체 후보(37,245개, zigzag_action bar 전부) 사용 시 예상 행 수가 ~3,150만 행에
  달해 로컬 CPU로 감당 불가능했다. 시드 고정 랜덤 서브샘플 1,500개 후보(원본 레시피의 732~813
  세그먼트 대비 여전히 1.85~2.05배 많고, 기간 스케일은 올바름)를 썼다 — 전체 모수 대신
  서브샘플을 쓴다는 사실을 축소하지 않고 명시한다.

포지션 피처 생성(`exit_head._position_feature_row`)과 `terminal_window=3`,
`adverse_unreal=-0.010`, `min_mfe_for_giveback=0.006`, `giveback_min=0.65` 분기 로직은 h48cons
시도와 동일하게 유지(원본 기본값 그대로) — 바뀐 건 배리어 폭 계산 방식과 후보 밀도뿐이다.

### 재학습/평가 도구

h48cons 스크립트와 동일하게 `train_eval_omega4_1_exit_head_price_move_sltp_retrain_20260622.py`의
`_fit_exit_head_only`를 그대로 재사용(encoder/direction_head/quality_head 동결, exit_head만
재학습)했고, VAL 평가는 `research_eth_omega461_exit_head_h48cons_relabel_20260813.py`의
`_evaluate_val`(내부적으로 `research_eth_omega461_exit_sweep_20260721.py`의
`prep_component`/`replay_exit_variant` 재사용, `EXIT_THRESHOLD=0.95` 고정)을 **모듈 import해서
직접 호출**했다 — 코드를 복붙하지 않고 이미 검증된 함수를 그대로 재사용.

## 중대한 운영 사고와 원인 수정 — 서버 다운

**중요: 아래 서술 중 서버 장애의 정확한 원인(디스크/메모리 알림 여부 등 세부 진단명)은
오케스트레이터가 전달한 정보이고, 이 세션이 직접 확인한 사실은 (1) 동일 지점에서 3회 연속
크래시, (2) 크래시 시점 전후로 서버가 `uptime` 기준 재부팅됐음, (3) 크래시 직전 마지막 로그
줄이 항상 `candidates_sampled=1500/37245` 직후였고 파이썬 트레이스백이 전혀 없었다는 점이다 —
트레이스백 부재는 예외가 아니라 강제 종료(OOM 등)를 시사한다.**

1,500개 전체 규모로 dev(GPU 없음)에서 2회, 서버(GPU 있음이지만 이 병목은 CPU/메모리 문제)에서
1회, 총 3회 연속으로 **정확히 동일한 지점**(`build_live_atr_barrier_exit_dataset` 시작 직후)에서
프로세스가 죽었다. 원인을 직접 진단: `exit_head._position_feature_row`가 행마다 약 290개
키(state.columns ~200개의 `cur_` prefix + entry/drift 열 ~80~100개 + `pos_` 13개)를 가진
딕셔너리를 만드는데, f-string으로 생성되는 키 문자열은 자동 인터닝되지 않아 raw 파이썬 리스트로
누적하면 후보 1,500개 × 평균 ~845bar/후보 ≈ 127만 행 전체를 메모리에 들고 있어야 했다(h48cons
시도의 54만 행보다 훨씬 크고, 그마저도 이미 청산 없이 한 번에 DataFrame 변환하는 구조였음).

**수정**(`_build_exit_dataset_entry_label_live_atr_barrier` 내부에만 적용, 공유 모듈인
`_position_feature_row`는 손대지 않음):

1. **청크 분할**: `CHUNK_SIZE=20,000`행마다 raw dict 리스트를 DataFrame으로 변환해 버리고
   비움(`_flush_chunk`), 끝에 `pd.concat`. transient 메모리를 컷 수와 무관하게 상한.
2. **메모리 안전장치**: 20개 후보 후보마다 `/proc/meminfo`의 `MemAvailable`과 자기 프로세스의
   `/proc/self/status` `VmRSS`를 확인, `available_gb < 3.0` 또는 `rss_gb > 6.0`이면 `--max-rows`를
   친 것과 동일하게 그 시점까지 모은 데이터로 우아하게 조기 종료(크래시 대신).

**검증**: 로컬 800후보(65.8만 행) 전 구간에서 RSS 1.2~3.7GB로 완전히 bounded(더 이상 후보 수에
비례해 무한정 증가하지 않음, `stopped_for_memory=False`로 정상 종료) 확인 후, 서버에서 300후보
소규모 검증(23.3만 행, RSS 최대 2.02GB, 이 스크립트가 서버에서 최초로 크래시 지점을 통과 —
`build loop done ... stopped_for_memory=False`) → 재학습·VAL평가까지 전체 파이프라인 정상 완주
확인 → 그 다음에야 원래 목표 규모인 1,500후보로 재실행했고, 123.4만 행 전체를 `stopped_for_memory=False`로
완주(피크 RSS 5.49GB, 서버 전체 메모리 31GB 중 19.58GB 여유, 안전장치 문턱을 전혀 건드리지
않음)했다. **"작은 규모로 먼저 안정성을 확인한 뒤에만 원래 규모로 진행"** 원칙을 지켰다.

## 0단계 체크포인트 — 배리어 기간 스케일

`_fast_timescale_checkpoint`(feature-row 생성 없이 배리어 해소 bar 수만 빠르게 계산, 학습 전
필수 게이트)로 전체 37,245개 후보 모집단에서 확인:

| | h48cons(실패한 선행 시도) | 이번(라이브 ATR) | 실제 라이브 baseline |
|---|---:|---:|---:|
| 해소기간 중앙값(long/short) | 10 / 9 bar | **622 / 595 bar** | - |
| 해소기간 평균(long/short) | 15.1 / 13.9 bar | 834.9 / 858.5 bar | - |
| 실측 평균 보유기간(h48qual/zig075) | - | - | 670.3 / 725.6 bar |
| horizon(6,000bar) 타임아웃 비율 | - | 2.22%(827/37,245) | - |

게이트 기준(중앙값 ≥30bar, 두 side 모두)을 명확히 통과했고, 자릿수 자체가 실제 라이브 평균
보유기간과 일치한다 — h48cons의 9~10bar와 대비되는 핵심 개선.

## 라벨 밀도/다양성 (전체 1,500후보, 123.4만 행)

| | h48cons(실패) | 이번(라이브 ATR) |
|---|---:|---:|
| 후보 수 | 37,158(전체 모집단 사용) | 1,500(서브샘플, 원본 대비 1.85~2.05배) |
| 행 수 | 540,088 | 1,234,431 |
| 양성 비율 | 19.64% | 19.90% |
| `near_barrier_resolution_exit` | 106,058(양성의 100%) | 4,500(양성의 1.83%) |
| `mfe_giveback_exit` | 0 | **185,788(양성의 75.6%)** |
| `adverse_unreal_exit` | 0 | **55,312(양성의 22.5%)** |

이게 이번 시도의 핵심 성과다 — 원본 라이브 레시피가 30,000행 중 단 3건만 가졌던
`mfe_giveback_exit`가, 그리고 h48cons도 0건이었던 `adverse_unreal_exit`가 처음으로 실질적인
표본 수를 확보했다. "지그재그 피벗 임박" 신호가 지배하던 구조에서 "실제 이익반납/손실확대
감지" 신호가 지배하는 구조로 완전히 뒤바뀌었다.

## VAL 평가 결과 (전체 1,500후보 스케일)

`EXIT_THRESHOLD=0.95` 고정, VAL=2025-10-01~12-31.

### h48qual (q050) — 개선

| | baseline | 라이브ATR 재라벨 |
|---|---:|---:|
| PnL | +5.45% | **+9.23%** |
| MDD | -11.62% | **-7.59%** |
| trades | 29 | 63 |
| WR | 41.4% | 30.2% |
| avg_hold_bars | 670.3 | 210.8 |
| exit_reasons | `stop_loss:17, take_profit:11, forced_end:1` | `exit_head:52, take_profit:8, stop_loss:3` |

PnL·MDD 둘 다 개선됐다. exit_head가 63건 중 52건(82.5%)을 주도하지만, 평균 보유기간이 여전히
210.8bar(baseline의 31%)로 h48cons 실패 사례의 6.3bar(baseline의 0.9%)와는 질적으로 다르다 —
포지션이 발전할 시간을 어느 정도 가진 뒤에 청산한다는 뜻.

### zig075 (q075) — 악화

| | baseline | 라이브ATR 재라벨 |
|---|---:|---:|
| PnL | +40.31% | **+0.70%** |
| MDD | -13.07% | **-19.91%** |
| trades | 29 | 65 |
| WR | 48.3% | 29.2% |
| avg_hold_bars | 725.6 | 275.7 |
| exit_reasons | `stop_loss:15, take_profit:13, forced_end:1` | `exit_head:49, stop_loss:7, take_profit:8, forced_end:1` |

PnL·MDD 둘 다 baseline보다 뚜렷이 나쁘다. zig075의 baseline 자체가 이 VAL 구간에서 유난히
강한 수익(+40.31%)을 냈는데, 새 exit_head(65건 중 49건, 75.4%가 exit_head 주도)가 그 수익의
대부분을 지워버렸다.

**소규모(300후보) 검증 런과 비교**: 300후보 단계에서는 zig075도 +17.26%(개선은 아니지만
완전한 붕괴도 아님)이었는데, 1,500후보 전체 규모에서는 +0.70%로 더 나빠졌다 — exit_head의
과발동 경향이 학습 데이터가 늘수록 완화되지 않고 오히려 심해졌다는 뜻이다(exit_head 비중
31.7%→75.4%). h48qual은 반대로 300후보(53.5%)에서 1,500후보(82.5%)로 exit_head 비중은
비슷하게 늘었지만 PnL은 되레 낮아지고(+16.58%→+9.23%) MDD는 개선(−11.56%→−7.59%)됐다 —
두 컴포넌트가 스케일에 반응하는 방향 자체가 다르다.

## 결론

**혼재된 결과(부분 성공) — 컴포넌트별로 상반된 방향.** h48qual은 PnL·MDD 둘 다 baseline보다
나아졌고 exit_head가 병적이지 않은 수준(평균 210.8bar 보유)으로 발동한다. zig075는 PnL·MDD
둘 다 baseline보다 나빠졌다. "라벨 소스를 실제 라이브 ATR 배리어로 바꾸면 exit head가
고쳐진다"는 가설은 **h48qual에서는 지지되고 zig075에서는 지지되지 않는다** — 라벨-소스 교체
하나만으로 두 컴포넌트를 동시에 고칠 수 있는 문제는 아닌 것으로 보인다.

0단계(기간 스케일)·라벨 다양성 체크포인트는 둘 다 명확히 의도대로 작동했다(중앙값 자릿수
일치, giveback/adverse 신호가 처음으로 실질적 표본 확보) — 이 축의 핵심 가설(h48_conservative
타이트 배리어가 병목이었다는 진단) 자체는 검증됐다. 다만 그 수정이 낳은 실제 청산 정책은
컴포넌트마다 다르게 작용했다.

오케스트레이터가 미리 정한 규칙대로, 세 번째 라벨 메커니즘을 이 세션이 스스로 찾아 나서지는
않는다 — h48qual/zig075의 이 비대칭 반응을 어떻게 다룰지(예: 컴포넌트별로 다른 정책 채택,
zig075만 원복, 둘 다 원복 등)는 오케스트레이터가 판단한다.

## 산출물

- 새 스크립트: `scripts/research_eth_omega461_exit_head_liveatr_relabel_20260813.py`
  (`_fast_timescale_checkpoint`, `_build_exit_dataset_entry_label_live_atr_barrier`,
  `_retrain_component_exit_head_liveatr`, 메모리 안전장치 `_available_memory_gb`/`_process_rss_gb`,
  CLI `--stage {checkpoint_only,full}` `--max-candidates` `--max-horizon-bars`)
- report.json(전체 1,500후보 최종): `tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_20260813_full1500/report.json`
- report.json(300후보 검증용, 참고): `tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_20260813_serverval300/report.json`
- 재학습된 번들(연구용, 라이브 미승격): 위 두 out-suffix 디렉토리 하위
  `{h48qual,zig075}/true_3head_tabm_bundle.pt`(모델 파일명에 `_exit_liveatr.pt` suffix로
  h48cons 산출물과 구분)
- 이 문서: `docs/experiments/eth_omega461_live_exit_head_liveatr_relabel_20260813.md`

## 준수 확인 (라벨-소스 실험 부분)

`fresh_forward_bar_by_bar=true`(VAL 리플레이는 `replay_exit_variant`의 단일 순방향 causal
루프), `trade_ledgers_used_as_input=false`, `saved_parent_exit_timestamps_used=false`,
`future_rows_used_for_entry=false`(direction/quality/TP-SL은 냉동 OOF 예측 CSV, exit_head만
매 bar 순방향으로 평가). direction_head/quality_head는 전혀 재학습하지 않았다. `EXIT_THRESHOLD=0.95`
유지. OOS(2026-01-01~03-31)는 이 실험에서 전혀 로딩되지 않았다. 라이브 파일
(`trading_bot_modules/omega4_6_1_live.py`, `trading_bot.py`, `runtime_config.py`, `.env`) 미변경
확인됨(`git status --porcelain` 결과 diff 없음).

---

## 후속 1 — zig075 단독 악화 원인 진단 (2026-08-13, 추가 수정 시도 없음)

오케스트레이터 지시: 원인만 한 문단으로 기록, 추가 수정 시도는 하지 않는다(세 번째 메커니즘을
찾지 말라는 원래 지시 유지). 이미 있는 리포트 수치만으로 진단했다(재실행 없음).

**진단**: h48cons 실패 사례("큰 이익 거래를 조기청산")와는 **다른 패턴**이다.
`max_trade_pnl`이 baseline과 새 레시피에서 완전히 동일하다(8.678202171093808% vs
8.678202171093762%, 부동소수점 오차 수준) — 전체 VAL 구간에서 가장 큰 단일 승리 거래는 전혀
잘려나가지 않았다. 대신 나타나는 건 "다수의 자잘한 조기청산"이다: 거래 수가 29건→65건으로
2.2배 늘고 평균 보유기간이 725.6bar→275.7bar(baseline의 38%)로 줄면서, 승률이 48.3%→29.2%로
크게 떨어졌다(50% 밑으로). exit_reasons도 `stop_loss:15,take_profit:13`(exit_head 0건)에서
`exit_head:49,stop_loss:7,take_profit:8`로 뒤집혀 새 exit_head가 65건 중 49건(75.4%)을
주도한다. `p95_trade_pnl`도 6.87%→5.89%로 소폭 하락(상위권도 약간은 깎였다는 뜻이지만
`max_trade_pnl`만큼 극단적이지는 않음). 종합하면: zig075의 새 exit_head는 h48cons처럼 큰
이익거래의 "머리를 자르는" 게 아니라, 원래 자연스러운 TP/SL로 흘러갔을 다수의 중간 규모
포지션을 그 전에 조기청산해 재진입 빈도만 늘리고 개별 거래의 승률/기댓값을 깎는 방식으로
전체 PnL을 잠식한 것으로 보인다.

## 후속 2 — 포트폴리오 레벨 검증 (2026-08-13)

### 왜 필요한가

이 문서(그리고 h48cons 선행 문서)의 모든 VAL 수치는 h48qual과 zig075를 **각각 독립된
전액자본 원장**으로 백테스트한 것이다. 그런데 실제 라이브 어댑터(`trading_bot_modules/omega4_6_1_live.py`,
읽기 전용 참고, 미수정)는 **계좌 포지션 슬롯을 하나만 공유**하며 `PRIORITY=("h48qual","zig075")`
우선순위로 매 bar 진입을 결정한다 — 어떤 bar에 h48qual이 side≠0 신호를 내면 zig075는 그 bar에
아예 평가조차 되지 않고, 포지션이 열리면 그 포지션을 연 컴포넌트 자신의 exit_head/TP-SL만
그 포지션의 청산을 결정한다(다른 컴포넌트의 신호는 무시). 컴포넌트 단독으로 좋아졌다고 해서
포트폴리오 전체가 좋아지는지는 별개 질문이다 — 오늘 밤 다른 이슈(TP/SL 플로어 사례)에서도
같은 교훈이 있었다.

### 방법

기존 코드를 재사용해서 이 정확한 결합 로직을 재구현하지 않았다: `scripts/replay_omega4_6_1_greedy_router_20260706.py`의
`greedy_replay`/`prepare_component`가 이미 라이브 어댑터와 `PRIORITY`/`SCALE_MAP`/`LEVERAGE_CAP=5.0`/
`NOTIONAL_CAP=1.8`/`DURATION_THRESHOLD=0.005417`가 완전히 동일한, 이미 검증된 단일계좌 그리디
라우터 재구현체다(런타임-네이티브 패리티 테스트 문서화됨). 새 스크립트
`scripts/research_eth_omega461_exit_head_portfolio_asymmetric_20260813.py`를 작성해 이 두 함수를
그대로 import해서 VAL 구간(2025-10-01~12-31)에 적용했다 — `greedy_replay` 자체는 수정하지
않았고, `prepare_component`만 `oof=True`로 바꾼 로컬 복사본(`_prepare_component_val`)을 새로
만들었다(원본은 OOS 전용이라 `oof=False`가 하드코딩돼 있어 VAL에는 그대로 못 씀). 프레임과
두 컴포넌트의 `validation_predictions_qXXX.csv` 타임스탬프 교집합으로 정렬하는 절차를
추가했다(`research_eth_omega461_exit_sweep_20260721.py`의 `prep_component`가 이미 하는 것과
동일한 보정, `greedy.prepare_component`엔 없어서 직접 구현).

비교한 두 구성:
- `baseline_both_original`: h48qual/zig075 둘 다 원본 라이브 번들(원본 exit_head).
- `asymmetric_h48qual_liveatr_zig075_original`: h48qual만 이번 실험에서 재학습한 라이브ATR
  재라벨 exit_head 번들로 교체, zig075는 원본 그대로 — 오케스트레이터가 결정한 실제 채택안.

듀레이션 게이트는 적용하지 않았다(현재 라이브가 게이트 off로 운영 중이라는 근거,
`docs/experiments/eth_omega461_exit_learning_20260724.md`의 구성 감사 절 참고 — 게이트 자체를
바꾸는 게 아니라 비교 대상 두 구성 모두 동일하게 "게이트 없음"으로 둬서 변수를 하나로
격리했다). `EXIT_THRESHOLD=0.95` 동일 유지. **OOS(2026-01-01~03-31)는 이 스크립트에서도 전혀
로딩하지 않았다.**

### 결과 (VAL, 포트폴리오 레벨)

| | baseline(둘 다 원본) | 비대칭(h48qual=신규, zig075=원본) |
|---|---:|---:|
| PnL | +36.82% | **+46.59%** |
| MDD | -24.34% | **-21.70%** |
| trades | 29 | 35 |
| WR | 41.4% | 37.1% |
| avg_hold_bars | 676.5 | 551.2 |
| max_trade_pnl | 14.76% | 14.76%(동일) |
| exit_reasons | `stop_loss:17, take_profit:12` | `take_profit:13, stop_loss:13, exit_head:9` |
| 포지션 슬롯 승자(source_component) | `zig075:22, h48qual:7` | `zig075:22, h48qual:13` |

**PnL·MDD 둘 다 baseline 대비 개선**(+9.77pp PnL, -2.64pp MDD 즉 낙폭 축소). zig075가 차지하는
거래 수는 22건으로 완전히 동일(zig075 자신의 진입·청산 로직이 전혀 안 바뀌었으니 당연) —
`max_trade_pnl`도 동일해서, 전체 VAL 구간 최대 단일 승리 거래는 두 구성 모두 (아마 같은)
zig075 거래로 보존됐다. 달라진 건 h48qual이 슬롯을 차지하는 횟수뿐이다(7건→13건, +6건) —
h48qual 자신의 원시 진입신호 빈도(`nonzero_side=0.018`)는 두 구성에서 완전히 동일했으니, 이건
"h48qual이 더 자주 신호를 낸다"가 아니라 **"h48qual의 새 exit_head가 평균 보유기간을 짧게
만들어(단독 평가에서 확인된 670bar→211bar) 슬롯을 더 자주 비워주고, 그 결과 h48qual 자신의
다음 신호가 슬롯을 잡을 기회 자체가 늘어난다"**는 포트폴리오 레벨 상호작용으로 해석된다 —
컴포넌트 단독 평가만으로는 안 보이는 효과다.

### 결론

**비대칭 채택(h48qual=새 라이브ATR 재라벨 exit_head, zig075=원본)이 포트폴리오 레벨 VAL에서
PnL·MDD 둘 다 개선을 확인시켜준다.** zig075 단독 결과가 나빴던 건 사실이지만, 실제 라이브가
운용하는 단일계좌·우선순위 결합 구조에서는 h48qual의 슬롯 점유 방식 자체가 바뀌면서 zig075의
불변 성과와 상쇄 이상으로 상쇄되고도 남는 순증가를 만들어냈다. 오케스트레이터의 비대칭 채택
결정은 포트폴리오 레벨 근거로 뒷받침된다.

**여전히 지켜야 할 것**: 이 포트폴리오 결과도 VAL 29~35건이라는 작은 표본이고, 세 번째 라벨
메커니즘을 찾는 방향으로 확장하지 않는다(오케스트레이터 원 지시 유지). OOS(2026-01-01~03-31)
오픈 여부는 오케스트레이터가 판단한다 — 이 세션은 열지 않았다.

### 산출물 (포트폴리오 검증)

- 새 스크립트: `scripts/research_eth_omega461_exit_head_portfolio_asymmetric_20260813.py`
- report.json: `tmp/causal_regen_20260516/eth_omega461_exit_head_portfolio_asymmetric_20260813/report.json`
- 거래 원장(diagnostic, 참고용): `tmp/causal_regen_20260516/eth_omega461_exit_head_portfolio_asymmetric_20260813/portfolio_ledger_{baseline_both_original,asymmetric_h48qual_liveatr_zig075_original}.csv`

## 준수 확인 (포트폴리오 검증 부분)

`fresh_forward_bar_by_bar=true`(`greedy_replay`는 단일 순방향 causal 루프, bar `i`는 그 시점까지
확정된 정보만 사용). `trade_ledgers_used_as_input=false`(원장은 출력 전용).
`saved_parent_exit_timestamps_used=false`. `future_rows_used_for_entry=false`.
direction_head/quality_head 미변경. `EXIT_THRESHOLD=0.95` 유지. OOS 미로딩. 라이브 파일
(`trading_bot_modules/omega4_6_1_live.py`, `trading_bot.py`, `runtime_config.py`, `.env`) 미변경.

---

## 후속 3 — OOS 단일 확인 (2026-08-13, 오케스트레이터 승인, 1회 한정)

오케스트레이터가 이 비대칭 구성(h48qual=신규 라이브ATR 재라벨 exit_head, zig075=원본)에 대해
표준 OOS(`research_eth_omega461_exit_sweep_20260721.py`의 `OOS_START/OOS_END`=2026-01-01~03-31,
이 exit-head 실험군이 오늘 밤 내내 써온 컨벤션)로 **1회 한정 확인**을 승인했다. 결과와 무관하게
재튜닝 후 재확인은 하지 않는다.

### ⚠️ 반드시 함께 읽을 유보 사항 (수치 옆에 명시)

같은 날 밤 별도 조사 `docs/experiments/eth_omega461_oos_selection_bias_scope_and_resolution_20260813.md`가
코드를 직접 읽어 확인한 사실: **h48qual/zig075의 배포된 `quality_threshold`(0.50/0.75)** —
baseline과 이 비대칭 후보가 **동일하게 공유하는** 값(둘 다 direction/quality head와
quality_threshold는 이 exit-head 실험 전체에서 얼려둠) — **이 자체가
`scripts/train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py:1173`에서
`(oos_pnl, validation_pnl)` — OOS pnl을 1순위로 — 정렬해서 선택된 값**이며, 그 선택이 최적화
타겟으로 직접 본 "oos" 프레임은 정확히 **2026-01-01~02-28**로 확인됐다 — 이번에 쓴 OOS 3개월
중 앞의 2개월과 완전히 겹친다. 즉 이 비교가 공유하는 진입-선택 레이어 자체가 이미 이 OOS
구간의 대부분에서 좋아 보이도록 최적화되어 있었다. baseline과 후보가 **똑같이** 그 오염된
레이어를 공유하므로 **"새 exit_head가 baseline보다 상대적으로 나은가"라는 이 비교 자체는
여전히 유효**하지만, **아래 절대 OOS PnL/MDD 수치 자체를 "깨끗한 미접촉 검증"으로 과대해석해선
안 된다.**

### 방법

`scripts/research_eth_omega461_exit_head_portfolio_asymmetric_oos_confirm_20260813.py`(신규,
1회용) — VAL 스크립트와 동일한 `greedy_replay`/`_component_cfg`/`_ledger_metrics`를 그대로
import해서 재사용하고, OOS 예측 파일(`oos_predictions_qXXX.csv`)에 맞는 `oof=False`
변환(`replay_omega4_6_1_greedy_router_20260706.py`의 원본 `prepare_component`가 이미 이
컨벤션이라 그대로 재사용, VAL 때처럼 로컬 복사본을 새로 만들 필요 없음)만 반영했다. VAL 수치는
재계산하지 않고 앞 절에서 이미 저장된 report.json을 그대로 읽어와 나란히 비교했다.

**실행 중 발견한 별도 데이터 결측(비교 로직과 무관, 수정 없이는 실행 자체가 안 됨)**:
`WIDE24_2026`(Regime3 레짐확률 오버레이)가 2026-02-28 16:05~23:55 구간 95개 bar(전체
OOS 25,633행의 0.37%)에서 결측이었다 — 실제 라이브도 레짐확률 없는 bar에서는 라우팅을 할 수
없으므로, 이 95개 bar를 리플레이에서 제외하는 게 인과적으로 정직한 처리다(수치를 유리하게
바꾸려는 조정이 아니라 스크립트가 크래시 없이 돌아가게 만든 데이터 정합성 수정). 정렬 후 남은
행수 25,538.

듀레이션 게이트 미적용(VAL 비교와 동일 컨벤션 유지, 변수 하나만 격리). `EXIT_THRESHOLD=0.95`
유지.

### 결과 — VAL vs OOS

| | VAL baseline | VAL 비대칭 | OOS baseline | OOS 비대칭 |
|---|---:|---:|---:|---:|
| PnL | +36.82% | +46.59% | +49.32% | **+93.27%** |
| MDD | -24.34% | -21.70% | -16.20% | **-15.48%** |
| trades | 29 | 35 | 24 | 24 |
| WR | 41.4% | 37.1% | 45.8% | 45.8% |
| avg_hold_bars | 676.5 | 551.2 | 783.3 | 775.5 |
| max_trade_pnl | 14.76% | 14.76%(동일) | 14.85% | 14.85%(동일) |

OOS에서 PnL이 +49.32%→+93.27%로 거의 2배가 됐고 MDD도 소폭 개선(-16.20%→-15.48%)됐다 —
방향은 VAL과 일치(PnL·MDD 둘 다 개선)한다. VAL과 다른 점: OOS는 baseline과 비대칭 구성의
**거래 수(24건)와 승률(45.8%)이 완전히 동일**하다(VAL은 29건/35건으로 달랐다) — 그런데도 PnL이
크게 벌어졌다는 건, 거래 수·승률이 아니라 **개별 거래의 청산 타이밍/규모 차이가 복리 효과로
누적**돼서 벌어진 격차라는 뜻이다(어느 특정 거래가 원인인지는 이번 1회 확인 범위에서 추가로
파고들지 않았다 — 오케스트레이터 지시대로 재튜닝/추가 조사 없이 결과만 보고).

### 결론

**OOS 단일 확인도 VAL과 같은 방향(PnL·MDD 둘 다 개선)을 재현했다.** 다만 위 유보 사항대로,
baseline과 후보가 공유하는 `quality_threshold` 진입-선택 레이어 자체가 이 OOS 구간(특히 앞
2개월)에 이미 유리하게 맞춰져 있었다는 사실 때문에, **절대 수치(OOS PnL +93.27%p 등)를 이
비대칭 구성의 진짜 미접촉 미래 성과로 읽으면 안 된다** — 다만 "같은 오염을 공유하는 두 구성
중 어느 쪽이 상대적으로 나은가"라는 이번 비교의 목적 자체는 유효하고, 그 상대비교에서 새
exit_head가 일관되게 이겼다. 이번이 1회 한정 확인이었으므로 이 결과로 재튜닝하거나 다른
파라미터로 재확인하지 않는다. 최종 승격/추가 조치 여부는 오케스트레이터가 판단한다.

### 산출물 (OOS 확인)

- 새 스크립트: `scripts/research_eth_omega461_exit_head_portfolio_asymmetric_oos_confirm_20260813.py`
- report.json: `tmp/causal_regen_20260516/eth_omega461_exit_head_portfolio_asymmetric_oos_confirm_20260813/report.json`(VAL 수치 인용 원본 포함)
- 거래 원장(diagnostic, 참고용): `tmp/causal_regen_20260516/eth_omega461_exit_head_portfolio_asymmetric_oos_confirm_20260813/portfolio_ledger_oos_{baseline_both_original,asymmetric_h48qual_liveatr_zig075_original}.csv`
- 인용한 유보 근거 문서: `docs/experiments/eth_omega461_oos_selection_bias_scope_and_resolution_20260813.md`(별도 세션 작성, 이 세션에서는 읽기만 함)

## 준수 확인 (OOS 확인 부분)

`fresh_forward_bar_by_bar=true`, `trade_ledgers_used_as_input=false`,
`saved_parent_exit_timestamps_used=false`, `future_rows_used_for_entry=false`.
direction_head/quality_head 미변경. `EXIT_THRESHOLD=0.95` 유지. **OOS는 이 절에서 오케스트레이터
승인 하에 1회만 열었다** — 재튜닝 후 재실행 없음. 라이브 파일
(`trading_bot_modules/omega4_6_1_live.py`, `trading_bot.py`, `runtime_config.py`, `.env`) 미변경
확인됨(`git status --porcelain` 결과 diff 없음).
