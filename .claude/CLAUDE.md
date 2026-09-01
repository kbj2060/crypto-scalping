CLAUDE.md
Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

Tradeoff: These guidelines bias toward caution over speed. For trivial tasks, use judgment.

1. Think Before Coding
Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:

State your assumptions explicitly. If uncertain, ask.
If multiple interpretations exist, present them - don't pick silently.
If a simpler approach exists, say so. Push back when warranted.
If something is unclear, stop. Name what's confusing. Ask.
2. Simplicity First
Minimum code that solves the problem. Nothing speculative.

No features beyond what was asked.
No abstractions for single-use code.
No "flexibility" or "configurability" that wasn't requested.
No error handling for impossible scenarios.
If you write 200 lines and it could be 50, rewrite it.
Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

3. Surgical Changes
Touch only what you must. Clean up only your own mess.

When editing existing code:

Don't "improve" adjacent code, comments, or formatting.
Don't refactor things that aren't broken.
Match existing style, even if you'd do it differently.
If you notice unrelated dead code, mention it - don't delete it.
When your changes create orphans:

Remove imports/variables/functions that YOUR changes made unused.
Don't remove pre-existing dead code unless asked.
The test: Every changed line should trace directly to the user's request.

4. Goal-Driven Execution
Define success criteria. Loop until verified.

Transform tasks into verifiable goals:

"Add validation" → "Write tests for invalid inputs, then make them pass"
"Fix the bug" → "Write a test that reproduces it, then make it pass"
"Refactor X" → "Ensure tests pass before and after"
For multi-step tasks, state a brief plan:

1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

Omega Artifact Integrity Promotion Gate
Omega/Omega4.x 모델 업그레이드, live 후보, baseline 승격은 scripts/audit_omega_artifact_integrity_20260630.py가 exit status 0과 promotion_pass=true를 반환해야 한다.
Parent artifact는 사용 quality threshold와 정확히 일치하는 train_predictions_qXXX.csv, validation_predictions_qXXX.csv, oos_predictions_qXXX.csv를 포함해야 한다. qXXX = round(quality_threshold * 100) zero-padded 값이다.
Risk sidecar는 report와 artifact에 risk_model.precomputed_prediction_dir와 risk_model.precomputed_prediction_tag를 기록하고, 해당 exact-threshold parent prediction artifact만 사용해야 한다.
저장된 trade ledger, candidate-event replay, 과거 비교 ledger는 diagnostic 전용이다. Per-bar parent prediction artifact를 대신해 promotion 근거로 쓰지 않는다.
정책 원문은 docs/model_contracts/omega_artifact_integrity_policy_20260630.md에 둔다.

Seed-Diversity Ensemble Promotion Gate
여러 학습 시드를 평균/배깅해서 승격을 주장하는 모델(Omega4.6.1의 bull/bear/chop MoE처럼 단일 시드 또는 전문가별 오프셋 구조는 제외)은 N≥5개의 진짜 다양한 시드(고정 간격 증가가 아닌 랜덤 추출)에서 OOS 부호 일치를 보여야 한다.
시드 리스트는 프로모션 리포트에 기록해야 한다.
시드 개수가 너무 적거나 너무 클러스터되어 신호와 시드-분산 노이즈를 구분할 수 없는 프로모션 주장은, 헤드라인 지표와 무관하게 무효다.
정책 배경: 2026-08-01 Sigma3-1h 감사에서 SEEDS=[270705,270710,270715,270720,270725] (한 base값의 +5 증분)로 만든 "5-seed ensemble"이 VAL은 진짜 다양한 8-seed 재앙상블과 거의 일치(+22.99% vs +23.85%)했지만 OOS는 부호가 뒤집혔다(+24.32%→-13.57%, MDD -32.64%).

Fresh-Forward Validation/OOS/Test Rule
Fresh-forward는 고정된 과거 validation/OOS 기간을 5분봉 bar 단위로 처음부터 끝까지 순차 진행하는 causal walk-forward 테스트를 뜻한다.
기본 split은 validation 2025-09-01부터 2025-12-31까지, OOS 2026-01-01부터 2026-03-31까지다. 날짜 경계가 바뀌면 리포트에 명시해야 한다.
각 bar에서는 그 시점까지 확정된 feature/state만 보고 신호를 생성한다. 이후 bar가 도착한 것처럼 한 칸씩 전진하면서 TP/SL/time-exit/PnL을 확정한다.
저장된 trade ledger, candidate-event ledger, parent exit timestamp, 또는 과거 원장의 entry/exit 결과를 입력으로 사용한 성과는 승격/모델 선택/test 근거로 쓰면 안 된다.
trading live path와 동일한 causal feature availability를 써야 하며, 미래 row에서 생성된 label/decision/ledger를 현재 decision에 조인하면 안 된다.
저장 원장 기반 replay는 diagnostic, accounting audit, historical reproduction 전용이다. 모델 선택, 승격, live 후보 성과, baseline 성과로 주장하지 않는다.
리포트는 fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false, saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false를 명시해야 한다.
기존 저장 원장 기반 validation/OOS 숫자는 research/dev score로만 취급한다. live promotion 또는 expected live PnL 근거로 쓰지 않는다.
이 규칙을 어기는 평가 결과는 성능 수치와 무관하게 promotion/test 근거로 무효다.
Futures Risk Sizing Contract
Futures sizing must distinguish margin, leverage, and notional explicitly:notional = margin_fraction * leverage
margin_fraction = notional / leverage
PnL = price_move * notional

For new Omega risk-sizing experiments, prefer predicting account risk as margin_fraction rather than predicting leverage directly.
If leverage is fixed, derive notional from margin:leverage = 3
notional = margin_fraction * 3

TP/SL model outputs should be interpreted as price-move targets before converting to account-PnL thresholds:take_profit = tp_price_move * notional
stop_loss = sl_price_move * notional

Canonical example:margin_fraction = 0.30
leverage = 3
notional = 0.90
tp_price_move = 0.04
sl_price_move = 0.015
take_profit = 0.04 * 0.90 = 0.036
stop_loss = 0.015 * 0.90 = 0.0135

Do not multiply TP/SL price lines by leverage again after notional is derived. That double-counts leverage because notional already includes exposure.

Position-Feature Train/Inference Parity Contract
exit_head 학습 라벨/피쳐 빌더가 만드는 pos_unrealized/pos_mfe/pos_mae/pos_dist_to_tp/pos_dist_to_sl/pos_notional/pos_leverage/pos_exposure 등 pos_* 피쳐는, 그 모델을 실제로 서빙하는 replay/live 경로가 같은 이름의 피쳐에 넣는 값과 스케일·단위·소스가 정확히 일치해야 한다.
가격변동 기반 피쳐(pos_unrealized/pos_mfe/pos_mae 등)는 notional/leverage를 곱해 스케일하면 안 된다. 이 값들은 live에서도 원시(비스케일) 가격변동률이다 — `trading_bot.py`의 `move = (current_price - entry) / entry` (예: 9178행), `omega4_6_1_live.py::exit_probability`가 그대로 받는 `unrealized_move` 인자.
리스크사이징 피쳐(pos_notional/pos_leverage/pos_exposure)는 실제 후보/캔들마다 달라지는 risk sidecar 예측값을 써야 한다. 아직 해당 candidate parent에 등록된 risk sidecar가 없어 고정 BASE_TEMPLATE 상수로 대체할 수밖에 없다면, report.json 등 산출물에 risk_sizing_source 같은 필드로 그 사실을 명시해야 한다 — 조용히 상수를 실제값인 것처럼 흘려보내면 안 된다. risk sidecar의 margin이 매 후보bar에서 실제로 양수인 비율이 낮을 수 있다(parent 자체 결정이 active한 bar에서만 양수) — candidate 인덱스에 그대로 곱하면 후보 풀이 그 비율까지 줄어들 수 있으니, 실제 active bar들의 (margin, leverage) 경험적 분포에서 페어를 보존해 리샘플하는 방식을 우선 검토한다.
pos_tp/pos_sl(TP/SL 배리어 레벨) 피쳐는 리스크사이징(margin/leverage)과 별개 축이다 — h48qual/zig075/BTC/SOL 전부 실제 라이브는 `omega4_6_1_live.py`의 `_ComponentConfig` 기본값(모든 `components_override` 호출부가 이 7개 필드를 그대로 둠, 자산 공통) 기반 ATR-adaptive 공식을 쓴다: `tp = clip(max(min_tp, atr_pct*tp_mult), 0, max_tp)`, `sl = clip(max(min_sl, atr_pct*sl_mult), 0, max_sl)` (`atr_window=192, tp_mult=12.0, sl_mult=6.0, min_tp=0.075, min_sl=0.040, max_tp=0.22, max_sl=0.12` — `omega4_6_1_live.py:91-97,181-185`). 학습 피쳐 빌더가 `take_profit=float(omega.BASE_TEMPLATE["take_profit"])`(2.6%/1.4%) 같은 고정 상수를 `_position_feature_row`에 그대로 흘리면 라이브 실제값과 어긋난다 — 2026-08-18 재점검에서 9개 파일 전부(h48cons 포함) 확인·수정됨. h48cons는 자신의 라벨 배리어(h48_conservative CSV, ATR mult 1.2/0.8·floor 0.6%/0.4%)가 라이브 ATR 공식과 다르므로, pos_tp/pos_sl은 라벨 배리어에서 유도하지 말고 위 라이브 공식으로 독립 계산해야 한다. 실제 ETH atr_pct는 이 window(192bar=16시간)에서 대부분(2025-05~09 구간 실측 99.7%) floor 미만이라 pos_tp/pos_sl이 대부분 상수처럼 보이지만(7.5%/4.0%), 이는 라이브 자체의 실제 특성이지 버그가 아니다 — 드문 고변동성 구간에서는 실제로 달라진다(같은 실측에서 2,000개 후보 샘플 기준 pos_tp 89종/pos_sl 17종 관측). audit 스크립트의 `SIZING_KWARGS`에 `take_profit`/`stop_loss`도 추가되어 이 패턴도 Pattern B로 자동 탐지된다.
배리어/청산 판정(barrier_end_i, tb_reason 등, "이 트레이드가 TP/SL로 언제 끝나는가")은 **intrabar 고가/저가 기준**이 h48qual/zig075(Omega4.6.1)의 실제 라이브 컨벤션이다 — `omega4_6_1_live.py::evaluate_exit`의 `bar_high_move`/`bar_low_move`(방금 완결된 bar의 실제 고가/저가로 계산, `trading_bot.py:9181-9202`에서 실제로 채워서 호출됨), "resting TP/SL 주문은 종가가 아니라 닿는 즉시 체결되고, 이미 확정된 bar만 쓰므로 lookahead 아님"이라는 문서화된 설계다. **학습된 exit_head 자신의 피쳐(pos_unrealized 등)는 여전히 종가/마크가격 기준**(위 문단) — 같은 라이브 호출 안에 두 컨벤션이 공존한다: 하드코딩 TP/SL 프리체크는 intrabar, 그 체크를 통과한 뒤에만 평가되는 학습된 exit_head의 입력은 종가. 배리어 판정 함수를 고칠 때 이 둘을 혼동해 하나로 통일하면 안 된다.
⚠️ 2026-08-18 정정: 같은 세션 안에서 이 항목을 처음엔 "종가 기준"으로 잘못 적었다 — `greedy_replay`/`_price_move`(둘 다 종가 기준)와 `trading_bot.py`의 일반적인 `current_price` 기반 unrealized 체크만 보고 `omega4_6_1_live.py::evaluate_exit`(h48qual/zig075를 실제로 지배하는 함수)를 직접 확인하지 않은 채 결론 내렸던 게 원인. `greedy_replay`가 종가만 쓰는 것 자체는 사실이지만, 그게 "라이브도 종가"라는 뜻은 아니었다 — replay 도구가 라이브의 bar_high_move/bar_low_move 개선을 반영 못한 채 뒤처져 있었을 가능성이 높다. 배리어 컨벤션을 고칠 때는 반드시 해당 자산의 실제 `evaluate_exit`류 라이브 함수(리플레이 스크립트 아님)를 직접 읽고 확인할 것.
리스크사이징 피쳐는 실제 후보/캔들마다 달라지는 risk sidecar 예측값을 써야 한다는 원칙과 별개로, ATR/변동성 계산 자체는 고가/저가를 써도 무방하다 — 여기서 다루는 것은 라벨/피쳐 해상 시점(barrier_end_i) 판정 컨벤션이다.
근거: 2026-08-18 h48qual/zig075 liveATR 재라벨 스크립트(scripts/research_eth_omega461_exit_head_liveatr_relabel_20260813.py)에서 이 패턴들이 동시에 깨져 있었던 게 발견됐고(배리어 판정은 위 정정으로 원래 intrabar 컨벤션이 맞았음이 확인됨), 스케일압축/무분산 리스크피쳐 2가지는 h48cons/BTC/SOL 변형 등 다수 파일에 코드 재사용/포팅으로 전파돼 있었다(그 중 다수가 실제 라이브 배포와 직결 — 자산별로 확인 필요, "research 파일이니 비라이브"라고 넘겨짚지 말 것). 전체 기록: docs/experiments/eth_odyssey4_exit_head_liveatr_barrier_and_label_reaudit_20260818.md.
새 exit_head/포지션-상태 학습 데이터 빌더를 작성하거나 기존 걸 복사할 때는 scripts/audit_position_feature_train_inference_parity_20260818.py를 재실행해 confirmed(Pattern A) 항목이 새로 늘지 않았는지 확인한다. needs_review(Pattern B/C)는 verdict가 아니라 사람이 직접 읽고 판단해야 하는 후보 목록이다 — 특히 Pattern C(배리어 고가/저가)는 "고가/저가=항상 버그"가 아니므로, 반드시 해당 자산의 실제 라이브 barrier 컨벤션부터 확인 후 판단한다.
Deploy/Git Two-Channel Conflict Contract
이 저장소는 서버 배포 경로가 **둘**이고 서로를 모른다. 이 긴장 때문에 2026-08-24와 2026-09-01 두 번 실제로 대시보드가 깨졌다.
(1) `scripts/ops/handoff.sh push` — rsync로 서버 파일을 직접 덮어쓴다. git을 전혀 거치지 않아 서버 워킹트리에 **커밋되지 않은 서빙 코드**를 남긴다.
(2) `scripts/ops/deploy_watcher.sh` — 10분 cron. origin/main을 폴링해 CI(`syntax-check`) 통과를 확인한 뒤 `git stash push -u` → `git merge --ff-only` → `git stash pop` 사이클을 돈다.
(1)로 배포하고 커밋하지 않은 파일이 있는 상태에서 **main이 전진하면**, (2)가 그 파일을 stash했다가 되돌리려다 같은 줄을 건드린 경우 **stash pop 충돌** → 서빙 파일에 `<<<<<<<` 마커가 문자 그대로 박힌다. watcher는 이때 설계대로 텔레그램 알림 후 정지하고 서비스는 건드리지 않는다(`deploy_watcher.sh` 224~237행) — 즉 **그 시점엔 아직 다운이 아니다**(실행 중 프로세스는 메모리에 옛 코드 보유). 이 상태에서 프로세스를 재시작하면 그제서야 크래시 루프에 빠져 다운된다. 2026-09-01 사고의 실제 다운 원인이 정확히 이 재시작이었다.

**main에 머지하기 전에 반드시 `bash scripts/ops/check_deploy_drift.sh`를 돌린다.** 머지가 watcher를 깨우는 방아쇠이므로 이 시점이 유일하게 예방 가능한 지점이다. 종료코드 0(안전)이 아니면 머지하지 말고 먼저 정리한다. 라이브 파일을 `handoff.sh push`로 배포한 직후, 그리고 대시보드가 이상할 때(다른 원인 추정보다 **먼저**)도 같은 스크립트를 돌린다.

**라이브 서빙 파일은 rsync만 하지 말고 커밋한다.** `dashboard/**`, `scripts/live_*.py`가 대상이다. 커밋하면 main == 배포본이 되어 stash 대상 자체가 사라진다. 서버에만 존재하는 코드는 그 자체가 리스크이고(오늘 실제로 터졌다), 커밋하면 CI 문법검사도 자동으로 걸린다.

**서버 파일을 덮어쓰기 전에 반드시 로컬과 md5를 대조한다.** 이 저장소는 동시 세션이 공유하므로 서버 쪽이 더 최신일 수 있다. 대조 없이 `handoff.sh push`하면 다른 세션이 배포한 작업을 소리없이 지운다. 대조 방법: `handoff.sh launch server <job> -- /usr/bin/md5sum <경로>` (읽기 전용). 내 변경만 되돌린 사본의 해시가 서버 것과 일치하면 "로컬 = 서버 + 내 변경분"이므로 덮어써도 안전하다.

**대시보드 프로세스를 죽이기 전에 디스크의 파일이 파싱되는지 먼저 확인한다.** `python -c "import ast; ast.parse(open(<파일>).read())"`. 충돌 마커가 박힌 파일 위에서 재시작하면 즉시 크래시 루프가 된다.

**충돌 발견 시 조치 순서**: (1) 로컬 정상본을 `handoff.sh push`로 재전송 → (2) 서버에서 `git add <파일>`로 `UU`→`M ` 해소(워킹트리 내용은 그대로 유지됨) → (3) 다음 watcher 사이클(최대 10분)이 `restored stashed local changes / deploy OK`로 자가회복하는지 로그로 확인 → (4) curl로 서빙 바이트 재검증. `last_deployed_sha`를 손으로 쓰지 말 것 — UU만 풀어주면 watcher가 스스로 회복한다.
