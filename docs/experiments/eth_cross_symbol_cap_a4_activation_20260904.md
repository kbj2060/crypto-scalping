# A4 크로스심볼 캡 배선 — 실태 확인·재구성·텔레메트리 (2026-09-04)

Status: 코드 변경 로컬 완료·테스트 통과(미커밋). 서버 `.env` 재구성은 **미적용**(auto 모드
분류기가 라이브 비밀파일 편집을 차단 → 사용자 실행/승인 대기, 스크립트 준비됨).

## 0. 결론 먼저

1. **A4 캡은 "미배선"이 아니었다.** 서버 `.env`에 `FINAL_GOVERNOR_OMEGA4_6_1_ETH_PORTFOLIO_CAP_ENABLE=True`,
   `FINAL_GOVERNOR_PORTFOLIO_TOTAL_NOTIONAL_CAP=3.0`, 지분 0.5/0.3/0.2가 이미 들어 있고
   (`live_model_v1_checkpoint_20260714.md`: 07-14엔 플래그 True + cap `uncapped`, 이후 3.0으로 설정),
   `FINAL_GOVERNOR_OMEGA4_6_1_SOL_BTC_REAL_EXECUTION_ENABLE=True`라 BTC/SOL 경로도 executor 분기를 타서
   캡이 적용되고 있었다. 저널이 증거다: ETH notional **정확히 1.5**(08-21, 08-22 — 컴포넌트 cap은 1.8,
   레버리지 cap 5.0이라 1.5는 포트폴리오 예산 3.0×0.5뿐), SOL **정확히 0.6**(=3.0×0.2, 7회). 08-31
   설계문서·메모리의 "기본 False라 지금은 no-op"은 **코드 기본값만 읽고 서버 .env를 안 본 오진**이었다.
2. **봇 전체가 페이퍼다.** `BINANCE_EXECUTION_ENABLED=False` → `binance_execution=OFF dry_run=True
   testnet=True` (journalctl, 08-13 이후 매 재시작). 저널 전 행 `exchange_execution_status=disabled`.
   "ETH 실경로"는 코드 경로 이름이지 실주문이 아니다. 따라서 A4 재구성은 실돈 리스크 0이고,
   "해보고 안되면"의 평가도 페이퍼 저널로 한다.
3. **남은 A4 작업 = 재구성 + 텔레메트리.** 리서치 최적점(cap 1.5 + 균등지분, fresh 07-01~08-30
   PnL 27.65%/MDD −4.65%)과 서버값(3.0 + 50/30/20)이 다르다. 그리고 지금 저널엔 "요청 notional"이
   없어 캡의 효과를 사후에 무캡과 비교할 수 없다 — 09-04 코드가 `portfolio_cap`(requested/approved/
   budget) 필드를 저널·로그에 남기도록 추가했다.
4. **실현 저널 재사이징(21건, 07-07~09-03)**: A4 구성이면 실현 PnL 30.8%→26.9%, 실현 MDD −10.3%→
   **−4.2%**. 리플레이 fresh 결과와 방향 일치(MDD 대폭↓), PnL은 리플레이(개선)와 달리 소폭↓ —
   수익의 84%가 SOL에서 났고 SOL도 0.6→0.5로 깎이기 때문. 표본 21건, 그중 SOL 10건.

## 1. 서버 실태 (2026-09-04 04:xx KST, 읽기 전용 프로브)

| 항목 | 값 |
|---|---|
| `.env` mtime | 2026-08-26 22:15 KST |
| ETH 캡 플래그 / cap / 지분 | True / 3.0 / 0.5·0.3·0.2 → 예산 ETH 1.5 · BTC 0.9 · SOL 0.6 |
| SOL/BTC real-execution 플래그 | True (executor 존재 → 캡 적용, 단 dry-run) |
| `BINANCE_EXECUTION_ENABLED` | **False** → 3자산 전부 dry-run |
| ETH 승수 / chop soft-size / SOL 승수 / BTC cmamba 게이트 | 1.0 / ON(thr 0.3) / 1.5 / ON |
| 컴포넌트 cap (`omega4_6_1_live.py`) | LEVERAGE_CAP 5.0, NOTIONAL_CAP 1.8 |
| 서버 HEAD | fd54b96 (origin/main과 동일), trading-bot.service active (pid 313, 09-03 21:25 부팅) |
| 오픈 포지션 | BTC LONG(08-21~, 0.26) · ETH SHORT(08-25~, 1.4247) — 둘 다 페이퍼 |

저널의 캡 바인딩 증거(`notional_exposure`): ETH 1.6129(07-07, 캡 전) → 1.5000(08-21, 08-22) /
1.4247(08-25, =mf 0.2849×lev 5.0 레버리지 cap) ; SOL 0.6000 ×7(08-03~09-02) ; BTC 0.26(예산 0.9 미달).

**왜 두 세션이 놓쳤나**: 설계문서는 `runtime_config.py`의 기본값(False/3.0)을 읽고 "no-op"이라 적었다.
서버 `.env`는 gitignore라 로컬 체크아웃에 없고, 봇은 캡 적용 시 아무 로그도 안 남겼다(요청→승인
축소가 조용히 일어남). → 이번에 시작 시 `SYSTEM portfolio_cap total_notional_cap=… budgets=…` 한 줄과
적용 시 `SYSTEM portfolio_cap asset=… requested=… approved=…` 한 줄을 추가했다.

## 2. 코드 변경 (로컬, 미커밋)

| 파일 | 변경 |
|---|---|
| `trading_bot_modules/portfolio_risk.py` | `asset_share()` 메서드, `portfolio_cap_trace(risk, asset, requested)` — JSON-safe 기록(asset_key, cap, share, budget, requested, approved, scaled, blocked, reason). 순수 함수. docstring의 "not yet wired" 정정 |
| `trading_bot_modules/runtime_config.py` | `FINAL_GOVERNOR_OMEGA4_6_1_SOL_BTC_SHADOW_PORTFOLIO_CAP_ENABLE`(기본 False) 신설 |
| `trading_bot.py` | ETH 실경로: trace 생성·로그·`info["portfolio_cap"]`(진입/차단 둘 다) → `_decision_bar_audit_context["portfolio_cap"]` → 저널. BTC/SOL: 캡 적용 조건을 `executor is not None **or** SHADOW 플래그`로 분리(실행 플래그를 꺼도 캡이 조용히 사라지지 않게), trace를 `active["portfolio_cap"]`에 보존 → `_omega461_shadow_audit_context` → 저널. main()에 시작 로그 1줄 |
| `trading_bot_modules/position_router.py` | `_journal_audit_fields` 화이트리스트에 `portfolio_cap` 추가(없으면 저널에 안 실림) |

의미 주의: `portfolio_cap.approved_notional`은 **캡 단계**의 값이다. ETH는 그 뒤 chop soft-size와
`finalize_sizing`이 더 줄일 수 있어 저널 `notional_exposure`(최종)와 다를 수 있다. 무캡 반사실의 최종
크기 = `notional_exposure × requested/approved`(하류 배수 동일 적용). 평가 스크립트가 이 식을 쓴다.

검증:
- `python scripts/test_portfolio_cap_prealloc_parity_20260904.py` — 6/6 통과. 라이브 `PortfolioRiskManager`
  ↔ 리플레이 `_replay_concurrent(cap_mode="prealloc")` 분기(원문 포팅 + 소스 텍스트 핀) 21×3 격자
  동일, 정규화·예산 동일, min_notional 0.05 동일, 저널 증거(1.6129→1.5, SOL→0.6, A4에서 1.4247→0.5)
  재현, trace 계약·저널 화이트리스트 통과.
- `python test/test_sizing_chain_parity.py` — 기존 2/2 통과(회귀 없음).
- 4개 파일 `ast.parse` OK. (로컬엔 torch가 없어 `trading_bot.py` import 실행은 불가 — 서버 배포 시
  CI `syntax-check`와 재시작 후 `SYSTEM portfolio_cap` 시작 로그로 확인할 것.)

## 3. 실현 저널 재사이징 평가 (백테스트 아님 — 이미 일어난 페이퍼 결정의 재집계)

`scripts/research_eth_cross_symbol_cap_realized_journal_eval_20260904.py`, 입력 서버 저널(09-04 pull,
43행, 마감 21건). prealloc은 크기만 `min(요청, 예산)`으로 바꾸고 진입·방향은 안 바꾸며 트레이드 손익은
notional에 선형(행별 검증 오차 ≤7e-18) → **기록보다 좁은 예산의 반사실은 정확**. 무캡 반사실은
요청값이 없어 불가(텔레메트리 이후 행부터 가능).

| 구성 | 실현 PnL | 실현 MDD | 축소 건수 | 평균 축소비 | 자산별 기여 (btc/eth/sol) |
|---|---:|---:|---:|---:|---|
| 기록 그대로 (= 서버 3.0/50-30-20) | 30.82% | −10.34% | 0 | — | +6.51 / −1.67 / +25.98 |
| cap 3.0 + 50/30/20 재적용 | 30.24% | −10.34% | 2 | 0.934 | +6.47 / −2.04 / +25.81 |
| **cap 1.5 + 균등 (A4 원 그리드 최적)** | **26.89%** | **−4.24%** | 14 | 0.634 | +6.17 / −0.88 / +21.61 |
| cap 1.0 + 균등 (확장 그리드 최적) | 20.62% | −2.83% | 16 | 0.475 | +6.01 / −0.44 / +15.05 |

- 방향은 리플레이 fresh 결과와 같다(MDD −59%). PnL은 리플레이가 "개선"이었던 것과 달리 −3.9pp — 이
  저널의 수익원이 SOL(10건, +26pp)이고 SOL도 0.6→0.5로 깎이기 때문. ETH(7건)는 순손실이라 축소가
  이득.
- 이 저널은 리플레이와 **다른 시스템**이다(ETH 승수 1.0 + chop soft-size ON, SOL 승수 1.5, BTC cmamba
  게이트 ON, 페이퍼 체결). 21건·2개월이라 확정 아님. cap 1.0은 PnL을 1/3 더 깎는다 → 1.5 채택.
- 겹침(07-07~09-03): btc-sol 12회 중 11회 동방향(92%), btc-eth 50%, eth-sol 38%. 08-31 실측(100/50/36)
  과 같은 그림.

## 4. 적용 절차 (서버 — 사용자 실행/승인 필요)

auto 모드 분류기가 서버 `.env` 편집(`sed -i`)을 차단해 여기서 멈췄다. 준비된 스크립트:

1. `scripts/ops/a4_env_apply_20260904.sh` — `.env` 백업(`.env.bak_pre_a4_20260904`) 후 cap 1.5 / 지분
   1.0·1.0·1.0(정규화 1/3) / SHADOW 캡 플래그 True. 재시작 안 함. 바뀐 값을 `load_dotenv→runtime_config`
   같은 경로로 파싱해 예산 0.5/0/0.5/0.5를 출력하고 디스크의 `trading_bot.py` 파싱을 확인한다.
2. 코드 머지 → `deploy_watcher`가 10분 내 재시작(권장, 재시작 1회로 env+코드 동시 적용). 머지 전
   `bash scripts/ops/check_deploy_drift.sh` 필수. 재시작 후 `journalctl -u trading-bot` 에서
   `SYSTEM portfolio_cap total_notional_cap=1.5 budgets={'eth_omega461': 0.5, … 'btc': 0.5, 'sol': 0.5}`
   확인.
3. 코드 머지 없이 env만 켤 거면 `scripts/ops/a4_restart_verify_20260904.sh`(sudoers 허용된 restart +
   45초 후 상태·시작로그·09-03 재부팅 이후 다른 유닛/대시보드 포트 확인).

되돌리기: `cp -p .env.bak_pre_a4_20260904 .env` + 재시작. 오픈 중인 페이퍼 포지션(ETH 1.4247)은 재시작
후에도 옛 크기를 유지한다(강제 리사이즈 없음) — 새 진입부터 0.5.

## 5. "안되면"의 판정 기준 (사전 등록 제안)

텔레메트리가 붙은 뒤의 마감 트레이드로, 같은 트레이드 집합에서 **적용된 크기 vs 무캡 반사실**
(`notional_exposure × requested/approved`) 두 실현 곡선을 만든다(평가 스크립트가 자동 계산).
- 판정 시점: 2026-09-30(달력 게이트) 또는 마감 15건 중 먼저 오는 쪽.
- 통과: 실현 MDD 상대개선 ≥ 30% **이고** PnL/|MDD| 비율이 무캡보다 높다.
- 실패: MDD 개선 < 30% 이거나 PnL 손실이 MDD 개선폭보다 크다 → A4 종료, 사용자 지시대로 새 모델 축.
- 리플레이(07-01~08-30)와 실현 저널(07-07~09-03)은 둘 다 이미 본 구간이라 판정에 재사용하지 않는다.

## 6. 정정된 기록

- `docs/eth_cross_symbol_exposure_cap_design_20260831.md`의 "활성화 게이트 기본 꺼짐 → 지금은 no-op"
  은 틀렸다(§1). 해당 문서에 정정 부록 추가.
- 메모리 `eth_cross_symbol_exposure_cap_design_20260831` 설명 정정, 피드백 메모리(라이브 플래그는
  서버 `.env`·시작 로그·저널 실값으로 확인) 신설.
