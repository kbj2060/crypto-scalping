#!/usr/bin/env bash
# A4 크로스심볼 캡 -- 서버 .env를 fresh-window 최적점(cap 1.5 + 균등지분)으로 바꾼다. 재시작은 하지 않는다.
# (2026-09-04 세션에서 auto 모드 분류기가 서버 .env 편집을 차단해, 사용자가 직접 실행하도록 저장한 스크립트)
#
# 실행(dev에서):
#   bash scripts/ops/handoff.sh launch server a4_env_apply --sync scripts/ops/a4_env_apply_20260904.sh \
#     -- bash /home/llewyn/crypto-scalping/scripts/ops/a4_env_apply_20260904.sh
#   bash scripts/ops/handoff.sh logs server a4_env_apply
#
# 바꾸는 키(2026-09-04 서버값 -> 새값):
#   FINAL_GOVERNOR_PORTFOLIO_TOTAL_NOTIONAL_CAP  3.0 -> 1.5
#   FINAL_GOVERNOR_PORTFOLIO_ETH/BTC/SOL_SHARE   0.5/0.3/0.2 -> 1.0/1.0/1.0 (생성 시 정규화 -> 1/3씩, 예산 0.5씩)
#   FINAL_GOVERNOR_OMEGA4_6_1_SOL_BTC_SHADOW_PORTFOLIO_CAP_ENABLE=True 추가(코드 배포 후 유효, 그 전엔 무해)
# 백업: .env.bak_pre_a4_20260904 (되돌리기: cp -p .env.bak_pre_a4_20260904 .env 후 재시작)
# 근거: docs/experiments/eth_cross_symbol_cap_a4_activation_20260904.md
set -u
cd /home/llewyn/crypto-scalping
cp -p .env .env.bak_pre_a4_20260904 || { echo "BACKUP FAILED"; exit 1; }
sed -i \
  -e 's/^FINAL_GOVERNOR_PORTFOLIO_TOTAL_NOTIONAL_CAP=.*/FINAL_GOVERNOR_PORTFOLIO_TOTAL_NOTIONAL_CAP=1.5/' \
  -e 's/^FINAL_GOVERNOR_PORTFOLIO_ETH_SHARE=.*/FINAL_GOVERNOR_PORTFOLIO_ETH_SHARE=1.0/' \
  -e 's/^FINAL_GOVERNOR_PORTFOLIO_BTC_SHARE=.*/FINAL_GOVERNOR_PORTFOLIO_BTC_SHARE=1.0/' \
  -e 's/^FINAL_GOVERNOR_PORTFOLIO_SOL_SHARE=.*/FINAL_GOVERNOR_PORTFOLIO_SOL_SHARE=1.0/' .env
if ! grep -q '^FINAL_GOVERNOR_OMEGA4_6_1_SOL_BTC_SHADOW_PORTFOLIO_CAP_ENABLE=' .env; then
  [ -n "$(tail -c1 .env)" ] && echo >> .env
  echo 'FINAL_GOVERNOR_OMEGA4_6_1_SOL_BTC_SHADOW_PORTFOLIO_CAP_ENABLE=True' >> .env
fi
echo "--- portfolio keys: before(<) -> after(>) ---"
diff <(grep -E '^FINAL_GOVERNOR_(PORTFOLIO_|OMEGA4_6_1_(ETH_PORTFOLIO|SOL_BTC_(REAL|SHADOW)))' .env.bak_pre_a4_20260904) \
     <(grep -E '^FINAL_GOVERNOR_(PORTFOLIO_|OMEGA4_6_1_(ETH_PORTFOLIO|SOL_BTC_(REAL|SHADOW)))' .env)
echo "--- mode/owner/size ---"; stat -c '%a %U %s %n' .env .env.bak_pre_a4_20260904
echo "--- parsed through the same load path trading_bot.py uses (load_dotenv -> runtime_config) ---"
python - <<'PY'
from dotenv import load_dotenv; load_dotenv()
from trading_bot_modules import runtime_config as rc
from trading_bot_modules.portfolio_risk import PortfolioRiskConfig, PortfolioRiskManager
r = PortfolioRiskManager(PortfolioRiskConfig(total_notional_cap=rc.FINAL_GOVERNOR_PORTFOLIO_TOTAL_NOTIONAL_CAP, asset_shares={
    "eth_omega461": rc.FINAL_GOVERNOR_PORTFOLIO_ETH_SHARE * rc.FINAL_GOVERNOR_PORTFOLIO_ETH_OMEGA461_SUBSHARE,
    "eth_sigma3_1h": rc.FINAL_GOVERNOR_PORTFOLIO_ETH_SHARE * rc.FINAL_GOVERNOR_PORTFOLIO_ETH_SIGMA3_1H_SUBSHARE,
    "btc": rc.FINAL_GOVERNOR_PORTFOLIO_BTC_SHARE, "sol": rc.FINAL_GOVERNOR_PORTFOLIO_SOL_SHARE}))
print("cap", r.config.total_notional_cap, "budgets", {k: round(r.asset_budget(k), 6) for k in ("eth_omega461", "eth_sigma3_1h", "btc", "sol")})
print("eth_cap_flag", rc.FINAL_GOVERNOR_OMEGA4_6_1_ETH_PORTFOLIO_CAP_ENABLE,
      "sol_btc_real_exec", rc.FINAL_GOVERNOR_OMEGA4_6_1_SOL_BTC_REAL_EXECUTION_ENABLE,
      "shadow_cap_flag", getattr(rc, "FINAL_GOVERNOR_OMEGA4_6_1_SOL_BTC_SHADOW_PORTFOLIO_CAP_ENABLE", "n/a (code not deployed yet)"))
PY
echo "--- disk trading_bot.py parses? (never restart on a file that does not) ---"
python -c "import ast; ast.parse(open('trading_bot.py').read()); print('trading_bot.py parses OK')"
echo DONE_ENV_APPLY
