#!/usr/bin/env bash
# 서버와 같은 꾸러미 판의 임시 venv(rtmodel/tools/server_env.sh 방식) -- 장 달력 표를 서버 판으로 뜨려고(2026-11-21 서버 확인: pandas_market_calendars 5.3.1.1 · duckdb 1.5.1).
#   bash rtduck/tools/server_env.sh <venv 디렉터리>
set -euo pipefail
V="${1:?venv 디렉터리}"
python3 -m venv "$V"
"$V/bin/pip" install -q "pandas_market_calendars==5.3.1.1" "pandas==2.2.3" "numpy==2.2.5"
"$V/bin/python" -c "import pandas_market_calendars as m, pandas; print(m.__version__, pandas.__version__)"
