"""
DeepSeek-R1:8b를 통해 이더리움 방향성 예측 질의
- Ollama 로컬 서버 사용 (http://localhost:11434)
- 최근 N개 캔들 데이터를 요약해서 LLM에 전달
- 사용법: python scripts/ask_llm.py [--rows 20] [--csv data/training_features_5m.csv]
"""

import argparse
import json
import urllib.request
import urllib.error
import csv
import sys
import os

# ─── 설정 ─────────────────────────────────────────────────────────────────────
DEFAULT_CSV = os.path.join(os.path.dirname(__file__), "../data/training_features_5m.csv")
DEFAULT_MODEL = "deepseek-r1:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"

# LLM에 전달할 핵심 컬럼 (전체 59개 중 해석 가능한 것만)
SUMMARY_COLS = [
    "timestamp", "close", "volume",
    "log_return", "volatility_z", "rsi", "macd_hist",
    "bb_width", "vwap_dist", "hma_slope",
    "whale_retail_ratio", "smart_money_flow", "net_taker_ratio",
    "funding_pressure", "last_funding_rate",
    "oi_change_rate", "squeeze_power",
    "mtf_trend_1h", "mtf_trend_4h",
    "session_us", "session_europe", "session_asia",
    "chop_index", "garman_klass_vol",
    # 전략 신호 (있을 경우)
    "sig_whale_sentiment", "sig_liq_squeeze", "sig_net_taker", "sig_ob_fvg",
    "sig_garch_regime", "sig_ou_mean_rev", "sig_jump_rebound", "sig_evt_tail",
    "garch_vol_z", "ou_halflife", "jump_flag", "evt_tail_flag",
    "regime_bull", "regime_bear", "regime_chop",
]


def read_last_rows(csv_path: str, n: int) -> list[dict]:
    """CSV 마지막 n행 읽기 (메모리 효율적)"""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows[-n:]


def format_rows_for_prompt(rows: list[dict]) -> str:
    """행 목록을 LLM이 읽기 쉬운 텍스트로 변환"""
    lines = []
    for row in rows:
        parts = []
        for col in SUMMARY_COLS:
            if col in row and row[col] not in ("", "nan", "NaN"):
                try:
                    val = float(row[col])
                    # 정수처럼 보이면 정수로
                    parts.append(f"{col}={val:.4g}")
                except ValueError:
                    parts.append(f"{col}={row[col]}")
        lines.append(f"[{row.get('timestamp','?')}] " + ", ".join(parts))
    return "\n".join(lines)


def build_prompt(data_text: str, n_rows: int) -> str:
    return f"""You are a professional cryptocurrency quantitative analyst specializing in Ethereum (ETH/USDT) scalping on 5-minute candles.

Below are the last {n_rows} candles of ETH/USDT with computed technical and market microstructure features:

{data_text}

Feature legend:
- log_return: 5-min log return
- volatility_z: realized volatility z-score
- rsi: RSI(14)
- macd_hist: MACD histogram
- bb_width: Bollinger Band width
- vwap_dist: distance from VWAP (normalized)
- hma_slope: Hull MA slope direction
- whale_retail_ratio: whale vs retail volume ratio
- smart_money_flow: net smart money pressure
- net_taker_ratio: taker buy / (taker buy + sell)
- funding_pressure: funding rate pressure signal
- last_funding_rate: perpetual funding rate
- oi_change_rate: open interest change rate
- squeeze_power: liquidation squeeze signal
- mtf_trend_1h/4h: higher timeframe trend (+1 up, -1 down, 0 flat)
- chop_index: choppiness index (high=choppy, low=trending)
- garch_vol_z: GARCH volatility z-score
- ou_halflife: OU process mean-reversion halflife (bars)
- jump_flag: 1 if price jump detected
- evt_tail_flag: 1 if extreme tail event
- regime_bull/bear/chop: current market regime flags
- sig_*: strategy signals (-1/0/1)
Task:
1. Summarize the current market state in 2-3 sentences.
2. Identify the dominant signal pattern.
3. Give a directional verdict: LONG / SHORT / NEUTRAL with confidence (low/medium/high).
4. State the key reasons (max 3 bullet points).
5. Suggest a stop-loss level based on recent volatility.

Answer in Korean.
"""


def ask_ollama(prompt: str, model: str) -> str:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 1024,
        }
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("response", "")
    except urllib.error.URLError as e:
        print(f"\n[오류] Ollama 서버에 연결할 수 없습니다: {e}")
        print("  → ollama serve 명령으로 서버를 먼저 시작하세요.")
        print("  → 미설치 시: curl -fsSL https://ollama.ai/install.sh | sh")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="DeepSeek-R1으로 ETH 방향성 예측")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="입력 CSV 경로")
    parser.add_argument("--rows", type=int, default=20, help="분석할 최근 캔들 수 (기본 20)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama 모델명")
    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv)
    if not os.path.exists(csv_path):
        print(f"[오류] CSV 파일 없음: {csv_path}")
        sys.exit(1)

    print(f"[1/3] CSV 읽는 중: {csv_path} (최근 {args.rows}행)")
    rows = read_last_rows(csv_path, args.rows)
    if not rows:
        print("[오류] 데이터가 없습니다.")
        sys.exit(1)

    print(f"[2/3] 프롬프트 생성 중... ({len(rows)}개 캔들)")
    data_text = format_rows_for_prompt(rows)
    prompt = build_prompt(data_text, len(rows))

    print(f"[3/3] {args.model} 에 질의 중... (최대 120초)\n")
    print("─" * 60)
    answer = ask_ollama(prompt, args.model)
    print(answer)
    print("─" * 60)


if __name__ == "__main__":
    main()
