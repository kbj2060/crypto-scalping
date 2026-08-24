#!/usr/bin/env python3
"""재량 매매 결정 로거 (룰북: docs/eth_discretionary_manual_strategy_rulebook_20260824.md)

사용법:
  python scripts/log_discretionary_decision_20260824.py          # 새 결정 기록 (대화형)
  python scripts/log_discretionary_decision_20260824.py --list   # 최근 기록 보기
  python scripts/log_discretionary_decision_20260824.py --stats  # 이탈률/건수 요약 (룰북 §6 판정용)

로그: data/discretionary/decision_log.jsonl (append 전용, 1결정 = 1줄 JSON)

스키마 (빈 입력으로 건너뛴 optional 필드는 null):
  ts_utc          str   결정 시각 UTC ISO8601 (정본)
  ts_kst          str   같은 시각 KST (가독용)
  symbol          str   기본 ETHUSDT
  action          str   entry | exit_tp | exit_sl | skip | note
  direction       str?  long | short (entry/skip 시)
  price           float 결정 시점 가격 (Binance fapi 자동 조회, 수정 가능)
  size_usdt       float? 증거금 (entry 시)
  leverage        float? 레버리지 (entry 시)
  rule_ids        list  근거 룰 ID (예: E1, X1) — 룰북 §3~4
  liq_support     float? 참조한 청산 지지 레벨
  liq_resistance  float? 참조한 청산 저항 레벨
  poc             float? VPVR POC 레벨
  stoch_k         float? Stoch RSI K
  stoch_d         float? Stoch RSI D
  stop_price      float? 손절 레벨 (entry 시)
  target_price    float? 익절 레벨 (entry 시)
  reason          str   자유 서술 근거
  rule_deviation  bool  룰북에 없는/어긋난 판단이 섞였는가
  deviation_note  str?  이탈 내용 (rule_deviation=true 시)
"""
import argparse
import json
import sys
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_PATH = REPO_ROOT / "data" / "discretionary" / "decision_log.jsonl"
KST = timezone(timedelta(hours=9))

ACTIONS = ["entry", "exit_tp", "exit_sl", "skip", "note"]
RULE_MENU = "E1(롱진입) E2(숏진입) S1(패스) X1(청산레벨 손절) X2(익절) X3(시간청산)"


def fetch_price(symbol: str):
    url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return float(json.load(resp)["price"])
    except Exception:
        return None


def ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    val = input(f"{prompt}{suffix}: ").strip()
    return val if val else (default or "")


def ask_float(prompt: str, default: float | None = None):
    while True:
        raw = ask(prompt + " (빈칸=건너뜀)" if default is None else prompt,
                  None if default is None else f"{default}")
        if not raw:
            return default
        try:
            return float(raw.replace(",", ""))
        except ValueError:
            print("  숫자가 아닙니다. 다시 입력하세요.")


def ask_choice(prompt: str, choices: list[str], default: str | None = None) -> str:
    while True:
        val = ask(f"{prompt} ({'/'.join(choices)})", default)
        if val in choices:
            return val
        print(f"  {choices} 중 하나를 입력하세요.")


def cmd_log(log_path: Path):
    symbol = ask("심볼", "ETHUSDT").upper()
    action = ask_choice("액션 entry=진입 exit_tp=익절 exit_sl=손절 skip=의도적패스 note=메모",
                        ACTIONS)

    direction = None
    if action in ("entry", "skip"):
        direction = ask_choice("방향", ["long", "short"])

    live = fetch_price(symbol)
    if live is not None:
        price = ask_float("가격", live)
    else:
        print("  (Binance 가격 자동 조회 실패 — 직접 입력)")
        price = ask_float("가격")
        while price is None:
            price = ask_float("가격(필수)")

    size_usdt = leverage = stop_price = target_price = None
    if action == "entry":
        size_usdt = ask_float("증거금 USDT")
        leverage = ask_float("레버리지")
        stop_price = ask_float("손절 레벨")
        target_price = ask_float("익절 레벨")

    rule_raw = ask(f"근거 룰 ID 콤마구분 — {RULE_MENU}", "")
    rule_ids = [r.strip().upper() for r in rule_raw.split(",") if r.strip()]

    print("-- 결정 당시 본 레벨/지표 (빈칸=건너뜀) --")
    liq_support = ask_float("청산 지지 레벨")
    liq_resistance = ask_float("청산 저항 레벨")
    poc = ask_float("VPVR POC")
    stoch_k = ask_float("Stoch RSI K")
    stoch_d = ask_float("Stoch RSI D")

    reason = ask("결정 근거 (자유 서술)")
    deviation = ask_choice("룰북에 없는/어긋난 판단이 섞였나", ["y", "n"], "n") == "y"
    deviation_note = ask("이탈 내용") if deviation else None

    now = datetime.now(timezone.utc)
    record = {
        "ts_utc": now.isoformat(timespec="seconds"),
        "ts_kst": now.astimezone(KST).isoformat(timespec="seconds"),
        "symbol": symbol,
        "action": action,
        "direction": direction,
        "price": price,
        "size_usdt": size_usdt,
        "leverage": leverage,
        "rule_ids": rule_ids,
        "liq_support": liq_support,
        "liq_resistance": liq_resistance,
        "poc": poc,
        "stoch_k": stoch_k,
        "stoch_d": stoch_d,
        "stop_price": stop_price,
        "target_price": target_price,
        "reason": reason,
        "rule_deviation": deviation,
        "deviation_note": deviation_note,
    }

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    n = sum(1 for _ in log_path.open(encoding="utf-8"))
    print(f"\n기록 완료 → {log_path} (총 {n}건)")


def load_records(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    return [json.loads(line) for line in log_path.open(encoding="utf-8") if line.strip()]


def cmd_list(log_path: Path, limit: int):
    records = load_records(log_path)
    if not records:
        print(f"기록 없음 ({log_path})")
        return
    for r in records[-limit:]:
        dev = " ⚠이탈" if r.get("rule_deviation") else ""
        dir_str = f" {r['direction']}" if r.get("direction") else ""
        rules = ",".join(r.get("rule_ids") or []) or "-"
        print(f"{r['ts_kst']}  {r['symbol']} {r['action']}{dir_str} @ {r['price']}"
              f"  룰[{rules}]{dev}  {r.get('reason', '')}")


def cmd_stats(log_path: Path):
    records = load_records(log_path)
    if not records:
        print(f"기록 없음 ({log_path})")
        return
    n = len(records)
    dev = sum(1 for r in records if r.get("rule_deviation"))
    print(f"총 {n}건 (판정 기준: 20건 이상)")
    for a in ACTIONS:
        c = sum(1 for r in records if r["action"] == a)
        if c:
            print(f"  {a}: {c}건")
    print(f"룰 이탈: {dev}건 ({dev / n * 100:.1f}%) — 판정 기준: 20% 미만")
    dev_notes = [r["deviation_note"] for r in records
                 if r.get("rule_deviation") and r.get("deviation_note")]
    if dev_notes:
        print("이탈 내용 (숨은 규칙 후보):")
        for note in dev_notes:
            print(f"  - {note}")


def main():
    parser = argparse.ArgumentParser(description="재량 매매 결정 로거")
    parser.add_argument("--list", action="store_true", help="최근 기록 보기")
    parser.add_argument("--stats", action="store_true", help="이탈률/건수 요약")
    parser.add_argument("--limit", type=int, default=10, help="--list 표시 건수")
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH,
                        help="로그 파일 경로 재정의(테스트용)")
    args = parser.parse_args()

    if args.list:
        cmd_list(args.log_path, args.limit)
    elif args.stats:
        cmd_stats(args.log_path)
    else:
        try:
            cmd_log(args.log_path)
        except (KeyboardInterrupt, EOFError):
            print("\n취소됨 (기록 안 함)")
            sys.exit(1)


if __name__ == "__main__":
    main()
