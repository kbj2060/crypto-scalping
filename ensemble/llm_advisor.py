"""
LLM 매매 어드바이저 — DeepSeek API 사용.

Playbook Router(HFT/MFT) 평가 JSON을 입력으로
{ reasoning, decision, confidence_score, kelly_weight }를 반환한다.

환경변수:
  LLM_ENABLED        : True/False (기본 True)
  LLM_API_KEY        : DeepSeek API 키
  DEEPSEEK_LLM_MODEL : 모델명 (기본 deepseek-chat)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

_SYSTEM_PROMPT = """[System Role]
You are the Chief Investment Officer (CIO) of a systematic quantitative fund.
Your job is to synthesize Hierarchical Playbook evaluations into one final execution decision.

[Market Context & Philosophy]
- Asset: BTC perpetual futures (long/short)
- Minimize overtrading and fee drag.
- You receive signals from 6 Unified Macro/Micro Playbook Groups.

[Data Interpretation Rules]
You must interpret the 6 unified playbooks by their strict hierarchical roles:

1) VETO & CRISIS (HFT Brakes - Top Priority):
- "PB_VETO_SHIELD": Extreme toxicity or liquidity vacuum. MUST HOLD.
- "PB_CRISIS_SNIPER": Flash crash or fake breakout trap. If active, prioritize its counter-trend direction.

2) DIRECTIONAL CORE (MFT Engine):
- "PB_TREND_SIGNAL", "PB_WHALE_SIGNAL", "PB_MEAN_REVERT_SIGNAL": These define the macro trend based on smart money accumulation and volume. Base your core direction (LONG/SHORT) on these.

3) SQUEEZE IGNITION:
- "PB_SQUEEZE_SNIPER": Indicates a highly explosive moment. If it aligns with the macro trend, increase kelly_weight.

[Strict Trading Rules]
1) VETO: If PB_VETO_SHIELD is matched -> decision MUST be HOLD.
2) OVERRIDE: If PB_CRISIS_SNIPER is matched -> prioritize its direction.
3) CONFLICT: If directional core signals contradict each other or market is too flat -> HOLD.
4) SYNERGY: If MFT core signals and Squeeze Ignition align -> Confident entry.

[Output Contract]
Respond with exactly one JSON object only (no markdown, no prose outside JSON):
{
  "reasoning": "2-3 concise sentences summarizing why based on the unified groups",
  "decision": "LONG" | "SHORT" | "HOLD",
  "confidence_score": <integer 0-100>,
  "kelly_weight": <float 0.0-1.0>
}
"""

_OUTPUT_SCHEMA = """{
  "reasoning": "<2-3 concise sentences>",
  "decision": "LONG" | "SHORT" | "HOLD",
  "confidence_score": <integer 0-100>,
  "kelly_weight": <float 0.0-1.0>
}"""


@dataclass
class LLMDecision:
    decision: str        # LONG/SHORT/HOLD
    conviction: int      # 0-100
    size: int            # 0-100 (% of max position)
    tp: float | None
    sl: float | None
    reasoning: str = ""

    def __str__(self) -> str:
        tp_str = f"{self.tp:.2f}" if self.tp else "—"
        sl_str = f"{self.sl:.2f}" if self.sl else "—"
        return (
            f"[LLM] {self.decision}  확신={self.conviction}%  "
            f"비중={self.size}%  TP={tp_str}  SL={sl_str}"
        )


def _sanitize_tp_sl(decision: str, tp: float | None, sl: float | None, ref_price: float) -> tuple[float | None, float | None]:
    """Make TP/SL directionally consistent with decision using current price as anchor."""
    if ref_price <= 0.0:
        return tp, sl
    d = str(decision).upper()
    t = float(tp) if tp is not None else None
    s = float(sl) if sl is not None else None
    if d == "LONG":
        # LONG: TP above, SL below current price.
        if t is not None and t <= ref_price and s is not None and s >= ref_price:
            t, s = s, t  # likely swapped output
        if t is not None and t <= ref_price:
            t = None
        if s is not None and s >= ref_price:
            s = None
    elif d == "SHORT":
        # SHORT: TP below, SL above current price.
        if t is not None and t >= ref_price and s is not None and s <= ref_price:
            t, s = s, t  # likely swapped output
        if t is not None and t >= ref_price:
            t = None
        if s is not None and s <= ref_price:
            s = None
    return t, s


def _build_prompt(ctx: dict[str, Any]) -> str:
    payload = (
        ctx.get("llm_router_payload")
        or {
            "portfolio_state": ctx.get("portfolio_state", {}),
            "market_environment": ctx.get("market_environment", {}),
            "playbook_consensus": ctx.get("playbook_consensus", {}),
        }
    )
    try:
        payload_json = json.dumps(payload, ensure_ascii=False)
    except Exception:
        payload_json = "{}"

    lines = [
        "Use the following structured JSON input:",
        payload_json,
        "",
        "Return ONLY JSON with this schema:",
        _OUTPUT_SCHEMA,
    ]
    return "\n".join(lines)


def _parse_response(raw: str) -> LLMDecision | None:
    match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
    if not match:
        return None
    try:
        d = json.loads(match.group())
    except json.JSONDecodeError:
        return None

    valid = {"LONG", "SHORT", "HOLD"}
    decision = str(d.get("decision", "LONG")).upper()
    if decision not in valid:
        decision = "HOLD"

    # New schema mapping (reasoning/decision/confidence_score/kelly_weight)
    conviction_raw = d.get("confidence_score", d.get("conviction", 0))
    try:
        conviction = int(max(0, min(100, int(round(float(conviction_raw))))))
    except Exception:
        conviction = 0

    kelly_raw = d.get("kelly_weight", None)
    if kelly_raw is None:
        size_raw = d.get("size", 0)
        try:
            size = int(max(0, min(100, int(round(float(size_raw))))))
        except Exception:
            size = 0
    else:
        try:
            size = int(max(0, min(100, int(round(float(kelly_raw) * 100.0)))))
        except Exception:
            size = 0

    return LLMDecision(
        decision=decision,
        conviction=conviction,
        size=size,
        tp=float(d["tp"]) if d.get("tp") not in (None, 0, "") else None,
        sl=float(d["sl"]) if d.get("sl") not in (None, 0, "") else None,
        reasoning=str(d.get("reasoning", "") or ""),
    )


def _call_deepseek(
    prompt: str,
    api_key: str,
    model: str,
    timeout: int,
    max_tokens: int,
    temperature: float = 0.7,
) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "temperature": float(temperature),
        "top_p": 0.7,
        "max_tokens": int(max_tokens),
    }).encode("utf-8")

    req = urllib.request.Request(
        _DEEPSEEK_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result["choices"][0]["message"]["content"]


class LLMAdvisor:
    """DeepSeek API 기반 비동기 매매 어드바이저."""

    def __init__(self) -> None:
        self.enabled: bool = os.getenv("LLM_ENABLED", "True").lower() == "true"
        self.api_key: str  = os.getenv("LLM_API_KEY", "")
        self.model: str    = os.getenv("DEEPSEEK_LLM_MODEL", "deepseek-chat")
        self.timeout: int  = int(os.getenv("LLM_TIMEOUT_SEC", "8"))
        self.max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "320"))
        self.last_advice: dict[str, Any] = {}

        if self.enabled and not self.api_key:
            logger.warning("LLMAdvisor: LLM_API_KEY 미설정 — 비활성화")
            self.enabled = False

    async def advise(self, ctx: dict[str, Any]) -> LLMDecision | None:
        """ctx 딕셔너리로 LLM 호출 후 LLMDecision 반환. 실패 시 None."""
        if not self.enabled:
            return None
        try:
            prompt = _build_prompt(ctx)
            loop = asyncio.get_event_loop()
            raw = await loop.run_in_executor(
                None,
                lambda: _call_deepseek(
                    prompt,
                    self.api_key,
                    self.model,
                    self.timeout,
                    self.max_tokens,
                    0.6,
                ),
            )
            decision = _parse_response(raw)
            if decision is None:
                logger.warning("LLMAdvisor: 1차 응답 파싱 실패 — raw=%s", raw[:200])
                repair_prompt = (
                    f"{prompt}\n\n"
                    "IMPORTANT: Reply with one compact JSON object only. "
                    "Keep reasoning concise (1 sentence)."
                )
                raw_retry = await loop.run_in_executor(
                    None,
                    lambda: _call_deepseek(
                        repair_prompt,
                        self.api_key,
                        self.model,
                        self.timeout,
                        max(self.max_tokens, 420),
                        0.2,
                    ),
                )
                decision = _parse_response(raw_retry)
                if decision is None:
                    logger.warning("LLMAdvisor: 2차 응답 파싱 실패 — raw=%s", raw_retry[:200])
                    return None
            ref_price = float(ctx.get("close", 0.0) or 0.0)
            decision.tp, decision.sl = _sanitize_tp_sl(decision.decision, decision.tp, decision.sl, ref_price)
            self.last_advice = {
                "decision": str(decision.decision),
                "confidence_score": int(decision.conviction),
                "kelly_weight": float(decision.size) / 100.0,
                "reasoning": str(decision.reasoning),
                "tp": decision.tp,
                "sl": decision.sl,
                "model": self.model,
            }
            return decision
        except Exception as e:
            logger.warning("LLMAdvisor: API 호출 실패 — %s", e)
            return None
