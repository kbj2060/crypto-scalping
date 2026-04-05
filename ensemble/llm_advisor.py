"""
LLM 매매 어드바이저 — DeepSeek API 사용.

DSAC 3종(primary/long/short) + M7 앙상블 + 시장 피처를 받아
{ decision, conviction, size, tp, sl } 5개 필드를 반환한다.

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

_SYSTEM_PROMPT = (
    "You are a professional quantitative trader for ETHUSDT 5-minute perpetual futures. "
    "You receive signals from three RL agents and a 7-model ML ensemble. "
    "Respond ONLY with a single valid JSON object — no markdown, no explanation."
)

_OUTPUT_SCHEMA = """{
  "decision": "LONG" | "SHORT" | "HOLD" | "EXIT" | "REDUCE" | "ADD",
  "conviction": <integer 0-100>,
  "size": <integer 0-100>,
  "tp": <float price or null>,
  "sl": <float price or null>
}"""

_RULES = """Rules:
- decision=ADD only if already in position and signals strengthen
- decision=REDUCE only if in position and signals weaken
- size=0 when decision=HOLD or EXIT
- If m7_gate_block=1 or any agent uncertainty > 0.8 → conviction <= 40
- If regime=chop or whipsaw and no strong agreement → HOLD"""


@dataclass
class LLMDecision:
    decision: str        # LONG/SHORT/HOLD/EXIT/REDUCE/ADD
    conviction: int      # 0-100
    size: int            # 0-100 (% of max position)
    tp: float | None
    sl: float | None

    def __str__(self) -> str:
        tp_str = f"{self.tp:.2f}" if self.tp else "—"
        sl_str = f"{self.sl:.2f}" if self.sl else "—"
        return (
            f"[LLM] {self.decision}  확신={self.conviction}%  "
            f"비중={self.size}%  TP={tp_str}  SL={sl_str}"
        )


def _build_prompt(ctx: dict[str, Any]) -> str:
    pos = ctx.get("position_type") or "NONE"
    regime = ctx.get("regime", "unknown")

    lines = [
        "=== MARKET ===",
        f"price={ctx.get('close', 0):.2f}  regime={regime}  "
        f"log_ret={ctx.get('log_return', 0):+.4f}",
        f"garch_z={ctx.get('garch_vol_z', 0):.2f}  "
        f"jump_z={ctx.get('jump_z', 0):.2f}  "
        f"evt_z={ctx.get('evt_excess_z', 0):.2f}",
        f"funding={ctx.get('last_funding_rate', 0):+.6f}  "
        f"pressure={ctx.get('funding_pressure', 0):.4f}",
        "",
        "=== AGENTS ===",
        f"primary : action={ctx.get('primary_action', 0)}  "
        f"lev={ctx.get('primary_lev', 0):.2f}  "
        f"std={ctx.get('primary_std', 0):.3f}",
        f"long    : action={ctx.get('long_action', 0)}  "
        f"lev={ctx.get('long_lev', 0):.2f}  "
        f"logit={ctx.get('long_logit', 0):+.3f}  "
        f"std={ctx.get('long_std', 0):.3f}",
        f"short   : action={ctx.get('short_action', 0)}  "
        f"lev={ctx.get('short_lev', 0):.2f}  "
        f"logit={ctx.get('short_logit', 0):+.3f}  "
        f"std={ctx.get('short_std', 0):.3f}",
        "",
        "=== M7 ENSEMBLE ===",
        f"DN={ctx.get('m7_prob_dn', 0):.2f}  "
        f"FL={ctx.get('m7_prob_fl', 0):.2f}  "
        f"UP={ctx.get('m7_prob_up', 0):.2f}  "
        f"conf={ctx.get('m7_confidence', 0):.2f}  "
        f"gate_block={ctx.get('m7_gate_block', 0)}",
        f"Q10={ctx.get('m7_q10', 0):+.4f}  "
        f"Q50={ctx.get('m7_q50', 0):+.4f}  "
        f"Q90={ctx.get('m7_q90', 0):+.4f}",
        f"tp_offset={ctx.get('m7_tp_offset', 0):+.4f}  "
        f"sl_offset={ctx.get('m7_sl_offset', 0):+.4f}",
        "",
        "=== ELITE SIGNALS ===",
        f"whale={ctx.get('sig_whale', 0):+.2f}  "
        f"oi_div={ctx.get('sig_oi_divergence', 0):+.2f}  "
        f"vol_confirm={ctx.get('sig_volume_confirm', 0):+.2f}  "
        f"trend_health={ctx.get('sig_trend_health', 0):+.2f}",
        "",
        "=== POSITION ===",
        f"type={pos}  "
        f"entry={ctx.get('entry_price', 0):.2f}  "
        f"unrealized={ctx.get('unrealized_pnl', 0):+.4f}  "
        f"hold={ctx.get('hold_count', 0)}candles",
        "",
        "=== CONSENSUS ===",
        f"agreement={ctx.get('agreement_count', 0)}/3  "
        f"net_score={ctx.get('net_score', 0):+.3f}  "
        f"kelly={ctx.get('kelly', 0):.3f}",
        "",
        _RULES,
        "",
        f"Output schema: {_OUTPUT_SCHEMA}",
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

    valid = {"LONG", "SHORT", "HOLD", "EXIT", "REDUCE", "ADD"}
    decision = str(d.get("decision", "HOLD")).upper()
    if decision not in valid:
        decision = "HOLD"

    return LLMDecision(
        decision=decision,
        conviction=int(max(0, min(100, d.get("conviction", 0)))),
        size=int(max(0, min(100, d.get("size", 0)))),
        tp=float(d["tp"]) if d.get("tp") not in (None, 0, "") else None,
        sl=float(d["sl"]) if d.get("sl") not in (None, 0, "") else None,
    )


def _call_deepseek(prompt: str, api_key: str, model: str, timeout: int) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 128,
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
                lambda: _call_deepseek(prompt, self.api_key, self.model, self.timeout),
            )
            decision = _parse_response(raw)
            if decision is None:
                logger.warning("LLMAdvisor: 응답 파싱 실패 — raw=%s", raw[:120])
            return decision
        except Exception as e:
            logger.warning("LLMAdvisor: API 호출 실패 — %s", e)
            return None
