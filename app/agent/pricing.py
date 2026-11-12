"""Published Groq on-demand prices, used only when a Groq call actually ran.

Source checked 2026-11-12: https://console.groq.com/docs/models
- openai/gpt-oss-20b: $0.075 / 1M input, $0.30 / 1M output
- openai/gpt-oss-120b: $0.15 / 1M input, $0.60 / 1M output

llama-3.1-8b-instant left the free and developer tiers on 2026-08-16.
Its last public list price was $0.05 / $0.08 per 1M tokens. It is kept so an
older key still gets a cost estimate. Unknown models return None rather than
a guessed price. The deterministic scorer does not call Groq, so its cost is 0.
"""

from __future__ import annotations

# USD per 1M tokens: (input, output)
GROQ_USD_PER_MILLION: dict[str, tuple[float, float]] = {
    "openai/gpt-oss-20b": (0.075, 0.30),
    "openai/gpt-oss-120b": (0.15, 0.60),
    "llama-3.1-8b-instant": (0.05, 0.08),
}


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    price = GROQ_USD_PER_MILLION.get(model)
    if price is None:
        return None
    input_rate, output_rate = price
    return (prompt_tokens / 1_000_000) * input_rate + (completion_tokens / 1_000_000) * output_rate
