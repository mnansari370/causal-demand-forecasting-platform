"""
LLM analytics assistant.

The LLM is used as the explanation layer of the platform, not as the source
of truth. All quantitative reasoning is done upstream by the forecasting,
causal inference, promotion analysis, and anomaly detection modules.

This module only takes a structured context and turns it into readable
business-language answers.
"""
from __future__ import annotations

import os
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a Senior Data Science Analyst at a retail company.

You have access only to outputs from a demand forecasting, causal inference,
promotion sensitivity, anomaly detection, and decision intelligence platform.

Rules:
1. Use only the numbers and facts in the provided CONTEXT.
2. Never invent values, metrics, or business events not in the context.
3. If something is not in the context, say:
   "This information is not available in the current results."
4. When citing a number, mention the model or method that produced it.
5. Keep explanations accessible to a non-technical retail manager.
6. When discussing causal estimates, include confidence intervals and p-values.
7. When discussing anomalies, state the anomaly type and confidence score.
8. You are explaining outputs, not generating new analysis.

Style:
- concise
- clear short paragraphs
- use bullets only when they genuinely improve clarity
"""

EVAL_QUERIES = [
    "How much did the LightGBM model improve forecast accuracy compared to the baseline?",
    "What does the 90% prediction interval coverage mean in practice for inventory planning?",
    "Why might MAPE be a misleading metric for this dataset?",
    "Which forecasting model performed best and why would you choose it?",
    "What does the interval width of the quantile forecast tell us about demand uncertainty?",
    "What is the true causal impact of promotions on demand, according to the analysis?",
    "Why is the naive promotion estimate misleading?",
    "What does the selection bias finding mean for how the business currently measures promotion ROI?",
    "Did the placebo test pass, and what does that tell us about the DiD methodology?",
    "Which items respond most strongly to promotions, and how confident are we?",
    "Which product families are most sensitive to promotions and by how much?",
    "What does the panel regression coefficient mean in plain English?",
    "How should a retail manager use promotion sensitivity estimates to plan next quarter?",
    "Why might some product families respond less strongly to promotions?",
    "What is the confidence interval on the demand uplift from promotions?",
    "How accurate is the visual anomaly detector, and which class is hardest to detect?",
    "Which series were flagged as anomalous in the test data?",
    "What is the difference between a spike anomaly and a structural break in demand terms?",
    "Why use a visual CNN classifier for anomaly detection instead of statistical thresholds?",
    "What action should a retail manager take when a drop anomaly is detected?",
    "Generate a weekly executive summary based on all available model outputs.",
    "Which stores should receive promotions in the next campaign based on causal estimates?",
    "If a store shows a structural break anomaly and low promotion sensitivity, what might that suggest?",
    "How do forecasting, causal inference, and anomaly detection work together in this platform?",
    "What are the top 3 business insights from this analysis?",
]


def _mock_response(question: str) -> dict[str, Any]:
    """
    Return a placeholder response when no live API is configured.
    """
    return {
        "answer": (
            "[Mock response — no API key configured]\n\n"
            f"Question: {question}\n\n"
            "To enable live responses, set ANTHROPIC_API_KEY and change "
            "llm.provider to 'anthropic' in configs/base.yaml."
        ),
        "model_used": "mock",
        "input_tokens": 0,
        "output_tokens": 0,
        "success": False,
        "error": "No API key configured",
    }


def _query_anthropic(
    question: str,
    context_str: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    """
    Query Anthropic if an API key is available.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set — falling back to mock response")
        return _mock_response(question)

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"CONTEXT:\n{context_str}\n\n"
                        f"QUESTION: {question}\n\n"
                        "Answer using only the context above."
                    ),
                }
            ],
        )

        answer = response.content[0].text if response.content else ""
        usage = getattr(response, "usage", None)

        return {
            "answer": answer,
            "model_used": model,
            "input_tokens": getattr(usage, "input_tokens", 0),
            "output_tokens": getattr(usage, "output_tokens", 0),
            "success": True,
            "error": None,
        }

    except Exception as exc:
        logger.error("Anthropic query failed: %s", exc)
        return {
            "answer": f"[Query failed: {exc}]",
            "model_used": model,
            "input_tokens": 0,
            "output_tokens": 0,
            "success": False,
            "error": str(exc),
        }


def query_llm(
    question: str,
    context_str: str,
    provider: str = "mock",
    model: str = "claude-3-5-haiku-latest",
    max_tokens: int = 600,
    temperature: float = 0.2,
) -> dict[str, Any]:
    """
    Query the configured LLM provider.
    """
    provider = (provider or "mock").strip().lower()

    if provider == "anthropic":
        return _query_anthropic(
            question=question,
            context_str=context_str,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    logger.info("Using mock LLM provider")
    return _mock_response(question)