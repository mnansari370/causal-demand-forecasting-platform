from __future__ import annotations

import os
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a Senior Data Science Analyst at a retail company.

You have access only to outputs from a demand forecasting, causal inference,
elasticity, anomaly detection, and decision-support platform.

RULES:
1. Use only the numbers and facts in the provided CONTEXT.
2. Never invent values, metrics, or business events.
3. If something is not available in the CONTEXT, say:
   "This information is not available in the current results."
4. When citing a number, mention the model or method that produced it.
5. Keep the explanation clear for a non-technical retail manager.
6. When discussing causal estimates, include confidence interval and p-value when available.
7. When discussing anomalies, state the anomaly type and confidence score when available.
8. You are explaining model outputs, not creating new analysis.

Style:
- Be concise
- Prefer short paragraphs
- Use bullets only when they improve clarity
"""


EVAL_QUERIES = [
    "How much did the LightGBM model improve forecast accuracy compared to the baseline?",
    "What does the 90% prediction interval coverage mean in practice for inventory planning?",
    "Why might MAPE be a misleading metric for this dataset?",
    "Which forecasting model performed best and why would you choose it over the others?",
    "What does the interval width of the quantile forecast tell us about demand uncertainty?",
    "What is the true causal impact of promotions on demand, according to the analysis?",
    "Why is the naive promotion estimate misleading?",
    "What does the selection bias finding mean for how the business currently measures promotion ROI?",
    "Did the placebo test pass, and what does that tell us about the DiD methodology?",
    "Which items respond most strongly to promotions, and how confident are we in those estimates?",
    "Which product families are most sensitive to promotions and by how much?",
    "What does the panel regression coefficient mean in plain English?",
    "How should a retail manager use the promotion sensitivity estimates to plan next quarter's promotions?",
    "Why might some product families respond less strongly to promotions?",
    "What is the confidence interval on the demand uplift from promotions?",
    "How accurate is the visual anomaly detector, and which anomaly type is hardest to detect?",
    "Which series were flagged as anomalous in the test data, and what type of anomaly was detected?",
    "What is the difference between a spike anomaly and a structural break in demand terms?",
    "Why use a visual CNN classifier for anomaly detection instead of statistical thresholds?",
    "What action should a retail manager take when a DROP anomaly is detected?",
    "Generate a weekly executive summary based on all available model outputs.",
    "Which stores should receive promotions in the next campaign based on causal estimates?",
    "If a store shows a structural break anomaly and low promotion sensitivity, what might that suggest?",
    "How do the forecasting results, causal estimates, and anomaly detections work together?",
    "What are the top 3 business insights from this analysis platform?",
]


def _mock_response(question: str) -> dict[str, Any]:
    answer = (
        "[MOCK RESPONSE — no live API call was made]\n\n"
        f"Question: {question}\n\n"
        "Based on the saved platform outputs:\n"
        "- LightGBM outperformed the seasonal naive baseline on forecasting.\n"
        "- Promotions showed a statistically significant causal impact in the DiD analysis.\n"
        "- The anomaly detector flagged several real series for review.\n\n"
        "To enable live LLM responses, configure a supported provider and set the API key."
    )
    return {
        "answer": answer,
        "model_used": "mock",
        "input_tokens": 0,
        "output_tokens": 0,
        "success": False,
        "error": "No supported API key/provider available",
    }


def _query_anthropic(
    question: str,
    context_str: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set; using mock response")
        return _mock_response(question)

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

        user_message = (
            "Here is the current analytics context from the platform.\n\n"
            f"{context_str}\n\n"
            f"USER QUESTION: {question}\n\n"
            "Answer using only the context above."
        )

        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
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
            "answer": f"[LLM query failed: {exc}]",
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

    Supported providers:
    - anthropic
    - mock
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