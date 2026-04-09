"""
LLM analytics assistant pipeline.

This script:
1. builds a structured context from saved outputs,
2. formats that context into an LLM-ready prompt,
3. runs a fixed evaluation question set,
4. saves responses and a human-evaluation template,
5. builds a compact main results table.

The pipeline works in mock mode by default.
To use Anthropic, set ANTHROPIC_API_KEY and update the config provider.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.load_data import load_config
from src.llm.assistant import EVAL_QUERIES, query_llm
from src.llm.context_builder import build_context, format_context_for_prompt
from src.utils.logger import get_logger


def build_main_results_table(results_dir: Path) -> str:
    """
    Build the main progression table using saved project outputs.
    """
    try:
        forecasting = json.loads((results_dir / "forecasting_results.json").read_text(encoding="utf-8"))
        did = json.loads((results_dir / "causal_did_result.json").read_text(encoding="utf-8"))

        def find_model(rows: list[dict], name_substring: str) -> dict:
            return next((row for row in rows if name_substring in str(row.get("model", ""))), {})

        naive = find_model(forecasting, "Naive")
        lgbm_point = find_model(forecasting, "LightGBM Point")
        lgbm_quantile = find_model(forecasting, "Quantile")

        cv_eval_path = results_dir / "cv_evaluation_results.json"
        cv_macro_f1 = None
        if cv_eval_path.exists():
            cv_eval = json.loads(cv_eval_path.read_text(encoding="utf-8"))
            cv_macro_f1 = cv_eval.get("macro_f1")

        rows = [
            (
                "Seasonal Naive (baseline)",
                naive.get("rmse"),
                naive.get("mae"),
                "—",
                "No",
                "No",
                "No",
            ),
            (
                "+ LightGBM point forecast",
                lgbm_point.get("rmse"),
                lgbm_point.get("mae"),
                "—",
                "No",
                "No",
                "No",
            ),
            (
                "+ Probabilistic intervals",
                lgbm_quantile.get("rmse"),
                lgbm_quantile.get("mae"),
                lgbm_quantile.get("coverage_90"),
                "No",
                "No",
                "No",
            ),
            (
                "+ Causal inference",
                lgbm_quantile.get("rmse"),
                lgbm_quantile.get("mae"),
                lgbm_quantile.get("coverage_90"),
                f"ATT={did.get('estimate')} (p={did.get('p_value')})",
                "No",
                "No",
            ),
            (
                "+ Visual anomaly detector",
                lgbm_quantile.get("rmse"),
                lgbm_quantile.get("mae"),
                lgbm_quantile.get("coverage_90"),
                f"ATT={did.get('estimate')}",
                f"macro F1={cv_macro_f1}" if cv_macro_f1 is not None else "Yes",
                "No",
            ),
            (
                "Full system",
                lgbm_quantile.get("rmse"),
                lgbm_quantile.get("mae"),
                lgbm_quantile.get("coverage_90"),
                f"ATT={did.get('estimate')}",
                f"macro F1={cv_macro_f1}" if cv_macro_f1 is not None else "Yes",
                "Yes",
            ),
        ]

        header = (
            f"{'Configuration':<42} "
            f"{'RMSE':>8} {'MAE':>6} {'Cov@90':>7} "
            f"{'Causal':>28} {'Vision':>18} {'LLM':>5}"
        )
        separator = "-" * len(header)
        lines = [separator, header, separator]

        for cfg, rmse, mae, cov, causal, vision, llm in rows:
            lines.append(
                f"{cfg:<42} "
                f"{str(rmse or 'N/A'):>8} "
                f"{str(mae or 'N/A'):>6} "
                f"{str(cov or 'N/A'):>7} "
                f"{str(causal):>28} "
                f"{str(vision):>18} "
                f"{str(llm):>5}"
            )

        lines.append(separator)
        return "\n".join(lines)

    except Exception as exc:
        return f"[Could not build results table: {exc}]"


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    logger = get_logger(
        "llm_pipeline",
        log_dir=PROJECT_ROOT / config["logs"]["log_dir"],
        level=config["logs"]["log_level"],
    )

    results_dir = PROJECT_ROOT / config["evaluation"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    llm_cfg = config.get("llm", {})
    provider = llm_cfg.get("provider", "mock")
    model_name = llm_cfg.get("model_name", "claude-3-5-haiku-latest")
    max_tokens = llm_cfg.get("max_tokens", 600)
    temperature = llm_cfg.get("temperature", 0.2)

    logger.info("Building analytics context from saved outputs")
    context = build_context(results_dir)
    context_str = format_context_for_prompt(context)

    (results_dir / "llm_context.json").write_text(
        json.dumps(context, indent=2, default=str),
        encoding="utf-8",
    )
    (results_dir / "llm_context_prompt.txt").write_text(
        context_str,
        encoding="utf-8",
    )

    logger.info("Context keys: %s", list(context.keys()))

    logger.info("Running %d evaluation queries (provider=%s)", len(EVAL_QUERIES), provider)
    responses = []
    n_success = 0

    for i, question in enumerate(EVAL_QUERIES, start=1):
        logger.info("[%d/%d] %s", i, len(EVAL_QUERIES), question[:80])

        result = query_llm(
            question=question,
            context_str=context_str,
            provider=provider,
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        responses.append(
            {
                "query_id": i,
                "question": question,
                "answer": result["answer"],
                "model_used": result["model_used"],
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "success": result["success"],
            }
        )

        if result["success"]:
            n_success += 1

        print("\n" + "=" * 60)
        print(f"Q{i}: {question}")
        print("-" * 60)
        preview = result["answer"]
        print(preview[:600] + ("..." if len(preview) > 600 else ""))

    (results_dir / "llm_responses.json").write_text(
        json.dumps(responses, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    eval_csv_path = results_dir / "llm_human_eval.csv"
    with open(eval_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "query_id",
                "question",
                "answer_preview",
                "accuracy_1_5",
                "usefulness_1_5",
                "groundedness_1_5",
                "clarity_1_5",
                "notes",
            ]
        )

        for row in responses:
            writer.writerow(
                [
                    row["query_id"],
                    row["question"],
                    row["answer"][:180].replace("\n", " "),
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )

    results_table = build_main_results_table(results_dir)
    (results_dir / "main_results_table.txt").write_text(
        results_table,
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("LLM PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Queries run:       {len(EVAL_QUERIES)}")
    print(f"Successful (API):  {n_success}")
    print(f"\nMAIN RESULTS TABLE:\n{results_table}")
    print(f"\nOutputs saved to: {results_dir}")
    print("=" * 70)

    logger.info("LLM pipeline complete")


if __name__ == "__main__":
    main()