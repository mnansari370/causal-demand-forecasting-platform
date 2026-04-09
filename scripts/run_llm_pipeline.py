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
    Build the main progression table for the report using actual saved outputs.
    """
    try:
        w1 = json.loads((results_dir / "week1_baseline_results.json").read_text(encoding="utf-8"))
        w2 = json.loads((results_dir / "week2_forecasting_results.json").read_text(encoding="utf-8"))
        did = json.loads((results_dir / "causal_did_result.json").read_text(encoding="utf-8"))

        baseline_test = next((r for r in w1 if r.get("evaluation_split") == "test"), {})
        lgbm_point = next((r for r in w2 if r.get("model") == "LightGBM Point"), {})
        lgbm_quant = next((r for r in w2 if "Quantile" in str(r.get("model", ""))), {})

        cv_eval_path = results_dir / "cv_evaluation_results.json"
        cv_macro_f1 = None
        if cv_eval_path.exists():
            cv_eval = json.loads(cv_eval_path.read_text(encoding="utf-8"))
            cv_macro_f1 = cv_eval.get("macro_f1")

        rows = [
            (
                "Baseline (Seasonal Naive)",
                baseline_test.get("rmse"),
                baseline_test.get("mae"),
                "—",
                "No",
                "No",
                "No",
            ),
            (
                "+ LightGBM Point Forecast",
                lgbm_point.get("rmse"),
                lgbm_point.get("mae"),
                "—",
                "No",
                "No",
                "No",
            ),
            (
                "+ Probabilistic Intervals",
                lgbm_quant.get("rmse"),
                lgbm_quant.get("mae"),
                lgbm_quant.get("coverage_90"),
                "No",
                "No",
                "No",
            ),
            (
                "+ Causal Inference Module",
                lgbm_quant.get("rmse"),
                lgbm_quant.get("mae"),
                lgbm_quant.get("coverage_90"),
                f"DiD ATT={did.get('estimate')} (p={did.get('p_value')})",
                "No",
                "No",
            ),
            (
                "+ Visual Anomaly Detector",
                lgbm_quant.get("rmse"),
                lgbm_quant.get("mae"),
                lgbm_quant.get("coverage_90"),
                f"DiD ATT={did.get('estimate')}",
                f"Macro F1={cv_macro_f1}" if cv_macro_f1 is not None else "Yes",
                "No",
            ),
            (
                "Full System (All Modules)",
                lgbm_quant.get("rmse"),
                lgbm_quant.get("mae"),
                lgbm_quant.get("coverage_90"),
                f"DiD ATT={did.get('estimate')}",
                f"Macro F1={cv_macro_f1}" if cv_macro_f1 is not None else "Yes",
                "Yes",
            ),
        ]

        header = (
            f"{'System Configuration':<40} "
            f"{'RMSE':>8} {'MAE':>7} {'Cov@90':>8} "
            f"{'Causal':>25} {'Vision':>18} {'LLM':>10}"
        )
        sep = "-" * len(header)
        lines = [sep, header, sep]

        for config_name, rmse, mae, cov90, causal, vision, llm in rows:
            lines.append(
                f"{config_name:<40} "
                f"{str(rmse or 'N/A'):>8} "
                f"{str(mae or 'N/A'):>7} "
                f"{str(cov90 or 'N/A'):>8} "
                f"{str(causal):>25} "
                f"{str(vision):>18} "
                f"{str(llm):>10}"
            )

        lines.append(sep)
        return "\n".join(lines)

    except Exception as exc:
        return f"[Could not build results table: {exc}]"


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")

    logger = get_logger(
        "run_llm_pipeline",
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

    # ------------------------------------------------------------------
    # STAGE 1 — Build context
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 1: Building analytics context")
    logger.info("=" * 60)

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
    logger.info("Context saved")

    # ------------------------------------------------------------------
    # STAGE 2 — Run evaluation queries
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 2: Running %d evaluation queries", len(EVAL_QUERIES))
    logger.info("=" * 60)

    responses = []
    total_input_tokens = 0
    total_output_tokens = 0
    n_success = 0

    for i, question in enumerate(EVAL_QUERIES, start=1):
        logger.info("[%d/%d] Query: %s", i, len(EVAL_QUERIES), question[:80])

        result = query_llm(
            question=question,
            context_str=context_str,
            provider=provider,
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        row = {
            "query_id": i,
            "question": question,
            "answer": result["answer"],
            "model_used": result["model_used"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "success": result["success"],
            "error": result["error"],
        }
        responses.append(row)

        if result["success"]:
            n_success += 1
            total_input_tokens += result["input_tokens"]
            total_output_tokens += result["output_tokens"]

        logger.info(
            "  [%d/%d] %s | tokens_in=%d out=%d",
            i,
            len(EVAL_QUERIES),
            "OK" if result["success"] else "MOCK/FAIL",
            result["input_tokens"],
            result["output_tokens"],
        )

        print("\n" + "=" * 70)
        print(f"Q{i}: {question}")
        print("-" * 70)
        preview = result["answer"]
        print(preview[:700] + ("..." if len(preview) > 700 else ""))

    # ------------------------------------------------------------------
    # STAGE 3 — Save responses + human evaluation CSV
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 3: Saving responses and human evaluation template")
    logger.info("=" * 60)

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

    logger.info("Saved responses and human eval template")

    # ------------------------------------------------------------------
    # STAGE 4 — Main results table
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 4: Building main results table")
    logger.info("=" * 60)

    results_table = build_main_results_table(results_dir)
    (results_dir / "main_results_table.txt").write_text(
        results_table,
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("WEEK 6 — LLM PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Queries run:         {len(EVAL_QUERIES)}")
    print(f"Successful API:      {n_success}")
    print(f"Total input tokens:  {total_input_tokens:,}")
    print(f"Total output tokens: {total_output_tokens:,}")

    print("\nMAIN RESULTS TABLE:")
    print(results_table)

    print("\nOUTPUTS:")
    print(f"  Context JSON:      {results_dir / 'llm_context.json'}")
    print(f"  Context prompt:    {results_dir / 'llm_context_prompt.txt'}")
    print(f"  LLM responses:     {results_dir / 'llm_responses.json'}")
    print(f"  Human eval CSV:    {results_dir / 'llm_human_eval.csv'}")
    print(f"  Results table:     {results_dir / 'main_results_table.txt'}")
    print("=" * 70)

    logger.info("Week 6 LLM pipeline complete")


if __name__ == "__main__":
    main()