"""
model_sensitivity_benchmark.py
================================
Benchmark script for Reviewer Question 1:
  "How robust is the agentic planner across model versions, model sizes,
   and prompt formulations?"

What this script measures
--------------------------
  1. Tool-calling accuracy  (Acc_tool)  — correct action selected
  2. Parameter accuracy     (Acc_param) — correct experiment name + key args
  3. JSON parse success     (JSON_ok)   — response is valid parseable JSON

Benchmark dimensions
---------------------
  A. Model sweep    : multiple Ollama models (size / family sensitivity)
  B. Prompt variant : 3 system-prompt formulations (prompt sensitivity)

Usage
-----
  # Run full benchmark (all models × all prompt variants × all queries)
  python benchmarks/model_sensitivity_benchmark.py

  # Quick smoke-test with one model
  python benchmarks/model_sensitivity_benchmark.py --models qwen2.5:7b --quick

  # Specific models only
  python benchmarks/model_sensitivity_benchmark.py --models qwen2.5:3b qwen2.5:7b llama3.1:8b

Outputs (written to benchmarks/results/)
------------------------------------------
  model_sensitivity_<timestamp>.csv   — per-query raw results
  model_sensitivity_<timestamp>.json  — summary tables (Table 1 + Table 2 for paper)
  model_sensitivity_<timestamp>.png   — heatmap figure

Requirements
------------
  pip install langchain-ollama pandas matplotlib seaborn tabulate
  Ollama must be running locally with at least one model pulled.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Resolve project root so imports work regardless of cwd
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# ---------------------------------------------------------------------------
# Benchmark query set  (30 queries × 6 categories)
# ---------------------------------------------------------------------------
# Each entry:
#   query        : user utterance sent to the router
#   expected_action : the correct JSON "action" value
#   expected_exp    : correct "experiment" value ("" if not applicable)
#   expected_args   : dict of key args that MUST appear (subset check)
#   category        : one of the 6 evaluation categories
#   notes           : human-readable rationale

BENCHMARK_QUERIES: List[Dict[str, Any]] = [
    # ── 1. Single-tool, explicit path ─────────────────────────────────────
    {
        "id": "ST01",
        "query": "Convert the raw .spe files in project_root/00_raw_data/Fe2p_batch01 to CSV.",
        "expected_action": "run",
        "expected_exp": "tool_xps_reader",
        "expected_args": {},
        "category": "single_tool_explicit",
    },
    {
        "id": "ST02",
        "query": "Run the XPS reader on project_root/00_raw_data/Si2p_set02.",
        "expected_action": "run",
        "expected_exp": "tool_xps_reader",
        "expected_args": {},
        "category": "single_tool_explicit",
    },
    {
        "id": "ST03",
        "query": "Fit the peaks for the spectra in 01_converted_csv/.",
        "expected_action": "run",
        "expected_exp": "tool_xps_fitter",
        "expected_args": {},
        "category": "single_tool_explicit",
    },
    {
        "id": "ST04",
        "query": "Calculate atomic percentages from the fitted results.",
        "expected_action": "run",
        "expected_exp": "tool_xps_quantifier",
        "expected_args": {},
        "category": "single_tool_explicit",
    },
    {
        "id": "ST05",
        "query": "Generate overlay plots for all fitted spectra.",
        "expected_action": "run",
        "expected_exp": "tool_xps_plotter",
        "expected_args": {},
        "category": "single_tool_explicit",
    },
    # ── 2. Full pipeline ───────────────────────────────────────────────────
    {
        "id": "FP01",
        "query": "Run the complete XPS analysis on all files in project_root/00_raw_data/.",
        "expected_action": "run",
        "expected_exp": "full_analysis",
        "expected_args": {},
        "category": "full_pipeline",
    },
    {
        "id": "FP02",
        "query": "Process everything end-to-end: convert, fit, quantify, and plot.",
        "expected_action": "run",
        "expected_exp": "full_analysis",
        "expected_args": {},
        "category": "full_pipeline",
    },
    {
        "id": "FP03",
        "query": "Run the full workflow on the LIB SEI dataset.",
        "expected_action": "run",
        "expected_exp": "full_analysis",
        "expected_args": {},
        "category": "full_pipeline",
    },
    {
        "id": "FP04",
        "query": "Run tool_real_xps_workflow on project_root.",
        "expected_action": "run",
        "expected_exp": "tool_real_xps_workflow",
        "expected_args": {},
        "category": "full_pipeline",
    },
    {
        "id": "FP05",
        "query": "Execute the production XPS pipeline including triage and quality gate.",
        "expected_action": "run",
        "expected_exp": "tool_real_xps_workflow",
        "expected_args": {},
        "category": "full_pipeline",
    },
    # ── 3. Map routing ─────────────────────────────────────────────────────
    {
        "id": "MAP01",
        "query": "Process the hyperspectral C1s map in project_root/00_raw_data/map_C1s.csv.",
        "expected_action": "run",
        "expected_exp": "tool_xps_mapper",
        "expected_args": {},
        "category": "map_routing",
    },
    {
        "id": "MAP02",
        "query": "Run MCR-ALS decomposition on the 2D XPS map data.",
        "expected_action": "run",
        "expected_exp": "tool_xps_mapper",
        "expected_args": {},
        "category": "map_routing",
    },
    {
        "id": "MAP03",
        "query": "Triage the file project_root/00_raw_data/O1s_map.csv to check if it is map data.",
        "expected_action": "triage",
        "expected_exp": "tool_xps_triage",
        "expected_args": {"file_path": "project_root/00_raw_data/O1s_map.csv"},
        "category": "map_routing",
    },
    {
        "id": "MAP04",
        "query": "Analyze the hyperspectral dataset with PCA clustering and component maps.",
        "expected_action": "run",
        "expected_exp": "tool_xps_mapper",
        "expected_args": {},
        "category": "map_routing",
    },
    {
        "id": "MAP05",
        "query": "Detect the data type of file.csv — is it a map or standard spectra?",
        "expected_action": "triage",
        "expected_exp": "tool_xps_triage",
        "expected_args": {},
        "category": "map_routing",
    },
    # ── 4. Ambiguous / clarification required ─────────────────────────────
    {
        "id": "CL01",
        "query": "Fit these spectra.",
        "expected_action": "clarify",
        "expected_exp": "",
        "expected_args": {},
        "category": "clarification",
    },
    {
        "id": "CL02",
        "query": "Group the samples by treatment condition.",
        "expected_action": "clarify",
        "expected_exp": "tool_create_groups",
        "expected_args": {},
        "category": "clarification",
    },
    {
        "id": "CL03",
        "query": "Run the correlator.",
        "expected_action": "clarify",
        "expected_exp": "",
        "expected_args": {},
        "category": "clarification",
    },
    {
        "id": "CL04",
        "query": "Process sample B.",
        "expected_action": "clarify",
        "expected_exp": "",
        "expected_args": {},
        "category": "clarification",
    },
    {
        "id": "CL05",
        "query": "Run quality check.",
        "expected_action": "clarify",
        "expected_exp": "tool_quality_gatekeeper",
        "expected_args": {},
        "category": "clarification",
    },
    # ── 5. Quality-gated rejection ────────────────────────────────────────
    {
        "id": "QG01",
        "query": "Run the quality gate on project_root/00_raw_data/noisy_Fe2p.csv to check if SNR is acceptable.",
        "expected_action": "run",
        "expected_exp": "tool_quality_gatekeeper",
        "expected_args": {},
        "category": "quality_gate",
    },
    {
        "id": "QG02",
        "query": "Validate data integrity for all files before fitting.",
        "expected_action": "run",
        "expected_exp": "tool_workflow_orchestrator",
        "expected_args": {},
        "category": "quality_gate",
    },
    {
        "id": "QG03",
        "query": "Check if sample_B7.csv passes the quality gate.",
        "expected_action": "run",
        "expected_exp": "tool_quality_gatekeeper",
        "expected_args": {},
        "category": "quality_gate",
    },
    {
        "id": "QG04",
        "query": "Run triage and quality validation on project_root/00_raw_data/LIB_batch03.spe.",
        "expected_action": "run",
        "expected_exp": "tool_workflow_orchestrator",
        "expected_args": {"file_path": "project_root/00_raw_data/LIB_batch03.spe"},
        "category": "quality_gate",
    },
    {
        "id": "QG05",
        "query": "Check the resolution and SNR of the survey spectra before processing.",
        "expected_action": "run",
        "expected_exp": "tool_quality_gatekeeper",
        "expected_args": {},
        "category": "quality_gate",
    },
    # ── 6. General / metadata / none ─────────────────────────────────────
    {
        "id": "GN01",
        "query": "What experiments are available?",
        "expected_action": "none",
        "expected_exp": "",
        "expected_args": {},
        "category": "general_query",
    },
    {
        "id": "GN02",
        "query": "What does the XPS fitter do?",
        "expected_action": "none",
        "expected_exp": "",
        "expected_args": {},
        "category": "general_query",
    },
    {
        "id": "GN03",
        "query": "List all supported raw file formats.",
        "expected_action": "none",
        "expected_exp": "",
        "expected_args": {},
        "category": "general_query",
    },
    {
        "id": "GN04",
        "query": "How does the triage router decide between map and standard workflow?",
        "expected_action": "none",
        "expected_exp": "",
        "expected_args": {},
        "category": "general_query",
    },
    {
        "id": "GN05",
        "query": "What is the default energy calibration reference?",
        "expected_action": "none",
        "expected_exp": "",
        "expected_args": {},
        "category": "general_query",
    },
]

# ---------------------------------------------------------------------------
# Models to benchmark
# ---------------------------------------------------------------------------
DEFAULT_MODELS = [
    "qwen2.5:3b",
    "qwen2.5:7b",
    "qwen2.5:14b",
    "llama3.1:8b",
    "mistral:7b",
    "gemma3:4b",
]

# ---------------------------------------------------------------------------
# System-prompt variants  (for prompt sensitivity analysis)
# ---------------------------------------------------------------------------

def _load_system_prompt_base() -> str:
    """Load the baseline SYSTEM_INSTRUCTION_BASE from the router module."""
    try:
        from llm_manager.experiment_router import SYSTEM_INSTRUCTION_BASE
        return SYSTEM_INSTRUCTION_BASE
    except ImportError:
        # Fallback minimal version
        return (
            'You are a controller for local python jobs/workflows.\n'
            'Return ONLY a single JSON object with fields:\n'
            '{"action": "run"|"clarify"|"none"|"execute"|"triage", '
            '"experiment": "<name>", "args": {}, "message": "<text>"}'
        )


def build_prompt_variants(base: str) -> Dict[str, str]:
    """
    Return 3 prompt variants:
      V1  baseline   — current system prompt (verbatim)
      V2  no_examples — same but with the Guidelines section removed
      V3  reworded   — action descriptions paraphrased
    """
    variants: Dict[str, str] = {}

    # V1: baseline
    variants["V1_baseline"] = base

    # V2: remove the Guidelines block (everything from "Guidelines:" onward)
    no_examples = re.sub(
        r'\nGuidelines:.*', '',
        base,
        flags=re.DOTALL,
    ).strip()
    no_examples += (
        '\nRespond with the JSON object only.'
    )
    variants["V2_no_guidelines"] = no_examples

    # V3: reword the action labels
    reworded = base.replace(
        '"run"', '"execute_tool"'
    ).replace(
        '"clarify"', '"ask_user"'
    ).replace(
        '"none"', '"answer_directly"'
    ).replace(
        '"triage"', '"detect_data_type"'
    ).replace(
        '- "run":', '- "execute_tool":'
    ).replace(
        '- "clarify":', '- "ask_user":'
    ).replace(
        '- "none":', '- "answer_directly":'
    ).replace(
        '- "execute":', '- "run_code":'
    )
    variants["V3_reworded_actions"] = reworded

    return variants


# ---------------------------------------------------------------------------
# JSON parsing helper (same logic as the real router)
# ---------------------------------------------------------------------------
_JSON_RE = re.compile(r'\{.*\}', re.DOTALL)

def parse_response(text: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Returns (json_valid, parsed_dict_or_None, error_msg).
    """
    m = _JSON_RE.search(text or "")
    if not m:
        return False, None, "No JSON object found"
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return False, None, str(e)
    if "action" not in obj:
        return False, None, "Missing 'action' field"
    if obj["action"] not in ("run", "clarify", "none", "execute", "triage"):
        return False, None, f"Invalid action: {obj['action']}"
    return True, obj, ""


# ---------------------------------------------------------------------------
# Accuracy helpers
# ---------------------------------------------------------------------------

def check_tool_accuracy(pred: Optional[Dict], expected: Dict) -> bool:
    """True when the predicted action matches the expected action."""
    if pred is None:
        return False
    return pred.get("action", "") == expected["expected_action"]


def check_param_accuracy(pred: Optional[Dict], expected: Dict) -> bool:
    """
    True when:
      - action is correct  AND
      - experiment name matches (empty string is a wildcard for non-run actions) AND
      - all keys in expected_args appear in pred['args'] (subset check)
    """
    if not check_tool_accuracy(pred, expected):
        return False
    exp_exp = expected["expected_exp"]
    if exp_exp and pred.get("experiment", "") != exp_exp:
        return False
    for k, v in expected["expected_args"].items():
        if k not in pred.get("args", {}):
            return False
    return True


# ---------------------------------------------------------------------------
# Single query runner
# ---------------------------------------------------------------------------

def run_single_query(
    llm: ChatOllama,
    system_prompt: str,
    user_query: str,
    retries: int = 2,
) -> Tuple[bool, Optional[Dict], str, float, str]:
    """
    Returns (json_valid, parsed, error, latency_s, raw_response).
    Retries on parse failure up to `retries` times.
    """
    system_msg = SystemMessage(content=system_prompt)
    human_msg = HumanMessage(content=f"User request:\n{user_query}")
    msgs = [system_msg, human_msg]

    last_raw = ""
    for attempt in range(retries + 1):
        t0 = time.perf_counter()
        try:
            resp = llm.invoke(msgs)
            latency = time.perf_counter() - t0
            raw = resp.content
            last_raw = raw
            valid, parsed, err = parse_response(raw)
            if valid:
                return valid, parsed, err, latency, raw
        except Exception as exc:
            latency = time.perf_counter() - t0
            last_raw = str(exc)
            if attempt == retries:
                return False, None, str(exc), latency, last_raw

    return False, None, "Max retries exceeded", latency, last_raw


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    models: List[str],
    queries: List[Dict],
    prompt_variants: Dict[str, str],
    output_dir: Path,
    temperature: float = 0.0,
    num_ctx: int = 4096,
    delay_between_calls: float = 0.5,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run the full benchmark matrix: models × prompt_variants × queries.

    Returns
    -------
    df      : raw per-query results DataFrame
    summary : nested dict suitable for JSON export (Tables 1 & 2)
    """
    records = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(models) * len(prompt_variants) * len(queries)
    done = 0

    for model in models:
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"{'='*60}")

        for variant_name, system_prompt in prompt_variants.items():
            print(f"\n  Prompt variant: {variant_name}")

            try:
                llm = ChatOllama(
                    model=model,
                    temperature=temperature,
                    num_ctx=num_ctx,
                    model_kwargs={"format": "json"},
                )
            except Exception as e:
                print(f"  [SKIP] Could not initialise {model}: {e}")
                for q in queries:
                    records.append(_make_error_record(model, variant_name, q, str(e)))
                continue

            for q in queries:
                done += 1
                print(f"  [{done}/{total}] {q['id']} ({q['category']}) ...", end=" ", flush=True)

                json_ok, parsed, err, latency, raw = run_single_query(
                    llm, system_prompt, q["query"]
                )
                acc_tool = check_tool_accuracy(parsed, q)
                acc_param = check_param_accuracy(parsed, q)

                predicted_action = parsed.get("action", "") if parsed else ""
                predicted_exp = parsed.get("experiment", "") if parsed else ""

                status = "✓" if acc_tool else "✗"
                print(f"{status}  action={predicted_action!r}  latency={latency:.2f}s")

                records.append({
                    "model": model,
                    "prompt_variant": variant_name,
                    "query_id": q["id"],
                    "category": q["category"],
                    "query": q["query"],
                    "expected_action": q["expected_action"],
                    "expected_exp": q["expected_exp"],
                    "predicted_action": predicted_action,
                    "predicted_exp": predicted_exp,
                    "json_ok": json_ok,
                    "acc_tool": acc_tool,
                    "acc_param": acc_param,
                    "latency_s": round(latency, 3),
                    "error": err,
                    "raw_response": raw[:300],  # truncate for readability
                })

                if delay_between_calls > 0:
                    time.sleep(delay_between_calls)

    df = pd.DataFrame(records)

    # ── Table 1: Model sensitivity (baseline prompt only) ─────────────────
    df_base = df[df["prompt_variant"] == "V1_baseline"].copy()
    table1_rows = []
    for model in models:
        m_df = df_base[df_base["model"] == model]
        if m_df.empty:
            continue
        # Approximate parameter size from name
        size = _extract_size(model)
        row = {
            "Model": model,
            "Size": size,
            "Acc_tool (%)": round(100 * m_df["acc_tool"].mean(), 1),
            "Acc_param (%)": round(100 * m_df["acc_param"].mean(), 1),
            "JSON_ok (%)": round(100 * m_df["json_ok"].mean(), 1),
            "Latency_mean (s)": round(m_df["latency_s"].mean(), 2),
            "N_queries": len(m_df),
        }
        # Per-category breakdown
        for cat in sorted(m_df["category"].unique()):
            cat_df = m_df[m_df["category"] == cat]
            row[f"Acc_tool_{cat} (%)"] = round(100 * cat_df["acc_tool"].mean(), 1)
        table1_rows.append(row)

    # ── Table 2: Prompt sensitivity (best model from Table 1, all variants) ─
    # Pick the model with highest overall Acc_tool on baseline
    if table1_rows:
        best_model = max(table1_rows, key=lambda r: r["Acc_tool (%)"])["Model"]
    else:
        best_model = models[0] if models else "unknown"

    df_best = df[df["model"] == best_model].copy()
    table2_rows = []
    for variant in sorted(df_best["prompt_variant"].unique()):
        v_df = df_best[df_best["prompt_variant"] == variant]
        table2_rows.append({
            "Prompt_variant": variant,
            "Acc_tool (%)": round(100 * v_df["acc_tool"].mean(), 1),
            "Acc_param (%)": round(100 * v_df["acc_param"].mean(), 1),
            "JSON_ok (%)": round(100 * v_df["json_ok"].mean(), 1),
            "N_queries": len(v_df),
        })

    summary = {
        "timestamp": ts,
        "n_queries": len(queries),
        "n_models": len(models),
        "n_prompt_variants": len(prompt_variants),
        "best_model_for_prompt_sensitivity": best_model,
        "table1_model_sensitivity": table1_rows,
        "table2_prompt_sensitivity": table2_rows,
    }

    # ── Save outputs ──────────────────────────────────────────────────────
    csv_path = output_dir / f"model_sensitivity_{ts}.csv"
    json_path = output_dir / f"model_sensitivity_{ts}.json"
    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n\nResults saved to:\n  {csv_path}\n  {json_path}")
    _print_tables(summary)
    _save_heatmap(df, output_dir, ts)

    return df, summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_error_record(model, variant, q, err):
    return {
        "model": model, "prompt_variant": variant,
        "query_id": q["id"], "category": q["category"], "query": q["query"],
        "expected_action": q["expected_action"], "expected_exp": q["expected_exp"],
        "predicted_action": "", "predicted_exp": "",
        "json_ok": False, "acc_tool": False, "acc_param": False,
        "latency_s": 0.0, "error": err, "raw_response": "",
    }


def _extract_size(model_name: str) -> str:
    """Extract a human-readable size tag from the model name."""
    m = re.search(r'(\d+(?:\.\d+)?)[bB]', model_name)
    return f"{m.group(1)}B" if m else "?"


def _print_tables(summary: Dict) -> None:
    try:
        from tabulate import tabulate
        print("\n\n── Table 1: Model Sensitivity (baseline prompt) ──")
        print(tabulate(summary["table1_model_sensitivity"], headers="keys", tablefmt="grid"))
        print("\n── Table 2: Prompt Sensitivity (best model) ──")
        print(tabulate(summary["table2_prompt_sensitivity"], headers="keys", tablefmt="grid"))
    except ImportError:
        print("\n[INFO] Install tabulate for formatted table output: pip install tabulate")
        print(json.dumps(summary["table1_model_sensitivity"], indent=2))


def _save_heatmap(df: pd.DataFrame, output_dir: Path, ts: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        df_base = df[df["prompt_variant"] == "V1_baseline"].copy()
        pivot = df_base.pivot_table(
            index="model",
            columns="category",
            values="acc_tool",
            aggfunc="mean"
        ) * 100

        fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(df["model"].unique()) * 0.8 + 2)))

        # Heatmap 1: model × category accuracy
        sns.heatmap(
            pivot, annot=True, fmt=".0f", cmap="RdYlGn",
            vmin=0, vmax=100, linewidths=0.5, ax=axes[0],
            cbar_kws={"label": "Acc_tool (%)"}
        )
        axes[0].set_title("Table 1 — Tool-Calling Accuracy\nby Model × Query Category (baseline prompt)")
        axes[0].set_xlabel("Query Category")
        axes[0].set_ylabel("Model")

        # Heatmap 2: prompt variant × category (best model only)
        best_model = df_base.groupby("model")["acc_tool"].mean().idxmax()
        df_best = df[df["model"] == best_model].copy()
        pivot2 = df_best.pivot_table(
            index="prompt_variant",
            columns="category",
            values="acc_tool",
            aggfunc="mean"
        ) * 100

        sns.heatmap(
            pivot2, annot=True, fmt=".0f", cmap="RdYlGn",
            vmin=0, vmax=100, linewidths=0.5, ax=axes[1],
            cbar_kws={"label": "Acc_tool (%)"}
        )
        axes[1].set_title(
            f"Table 2 — Prompt Sensitivity\nModel: {best_model}"
        )
        axes[1].set_xlabel("Query Category")
        axes[1].set_ylabel("Prompt Variant")

        plt.tight_layout()
        fig_path = output_dir / f"model_sensitivity_{ts}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Figure saved to: {fig_path}")
    except ImportError:
        print("[INFO] Install matplotlib + seaborn for heatmap output.")
    except Exception as e:
        print(f"[WARN] Could not save heatmap: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="XPS LLM Router — Model & Prompt Sensitivity Benchmark"
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Ollama model names to benchmark (default: see DEFAULT_MODELS in script)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Use first 5 queries only (smoke test)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="LLM sampling temperature (default 0.0 for determinism)"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "benchmarks" / "results",
        help="Directory to write results (default: benchmarks/results/)"
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Seconds to wait between API calls (default 0.5)"
    )
    parser.add_argument(
        "--prompt-variants-only", action="store_true",
        help="Only run prompt sensitivity (skip model sweep, use first model)"
    )
    args = parser.parse_args()

    models = args.models or DEFAULT_MODELS
    if args.prompt_variants_only:
        models = [models[0]]

    queries = BENCHMARK_QUERIES[:5] if args.quick else BENCHMARK_QUERIES

    base_prompt = _load_system_prompt_base()
    prompt_variants = build_prompt_variants(base_prompt)

    print("XPS LLM Router — Model Sensitivity Benchmark")
    print(f"  Models         : {models}")
    print(f"  Prompt variants: {list(prompt_variants.keys())}")
    print(f"  Queries        : {len(queries)}")
    print(f"  Temperature    : {args.temperature}")
    print(f"  Output dir     : {args.output_dir}")
    print()

    run_benchmark(
        models=models,
        queries=queries,
        prompt_variants=prompt_variants,
        output_dir=args.output_dir,
        temperature=args.temperature,
        delay_between_calls=args.delay,
    )


if __name__ == "__main__":
    main()
