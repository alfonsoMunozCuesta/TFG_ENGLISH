"""
Simple local AI benchmark for the techniques available in this Dash app.

Measure the same performance metrics as the original benchmark:
response time, response length, approximate RAM usage, and approximate CPU usage.
Prompts use only the techniques and variables available in this application:
Kaplan-Meier, Cox regression, and the Log-Rank test.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from lifelines import CoxPHFitter, KaplanMeierFitter

from log_rank_test import perform_log_rank_test


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = BASE_DIR / "dataset_limpio.csv"
DEFAULT_ENDPOINT = "http://127.0.0.1:11434/api/chat"
DEFAULT_MODEL = "llama3:latest"

DASH_VARIABLES = [
    {
        "column": "gender_F",
        "label": "Gender",
        "groups": {0: "Male", 1: "Female"},
    },
    {
        "column": "disability_N",
        "label": "Disability",
        "groups": {0: "Without disability", 1: "With disability"},
    },
]

DEFAULT_PROMPT = (
    "Provide a brief academic interpretation of dropout results from the "
    "survival dashboard using Kaplan-Meier, Cox regression, or the "
    "Log-Rank test as appropriate."
)


def _parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments for the benchmark."""
    parser = argparse.ArgumentParser(
        description="Measure response times and approximate resource usage of the local AI server."
    )
    parser.add_argument(
        "--mode",
        choices=("techniques", "repeat"),
        default="techniques",
        help=(
            "techniques measures queries for Dash techniques; repeat sends the same prompt several times."
        ),
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Repetitions per technique in techniques mode, or number of queries in repeat mode.",
    )
    parser.add_argument("--warmup", type=int, default=1, help="Unmeasured warmup queries.")
    parser.add_argument("--timeout", type=int, default=180, help="Timeout per query in seconds.")
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="Chat completions compatible endpoint.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt used in repeat mode.")
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=512,
        help="Ollama context size. A smaller value reduces KV cache memory usage.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=180,
        help="Maximum generated tokens per response.",
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET_PATH),
        help="Clean CSV used by Dash.",
    )
    parser.add_argument(
        "--process-name",
        default="ollama",
        help="Process name used to estimate RAM/CPU usage.",
    )
    parser.add_argument(
        "--out-dir",
        default="benchmarks",
        help="Directory for CSV and JSON results.",
    )
    return parser.parse_args()


def _load_dataset(dataset_path: str) -> pd.DataFrame:
    """Load the clean dataset with the minimum columns used by Dash."""
    df = pd.read_csv(dataset_path, sep=";")
    required = {"date", "final_result", "gender_F", "disability_N"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"""Missing dataset columns: {', '.join(missing)}""")
    return df


def _query_process_snapshot(process_name: str) -> dict | None:
    """Return accumulated CPU time and RAM usage for a Windows process, if found."""
    if platform.system().lower() != "windows":
        return None

    ps_script = f"""
    $p = Get-Process -ErrorAction SilentlyContinue |
        Where-Object {{ $_.ProcessName -like '*{process_name}*' }} |
        Sort-Object WorkingSet64 -Descending |
        Select-Object -First 1 Id,ProcessName,CPU,WorkingSet64
    if ($null -ne $p) {{ $p | ConvertTo-Json -Compress }}
    """

    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return None

    output = completed.stdout.strip()
    if not output:
        return None

    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return None

    return {
        "pid": data.get("Id"),
        "name": data.get("ProcessName"),
        "cpu_seconds": float(data.get("CPU") or 0.0),
        "ram_mb": round(float(data.get("WorkingSet64") or 0.0) / (1024 * 1024), 2),
    }


def _build_payload(prompt: str, model: str, endpoint: str, num_ctx: int, max_tokens: int) -> dict:
    """Build the JSON body sent to the model endpoint."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are an academic assistant who interprets "
                "survival analysis results clearly, cautiously, and briefly."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    if endpoint.rstrip("/").endswith("/api/chat"):
        return {
            "model": model,
            "messages": messages,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
                "num_ctx": num_ctx,
                "num_predict": max_tokens,
            },
            "stream": False,
        }

    return {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": max_tokens,
        "stream": False,
    }


def _call_model(
    prompt: str,
    timeout: int,
    endpoint: str,
    model: str,
    num_ctx: int,
    max_tokens: int,
) -> tuple[str, float]:
    """Call the model, measure response time, and return the generated text."""
    start = time.perf_counter()
    response = requests.post(
        endpoint,
        json=_build_payload(prompt, model, endpoint, num_ctx, max_tokens),
        timeout=timeout,
    )
    elapsed = time.perf_counter() - start
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(f"""{exc}. Server response: {response.text}""") from exc

    data = response.json()
    if "choices" in data:
        content = data["choices"][0]["message"]["content"]
    else:
        content = data["message"]["content"]
    return content, elapsed


def _survival_probability(df: pd.DataFrame, day: int) -> float:
    """Calculate approximate Kaplan-Meier survival on a given day."""
    kmf = KaplanMeierFitter()
    kmf.fit(df["date"], event_observed=df["final_result"])
    return float(kmf.predict(day))


def _format_counts(df: pd.DataFrame, column: str, groups: dict[int, str]) -> str:
    """Return a readable summary of group frequencies."""
    counts = df[column].value_counts().sort_index()
    return ", ".join(f"{groups.get(int(value), value)}: {int(count)}" for value, count in counts.items())


def _build_kaplan_prompt(df: pd.DataFrame, variable: dict | None = None) -> dict:
    """Create a global or covariate-stratified Kaplan-Meier prompt."""
    total = len(df)
    events = int(df["final_result"].sum())
    min_day = int(df["date"].min())
    max_day = int(df["date"].max())
    mean_day = round(float(df["date"].mean()), 2)
    surv_100 = round(_survival_probability(df, 100), 3)
    surv_200 = round(_survival_probability(df, 200), 3)

    if variable is None:
        group_parts = []
        for item in DASH_VARIABLES:
            group_parts.append(
                f"{item['label']} ({_format_counts(df, item['column'], item['groups'])})"
            )
        return {
            "technique": "Kaplan-Meier",
            "prompt": (
                "Provide a brief academic interpretation of the global Kaplan-Meier curve "
                "from the academic dropout dashboard. "
                f"Dataset: {total} students, {events} dropout events, "
                f"follow-up between {min_day} and {max_day} days, mean {mean_day} days. "
                f"Estimated survival is {surv_100} at day 100 and {surv_200} "
                "at day 200. The variables available for stratifying curves "
                f"in this Dash app are: {'; '.join(group_parts)}."
            ),
        }

    column = variable["column"]
    groups = variable["groups"]
    group_parts = []
    for value, label in groups.items():
        subset = df[df[column] == value]
        if subset.empty:
            continue
        group_parts.append(
            f"{label}: n={len(subset)}, dropouts={int(subset['final_result'].sum())}, "
            f"S(100)={round(_survival_probability(subset, 100), 3)}, "
            f"S(200)={round(_survival_probability(subset, 200), 3)}"
        )

    return {
        "technique": "Kaplan-Meier",
        "prompt": (
            "Provide a brief academic interpretation of the Kaplan-Meier curves in the "
            f"dashboard, comparing groups by {variable['label']}. "
            f"Group summary: {'; '.join(group_parts)}."
        ),
    }


def _build_cox_prompt(df: pd.DataFrame, covariables: list[str], variable_label: str) -> dict:
    """Create a Cox regression prompt using actual Dash covariates."""
    df_cox = df[["date", "final_result"] + covariables].copy()
    cph = CoxPHFitter()
    cph.fit(df_cox, duration_col="date", event_col="final_result")
    summary = cph.summary.reset_index()

    parts = []
    for _, row in summary.iterrows():
        parts.append(
            f"{row['covariate']}: HR={round(float(row['exp(coef)']), 3)}, "
            f"coef={round(float(row['coef']), 3)}, p={round(float(row['p']), 4)}"
        )

    return {
        "technique": "Cox Regression",
        "prompt": (
            "Provide a brief academic interpretation of Cox regression in the "
            "academic dropout dashboard. "
            f"Included covariates: {', '.join(covariables)}. "
            f"Results: {'; '.join(parts)}. "
            "Explain how the covariates affect dropout probability."
        ),
    }


def _build_logrank_prompt(df: pd.DataFrame) -> dict:
    """Create a Log-Rank test prompt using actual Dash covariates."""
    parts = []
    for variable in DASH_VARIABLES:
        result = perform_log_rank_test(df, variable["column"])
        row = result.iloc[0]
        parts.append(
            f"{variable['label']} ({_format_counts(df, variable['column'], variable['groups'])}): "
            f"statistic={round(float(row['test_statistic']), 4)}, "
            f"p-value={round(float(row['p_value']), 6)}, "
            f"decision={row['Decisión']}, conclusion={row['Conclusión']}"
        )

    return {
        "technique": "Log-Rank Test",
        "prompt": (
            "Provide a brief academic interpretation of a Log-Rank test from the dashboard "
            "comparing survival curves using the available variables. "
            f"Results: {'; '.join(parts)}."
        ),
    }


def _build_dash_prompts(df: pd.DataFrame) -> list[dict]:
    """Prepare prompts only for the techniques and variables available in Dash."""
    prompts = [_build_kaplan_prompt(df)]

    prompts.append(
        _build_cox_prompt(
            df,
            [variable["column"] for variable in DASH_VARIABLES],
            "Gender + Disability",
        )
    )

    prompts.append(_build_logrank_prompt(df))

    return prompts


def _run_once(
    index: int,
    prompt: str,
    timeout: int,
    process_name: str,
    endpoint: str,
    model: str,
    num_ctx: int,
    max_tokens: int,
    technique: str,
) -> dict:
    """Run one trial and record system metrics."""
    before = _query_process_snapshot(process_name)
    content, elapsed = _call_model(prompt, timeout, endpoint, model, num_ctx, max_tokens)
    after = _query_process_snapshot(process_name)

    cpu_percent = None
    ram_mb = None
    if before and after and before.get("pid") == after.get("pid") and elapsed > 0:
        cpu_delta = max(0.0, after["cpu_seconds"] - before["cpu_seconds"])
        cpu_percent = round((cpu_delta / elapsed / max(1, os.cpu_count() or 1)) * 100, 2)
        ram_mb = after["ram_mb"]
    elif after:
        ram_mb = after["ram_mb"]

    return {
        "run": index,
        "technique": technique,
        "elapsed_seconds": round(elapsed, 3),
        "response_chars": len(content),
        "cpu_percent_approx": cpu_percent,
        "ram_mb_approx": ram_mb,
        "process": after["name"] if after else None,
        "pid": after["pid"] if after else None,
    }


def _save_results(
    results: list[dict],
    args: argparse.Namespace,
    summary: dict,
    technique_prompts: list[dict],
) -> tuple[Path, Path]:
    """Save detailed results and the summary to disk."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"ia_benchmark_{timestamp}.csv"
    json_path = out_dir / f"ia_benchmark_{timestamp}.json"

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "endpoint": args.endpoint,
        "model": args.model,
        "num_ctx": args.num_ctx,
        "max_tokens": args.max_tokens,
        "mode": args.mode,
        "runs": args.runs,
        "warmup": args.warmup,
        "dataset": args.dataset,
        "prompt": args.prompt if args.mode == "repeat" else None,
        "technique_prompts": technique_prompts if args.mode == "techniques" else None,
        "summary": summary,
        "results": results,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, json_path


def _build_summary(results: list[dict]) -> dict:
    """Summarize the main metrics across all runs."""
    times = [row["elapsed_seconds"] for row in results]
    cpu_values = [row["cpu_percent_approx"] for row in results if row["cpu_percent_approx"] is not None]
    ram_values = [row["ram_mb_approx"] for row in results if row["ram_mb_approx"] is not None]

    return {
        "time_mean_seconds": round(statistics.mean(times), 3),
        "time_median_seconds": round(statistics.median(times), 3),
        "time_max_seconds": round(max(times), 3),
        "time_min_seconds": round(min(times), 3),
        "response_chars_mean": round(statistics.mean(row["response_chars"] for row in results), 1),
        "ram_mb_max_approx": round(max(ram_values), 2) if ram_values else None,
        "cpu_percent_mean_approx": round(statistics.mean(cpu_values), 2) if cpu_values else None,
        "cpu_percent_max_approx": round(max(cpu_values), 2) if cpu_values else None,
    }


def _build_tasks(args: argparse.Namespace, technique_prompts: list[dict]) -> list[dict]:
    """Prepare the list of tasks to run during the benchmark."""
    if args.mode == "repeat":
        return [
            {
                "run": index,
                "technique": "Prompt repetido",
                "prompt": args.prompt,
            }
            for index in range(1, args.runs + 1)
        ]

    tasks = []
    run_index = 1
    for item in technique_prompts:
        for _ in range(args.runs):
            tasks.append(
                {
                    "run": run_index,
                    "technique": item["technique"],
                    "prompt": item["prompt"],
                }
            )
            run_index += 1
    return tasks


def main() -> int:
    """Orchestrate the complete benchmark from arguments to output files."""
    args = _parse_args()

    if args.runs <= 0:
        print("ERROR: --runs must be greater than 0.")
        return 1

    try:
        df = _load_dataset(args.dataset)
        technique_prompts = _build_dash_prompts(df)
    except Exception as exc:
        print("ERROR: could not prepare the Dash techniques.")
        print(f"""Details: {exc}""")
        return 1

    print("Dash local AI benchmark")
    print(f"Endpoint: {args.endpoint}")
    print(f"""Model:    {args.model}""")
    print(f"""Context:  {args.num_ctx}""")
    print(f"Tokens:   {args.max_tokens}")
    print(f"Dataset:  {args.dataset}")
    print(f"""Mode:     {args.mode}""")
    if args.mode == "techniques":
        print(f"""Queries: {len(technique_prompts)} techniques x {args.runs} repetition(s)""")
        print("Techniques:")
        for item in technique_prompts:
            print(f"- {item['technique']}")
    else:
        print(f"Runs:     {args.runs} (+ {args.warmup} warmup)")
    print()

    try:
        for index in range(args.warmup):
            print(f"Warmup {index + 1}/{args.warmup}...")
            _call_model(
                args.prompt,
                args.timeout,
                args.endpoint,
                args.model,
                args.num_ctx,
                args.max_tokens,
            )
    except Exception as exc:
        print("ERROR: could not contact the AI server.")
        print(f"""Details: {exc}""")
        print("Check that Ollama is running and the model has been downloaded.")
        return 1

    tasks = _build_tasks(args, technique_prompts)
    results = []
    for position, task in enumerate(tasks, start=1):
        print(
            f"{task['technique']} ({position}/{len(tasks)})...",
            end=" ",
            flush=True,
        )
        try:
            row = _run_once(
                task["run"],
                task["prompt"],
                args.timeout,
                args.process_name,
                args.endpoint,
                args.model,
                args.num_ctx,
                args.max_tokens,
                task["technique"],
            )
        except Exception as exc:
            print("ERROR")
            print(f"""Details: {exc}""")
            return 1
        results.append(row)
        print(f"{row['elapsed_seconds']} s")

    summary = _build_summary(results)
    csv_path, json_path = _save_results(results, args, summary, technique_prompts)

    print()
    print("Summary")
    print(f"""- Mean time:     {summary['time_mean_seconds']} s""")
    print(f"""- Maximum time:  {summary['time_max_seconds']} s""")
    print(f"""- Approx. RAM:   {summary['ram_mb_max_approx'] or 'N/A'} MB""")
    print(f"""- Approx. CPU:   {summary['cpu_percent_mean_approx'] or 'N/A'} % mean""")
    print()
    print("Details by technique")
    for row in results:
        print(
            f"- {row['technique']}: {row['elapsed_seconds']} s, "
            f"RAM {row['ram_mb_approx'] or 'N/A'} MB, "
            f"CPU {row['cpu_percent_approx'] or 'N/A'} %"
        )
    print()
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
