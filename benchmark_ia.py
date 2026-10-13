"""
Simple benchmark for the local AI module.

Measure response times against the endpoint configured in config.py and, on
Windows, try to estimate RAM/CPU usage of the llama-server process.
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

import requests

from config import LLAMA_SERVER_URL, MODEL_NAME


DEFAULT_PROMPT = (
    "Provide a brief academic interpretation of these survival results: "
    "1200 students, 310 dropout events, mean follow-up of 145 days, "
    "Hazard Ratio of 1.42 for studied credits and a p-value of 0.03."
)

TECHNIQUE_PROMPTS = [
    {
        "technique": "Kaplan-Meier",
        "prompt": (
            "Provide a brief academic interpretation of a Kaplan-Meier analysis with "
            "1200 students, 310 observed dropouts, and estimated survival "
            "of 0.82 at 100 days and 0.69 at 200 days. The curve for the group "
            "with more studied credits declines earlier than the others."
        ),
    },
    {
        "technique": "Cox Regression",
        "prompt": (
            "Provide a brief academic interpretation of Cox regression applied "
            "to academic dropout. The studied credits variable has a "
            "Hazard Ratio of 1.42 with p-value 0.03, age 35-55 has Hazard Ratio "
            "0.91 with p-value 0.28, and disability has Hazard Ratio 1.18 with "
            "p-value 0.12."
        ),
    },
    {
        "technique": "Log-Rank Test",
        "prompt": (
            "Provide a brief academic interpretation of a Log-Rank test comparing "
            "survival curves by prior education level. The test "
            "statistic is 12.84 and the p-value is 0.004. The curves show "
            "visible separation between groups from the midpoint of follow-up."
        ),
    },
    {
        "technique": "Weibull",
        "prompt": (
            "Provide a brief academic interpretation of a Weibull model for academic "
            "dropout. The shape parameter rho is 1.31, estimated median survival "
            "is 176 days, log-likelihood is -842.5, and AIC is 1689.0. "
            "Risk appears to increase slightly over time."
        ),
    },
    {
        "technique": "Exponential",
        "prompt": (
            "Provide a brief academic interpretation of an Exponential model for "
            "academic dropout. The estimated lambda parameter is 0.0048, median "
            "survival is 144 days, log-likelihood is -861.2, and AIC is "
            "1724.4. The model assumes constant risk during follow-up."
        ),
    },
    {
        "technique": "Random Survival Forest",
        "prompt": (
            "Provide a brief academic interpretation of a Random Survival Forest applied "
            "to academic dropout. The concordance index is 0.71, the "
            "out-of-bag score is 0.66, and the most important variables are studied credits, "
            "age, and prior education level. High-risk profiles show "
            "clearly lower survival curves."
        ),
    },
]


def _parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments for the AI benchmark."""
    parser = argparse.ArgumentParser(
        description="Measure response times and approximate resource usage of the local AI server."
    )
    parser.add_argument(
        "--mode",
        choices=("techniques", "repeat"),
        default="techniques",
        help=(
            "techniques measures one query per technique; repeat sends the same prompt several times."
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
    parser.add_argument("--endpoint", default=LLAMA_SERVER_URL, help="Chat completions compatible endpoint.")
    parser.add_argument("--model", default=MODEL_NAME, help="Model name sent to the endpoint.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt used for each query.")
    parser.add_argument(
        "--process-name",
        default="llama-server",
        help="Process name used to estimate RAM/CPU usage.",
    )
    parser.add_argument(
        "--out-dir",
        default="benchmarks",
        help="Directory for CSV and JSON results.",
    )
    return parser.parse_args()


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


def _build_payload(prompt: str, model: str) -> dict:
    """Build the JSON body sent to the model endpoint."""
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an academic assistant who interprets "
                    "survival analysis results clearly, cautiously, and briefly."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 320,
        "stream": False,
    }


def _call_model(prompt: str, timeout: int, endpoint: str, model: str) -> tuple[str, float]:
    """Call the model, measure response time, and return the generated text."""
    start = time.perf_counter()
    response = requests.post(endpoint, json=_build_payload(prompt, model), timeout=timeout)
    elapsed = time.perf_counter() - start
    response.raise_for_status()

    data = response.json()
    content = data["choices"][0]["message"]["content"]
    return content, elapsed


def _run_once(
    index: int,
    prompt: str,
    timeout: int,
    process_name: str,
    endpoint: str,
    model: str,
    technique: str,
) -> dict:
    """Run one benchmark trial and record system metrics."""
    before = _query_process_snapshot(process_name)
    content, elapsed = _call_model(prompt, timeout, endpoint, model)
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


def _save_results(results: list[dict], args: argparse.Namespace, summary: dict) -> tuple[Path, Path]:
    """Save detailed results and the benchmark summary to disk."""
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
        "mode": args.mode,
        "runs": args.runs,
        "warmup": args.warmup,
        "prompt": args.prompt if args.mode == "repeat" else None,
        "technique_prompts": TECHNIQUE_PROMPTS if args.mode == "techniques" else None,
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

    summary = {
        "time_mean_seconds": round(statistics.mean(times), 3),
        "time_median_seconds": round(statistics.median(times), 3),
        "time_max_seconds": round(max(times), 3),
        "time_min_seconds": round(min(times), 3),
        "response_chars_mean": round(statistics.mean(row["response_chars"] for row in results), 1),
        "ram_mb_max_approx": round(max(ram_values), 2) if ram_values else None,
        "cpu_percent_mean_approx": round(statistics.mean(cpu_values), 2) if cpu_values else None,
        "cpu_percent_max_approx": round(max(cpu_values), 2) if cpu_values else None,
    }
    return summary


def _build_tasks(args: argparse.Namespace) -> list[dict]:
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
    for item in TECHNIQUE_PROMPTS:
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

    print("Local AI benchmark")
    print(f"Endpoint: {args.endpoint}")
    print(f"""Model:    {args.model}""")
    print(f"""Mode:     {args.mode}""")
    if args.mode == "techniques":
        print(f"""Queries: {len(TECHNIQUE_PROMPTS)} techniques x {args.runs} repetition(s)""")
    else:
        print(f"Runs:     {args.runs} (+ {args.warmup} warmup)")
    print()

    try:
        for index in range(args.warmup):
            print(f"Warmup {index + 1}/{args.warmup}...")
            _call_model(args.prompt, args.timeout, args.endpoint, args.model)
    except Exception as exc:
        print("ERROR: could not contact the AI server.")
        print(f"""Details: {exc}""")
        print("Check that START_LLAMA_SERVER.bat is running.")
        return 1

    tasks = _build_tasks(args)
    results = []
    for position, task in enumerate(tasks, start=1):
        print(f"{task['technique']} ({position}/{len(tasks)})...", end=" ", flush=True)
        try:
            row = _run_once(
                task["run"],
                task["prompt"],
                args.timeout,
                args.process_name,
                args.endpoint,
                args.model,
                task["technique"],
            )
        except Exception as exc:
            print("ERROR")
            print(f"""Details: {exc}""")
            return 1
        results.append(row)
        print(f"{row['elapsed_seconds']} s")

    summary = _build_summary(results)
    csv_path, json_path = _save_results(results, args, summary)

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
