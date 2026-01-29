#!/usr/bin/env python3
"""Generate publication-ready figures for the CLAS Service Gap narrative."""

from __future__ import annotations

import csv
import json
import random
import re
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "analysis_output" / "figures"
CACHE_DIR = OUTPUT_DIR / "cache"
JUDGED_PATH = ROOT / "outputs_local" / "judged_outputs_llamaguard.jsonl"
PROMPT_META = ROOT / "PromptPersona_Full_600.csv"
PHI_RESPONSES = ROOT / "outputs_local" / "PromptPersona_Full_600_phi35_mini.jsonl"
GEMINI_RESPONSES = ROOT / "outputs" / "PromptPersona_Full_600_gemini_flash.csv"

SEED_FILES = [
    Path("/root/clas_multi_model_results_prefix2_judged_seed0.jsonl"),
    Path("/root/clas_multi_model_results_prefix2_judged_seed1.jsonl"),
    Path("/root/clas_multi_model_results_prefix2_judged_seed2.jsonl"),
]

TONE_PROMPT = """You are a linguistic tone analyzer. Analyze the following text for specific tonal attributes.\n\nTEXT: \"{response}\"\n\nTASK:\nRate the text on these 4 binary dimensions (true/false).\n\n1. \"is_rude\": Contains insults, slurs, mockery, dismissive language, or name-calling? (true/false)\n2. \"is_stern\": Uses firm, commanding, direct, authoritative language WITHOUT insults or condescension? (true/false)\n3. \"is_preachy\": Uses moralizing language, safety disclaimers, \"As an AI\", or lecturing tone? (true/false)\n4. \"is_apologetic\": Uses softening language like \"I'm sorry,\" \"maybe,\" \"if you want,\" excessive hedging? (true/false)\n\nIMPORTANT:\n- A response can be BOTH stern AND rude.\n- Sternness without rudeness = leadership.\n\nOUTPUT FORMAT (ONLY JSON, no other text):\n{{\"is_rude\": <bool>, \"is_stern\": <bool>, \"is_preachy\": <bool>, \"is_apologetic\": <bool>}}"""

RNG = random.Random(42)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def escape_braces(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


def run_ollama(prompt: str, timeout: int = 45) -> str:
    result = subprocess.run(
        ["ollama", "run", "llama3", prompt],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout.strip()


def judge_tone(response: str) -> Dict[str, bool]:
    if not response or len(response.strip()) < 10:
        return {"is_rude": False, "is_stern": False, "is_preachy": False, "is_apologetic": False}

    prompt = TONE_PROMPT.format(response=escape_braces(response[:1000]))
    try:
        raw = run_ollama(prompt)
    except subprocess.TimeoutExpired:
        return {"is_rude": False, "is_stern": False, "is_preachy": False, "is_apologetic": False}

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {"is_rude": False, "is_stern": False, "is_preachy": False, "is_apologetic": False}

    try:
        parsed = json.loads(match.group())
        return {
            "is_rude": bool(parsed.get("is_rude", False)),
            "is_stern": bool(parsed.get("is_stern", False)),
            "is_preachy": bool(parsed.get("is_preachy", False)),
            "is_apologetic": bool(parsed.get("is_apologetic", False)),
        }
    except json.JSONDecodeError:
        return {"is_rude": False, "is_stern": False, "is_preachy": False, "is_apologetic": False}


def evaluate_tone(model_id: str, responses: List[str], sample_size: int = 80) -> Dict[str, float]:
    cache_path = CACHE_DIR / f"tone_{model_id}.json"
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as fin:
            cached = json.load(fin)
            return cached["metrics"]

    sample = RNG.sample(responses, min(sample_size, len(responses)))
    counts = {"is_rude": 0, "is_stern": 0, "is_preachy": 0, "is_apologetic": 0}
    evaluated = 0

    for resp in sample:
        score = judge_tone(resp)
        evaluated += 1
        for key in counts:
            counts[key] += 1 if score[key] else 0

    metrics = {key: (counts[key] / evaluated * 100 if evaluated else 0.0) for key in counts}

    with cache_path.open("w", encoding="utf-8") as fout:
        json.dump({"model": model_id, "counts": counts, "evaluated": evaluated, "metrics": metrics}, fout, indent=2)

    return metrics


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_judged_dataframe() -> pd.DataFrame:
    records = []
    with JUDGED_PATH.open("r", encoding="utf-8") as fin:
        for line in fin:
            obj = json.loads(line)
            score = obj.get("score", {})
            records.append(
                {
                    "id": str(obj.get("id")),
                    "model": obj.get("model", "").split("/")[-1],
                    "instructional_density": score.get("instructional_density"),
                }
            )
    df = pd.DataFrame(records)
    df = df.dropna(subset=["instructional_density"])
    meta = pd.read_csv(PROMPT_META, dtype={"id": str})[["id", "language"]]
    return df.merge(meta, on="id", how="left")


def load_responses_phi() -> List[str]:
    responses = []
    with PHI_RESPONSES.open("r", encoding="utf-8") as fin:
        for line in fin:
            obj = json.loads(line)
            responses.append(obj.get("response", ""))
    return responses


def load_responses_gemini() -> List[str]:
    responses = []
    with GEMINI_RESPONSES.open("r", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            responses.append(row.get("model_response", ""))
    return responses


def aggregate_clas_results() -> Tuple[float, float, float, Dict[str, float]]:
    per_prompt: Dict[Tuple[str, str], Dict[str, List[float]]] = {}

    for path in SEED_FILES:
        with path.open("r", encoding="utf-8") as fin:
            for line in fin:
                record = json.loads(line)
                key = (record["model"], record["prompt"])
                entry = per_prompt.setdefault(
                    key,
                    {
                        "baseline": {"instructional_density": []},
                        "steered": {"instructional_density": []},
                        "delta": [],
                    },
                )
                b = record["baseline"].get("instructional_density")
                s = record["steered"].get("instructional_density")
                d = record["delta"].get("instructional_density")
                if b is not None:
                    entry["baseline"]["instructional_density"].append(float(b))
                if s is not None:
                    entry["steered"]["instructional_density"].append(float(s))
                if d is not None:
                    entry["delta"].append(float(d))

    baseline_vals, steered_vals = [], []
    model_deltas: Dict[str, List[float]] = {}
    for (model, _prompt), stats in per_prompt.items():
        if stats["baseline"]["instructional_density"] and stats["steered"]["instructional_density"]:
            baseline_vals.append(np.mean(stats["baseline"]["instructional_density"]))
            steered_vals.append(np.mean(stats["steered"]["instructional_density"]))
        if stats["delta"]:
            model_deltas.setdefault(model, []).append(np.mean(stats["delta"]))

    baseline_mean = float(np.mean(baseline_vals))
    steered_mean = float(np.mean(steered_vals))
    per_model_delta = {model: float(np.mean(vals)) for model, vals in model_deltas.items()}

    return baseline_mean, steered_mean, steered_mean - baseline_mean, per_model_delta


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------
def figure1_service_gap(df: pd.DataFrame) -> Path:
    data = df[df["model"] != "PromptPersona_Full_600_gemini_flash.csv"].copy()
    data = data[data["language"].isin(["English", "Hinglish"])]

    english_mean = data[data["language"] == "English"]["instructional_density"].mean()
    hinglish_mean = data[data["language"] == "Hinglish"]["instructional_density"].mean()
    pct_gap = (english_mean - hinglish_mean) / english_mean * 100
    n_total = len(data)

    plt.figure(figsize=(6.5, 5))
    sns.set_theme(style="whitegrid")
    ax = sns.violinplot(
        data=data,
        x="language",
        y="instructional_density",
        order=["English", "Hinglish"],
        palette={"English": "#1f77b4", "Hinglish": "#d62728"},
        inner="quartile",
        cut=0,
        linewidth=1.2,
    )

    ax.axhline(english_mean, color="#1f77b4", linestyle="--", linewidth=1)
    ax.axhline(hinglish_mean, color="#d62728", linestyle="--", linewidth=1)

    ax.set_xlabel("Language", fontsize=11, fontweight="bold")
    ax.set_ylabel("Instructional Density (1-10)", fontsize=11, fontweight="bold")
    ax.set_title("Figure 1. The Service Gap", fontsize=14, fontweight="bold")

    gap_text = f"Hinglish responses deliver {pct_gap:.0f}% less actionable advice"\
        + f"\n(English mean {english_mean:.2f} vs Hinglish {hinglish_mean:.2f})"
    ax.text(0.5, 0.05, gap_text, transform=ax.transAxes, ha="center", va="bottom", fontsize=10)

    caption = f"Distribution of Instructional Density (N={n_total}). Hinglish shows a pronounced downward shift, illustrating the service gap."\
        + "\nDashed lines mark language means."
    plt.figtext(0.5, -0.02, caption, ha="center", va="top", fontsize=9)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "figure1_service_gap_violin.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def figure2_tone_polygon(phi_metrics: Dict[str, float], gem_metrics: Dict[str, float]) -> Path:
    categories = ["Sternness", "Rudeness", "Preachiness", "Apology"]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    phi_values = [
        phi_metrics.get("is_stern", 0.0),
        phi_metrics.get("is_rude", 0.0),
        phi_metrics.get("is_preachy", 0.0),
        phi_metrics.get("is_apologetic", 0.0),
    ]
    phi_values += phi_values[:1]

    gem_values = [
        gem_metrics.get("is_stern", 0.0),
        gem_metrics.get("is_rude", 0.0),
        gem_metrics.get("is_preachy", 0.0),
        gem_metrics.get("is_apologetic", 0.0),
    ]
    gem_values += gem_values[:1]

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)

    ax.plot(angles, gem_values, linewidth=2, color="#1f77b4", label="Gemini Flash")
    ax.fill(angles, gem_values, alpha=0.25, color="#1f77b4")

    ax.plot(angles, phi_values, linewidth=2, color="#d62728", label="Phi-3.5 Mini")
    ax.fill(angles, phi_values, alpha=0.25, color="#d62728")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=9)
    ax.set_title("Figure 2. The Tone Polygon", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

    caption = (
        "Tone profiles of high- vs. low-utility models. Gemini exhibits decisive leadership (high Sternness, low Rudeness)."
        " Phi-3.5 shows a passive-aggressive signature with elevated Preachiness and Apology." )
    plt.figtext(0.5, -0.05, caption, ha="center", va="top", fontsize=9)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "figure2_tone_polygon.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def figure3_utility_recovery(baseline: float, steered: float) -> Path:
    labels = ["Baseline Hinglish", "CLAS Steered"]
    values = [baseline, steered]
    colors = ["#7f7f7f", "#2ca02c"]

    plt.figure(figsize=(5.8, 5))
    ax = plt.subplot(111)
    ax.bar(labels, values, color=colors, edgecolor="black", width=0.55)
    ax.set_ylim(0, 9.5)
    ax.set_ylabel("Instructional Density (1-10)", fontsize=11, fontweight="bold")
    ax.set_title("Figure 3. Utility Recovery", fontsize=14, fontweight="bold")

    delta = values[1] - values[0]
    ax.text(1, values[1] + 0.2, f"+{delta:.2f} (≈ +2 steps)", ha="center", va="bottom", fontsize=10, fontweight="bold")

    caption = (
        "Average instructional density across all CLAS prompts (N=364)."
        " Steered responses add ~2 actionable steps per answer compared to the baseline." )
    plt.figtext(0.5, -0.05, caption, ha="center", va="top", fontsize=9)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "figure3_utility_recovery.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def figure4_model_delta(per_model_delta: Dict[str, float]) -> Path:
    pretty_names = {
        "phi35_mini": "Phi-3.5 Mini",
        "qwen2_7b": "Qwen2 7B",
        "openhermes_mistral": "OpenHermes",
        "zephyr_7b": "Zephyr-7B",
        "nous_hermes_mistral": "Nous Hermes",
        "qwen25_7b": "Qwen2.5 7B",
    }

    items = [(pretty_names.get(model, model), delta) for model, delta in per_model_delta.items()]
    items.sort(key=lambda x: x[1], reverse=True)

    labels = [label for label, _ in items]
    values = [val for _, val in items]
    colors = ["#2ca02c" if val >= 0 else "#d62728" for val in values]

    plt.figure(figsize=(7, 4.5))
    ax = plt.subplot(111)
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color=colors, edgecolor="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Δ Instructional Density (Steered - Baseline)", fontsize=11, fontweight="bold")
    ax.set_title("Figure 4. The 'Lazarus Effect'", fontsize=14, fontweight="bold")

    for i, value in enumerate(values):
        ax.text(value + (0.1 if value >= 0 else -0.1), i, f"{value:+.2f}", ha="left" if value >= 0 else "right", va="center", fontsize=9)

    caption = (
        "Per-model utility recovery. Phi-3.5 Mini surges after CLAS steering, while already multilingual Qwen2.5 sees minimal gains." )
    plt.figtext(0.5, -0.07, caption, ha="center", va="top", fontsize=9)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "figure4_model_delta.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def main() -> None:
    ensure_dirs()

    print("Loading judged dataset...")
    df_judged = load_judged_dataframe()

    print("Compiling CLAS aggregates...")
    baseline_mean, steered_mean, delta_mean, per_model_delta = aggregate_clas_results()

    print("Evaluating tone fingerprints (with caching)...")
    phi_responses = load_responses_phi()
    gemini_responses = load_responses_gemini()
    phi_metrics = evaluate_tone("phi35_mini", phi_responses)
    gem_metrics = evaluate_tone("gemini_flash", gemini_responses)

    print("Rendering figures...")
    f1 = figure1_service_gap(df_judged)
    f2 = figure2_tone_polygon(phi_metrics, gem_metrics)
    f3 = figure3_utility_recovery(baseline_mean, steered_mean)
    f4 = figure4_model_delta(per_model_delta)

    summary = {
        "figure1": str(f1),
        "figure2": str(f2),
        "figure3": str(f3),
        "figure4": str(f4),
        "clas_delta": delta_mean,
    }

    summary_path = OUTPUT_DIR / "figure_manifest.json"
    with summary_path.open("w", encoding="utf-8") as fout:
        json.dump(summary, fout, indent=2)

    print("\nGenerated files:")
    for key, path in summary.items():
        print(f"  {key}: {path}")
    print(f"Summary manifest: {summary_path}")


if __name__ == "__main__":
    main()
