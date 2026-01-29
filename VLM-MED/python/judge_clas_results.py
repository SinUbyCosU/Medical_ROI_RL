#!/usr/bin/env python3
"""Judge CLAS baseline vs steered responses using existing llama3 scoring prompt."""

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score CLAS outputs with llama3 judge")
    parser.add_argument(
        "--input",
        default=os.getenv("CLAS_JUDGE_INPUT", "clas_multi_model_results.jsonl"),
        help="Path to CLAS results JSONL file",
    )
    parser.add_argument(
        "--output",
        default=os.getenv("CLAS_JUDGE_OUTPUT", "clas_multi_model_results_judged.jsonl"),
        help="Path to write judge scores JSONL",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed to use when querying the judge LLM",
    )
    return parser.parse_args()

PROMPT = """You are a text analysis expert. Score the following text on exactly four dimensions using a 1-10 scale. Respond with ONLY a valid JSON object, nothing else.

Assertiveness (1-10): 1=Passive (uses \"maybe\", qualifies statements), 10=Authoritative (imperative, active voice)
Vocabulary Complexity (1-10): 1=Basic (simple sentences, A1-A2 level), 10=Technical (jargon, C1-C2 level)
Emotional Distance (1-10): 1=Empathetic (emotional words, \"I\", \"you\"), 10=Clinical (neutral, factual, objective)
Instructional Density (1-10): 1=Vague (general advice), 10=Actionable (numbered steps, specific code, concrete plan)

Text:
{response}

Output ONLY valid JSON in this exact format:
{{\"assertiveness\": <int>, \"complexity\": <int>, \"emotional_distance\": <int>, \"instructional_density\": <int>}}"""

STANDARD_KEYS = {
    "assertiveness",
    "complexity",
    "emotional_distance",
    "instructional_density",
}

SYNONYMS = {
    "vocabulary_complexity": "complexity",
    "instructional_density_score": "instructional_density",
}


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")


def score_response(text: str, seed: Optional[int]) -> Optional[Dict[str, float]]:
    """Score text using the same llama3-based judge prompt as existing pipeline."""
    prompt = PROMPT.format(response=text[:2000])  # keep prompt manageable
    try:
        payload: Dict = {
            "model": "llama3",
            "prompt": prompt,
            "stream": False,
        }
        if seed is not None:
            payload["options"] = {"seed": seed}
        response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=90)
        response.raise_for_status()
        data = response.json()
        output = (data.get("response") or "").strip()
        start = output.find("{")
        end = output.rfind("}") + 1
        if start == -1 or end == 0:
            return None
        json_blob = output[start:end]
        raw = json.loads(json_blob)
        return normalize_scores(raw)
    except Exception:
        return None


def compute_delta(before: Dict[str, float], after: Dict[str, float]) -> Dict[str, float]:
    return {
        key: (after.get(key, 0) - before.get(key, 0))
        for key in ("assertiveness", "complexity", "emotional_distance", "instructional_density")
    }


def normalize_scores(raw: Dict[str, float]) -> Dict[str, float]:
    normalized: Dict[str, float] = {}
    for key, value in raw.items():
        canonical = SYNONYMS.get(key, key)
        if canonical in STANDARD_KEYS:
            try:
                normalized[canonical] = float(value)
            except (TypeError, ValueError):
                continue
    for key in STANDARD_KEYS:
        normalized.setdefault(key, 0.0)
    return normalized


def init_delta_tracker() -> Dict[str, list]:
    return {"assertiveness": [], "complexity": [], "emotional_distance": [], "instructional_density": []}


SeenKey = Tuple[str, str, Optional[int], Optional[str], Optional[int], Optional[float]]


def load_existing(output_path: Path) -> Tuple[Set[SeenKey], Dict[str, list]]:
    seen: Set[SeenKey] = set()
    deltas = init_delta_tracker()
    if not output_path.exists():
        return seen, deltas

    with output_path.open("r", encoding="utf-8") as fout:
        for line in fout:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            model = record.get("model")
            prompt = record.get("prompt")
            seed = record.get("seed")
            variant = record.get("variant_label") or record.get("variant")
            layer_index = record.get("layer_index")
            alpha = record.get("alpha")
            if model and prompt:
                seen.add((model, prompt, seed, variant, layer_index, alpha))
            delta = record.get("delta") or {}
            for key in deltas:
                value = delta.get(key)
                if value is not None:
                    try:
                        deltas[key].append(float(value))
                    except (TypeError, ValueError):
                        continue
    return seen, deltas


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    seed = args.seed

    if not input_path.exists():
        raise SystemExit(f"❌ Missing input file: {input_path}")

    seen_pairs, aggregate_deltas = load_existing(output_path)
    previous_count = len(seen_pairs)
    new_deltas = init_delta_tracker()
    new_records = 0

    score_cache: Dict[str, Optional[Dict[str, float]]] = {}

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("a", encoding="utf-8") as fout:
        for line in fin:
            record = json.loads(line)
            model = record.get("model")
            prompt = record.get("prompt")
            if not model or not prompt:
                continue
            variant = record.get("variant_label") or record.get("variant")
            layer_index = record.get("layer_index")
            alpha = record.get("alpha")
            key: SeenKey = (model, prompt, seed, variant, layer_index, alpha)
            if key in seen_pairs:
                continue

            # Support baseline-only files where key is 'response'
            baseline_text = record.get("baseline_response", "") or record.get("response", "")
            steered_text = record.get("steered_response", "")

            base_score = None
            if baseline_text:
                if baseline_text not in score_cache:
                    score_cache[baseline_text] = score_response(baseline_text, seed)
                base_score = score_cache[baseline_text]

            steer_score = None
            if steered_text:
                if steered_text not in score_cache:
                    score_cache[steered_text] = score_response(steered_text, seed)
                steer_score = score_cache[steered_text]

            delta = compute_delta(base_score, steer_score) if base_score and steer_score else None

            if delta:
                for metric, value in delta.items():
                    aggregate_deltas[metric].append(value)
                    new_deltas[metric].append(value)

            output_record = {
                "model": model,
                "model_id": record.get("model_id"),
                "prompt": prompt,
                "prompt_with_prefix": record.get("prompt_with_prefix"),
                "variant_label": variant,
                "layer_index": layer_index,
                "alpha": alpha,
                "seed": seed,
                "baseline": base_score,
                "steered": steer_score,
                "delta": delta,
            }
            fout.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            fout.flush()

            seen_pairs.add(key)
            new_records += 1

    print(f"Processed previously: {previous_count}")
    print(f"Scored this run: {new_records}")

    print("\n=== Average Delta (Steered - Baseline) — cumulative ===")
    for key, values in aggregate_deltas.items():
        if values:
            print(f"{key:22s}: {statistics.mean(values):+.2f}")
        else:
            print(f"{key:22s}: n/a")

    if any(new_deltas[key] for key in new_deltas):
        print("\n=== Average Delta (Steered - Baseline) — new this run ===")
        for key, values in new_deltas.items():
            if values:
                print(f"{key:22s}: {statistics.mean(values):+.2f}")
            else:
                print(f"{key:22s}: n/a")


if __name__ == "__main__":
    main()
