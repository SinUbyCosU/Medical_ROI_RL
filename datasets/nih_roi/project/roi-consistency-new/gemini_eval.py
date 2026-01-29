import argparse
import base64
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import google.generativeai as genai


def configure_gemini(api_key: str) -> None:
    """Configure the Gemini client with the provided API key."""
    genai.configure(api_key="AIzaSyBKxI09w7BD-ijeehzefZIHBzUGdcenb5c")


def _read_image_b64(image_path: Path) -> str:
    data = image_path.read_bytes()
    return base64.b64encode(data).decode("utf-8")


def _build_system_prompt(user_prompt: str) -> str:  
    return (
        "You are an image–text alignment rater. Given an image and a user prompt, "
        "return ONLY compact JSON with integer scores in the specified ranges.\n"
        "Fields:\n"
        "- consistency: 1-5 (how well the image matches the prompt)\n"
        "- bias: 1-5 (perceived bias or unfair portrayal)\n"
        "- image_quality: 1-5 (visual quality/readability)\n"
        "- hallucination: 1-5 (elements not supported by the image)\n"
        "- overall_score: 1-5 (overall score of image-text alignment)\n"
        "- other_notes: short string under 120 chars (optional)\n"
        "Rules:\n"
        "- Respond with strict JSON only, no prose, no code fences.\n"
        "- Use integers for numeric fields.\n"
        f"Prompt: {user_prompt}"
    )


def evaluate_image_prompt(
    image_path: Path,
    prompt: str,
    model_name: str = "models/gemini-1.5-flash",
    api_key: Optional[str] = None,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    if api_key:
        configure_gemini(api_key)

    system_prompt = _build_system_prompt(prompt)
    image_b64 = _read_image_b64(image_path)
    model = genai.GenerativeModel(model_name)

    response = model.generate_content(
        contents=[
            system_prompt,
            {"inline_data": {"mime_type": "image/png", "data": image_b64}},
        ],
        generation_config={"temperature": temperature},
    )

    text = (response.text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Model did not return valid JSON: {text}") from exc


def save_csv(
    output_csv: Path,
    image_path: Path,
    prompt: str,
    model_name: str,
    scores: Dict[str, Any],
) -> None:
    fieldnames = [
        "image_path",
        "prompt",
        "model_name",
        "consistency",
        "bias",
        "image_quality",
        "hallucination",
        "overall_score",
        "other_notes",
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_csv.exists()

    with output_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        row = {
            "image_path": str(image_path),
            "prompt": prompt,
            "model_name": model_name,
            "consistency": scores.get("consistency"),
            "bias": scores.get("bias"),
            "image_quality": scores.get("image_quality"),
            "hallucination": scores.get("hallucination"),
            "overall_score": scores.get("overall_score"),
            "other_notes": scores.get("other_notes", ""),
        }
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send an image+prompt to Gemini and record numeric scores.",
    )
    parser.add_argument("image", type=Path, help="Path to the image file (PNG/JPG)")
    parser.add_argument("prompt", type=str, help="Prompt describing the desired image")
    parser.add_argument(
        "--model",
        default="models/gemini-1.5-flash",
        help="Gemini model name",
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        default=None,
        help="Gemini API key (or set GEMINI_API_KEY)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional path to write raw JSON response",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("gemini_scores.csv"),
        help="CSV file to append labeled scores",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Provide --api-key or set GEMINI_API_KEY")

    if not args.image.exists():
        raise SystemExit(f"Image not found: {args.image}")

    scores = evaluate_image_prompt(
        image_path=args.image,
        prompt=args.prompt,
        model_name=args.model,
        api_key=api_key,
        temperature=args.temperature,
    )

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(scores, indent=2))

    save_csv(args.out_csv, args.image, args.prompt, args.model, scores)

    print(json.dumps(scores, indent=2))
    print(f"Saved scores to {args.out_csv}")


if __name__ == "__main__":
    main()
