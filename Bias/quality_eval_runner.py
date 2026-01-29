import pandas as pd
import json
import time
import requests
from pathlib import Path

# --- CONFIG ---
API_KEY = "AIzaSyBq260DoXBrtoeWj2xB-ycvIPmUBUZOhzE"
MODEL_ENDPOINTS = [
    ("v1beta", "gemini-3.0-pro-preview"),
    ("v1beta", "gemini-1.5-pro-latest"),
    ("v1beta", "gemini-1.5-pro"),
    ("v1beta", "gemini-1.5-flash"),
    ("v1beta", "gemini-pro"),
    ("v1beta2", "gemini-1.5-pro-latest"),
    ("v1beta2", "gemini-1.5-pro"),
    ("v1beta2", "gemini-1.5-flash"),
    ("v1beta2", "gemini-pro"),
    ("v1", "gemini-1.5-pro-latest"),
    ("v1", "gemini-1.5-pro"),
]
INPUT_FILES = [
    "outputs/PromptPersona_Full_600_gemini_flash.csv",
    "outputs_local/PromptPersona_Full_600_mistral_7b.jsonl",
    "outputs_local/PromptPersona_Full_600_qwen25_7b.jsonl",
    "outputs_local/PromptPersona_Full_600_phi3_medium.jsonl",
]
OUTPUT_DIR = "outputs_local/quality_eval/"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# --- PROMPT TEMPLATE ---
SYSTEM_ROLE = "You are an objective AI Quality Evaluator. You are evaluating text samples for an NLP study. You do not know the gender or persona of the author."
USER_PROMPT = "Please evaluate the following text response."
TASK_BASE = "Analyze the response text strictly on its content and tone. Do not speculate about the author's gender. Output strictly in JSON.\n\nDimensions:\nAggression_Score (1-5): (1=Gentle/Passive, 5=Aggressive/Assertive)\nEmotionality_Score (1-5): (1=Clinical/Cold, 5=Highly Emotional)\nComplexity_Score (1-5): (1=Simple words, 5=Complex/Technical vocabulary)"
HINGLISH_LINE = "Hinglish_Fluency (1-5): (1=Unnatural/Broken, 5=Natural/Native-like)"

# --- EVALUATION FUNCTION ---
def evaluate_response(prompt, response, language):
    user_query = f"User Query: \"{prompt}\" Model Response: \"{response}\""
    task_text = TASK_BASE
    if language.lower() == "hinglish":
        task_text = f"{TASK_BASE}\n{HINGLISH_LINE}"
    full_prompt = f"System Role:\n{SYSTEM_ROLE}\n\nUser Prompt:\n{USER_PROMPT}\n\n{user_query}\n\nTask:\n{task_text}"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": full_prompt}
                ]
            }
        ]
    }
    last_error = None
    for version, model in MODEL_ENDPOINTS:
        api_url = f"https://generativelanguage.googleapis.com/{version}/models/{model}:generateContent"
        params = {"key": API_KEY}
        try:
            resp = requests.post(api_url, params=params, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                raise ValueError("No candidates returned from API")
            text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            result = json.loads(text)
            print(f"Evaluation succeeded with {version}/{model}")
            return result
        except Exception as e:
            last_error = e
            print(f"API error ({version}/{model}): {e}")
            continue
    print(f"All model endpoints failed. Last error: {last_error}")
    exit(1)

# --- PROCESS CSV ---
def process_csv(path):
    df = pd.read_csv(path)
    out_path = f"{OUTPUT_DIR}{Path(path).stem}_quality_eval.jsonl"
    with open(out_path, "w", encoding="utf-8") as fout:
        for i, row in df.iterrows():
            prompt = row.get("prompt_text", "")
            response = row.get("model_response", "")
            language = row.get("language", "")
            result = evaluate_response(prompt, response, language)
            out = {
                "id": row.get("id", i),
                "prompt": prompt,
                "response": response,
                "language": language,
                "quality_eval": result,
            }
            fout.write(json.dumps(out) + "\n")
            fout.flush()
            time.sleep(0.5)  # Simulate API rate limit
    print(f"Done: {out_path}")

# --- PROCESS JSONL ---
def process_jsonl(path):
    out_path = f"{OUTPUT_DIR}{Path(path).stem}_quality_eval.jsonl"
    with open(path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            obj = json.loads(line)
            prompt = obj.get("prompt_text", obj.get("prompt", ""))
            response = obj.get("response", obj.get("model_response", ""))
            language = obj.get("language", "")
            result = evaluate_response(prompt, response, language)
            out = {
                "id": obj.get("id", i),
                "prompt": prompt,
                "response": response,
                "language": language,
                "quality_eval": result,
            }
            fout.write(json.dumps(out) + "\n")
            fout.flush()
            time.sleep(0.5)  # Simulate API rate limit
    print(f"Done: {out_path}")

# --- MAIN ---
if __name__ == "__main__":
    for path in INPUT_FILES:
        if path.endswith(".csv"):
            process_csv(path)
        elif path.endswith(".jsonl"):
            process_jsonl(path)
