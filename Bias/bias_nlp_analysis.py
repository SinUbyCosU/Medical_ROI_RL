import pandas as pd
import json
from pathlib import Path
from transformers import AutoTokenizer
from textblob import TextBlob
from detoxify import Detoxify
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIG ---
CSV_PATH = "outputs/PromptPersona_Full_600_gemini_flash.csv"
JSONL_PATHS = [
    "outputs_local/PromptPersona_Full_600_mistral_7b.jsonl",
    "outputs_local/PromptPersona_Full_600_qwen25_7b.jsonl",
    "outputs_local/PromptPersona_Full_600_phi3_medium.jsonl",
]
TOKENIZER_MODEL = "gpt2"  # Change to match your model if needed

# --- TOKENIZER ---
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)

def sentiment_score(text):
    try:
        return TextBlob(str(text)).sentiment.polarity
    except Exception:
        return np.nan

def toxicity_score(text):
    try:
        scores = Detoxify('original').predict(str(text))
        return scores['toxicity']
    except Exception:
        return np.nan

def lexical_diversity(text):
    words = str(text).split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)

def analyze_csv(path):
    df = pd.read_csv(path)
    df["sentiment"] = df["model_response"].apply(sentiment_score)
    df["toxicity"] = df["model_response"].apply(toxicity_score)
    df["lexical_diversity"] = df["model_response"].apply(lexical_diversity)
    summary = df.groupby(["gender", "language"]).agg(
        avg_sentiment=("sentiment", "mean"),
        avg_toxicity=("toxicity", "mean"),
        avg_lexdiv=("lexical_diversity", "mean"),
    ).reset_index()
    print(f"\n--- CSV: {path} ---")
    print(summary)
    summary.plot(x="gender", y=["avg_sentiment", "avg_toxicity", "avg_lexdiv"], kind="bar", figsize=(10,6))
    plt.title(f"Gemini Flash: Sentiment, Toxicity, Lexical Diversity by Gender/Language")
    plt.tight_layout()
    plt.savefig('outputs_local/flash_nlp_metrics.png')
    plt.close()
    return df

def analyze_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            gender = obj.get("gender", obj.get("persona_code", "Unknown"))
            lang = obj.get("language", "Unknown")
            response = obj.get("response", obj.get("model_response", ""))
            sentiment = sentiment_score(response)
            toxicity = toxicity_score(response)
            lexdiv = lexical_diversity(response)
            rows.append({"gender": gender, "language": lang, "sentiment": sentiment, "toxicity": toxicity, "lexical_diversity": lexdiv})
    df = pd.DataFrame(rows)
    summary = df.groupby(["gender", "language"]).agg(
        avg_sentiment=("sentiment", "mean"),
        avg_toxicity=("toxicity", "mean"),
        avg_lexdiv=("lexical_diversity", "mean"),
    ).reset_index()
    print(f"\n--- JSONL: {path} ---")
    print(summary)
    summary.plot(x="gender", y=["avg_sentiment", "avg_toxicity", "avg_lexdiv"], kind="bar", figsize=(10,6))
    plt.title(f"{Path(path).stem}: Sentiment, Toxicity, Lexical Diversity by Gender/Language")
    plt.tight_layout()
    plt.savefig(f'outputs_local/{Path(path).stem}_nlp_metrics.png')
    plt.close()
    return df

# --- RUN ---
analyze_csv(CSV_PATH)
for path in JSONL_PATHS:
    if Path(path).exists():
        analyze_jsonl(path)
