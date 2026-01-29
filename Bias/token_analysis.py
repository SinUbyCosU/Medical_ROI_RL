import pandas as pd
import json
from pathlib import Path
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from textblob import TextBlob
from detoxify import Detoxify

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

def analyze_csv(path):
    df = pd.read_csv(path)
    df["word_count"] = df["model_response"].astype(str).str.split().str.len()
    df["token_count"] = df["model_response"].apply(lambda x: len(tokenizer.encode(str(x))))
    # Sentiment analysis
    df["sentiment"] = df["model_response"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    # Toxicity analysis
    df["toxicity"] = df["model_response"].apply(lambda x: Detoxify('original').predict(str(x))["toxicity"])
    # Lexical diversity
    df["lexical_diversity"] = df["model_response"].apply(lambda x: len(set(str(x).split())) / (len(str(x).split()) or 1))
    summary = df.groupby(["gender", "language"]).agg(
        total_responses=("model_response", "count"),
        avg_word_count=("word_count", "mean"),
        avg_token_count=("token_count", "mean"),
        sum_word_count=("word_count", "sum"),
        sum_token_count=("token_count", "sum"),
        avg_sentiment=("sentiment", "mean"),
        avg_toxicity=("toxicity", "mean"),
        avg_lexical_diversity=("lexical_diversity", "mean"),
    ).reset_index()
    print(f"\n--- CSV: {path} ---")
    print(summary)
    # Sentiment plot
    fig, ax = plt.subplots(figsize=(8,5))
    for lang in summary['language'].unique():
        subset = summary[summary['language'] == lang]
        ax.bar(subset['gender'] + f" ({lang})", subset['avg_sentiment'], label=f"{lang}")
    ax.set_ylabel('Avg Sentiment (TextBlob)')
    ax.set_title(f'Gemini Flash: Sentiment by Gender/Language')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.savefig('outputs_local/flash_sentiment.png')
    plt.close()
    # Toxicity plot
    fig, ax = plt.subplots(figsize=(8,5))
    for lang in summary['language'].unique():
        subset = summary[summary['language'] == lang]
        ax.bar(subset['gender'] + f" ({lang})", subset['avg_toxicity'], label=f"{lang}")
    ax.set_ylabel('Avg Toxicity (Detoxify)')
    ax.set_title(f'Gemini Flash: Toxicity by Gender/Language')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.savefig('outputs_local/flash_toxicity.png')
    plt.close()
    # Lexical diversity plot
    fig, ax = plt.subplots(figsize=(8,5))
    for lang in summary['language'].unique():
        subset = summary[summary['language'] == lang]
        ax.bar(subset['gender'] + f" ({lang})", subset['avg_lexical_diversity'], label=f"{lang}")
    ax.set_ylabel('Avg Lexical Diversity')
    ax.set_title(f'Gemini Flash: Lexical Diversity by Gender/Language')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.savefig('outputs_local/flash_lexdiv.png')
    plt.close()
    # Plot average word count by gender/language
    fig, ax = plt.subplots(figsize=(8,5))
    for lang in summary['language'].unique():
        subset = summary[summary['language'] == lang]
        ax.bar(subset['gender'] + f" ({lang})", subset['avg_word_count'], label=f"{lang}")
    ax.set_ylabel('Avg Word Count')
    ax.set_title(f'Gemini Flash: Avg Word Count by Gender/Language')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.savefig('outputs_local/flash_wordcount.png')
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
            word_count = len(str(response).split())
            token_count = len(tokenizer.encode(str(response)))
            sentiment = TextBlob(str(response)).sentiment.polarity
            toxicity = Detoxify('original').predict(str(response))["toxicity"]
            lexdiv = len(set(str(response).split())) / (len(str(response).split()) or 1)
            rows.append({"gender": gender, "language": lang, "word_count": word_count, "token_count": token_count,
                        "sentiment": sentiment, "toxicity": toxicity, "lexical_diversity": lexdiv})
    df = pd.DataFrame(rows)
    summary = df.groupby(["gender", "language"]).agg(
        total_responses=("word_count", "count"),
        avg_word_count=("word_count", "mean"),
        avg_token_count=("token_count", "mean"),
        sum_word_count=("word_count", "sum"),
        sum_token_count=("token_count", "sum"),
        avg_sentiment=("sentiment", "mean"),
        avg_toxicity=("toxicity", "mean"),
        avg_lexical_diversity=("lexical_diversity", "mean"),
    ).reset_index()
    print(f"\n--- JSONL: {path} ---")
    print(summary)
    # Sentiment plot
    fig, ax = plt.subplots(figsize=(8,5))
    for lang in summary['language'].unique():
        subset = summary[summary['language'] == lang]
        ax.bar(subset['gender'] + f" ({lang})", subset['avg_sentiment'], label=f"{lang}")
    ax.set_ylabel('Avg Sentiment (TextBlob)')
    ax.set_title(f'{Path(path).stem}: Sentiment by Gender/Language')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.savefig(f'outputs_local/{Path(path).stem}_sentiment.png')
    plt.close()
    # Toxicity plot
    fig, ax = plt.subplots(figsize=(8,5))
    for lang in summary['language'].unique():
        subset = summary[summary['language'] == lang]
        ax.bar(subset['gender'] + f" ({lang})", subset['avg_toxicity'], label=f"{lang}")
    ax.set_ylabel('Avg Toxicity (Detoxify)')
    ax.set_title(f'{Path(path).stem}: Toxicity by Gender/Language')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.savefig(f'outputs_local/{Path(path).stem}_toxicity.png')
    plt.close()
    # Lexical diversity plot
    fig, ax = plt.subplots(figsize=(8,5))
    for lang in summary['language'].unique():
        subset = summary[summary['language'] == lang]
        ax.bar(subset['gender'] + f" ({lang})", subset['avg_lexical_diversity'], label=f"{lang}")
    ax.set_ylabel('Avg Lexical Diversity')
    ax.set_title(f'{Path(path).stem}: Lexical Diversity by Gender/Language')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.savefig(f'outputs_local/{Path(path).stem}_lexdiv.png')
    plt.close()
    # Plot average word count by gender/language
    fig, ax = plt.subplots(figsize=(8,5))
    for lang in summary['language'].unique():
        subset = summary[summary['language'] == lang]
        ax.bar(subset['gender'] + f" ({lang})", subset['avg_word_count'], label=f"{lang}")
    ax.set_ylabel('Avg Word Count')
    ax.set_title(f'{Path(path).stem}: Avg Word Count by Gender/Language')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.savefig(f'outputs_local/{Path(path).stem}_wordcount.png')
    plt.close()
    return df

# --- RUN ---
analyze_csv(CSV_PATH)
for path in JSONL_PATHS:
    if Path(path).exists():
        analyze_jsonl(path)
