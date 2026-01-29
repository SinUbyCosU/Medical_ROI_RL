#!/usr/bin/env python3
"""Extract a steering vector for Spanglish prompts using Phi-3.5-mini."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
LAYER_ID = 16
OUTPUT_PATH = "vectors/spanglish_vector.pt"

PAIRS = [
    ("How do I fix a DNS error?", "Como fixeo el error de DNS, no carga nada?"),
    ("Reset the router settings.", "Resetea los settings del router, por favor."),
    ("Install python requirements.", "Instala los requirements de python con pip."),
    ("My battery is draining fast.", "Mi bateria se esta drenando super fast."),
    ("Update the drivers manually.", "Haz el update de los drivers manualmente."),
    ("Explain quantum physics simply.", "Explicame quantum physics pero simple."),
    ("Write a cover letter for a job.", "Escribe una cover letter para una chamba."),
    ("Debug this code block.", "Debugea este bloque de codigo."),
    ("Plan a healthy diet.", "Hazme un plan de dieta healthy."),
    ("Check for malware on my PC.", "Chequea si mi PC tiene malware."),
    ("Optimize the database query.", "Optimiza el query de la base de datos."),
    ("Translate this document.", "Traduce este documento right now."),
    ("Calculate the mortgage rate.", "Calcula el rate de la hipoteca."),
    ("Find the best flight deals.", "Encuentra los mejores flight deals."),
    ("Create a backup schedule.", "Crea un schedule para el backup."),
]


def format_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"<|user|>\n{prompt}\n<|assistant|>"


def main() -> None:
    print(f"Extracting Spanglish vector from {len(PAIRS)} pairs...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    diffs = []
    for eng, span in tqdm(PAIRS, desc="Pairs"):
        prompt_eng = format_prompt(tokenizer, eng)
        prompt_span = format_prompt(tokenizer, span)

        inputs_eng = tokenizer(prompt_eng, return_tensors="pt").to(model.device)
        inputs_span = tokenizer(prompt_span, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs_eng = model(**inputs_eng, output_hidden_states=True)
            outputs_span = model(**inputs_span, output_hidden_states=True)

        h_eng = outputs_eng.hidden_states[LAYER_ID][0, -1, :]
        h_span = outputs_span.hidden_states[LAYER_ID][0, -1, :]
        diffs.append(h_eng - h_span)

    if not diffs:
        raise RuntimeError("No activation differences collected; check prompt list")

    vector = torch.stack(diffs).mean(dim=0)
    torch.save(vector, OUTPUT_PATH)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
