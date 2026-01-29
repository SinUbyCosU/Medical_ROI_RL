#!/usr/bin/env python3
"""Extract English/Hinglish hidden states for paired prompts and save JSONL.

Writes: analysis_output/extracted_pairs.jsonl
"""
from __future__ import annotations
import csv, json
from pathlib import Path
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_PROMPT_CSV = Path('Bias/PromptPersona_Full_600.csv')
DEFAULT_OUT = Path('analysis_output/extracted_pairs.jsonl')


def load_pairs(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Prompt CSV not found: {path}")
    eng_lookup = {}
    hinglish_rows = []
    with path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            lang = (row.get('language') or '').strip().lower()
            key = (row.get('domain') or '').strip(), (row.get('topic') or '').strip(), (row.get('gender') or '').strip()
            prompt = (row.get('prompt_text') or '').strip()
            if not prompt:
                continue
            if lang == 'english':
                eng_lookup[key] = prompt
            elif lang in ('hinglish','code-mixed','cm') or 'hinglish' in (row.get('language') or '').lower():
                hinglish_rows.append((key, prompt))
    pairs = []
    for key, hin in hinglish_rows:
        eng = eng_lookup.get(key)
        if eng:
            pairs.append((eng, hin))
    return pairs


def format_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"<|user|>\n{prompt}\n<|assistant|>"


def capture_all_layers(model, tokenizer, prompt: str):
    formatted = format_prompt(tokenizer, prompt)
    inputs = tokenizer(formatted, return_tensors='pt')
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    # hidden_states is tuple length L+1; take last-token vector for each layer
    arrs = []
    for layer in hidden_states:
        vec = layer[0, -1, :].cpu().numpy().tolist()
        arrs.append(vec)
    return arrs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt-csv', type=Path, default=DEFAULT_PROMPT_CSV)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--model-id', default='microsoft/Phi-3.5-mini-instruct')
    parser.add_argument('--max-pairs', type=int, default=600)
    args = parser.parse_args()

    pairs = load_pairs(args.prompt_csv)
    if not pairs:
        raise SystemExit('No aligned pairs found in prompt CSV')
    pairs = pairs[:args.max_pairs]

    print(f'Found {len(pairs)} pairs; loading model {args.model_id}')
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16, device_map='auto')
    model.eval()

    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as fh:
        for idx,(eng,hin) in enumerate(pairs, start=1):
            try:
                h_eng = capture_all_layers(model, tokenizer, eng)
                h_cm  = capture_all_layers(model, tokenizer, hin)
            except Exception as e:
                print(f'Error capturing pair {idx}: {e}')
                continue
            rec = {'prompt_eng': eng, 'prompt_cm': hin, 'layer_h': [{'eng':e,'cm':c} for e,c in zip(h_eng,h_cm)]}
            fh.write(json.dumps(rec) + '\n')
            if idx % 5 == 0 or idx==len(pairs):
                print(f'Wrote {idx}/{len(pairs)}')
    print(f'Wrote extracted pairs to {out}')

if __name__=='__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
