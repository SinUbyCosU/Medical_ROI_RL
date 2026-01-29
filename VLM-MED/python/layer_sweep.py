#!/usr/bin/env python3
"""Layer sweep experiment

Creates a CSV of model generations while sweeping injection layers and alpha values.

Usage (recommended in tmux):
    tmux new -s layer_sweep
    python -u layer_sweep.py --prompts test_prompts.json --num-prompts 50 --output layer_sweep_results.csv

Notes:
- Expects a model wrapper `CLAS_Model` with `generate(prompt, injection_layer, alpha)` available on PYTHONPATH.
"""

import argparse
import json
import os
import sys
from tqdm import tqdm
import csv

try:
    from steering import SteeringVector  # optional, if used
except Exception:
    SteeringVector = None

try:
    from clas_model import CLAS_Model
except Exception:
    CLAS_Model = None


def get_utility_score(response_text):
    """Placeholder scoring function. Replace with project-specific scorer.

    Returns a float utility score for a response.
    """
    # TODO: implement or import the real scoring function
    return 0.0


def extract_prompts_from_dict(data, key=None, max_n=200):
    candidates = []
    keys = [key] if key else list(data.keys())
    for k in keys:
        if k not in data:
            continue
        val = data[k]
        if isinstance(val, list):
            for item in val:
                if isinstance(item, str) and item.strip():
                    candidates.append(item.strip())
                elif isinstance(item, dict):
                    for fld in ('content', 'prompt', 'text'):
                        if fld in item and isinstance(item[fld], str) and item[fld].strip():
                            candidates.append(item[fld].strip())
                            break
                    else:
                        # fallback: stringify the dict
                        candidates.append(json.dumps(item, ensure_ascii=False))
        elif isinstance(val, str) and val.strip():
            candidates.append(val.strip())
    seen = set()
    out = []
    for s in candidates:
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= max_n:
            break
    return [{'id': str(i), 'prompt': p} for i, p in enumerate(out)]


def load_prompts(path, key=None, max_n=200):
    # Try JSONL first
    if path.endswith('.jsonl'):
        prompts = []
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_n:
                    break
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                # Try to extract a prompt field
                prompt = item.get('prompt_with_prefix') or item.get('prompt') or item.get('text') or item.get('content')
                if not prompt:
                    # fallback: stringify the dict
                    prompt = json.dumps(item, ensure_ascii=False)
                pid = item.get('id') or item.get('prompt_id') or str(i)
                prompts.append({'id': pid, 'prompt': prompt})
        return prompts
    # Otherwise, treat as JSON
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        prompts = []
        for i, item in enumerate(data):
            if isinstance(item, dict):
                pid = item.get('id', str(i))
                prompt = item.get('prompt') or item.get('text') or item.get('content') or ''
            else:
                pid = str(i)
                prompt = str(item)
            prompts.append({'id': pid, 'prompt': prompt})
        return prompts[:max_n]
    elif isinstance(data, dict):
        return extract_prompts_from_dict(data, key=key, max_n=max_n)
    else:
        raise ValueError('Prompts file must contain a JSON list or dict')


def load_prompts_from_csv(path, column='Sentences', num_prompts=None):
    prompts = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError('CSV has no header')
        if column not in reader.fieldnames:
            column = reader.fieldnames[0]
        for i, row in enumerate(reader):
            if num_prompts and i >= num_prompts:
                break
            text = row.get(column, '')
            if text is None:
                text = ''
            prompts.append({'id': str(i), 'prompt': text})
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts', type=str, default='test_prompts.json')
    parser.add_argument('--prompts-csv', type=str, default=None, help='Path to a CSV to load prompts from (overrides --prompts when set)')
    parser.add_argument('--prompt-csv-column', type=str, default='Sentences', help='CSV column to read prompts from (default: Sentences)')
    parser.add_argument('--prompt-key', type=str, default=None, help='Top-level key to extract prompts from when prompts file is a dict')
    parser.add_argument('--num-prompts', type=int, default=50, help='Number of prompts to load (default: 50)')
    parser.add_argument('--prefix', type=str, default=None, help='Optional prefix string to prepend to every prompt')
    parser.add_argument('--output', type=str, default='layer_sweep_results.csv')
    parser.add_argument('--model', type=str, default=None, help='Optional model identifier to pass to CLAS_Model')
    args = parser.parse_args()

    # Load prompts: prefer CSV when provided (or when --prompts endswith .csv)
    if args.prompts_csv or (args.prompts and args.prompts.lower().endswith('.csv')):
        csv_path = args.prompts_csv if args.prompts_csv else args.prompts
        prompts = load_prompts_from_csv(csv_path, column=args.prompt_csv_column, num_prompts=args.num_prompts)
    else:
        prompts = load_prompts(args.prompts, key=args.prompt_key, max_n=args.num_prompts)

    # Apply prefix if provided
    if args.prefix:
        for p in prompts:
            p['prompt'] = f"{args.prefix} {p['prompt']}"

    layers_to_test = [10, 12, 13, 14, 16, 18, 20, 24, 28]
    alphas = [0.05]

    if CLAS_Model is None:
        print('Error: CLAS_Model is not importable. Ensure the model wrapper is on PYTHONPATH.', file=sys.stderr)
        sys.exit(2)

    model = CLAS_Model(args.model) if args.model else CLAS_Model()

    # We'll include alpha in the CSV so each run is unambiguous.
    fieldnames = ['prompt_id', 'layer', 'response_text', 'utility_score']

    tmp_output = args.output + '.tmp'
    with open(tmp_output, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # outer loop: prompts, inner: layer x alpha (tqdm on total combinations)
        total = len(prompts) * len(layers_to_test) * len(alphas)
        pbar = tqdm(total=total, desc='Layer sweep')

        for p in prompts:
            prompt_id = p['id']
            prompt_text = p['prompt']
            for layer in layers_to_test:
                for alpha in alphas:
                    # Call model.generate(prompt, injection_layer, alpha)
                    try:
                        response = model.generate(prompt_text, injection_layer=layer, alpha=alpha)
                        # If generate returns object, extract text
                        response_text = response if isinstance(response, str) else getattr(response, 'text', str(response))
                    except Exception as e:
                        response_text = f'ERROR: {e}'

                    try:
                        score = get_utility_score(response_text)
                    except Exception:
                        score = None

                    writer.writerow({
                        'prompt_id': prompt_id,
                        'layer': layer,
                        'response_text': response_text,
                        'utility_score': score,
                    })
                    csvfile.flush()
                    pbar.update(1)

        pbar.close()

    os.replace(tmp_output, args.output)
    print('Wrote results to', args.output)


if __name__ == '__main__':
    main()
