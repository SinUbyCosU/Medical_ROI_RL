#!/usr/bin/env python3
"""CMI calculator for Hinglish text.

Computes Code-Mixing Index using a simple script-based token classifier.
"""

import re


DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
ASCII_RE = re.compile(r"^[\x00-\x7F]+$")


def classify_token(tok):
    if not tok or tok.isnumeric():
        return None
    if DEVANAGARI_RE.search(tok):
        return 'HI'
    if ASCII_RE.match(tok):
        return 'EN'
    return None


def calculate_cmi(text):
    tokens = text.split()
    counts = {'EN': 0, 'HI': 0}
    for tok in tokens:
        lang = classify_token(tok)
        if lang:
            counts[lang] += 1
    total = counts['EN'] + counts['HI']
    if total == 0:
        return 0.0
    dominant = max(counts.values())
    cmi = 1 - (dominant / total)
    return float(cmi)


if __name__ == '__main__':
    sample = "You are a smart Hinglish AI. Tum bahut accha kaam karte ho."
    print('CMI:', calculate_cmi(sample))
