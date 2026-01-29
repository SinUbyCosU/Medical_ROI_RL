#!/usr/bin/env python3
"""Compute PC1 (first principal component) for per-layer deltas and save vectors + metadata.

Reads: analysis_output/extracted_pairs.jsonl
Writes: vectors/pc1_layer_{ell}.pt and analysis_output/pc1_summary.json
"""
from pathlib import Path
import json
import numpy as np
from sklearn.decomposition import PCA
import torch

INPATH = Path('analysis_output/extracted_pairs.jsonl')
OUT_JSON = Path('analysis_output/pc1_summary.json')
OUT_VEC_DIR = Path('vectors')
OUT_VEC_DIR.mkdir(parents=True, exist_ok=True)

if not INPATH.exists():
    raise SystemExit(f"Missing input file: {INPATH}")

# load records
recs = []
with INPATH.open(encoding='utf-8') as f:
    for line in f:
        line=line.strip()
        if not line: continue
        rec=json.loads(line)
        recs.append(rec)

if not recs:
    raise SystemExit('No extracted pairs found')

L = max(len(r.get('layer_h',[])) for r in recs)
summary={}
for ell in range(L):
    # collect deltas where both eng and cm present
    deltas = []
    for r in recs:
        layer = r.get('layer_h',[])
        if ell < len(layer):
            ent = layer[ell].get('eng') if isinstance(layer[ell], dict) else None
            cmt = layer[ell].get('cm')  if isinstance(layer[ell], dict) else None
            if ent is None or cmt is None:
                # try alternative formats
                try:
                    ent = layer[ell][0]
                    cmt = layer[ell][1]
                except Exception:
                    ent=None; cmt=None
            if ent is not None and cmt is not None:
                deltas.append(np.asarray(ent, dtype=float) - np.asarray(cmt, dtype=float))
    if len(deltas)==0:
        summary[ell]={'n':0,'evr1':None,'dim':None,'pc1_path':None,'mean_abs_proj':None}
        continue
    D = np.stack(deltas)
    n,dim = D.shape
    # center
    Dc = D - D.mean(axis=0, keepdims=True)
    # PCA
    pca = PCA(n_components=min(dim, min(50, n)))
    pca.fit(Dc)
    evr1 = float(pca.explained_variance_ratio_[0]) if pca.explained_variance_ratio_.size>0 else None
    pc1 = pca.components_[0]
    # project deltas onto pc1
    projs = Dc.dot(pc1)
    mean_abs_proj = float(np.mean(np.abs(projs)))
    # save pc1
    vec_path = OUT_VEC_DIR/f'pc1_layer_{ell}.pt'
    torch.save(torch.tensor(pc1, dtype=torch.float32), str(vec_path))
    summary[ell] = {'n':int(n),'evr1':evr1,'dim':int(dim),'pc1_path':str(vec_path),'mean_abs_proj':mean_abs_proj}

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps(summary, indent=2))
print(f'Wrote {OUT_JSON} and pc1 vectors to {OUT_VEC_DIR}')
