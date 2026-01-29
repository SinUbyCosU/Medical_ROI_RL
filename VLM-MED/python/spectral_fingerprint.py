# spectral_fingerprint.py
import json, numpy as np
from pathlib import Path
from sklearn.decomposition import PCA

def load_pairs(path):
    p=Path(path)
    if not p.exists():
        raise SystemExit(f"Missing input file: {p}")
    recs=[]
    with p.open(encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            rec=json.loads(line)
            recs.append(rec)
    return recs

def per_layer_stats(recs, layer_key='layer_h'):
    L = max(len(r.get(layer_key, [])) for r in recs)
    out={}
    for ell in range(L):
        H_eng=[]
        H_cm=[]
        for r in recs:
            layer = r.get(layer_key, [])
            if ell < len(layer):
                ent = layer[ell].get('eng') if isinstance(layer[ell], dict) else None
                cmt = layer[ell].get('cm')  if isinstance(layer[ell], dict) else None
                if ent is None or cmt is None:
                    # fall back to numeric arrays stored directly
                    ent = layer[ell][0] if isinstance(layer[ell], (list,tuple)) and len(layer[ell])>0 else None
                    cmt = layer[ell][1] if isinstance(layer[ell], (list,tuple)) and len(layer[ell])>1 else None
                if ent is not None and cmt is not None:
                    H_eng.append(np.asarray(ent, dtype=float))
                    H_cm.append(np.asarray(cmt, dtype=float))
        if len(H_eng)==0:
            out[ell]={'evr1':None,'mean_sim':None,'n':0}
            continue
        H_eng = np.stack(H_eng)
        H_cm  = np.stack(H_cm)
        deltas = H_eng - H_cm
        pca = PCA().fit(deltas)
        evr1 = float(pca.explained_variance_ratio_[0]) if pca.explained_variance_ratio_.size>0 else 0.0
        sims = np.sum(H_eng*H_cm,axis=1) / (np.linalg.norm(H_eng,axis=1)*np.linalg.norm(H_cm,axis=1)+1e-12)
        out[ell] = {'evr1': evr1, 'mean_sim': float(np.mean(sims)), 'n': int(len(sims))}
    return out

if __name__=='__main__':
    inpath = Path('analysis_output/extracted_pairs.jsonl')
    recs = load_pairs(inpath)
    stats = per_layer_stats(recs, layer_key='layer_h')
    out = Path('analysis_output/spectral_stats.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2))
    print(f'Wrote {out}')
