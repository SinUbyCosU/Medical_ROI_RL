import pandas as pd
from pathlib import Path

root = Path("/root/data/CheXpert-v1.0-small")
df = pd.read_csv(root/"train.csv")  # use valid.csv for val if needed
classes = ["Atelectasis","Cardiomegaly","Consolidation","Edema","Pleural Effusion"]

rows = []
for _, r in df.iterrows():
    img_rel = r["Path"]
    labs = []
    for c in classes:
        v = r[c]
        if pd.notna(v) and v == 1:
            labs.append(c)
    rows.append({"image": img_rel, "labels": ",".join(labs)})

pd.DataFrame(rows).to_csv("chexpert_train.csv", index=False)
print("wrote chexpert_train.csv", len(rows))
