
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import torch
from clas_model import CLAS_Model

# Load calibration prompts

# Use eng_hin_pairs.csv for calibration pairs
pairs = pd.read_csv('../eng_hin_pairs.csv')
model = CLAS_Model()

# Extract mean hidden states for all English and Hinglish prompts

eng_hiddens = []
hin_hiddens = []
for _, row in pairs.iterrows():
    h_eng = model.get_hidden_states(row['prompt_text_eng'])
    h_hin = model.get_hidden_states(row['prompt_text_hin'])
    # Average over sequence dimension (dim=1), result shape: (1, hidden_dim)
    eng_hiddens.append(h_eng.mean(dim=1))
    hin_hiddens.append(h_hin.mean(dim=1))

# Stack and average over all prompts
eng_mean = torch.cat(eng_hiddens, dim=0).mean(dim=0)
hin_mean = torch.cat(hin_hiddens, dim=0).mean(dim=0)

v_delta = eng_mean - hin_mean

# Save vector
torch.save(v_delta, 'results/caa_vector.pt')
print('Saved CAA vector to results/caa_vector.pt')
