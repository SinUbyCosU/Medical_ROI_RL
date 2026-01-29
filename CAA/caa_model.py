
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from clas_model import CLAS_Model

class CAA_Model(CLAS_Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.v_delta = torch.load('results/caa_vector.pt')

    def generate(self, prompt, alpha=1.0, **kwargs):
        # For now, just call the base generate without steer_vector/steer_everywhere
        return super().generate(
            prompt,
            alpha=alpha,
            **kwargs
        )
