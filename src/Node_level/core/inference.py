import sys
from pathlib import Path
import torch
import json
from data_provider.constants import LABEL_WHOL_DICT

project_root = Path(__file__).parent.parent.parent.resolve()
hs2c_path = project_root / '7_HS2C_pipeline'
sys.path.insert(0, str(hs2c_path))


class LLMClassifier:
    def __init__(self, config, llm_engine):
        self.config = config
        self.llm_engine = llm_engine
        self.candidate_labels = list(self.config['label_whol_dict'].values())
        self.candidate_abbr_labels = list(self.config['label_abbr_dict'].values())

    def predict(self, prompt):
        if not self.candidate_labels:
            return "Error: No candidate labels"

        return self.llm_engine.compute_label_likelihood(prompt, self.candidate_labels)
