import os
import sys
from pathlib import Path
import torch
from torch.utils.data import Dataset
from .constants import *

project_root = Path(__file__).parent.parent.parent.resolve()
hs2c_path = project_root / '7_HS2C_pipeline'
sys.path.insert(0, str(hs2c_path))

from utils.utils import set_random_seed

class HS2CDataset(Dataset):
    def __init__(self, config, data_type='base'):
        self.config = config
        self.dataset_name = config['dataset']
        self.data_root = config['data_root']
        
        self.data = None
        self.num_features = 0
        self.num_classes = 0
        self.hs2c_data_dir = os.path.join(hs2c_path, 'generated_data')
        os.makedirs(self.hs2c_data_dir, exist_ok=True)
        
        if data_type == 'base':
            self._load_base_data()
        else:
            self._load_enhanced_data()

    def _load_base_data(self):
        """Load base graph data from disk"""
        file_path = os.path.join(self.data_root, f"{self.config['new_date']}_{self.dataset_name}")
        data_file_name = f"baseline_{self.config['new_date']}_{self.dataset_name}.pt"
        full_path = os.path.join(file_path, data_file_name)
        if os.path.exists(full_path):
            self.data = torch.load(full_path, weights_only=False)
            print(self.data)
            print(f"✓ Loaded base data from {full_path}")
        else:
            print(f"[Warn] Base data file {full_path} not found.")

        if self.data:
            self.num_nodes = self.data.num_nodes
            self.num_features = self.data.x.shape[1] if self.data.x is not None else 0
            self.num_classes = self.data.y.max().item() + 1 if self.data.y is not None else 0
        
    def _load_enhanced_data(self):
        """Load enhanced graph data from disk"""
        file_path = os.path.join(self.hs2c_data_dir, f"{self.config['new_date']}_{self.dataset_name}")
        data_file_name = f"{self.config['new_date']}_{self.dataset_name}_knn_target_graph.pt"
        full_path = os.path.join(file_path, data_file_name)
        
        if os.path.exists(full_path):
            self.data = torch.load(full_path, weights_only=False)
            print(self.data)
            print(f"✓ Loaded enhanced data from {full_path}")
        else:
            print(f"[Warn] Enhanced data File {full_path} not found. Please ensure data is prepared.")
            
        if self.data:
            self.num_nodes = self.data.num_nodes
            self.num_features = self.data.x.shape[1] if self.data.x is not None else 0
            self.num_classes = self.data.y.max().item() + 1 if self.data.y is not None else 0
            
    def get_data(self):
        return self.data

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.data
