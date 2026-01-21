import os
import torch
from dataset import get_dataset
from feature_adapter import UnifiedAdapter

class DataHandler:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
        
        self.dataset_name = args.dataset
        self.data_dir = args.data_dir
        self.seed = args.data_seed
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.num_classes = 0
        self.num_features = 0
        self.adapter = None
        
        self._load_dataset()
        self._init_adapter()

    def _load_dataset(self):
        print(f"Loading dataset: {self.dataset_name}...")
        datasets, self.num_classes, self.num_features = get_dataset(
            self.dataset_name, self.data_dir, self.seed
        )
        self.train_dataset, self.val_dataset, self.test_dataset = datasets
        self.dataset_desc = self.train_dataset.dataset.dataset_desc

    def _init_adapter(self):
        self.adapter = UnifiedAdapter(self.device)

    def get_adapter(self):
        return self.adapter
