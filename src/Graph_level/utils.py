import os.path

import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, sort_edge_index, subgraph
from torch_geometric.loader import DataLoader
import networkx as nx
from tqdm import tqdm


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

