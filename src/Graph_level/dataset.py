
import os
import glob
import json
import os.path as osp
import torch
import copy
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
import torch_geometric.transforms as T
import pickle as pkl
import random
import numpy as np
from tqdm import tqdm
from mmcv import Config
import sys

from torch_geometric.utils import dense_to_sparse, remove_self_loops, add_self_loops
from scipy.spatial.distance import cdist
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from utils import set_random_seed


def compute_adjacency_matrix_images(coord, sigma=0.1):
    coord = coord.reshape(-1, 2)
    dist = cdist(coord, coord)
    A = np.exp(- dist / (sigma * np.pi) ** 2)
    A[np.diag_indices_from(A)] = 0
    return A

def list_to_torch(data):
    for i in range(len(data)):
        if data[i] is None:
            continue
        elif isinstance(data[i], np.ndarray):
            if data[i].dtype == np.bool:
                data[i] = data[i].astype(np.float32)
            data[i] = torch.from_numpy(data[i]).float()
        elif isinstance(data[i], list):
            data[i] = list_to_torch(data[i])
    return data

class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data

def undirected_graph(data):
    data.edge_index = torch.cat([torch.stack([data.edge_index[1], data.edge_index[0]], dim=0),
                                 data.edge_index], dim=1)
    return data


def split(data, batch):
    # i-th contains elements from slice[i] to slice[i+1]
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])
    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)
    data.__num_nodes__ = np.bincount(batch).tolist()

    slices = dict()
    slices['x'] = node_slice
    slices['edge_index'] = edge_slice
    slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    slices['sentence_tokens'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    return data, slices

def read_file(folder, prefix, name):
    file_path = osp.join(folder, prefix + f'_{name}.txt')
    return np.genfromtxt(file_path, dtype=np.int64)

def read_sentigraph_data(folder: str, prefix: str):
    txt_files = glob.glob(os.path.join(folder, "{}_*.txt".format(prefix)))
    json_files = glob.glob(os.path.join(folder, "{}_*.json".format(prefix)))
    txt_names = [f.split(os.sep)[-1][len(prefix) + 1:-4] for f in txt_files]
    json_names = [f.split(os.sep)[-1][len(prefix) + 1:-5] for f in json_files]
    names = txt_names + json_names

    with open(os.path.join(folder, prefix + "_node_features.pkl"), 'rb') as f:
        x: np.array = pkl.load(f)
    x: torch.FloatTensor = torch.from_numpy(x)
    edge_index: np.array = read_file(folder, prefix, 'edge_index')
    edge_index: torch.tensor = torch.tensor(edge_index, dtype=torch.long).T
    batch: np.array = read_file(folder, prefix, 'node_indicator') - 1  # from zero
    y: np.array = read_file(folder, prefix, 'graph_labels')
    y: torch.tensor = torch.tensor(y, dtype=torch.long)
    edge_attr = torch.ones((edge_index.size(1), 1)).float()
    name = torch.tensor(range(y.size(0)))
    supplement = dict()
    if 'split_indices' in names:
        split_indices: np.array = read_file(folder, prefix, 'split_indices')
        split_indices = torch.tensor(split_indices, dtype=torch.long)
        supplement['split_indices'] = split_indices
    if 'sentence_tokens' in names:
        with open(os.path.join(folder, prefix + '_sentence_tokens.json')) as f:
            sentence_tokens: dict = json.load(f)
        supplement['sentence_tokens'] = sentence_tokens
    data = Data(name=name, x=x, edge_index=edge_index, y=y, sentence_tokens=list(sentence_tokens.values()))
    data, slices = split(data, batch)

    return data, slices, supplement

class SentiGraphTransform(object):

    def __init__(self, transform=None):
        self.transform = transform

    def __call__(self, data):
        data.edge_attr = torch.ones(data.edge_index.size(1), 1)
        if self.transform is not None:
            return self.transform(data)
        return data


class SentiGraphDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=undirected_graph):
        self.name = name
        super(SentiGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices, self.supplement = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['node_features', 'node_indicator', 'sentence_tokens', 'edge_index',
                'graph_labels', 'split_indices']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # Read data into huge `Data` list.
        self.data, self.slices, self.supplement \
            = read_sentigraph_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices, self.supplement), self.processed_paths[0])


def load_SeniGraph(dataset_dir, dataset_name, transform=None):
    sent_transform = SentiGraphTransform(transform)
    dataset = SentiGraphDataset(root=dataset_dir, name=dataset_name, transform=sent_transform)
    return dataset

def get_dataset(args, sparse=True, cleaned=False):
    name = args.dataset
    set_random_seed(args.data_seed)
    data_dir = args.data_dir

    if name in ["Graph-SST2"]:
        datadir = data_dir
        dataset = load_SeniGraph(datadir, name)
        num_classes = dataset.num_classes
        num_features = dataset.num_features
        test_idx = []
        train = []
        for idx, g in enumerate(dataset):
            if g.num_edges <= 2:
                continue
            degree = float(g.num_edges) / g.num_nodes
            if degree >= 1.76785714:
                train.append(idx)
            elif degree <= 1.57142857:
                test_idx.append(idx)

        val_idx = train[:int(len(train) * 0.1)]
        train_idx = train[int(len(train) * 0.1):]

        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        val_dataset = dataset[val_idx]
        dataset = [train_dataset, val_dataset, test_dataset]
        return dataset, num_classes, num_features

    elif 'ogbg' in name:
        datasets = PygGraphPropPredDataset(root=data_dir, name='-'.join(name.split('_')))
        num_classes = datasets.num_classes
        num_features = datasets.num_features
        split_idx = datasets.get_idx_split()
        train_dataset = datasets[split_idx["train"]]
        val_dataset = datasets[split_idx["valid"]]
        test_dataset = datasets[split_idx["test"]]
        dataset = [train_dataset, val_dataset, test_dataset]
        return dataset, num_classes, num_features

    else:
        raise NotImplementedError
        return dataset, num_classes, num_features


def get_dataloader(dataset_name, dataset, batch_size, args):
    train_dataset, val_dataset, test_dataset = dataset

    dataloader = dict()
    dataloader['train'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dataloader['val'] = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    dataloader['test'] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return dataloader

