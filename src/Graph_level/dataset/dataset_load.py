import os
import re
import glob
import json
import copy
import random
import os.path as osp
import pickle as pkl
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import random_split
from torch_geometric.utils import degree, dense_to_sparse, remove_self_loops, add_self_loops
from torch_geometric.utils import from_smiles as _from_smiles
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from scipy.spatial.distance import cdist
from torch_geometric.data import InMemoryDataset, Data, InMemoryDataset, download_url
from torch_geometric.loader import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from ogb.utils import smiles2graph
import traceback
from rdkit import Chem
from typing import Callable, Dict, Optional, Tuple, Union
import zipfile
import gzip
# from dataset.info import *

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
gl_root = PROJECT_ROOT / '5_Graph_level'
data_root = gl_root / 'dataset'
sys.path.insert(0, str(gl_root))
from utils import set_random_seed
from dataset.info import *


def undirected_graph(data):
    data.edge_index = torch.cat([torch.stack([data.edge_index[1], data.edge_index[0]], dim=0), data.edge_index], dim=1)
    return data


def split(data, batch):
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])
    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    data.edge_index -= node_slice[batch[row]].unsqueeze(0)
    data.__num_nodes__ = np.bincount(batch).tolist()

    slices = dict()
    slices['x'] = node_slice
    slices['edge_index'] = edge_slice
    slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    slices['raw_texts'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    if hasattr(data, 'name'):
        slices['name'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
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
    dataset_name = prefix.lower()
    dataset_desc = GRAPH_DESC[dataset_name]
    data = Data(name=name, x=x, edge_index=edge_index, y=y, raw_texts=list(sentence_tokens.values()), dataset_desc=dataset_desc)
    data, slices = split(data, batch)

    return data, slices, supplement, dataset_desc


class SentiGraphTransform(object):

    def __init__(self, transform=None):
        self.transform = transform

    def __call__(self, data):
        data.edge_attr = torch.ones(data.edge_index.size(1), 1)
        if self.transform is not None:
            return self.transform(data)
        return data


class SentiGraphDataset(InMemoryDataset):
    r"""
        The SentiGraph datasets from `Explainability in Graph Neural Networks: A Taxonomic Survey
        <https://arxiv.org/abs/2012.15445>`_.
        The datasets take pretrained BERT as node feature extractor
        and dependency tree as edges to transfer the text sentiment datasets into
        graph classification datasets.

        The dataset `Graph-SST2 <https://drive.google.com/file/d/1-PiLsjepzT8AboGMYLdVHmmXPpgR8eK1/view?usp=sharing>`_
        should be downloaded to the proper directory before running. All the three datasets Graph-SST2, Graph-SST5, and
        Graph-Twitter can be download in this
        `link <https://drive.google.com/drive/folders/1dt0aGMBvCEUYzaG00TYu1D03GPO7305z?usp=sharing>`_.

        Args:
            root (:obj:`str`): Root directory where the datasets are saved
            name (:obj:`str`): The name of the datasets.
            transform (:obj:`Callable`, :obj:`None`): A function/transform that takes in an
                :class:`torch_geometric.data.Data` object and returns a transformed
                version. The data object will be transformed before every access.
                (default: :obj:`None`)
            pre_transform (:obj:`Callable`, :obj:`None`):  A function/transform that takes in
                an :class:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                being saved to disk. (default: :obj:`None`)

        .. note:: The default parameter of pre_transform is :func:`~undirected_graph`
            which transfers the directed graph in original data into undirected graph before
            being saved to disk.
        """
    def __init__(self, root, name, transform=None, pre_transform=undirected_graph):
        self.name = name
        super(SentiGraphDataset, self).__init__(root, transform, pre_transform)
        try:
            loaded_data = torch.load(self.processed_paths[0], weights_only=False)
        except TypeError:
            loaded_data = torch.load(self.processed_paths[0])
        
        if isinstance(loaded_data, tuple):
            if len(loaded_data) == 4:
                self.data, self.slices, self.supplement, self.dataset_desc = loaded_data
            elif len(loaded_data) == 3:
                self.data, self.slices, self.supplement = loaded_data
                self.dataset_desc = GRAPH_DESC.get(self.name.lower(), {}).get('dataset_desc', f"Dataset: {self.name}")
            else:
                raise ValueError(f"Unexpected number of values in processed file: {len(loaded_data)}")
        else:
             raise ValueError(f"Loaded data is not a tuple, got {type(loaded_data)}")

        if 'name' not in self.slices:
            num_graphs = self.slices['y'].numel() - 1
            if hasattr(self, '_data'):
                self._data.name = torch.arange(num_graphs)
            else:
                self.data.name = torch.arange(num_graphs)
            self.slices['name'] = torch.arange(0, num_graphs + 1)

        if 'raw_texts' not in self.slices:
            json_path = osp.join(self.raw_dir, f"{self.name}_sentence_tokens.json")
            if not osp.exists(json_path):
                name_underscore = self.name.replace('-', '_')
                json_path = osp.join(self.raw_dir, f"{name_underscore}_sentence_tokens.json")
            
            if osp.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        sentence_tokens = json.load(f)
                    
                    num_graphs = self.slices['y'].numel() - 1
                    raw_texts_list = []
                    if isinstance(sentence_tokens, list):
                         raw_texts_list = sentence_tokens
                    else:
                        for i in range(num_graphs):
                            key = str(i)
                            if key in sentence_tokens:
                                raw_texts_list.append(sentence_tokens[key])
                            else:
                                raw_texts_list.append([f"node_{x}" for x in range(10)])

                    if len(raw_texts_list) == num_graphs:
                        if hasattr(self, '_data'):
                            self._data.raw_texts = raw_texts_list
                        else:
                            self.data.raw_texts = raw_texts_list
                        self.slices['raw_texts'] = torch.arange(0, num_graphs + 1, dtype=torch.long)
                        print("Successfully injected raw_texts")
                    else:
                        print(f"Mismatch in raw_texts length: {len(raw_texts_list)} vs {num_graphs}")

                except Exception as e:
                    print(f"Failed to inject raw_texts: {e}")
            else:
                print(f"Could not find sentence_tokens.json at {json_path}")

        self.dataset_desc = GRAPH_DESC[self.name.lower()]

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
        try:
            self.data, self.slices, self.supplement, self.dataset_desc = read_sentigraph_data(self.raw_dir, self.name)
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            if type(e) is FileNotFoundError:
                print("Please download the required datasets file to the root directory.")
                print("The google drive link is "
                      "https://drive.google.com/drive/folders/1dt0aGMBvCEUYzaG00TYu1D03GPO7305z?usp=sharing")
            raise SystemExit()

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices, self.supplement, self.dataset_desc), self.processed_paths[0])


def load_SeniGraph(dataset_dir, dataset_name, transform=None):
    sent_transform = SentiGraphTransform(transform)
    dataset = SentiGraphDataset(root=dataset_dir, name=dataset_name, transform=sent_transform)
    return dataset

def maybe_log(path, log=True):
    if log:
        print('Extracting', path)


def extract_gz(path, folder, log=True):
    maybe_log(path, log)
    with gzip.open(path, 'r') as r:
        with open(osp.join(folder, '.'.join(os.path.basename(path).split('.')[:-1])), 'wb') as w:
            w.write(r.read())


def extract_zip(path, folder, log=True):
    maybe_log(path, log)
    with zipfile.ZipFile(path, 'r') as f:
        f.extractall(folder)


def attach_label_text(data, dataset_name, task_name, desc_dict=MOLECULENET_DESC):
    task_info = desc_dict[dataset_name]["tasks"][task_name]

    y_tensor = data.y
    if y_tensor.dim() > 1 and y_tensor.shape[1] > 1:
        y_val = y_tensor[0, 0].item()
    elif y_tensor.numel() == 1:
        y_val = y_tensor.item()
    else:
        y_val = y_tensor[0].item()

    if torch.isnan(torch.tensor(y_val)):
        data.label_text = "Unknown (NaN)"
        data.label_description = f"Task: {task_name}. Label: NaN"
        return data

    y = int(y_val)

    if y in task_info["label_desc"]:
        data.label_text = task_info["label_desc"][y]
        data.label_description = (
            f"Task: {task_name}. "
            f"Meaning: {task_info['description']}. "
            f"Label = {y} ({task_info['label_desc'][y]})."
        )
    else:
        data.label_description = f"Task: {task_name}. Label = {y} (Description not found)"
    return data

class MoleculeDataset(InMemoryDataset):
    r"""The `MoleculeNet <http://moleculenet.org/datasets-1>`_ benchmark
    collection  from the `"MoleculeNet: A Benchmark for Molecular Machine
    Learning" <https://arxiv.org/abs/1703.00564>`_ paper, containing datasets
    from physical chemistry, biophysics and physiology.
    All datasets come with the additional node and edge features introduced by
    the :ogb:`null`
    `Open Graph Benchmark <https://ogb.stanford.edu/docs/graphprop/>`_.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (:obj:`"MUV"`, :obj:`"BBBP"`, 
            :obj:`"SIDER"`, :obj:`"ClinTox"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
        from_smiles (callable, optional): A custom function that takes a SMILES
            string and outputs a :obj:`~torch_geometric.data.Data` object.
            If not set, defaults to :meth:`~torch_geometric.utils.from_smiles`.
            (default: :obj:`None`)

    **STATS:**

    .. list-table::
        :widths: 20 10 10 10 10 10
        :header-rows: 1

        * - Name
          - #graphs
          - #nodes
          - #edges
          - #features
          - #classes
        * - MUV
          - 93,087
          - ~24.2
          - ~52.6
          - 9
          - 17
        * - BBBP
          - 2,050
          - ~23.9
          - ~51.6
          - 9
          - 1
        * - SIDER
          - 1,427
          - ~33.6
          - ~70.7
          - 9
          - 27
        * - ClinTox
          - 1,484
          - ~26.1
          - ~55.5
          - 9
          - 2
    """

    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/{}'
    mutag_url = 'https://github.com/divelab/DIG_storage/raw/main/xgraph/datasets/{}'

    names: Dict[str, Tuple[str, str, str, int, Union[int, slice]]] = {
        'bbbp': ('BBBP', 'BBBP.csv', 'BBBP', -1, -2),                               # classification, binary, 血脑屏障可渗透性
        'muv': ('MUV', 'muv.csv.gz', 'muv', -1, slice(0, 17)),                      # classification, multi(17), 非常稀疏的生物活性标签
        'sider': ('SIDER', 'sider.csv.gz', 'sider', 0, slice(1, 28)),               # classification, multi(27), 不良药物反应
        'clintox': ('ClinTox', 'clintox.csv.gz', 'clintox', 0, slice(1, 3)),        # classification, multi(2), 临床毒性预测
    }

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
        use_ogb_features: bool = True,
        add_atom_text: bool = True,
    ) -> None:
        self.name = name.lower()
        assert self.name in self.names.keys(), f"Unknown dataset name: {name}"
        self.use_ogb_features = use_ogb_features
        self.add_atom_text = add_atom_text

        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        self.dataset_desc = MOLECULENET_DESC[self.name]
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        return f'{self.names[self.name][2]}.csv'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        url = self.url.format(self.names[self.name][1])
        path = download_url(url, self.raw_dir)
        if self.names[self.name][1][-2:] == 'gz':
            extract_gz(path, self.raw_dir)
            os.unlink(path)

    def process(self) -> None:
        if self.name.lower() == 'MUTAG'.lower():
            with open(os.path.join(self.raw_dir, 'MUTAG_node_labels.txt'), 'r') as f:
                nodes_all_temp = f.read().splitlines()
                nodes_all = [int(i) for i in nodes_all_temp]

            adj_all = np.zeros((len(nodes_all), len(nodes_all)))
            with open(os.path.join(self.raw_dir, 'MUTAG_A.txt'), 'r') as f:
                adj_list = f.read().splitlines()
            for item in adj_list:
                lr = item.split(', ')
                l = int(lr[0])
                r = int(lr[1])
                adj_all[l - 1, r - 1] = 1

            with open(os.path.join(self.raw_dir, 'MUTAG_graph_indicator.txt'), 'r') as f:
                graph_indicator_temp = f.read().splitlines()
                graph_indicator = [int(i) for i in graph_indicator_temp]
                graph_indicator = np.array(graph_indicator)

            with open(os.path.join(self.raw_dir, 'MUTAG_graph_labels.txt'), 'r') as f:
                graph_labels_temp = f.read().splitlines()
                graph_labels = [int(i) for i in graph_labels_temp]

            data_list = []
            for i in range(1, 189):
                idx = np.where(graph_indicator == i)
                graph_len = len(idx[0])
                adj = adj_all[idx[0][0]:idx[0][0] + graph_len, idx[0][0]:idx[0][0] + graph_len]
                label = int(graph_labels[i - 1] == 1)
                feature = nodes_all[idx[0][0]:idx[0][0] + graph_len]
                nb_clss = 7
                targets = np.array(feature).reshape(-1)
                one_hot_feature = np.eye(nb_clss)[targets]
                from torch_geometric.data import Data
                data_example = Data(
                    x=torch.from_numpy(one_hot_feature).float(),
                    edge_index=dense_to_sparse(torch.from_numpy(adj))[0],
                    y=label
                )
                data_list.append(data_example)
        else:
            with open(self.raw_paths[0]) as f:
                dataset = f.read().split('\n')[1:-1]
                dataset = [x for x in dataset if len(x) > 0]  # Filter empty lines.

            data_list = []
            for line in dataset:
                line = re.sub(r'\".*\"', '', line)  # Replace ".*" strings.
                values = line.split(',')

                smiles = values[self.names[self.name][3]]
                labels = values[self.names[self.name][4]]
                labels = labels if isinstance(labels, list) else [labels]

                ys = [float(y) if len(y) > 0 else float('NaN') for y in labels]
                y = torch.tensor(ys, dtype=torch.float).view(1, -1)

                if self.use_ogb_features:
                    try:
                        # Use OGB features
                        graph = smiles2graph(smiles)
                        from torch_geometric.data import Data
                        data = Data(
                            x=torch.from_numpy(graph['node_feat']).to(torch.long),
                            edge_index=torch.from_numpy(graph['edge_index']).to(torch.long),
                            edge_attr=torch.from_numpy(graph['edge_feat']).to(torch.long),
                            num_nodes=int(graph['num_nodes']),
                            y=y,
                            smiles=smiles
                        )

                        if self.add_atom_text:
                            mol = Chem.MolFromSmiles(smiles)
                            if mol is not None:
                                atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
                                if len(atoms) == data.num_nodes:
                                    data.raw_texts = atoms
                                else:
                                    # Mismatch in node count between OGB and RDKit
                                    raise ValueError(f"Node count mismatch: OGB={data.num_nodes}, RDKit={len(atoms)}")
                            else:
                                # RDKit failed to parse SMILES
                                raise ValueError("RDKit failed to parse SMILES")

                        data.dataset_desc = MOLECULENET_DESC[self.name]

                    except Exception as e:
                        print(f"Warning: Failed to process SMILES {smiles} with OGB: {e}")
                        continue
                else:
                    # Use default PyG features
                    data = _from_smiles(smiles)
                    data.y = y
                    data.smiles = smiles

                if data.num_nodes == 0:
                    continue

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

            self.save(data_list, self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.names[self.name][0]}({len(self)})'


def get_dataset(dataset_name, data_dir, data_seed, sparse=True, cleaned=False):
    name = dataset_name
    set_random_seed(data_seed)

    if name in ["Graph-SST2", "Graph-SST5", "Graph-Twitter"]:
        datadir = data_dir
        dataset = load_SeniGraph(datadir, name)
        num_classes = dataset.num_classes
        num_features = dataset.num_features
        n = len(dataset)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        n_test = n - n_train - n_val
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(data_seed)
        )
        dataset = [train_dataset, val_dataset, test_dataset]
        return dataset, num_classes, num_features

    elif name in ['BBBP', 'ClinTox', 'SIDER', 'MUV']:
        dataset = MoleculeDataset(data_dir, name=name, use_ogb_features=True, add_atom_text=True)
        num_features = dataset[0].x.size(-1)
        y0 = dataset[0].y
        if y0.dim() == 1:
            num_classes = 1
        else:
            num_classes = y0.size(-1)

        data0 = dataset[0]
        mol = Chem.MolFromSmiles(data0.smiles)
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        print(atoms)

        n = len(dataset)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        n_test = n - n_train - n_val
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(data_seed)
        )
        dataset = [train_dataset, val_dataset, test_dataset]
        return dataset, num_classes, num_features

    else:
        raise ValueError(f"Unknown dataset name: {name}")

def get_dataloader(dataset_name, dataset, batch_size, args):
    train_dataset, val_dataset, test_dataset = dataset

    dataloader = dict()
    dataloader['train'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dataloader['val'] = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    dataloader['test'] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return dataloader


if __name__ == "__main__":
    dataset_name = 'MUV'
    data_dir = data_root
    data_seed = 0
    dataset = get_dataset(dataset_name, data_dir, data_seed)
    data0 = dataset[0]
    print('-' * 80)
    print('First data object:')
    print(data0)

