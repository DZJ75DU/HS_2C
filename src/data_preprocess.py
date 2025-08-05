import argparse
import os
import re
from datetime import datetime
import igraph as ig
import numpy as np
import leidenalg
import networkx as nx
import gc
import random
import copy
import json
import faiss
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import defaultdict
from torch_cluster import knn_graph
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_dense_adj
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from coding_tree_build import PartitionTree
from logger import Logger
from utils import *


BACKGROUND_INFO_DEFAULT_TOKEN = '<BACKGROUND_NODE_INFO>'
TARGET_INFO_DEFAULT_TOKEN = '<TARGET_NODE_INFO>'
DEFAULT_G_NEIGHBOR_INFO_TOKEN = "<NEIGHBOR_INFO>"
DEFAULT_SUMMARY_CORPUS_TOKEN = "<SUMMARY_CORPUS>"
DEFAULT_CATEGORY_TOKEN = "<category_list>"
DEFAULT_DATASET_NAME_TOKEN = "<DATASET_TOKEN>"
DEFAULT_GRAPH_TYPE_TOKEN = "<GRAPH_TYPE_TOKEN>"
DEFAULT_NODE_TYPE_TOKEN = "<NODE_TYPE_TOKEN>"
DEFAULT_CENTER_RAW_TEXT_TOKEN = "<CENTER_RAW_TEXT>"
DEFAULT_CONDENSATION_FAILURE_TOKEN = "<CONDENSATION_FAILURE>"

label_whol_dict = {
    'ogb_arxiv': {
        0: "Numerical Analysis",
        1: "Multimedia",
        2: "Logic in Computer Science",
        3: "Computers and Society",
        4: "Cryptography and Security",
        5: "Distributed, Parallel, and Cluster Computing",
        6: "Human-Computer Interaction",
        7: "Computational Engineering, Finance, and Science",
        8: "Networking and Internet Architecture",
        9: "Computational Complexity",
        10: "Artificial Intelligence",
        11: "Multiagent Systems",
        12: "General Literature",
        13: "Neural and Evolutionary Computing",
        14: "Symbolic Computation",
        15: "Hardware Architecture",
        16: "Computer Vision and Pattern Recognition",
        17: "Graphics",
        18: "Emerging Technologies",
        19: "Systems and Control",
        20: "Computational Geometry",
        21: "Other Computer Science",
        22: "Programming Languages",
        23: "Software Engineering",
        24: "Machine Learning",
        25: "Sound",
        26: "Social and Information Networks",
        27: "Robotics",
        28: "Information Theory",
        29: "Performance",
        30: "Computation and Language",
        31: "Information Retrieval",
        32: "Mathematical Software",
        33: "Formal Languages and Automata Theory",
        34: "Data Structures and Algorithms",
        35: "Operating Systems",
        36: "Computer Science and Game Theory",
        37: "Databases",
        38: "Digital Libraries",
        39: "Discrete Mathematics",
    },
    'arxiv_2023': {
        0: "Numerical Analysis",
        1: "Multimedia",
        2: "Logic in Computer Science",
        3: "Computers and Society",
        4: "Cryptography and Security",
        5: "Distributed, Parallel, and Cluster Computing",
        6: "Human-Computer Interaction",
        7: "Computational Engineering, Finance, and Science",
        8: "Networking and Internet Architecture",
        9: "Computational Complexity",
        10: "Artificial Intelligence",
        11: "Multiagent Systems",
        12: "General Literature",
        13: "Neural and Evolutionary Computing",
        14: "Symbolic Computation",
        15: "Hardware Architecture",
        16: "Computer Vision and Pattern Recognition",
        17: "Graphics",
        18: "Emerging Technologies",
        19: "Systems and Control",
        20: "Computational Geometry",
        21: "Other Computer Science",
        22: "Programming Languages",
        23: "Software Engineering",
        24: "Machine Learning",
        25: "Sound",
        26: "Social and Information Networks",
        27: "Robotics",
        28: "Information Theory",
        29: "Performance",
        30: "Computation and Language",
        31: "Information Retrieval",
        32: "Mathematical Software",
        33: "Formal Languages and Automata Theory",
        34: "Data Structures and Algorithms",
        35: "Operating Systems",
        36: "Computer Science and Game Theory",
        37: "Databases",
        38: "Digital Libraries",
        39: "Discrete Mathematics",
    },
    'products': {
        0: "Home & Kitchen",
        1: "Health & Personal Care",
        2: "Beauty",
        3: "Sports & Outdoors",
        4: "Books",
        5: "Patio, Lawn & Garden",
        6: "Toys & Games",
        7: "CDs & Vinyl",
        8: "Cell Phones & Accessories",
        9: "Grocery & Gourmet Food",
        10: "Arts, Crafts & Sewing",
        11: "Clothing, Shoes & Jewelry",
        12: "Electronics",
        13: "Movies & TV",
        14: "Software",
        15: "Video Games",
        16: "Automotive",
        17: "Pet Supplies",
        18: "Office Products",
        19: "Industrial & Scientific",
        20: "Musical Instruments",
        21: "Tools & Home Improvement",
        23: "Baby Products",
        24: "label 25",
        25: "Appliances",
        26: "Kitchen & Dining",
        27: "Collectibles & Fine Art",
        28: "All Beauty",
        30: "Amazon Fashion",
        31: "Computers",
        32: "All Electronics",
        34: "MP3 Players & Accessories",
        36: "Office & School Supplies",
        37: "Home Improvement",
        38: "Camera & Photo",
        39: "GPS & Navigation",
        40: "Digital Music",
        41: "Car Electronics",
        42: "Baby",
    },
    'instagram': {
        0: "Normal Users",
        1: "Commercial Users",
    },
    'citeseer': {
        0: "Agents",
        1: "ML (Machine Learning)",
        2: "IR (Information Retrieval)",
        3: "DB (Databases)",
        4: "HCI (Human-Computer Interaction)",
        5: "AI (Artificial Intelligence)",
    },
    'reddit': {
        0: "Normal Users",
        1: "Popular Users",
    }
}

label_abbr_dict = {
    'ogb_arxiv': {
        0: "cs.NA",
        1: "cs.MM",
        2: "cs.LO",
        3: "cs.CY",
        4: "cs.CR",
        5: "cs.DC",
        6: "cs.HC",
        7: "cs.CE",
        8: "cs.NI",
        9: "cs.CC",
        10: "cs.AI",
        11: "cs.MA",
        12: "cs.GL",
        13: "cs.NE",
        14: "cs.SC",
        15: "cs.AR",
        16: "cs.CV",
        17: "cs.GR",
        18: "cs.ET",
        19: "cs.SY",
        20: "cs.CG",
        21: "cs.OH",
        22: "cs.PL",
        23: "cs.SE",
        24: "cs.LG",
        25: "cs.SD",
        26: "cs.SI",
        27: "cs.RO",
        28: "cs.IT",
        29: "cs.PF",
        30: "cs.CL",
        31: "cs.IR",
        32: "cs.MS",
        33: "cs.FL",
        34: "cs.DS",
        35: "cs.OS",
        36: "cs.GT",
        37: "cs.DB",
        38: "cs.DL",
        39: "cs.DM",
    },
    'arxiv_2023': {
        0: "cs.NA",
        1: "cs.MM",
        2: "cs.LO",
        3: "cs.CY",
        4: "cs.CR",
        5: "cs.DC",
        6: "cs.HC",
        7: "cs.CE",
        8: "cs.NI",
        9: "cs.CC",
        10: "cs.AI",
        11: "cs.MA",
        12: "cs.GL",
        13: "cs.NE",
        14: "cs.SC",
        15: "cs.AR",
        16: "cs.CV",
        17: "cs.GR",
        18: "cs.ET",
        19: "cs.SY",
        20: "cs.CG",
        21: "cs.OH",
        22: "cs.PL",
        23: "cs.SE",
        24: "cs.LG",
        25: "cs.SD",
        26: "cs.SI",
        27: "cs.RO",
        28: "cs.IT",
        29: "cs.PF",
        30: "cs.CL",
        31: "cs.IR",
        32: "cs.MS",
        33: "cs.FL",
        34: "cs.DS",
        35: "cs.OS",
        36: "cs.GT",
        37: "cs.DB",
        38: "cs.DL",
        39: "cs.DM",
    },
    'products': {},
    'instagram': {},
    'citeseer': {},
    'reddit': {}
}

catogory_arxiv = """ Artificial Intelligence; Hardware Architecture; Computational Complexity; Computational Engineering, Finance, and Science; Computational Geometry; Computation and Language; Cryptography and Security; Computer Vision and Pattern Recognition; Computers and Society; Databases; Distributed, Parallel, and Cluster Computing; Digital Libraries; Discrete Mathematics; Data Structures and Algorithms; Emerging Technologies; Formal Languages and Automata Theory; General Literature; Graphics; Computer Science and Game Theory; Human-Computer Interaction; Information Retrieval; Information Theory; Machine Learning; Logic in Computer Science; Multiagent Systems; Multimedia; Mathematical Software; Numerical Analysis; Neural and Evolutionary Computing; Networking and Internet Architecture; Other Computer Science; Operating Systems; Performance; Programming Languages; Robotics; Symbolic Computation; Sound; Software Engineering; Social and Information Networks; Systems and Control """
category_cora = """ Rule Learning; Neural Networks; Case-Based; Genetic Algorithms; Theory; Reinforcement Learning; Probabilistic Methods """
category_pubmed = """ Diabetes Mellitus, Experimental; Diabetes Mellitus Type 1; Diabetes Mellitus Type 2 """
category_instagram = """ Normal Users; Commercial Users """
category_citeseer = """ IR (Information Retrieval); HCI (Human-Computer Interaction); Agents; AI (Artificial Intelligence); ML (Machine Learning); DB (Databases) """
category_reddit = """ Normal Users; Popular Users """
category_wikics = """ Computational Linguistics; Databases; Operating Systems; Computer Architecture; Computer Security; Internet Protocols; Computer File Systems; Distributed Computing Architecture; Web Technology; Programming Language Topics """
category_products = """ Home & Kitchen; Health & Personal Care; Beauty; Sports & Outdoors; Books; Patio, Lawn & Garden; Toys & Games; CDs & Vinyl; Cell Phones & Accessories; Grocery & Gourmet Food; Arts, Crafts & Sewing; Clothing, Shoes & Jewelry; Electronics; Movies & TV; Software; Video Games; Automotive; Pet Supplies; Office Products; Industrial & Scientific; Musical Instruments; Tools & Home Improvement; Magazine Subscriptions; Baby Products; Appliances; Kitchen & Dining; Collectibles & Fine Art; All Beauty; Luxury Beauty; Amazon Fashion; Computers; All Electronics; Purchase Circles; MP3 Players & Accessories; Gift Cards; Office & School Supplies; Home Improvement; Camera & Photo; GPS & Navigation; Digital Music; Car Electronics; Baby; Kindle Store; Buy a Kindle; Furniture & decoration """


class graph_dataset(Dataset):
    def __init__(self, args, graph_name='ogb_arxiv', tokenizer=None):
        super(graph_dataset, self).__init__()
        self.args = args
        if torch.cuda.is_available():
            self.args.device = torch.device(f'cuda')
        else:
            self.args.device = torch.device('cpu')

        self.tokenizer = AutoTokenizer.from_pretrained(f"{self.args.model_path}/{self.args.model_name}")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # 使用eos_token作为pad_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # 设置pad_token_id为eos_token_id
        self.tokenizer.padding_side = "right"
        self.model = AutoModelForCausalLM.from_pretrained(
            f"{self.args.model_path}/{self.args.model_name}",
            torch_dtype=torch.float16,
        ).to(self.args.device)

        if self.args.dataset in ['ogb_arxiv']:
            self.category_list = catogory_arxiv
            self.dataset_name_in_prompt = "ArXiv CS"
            self.graph_type = "academic citation"
            self.node_type = "paper"
        elif self.args.dataset in ['arxiv_2023']:
            self.category_list = catogory_arxiv
            self.dataset_name_in_prompt = "ArXiv CS"
            self.graph_type = "academic citation"
            self.node_type = "paper"
        elif self.args.dataset in ['instagram']:
            self.category_list = category_instagram
            self.dataset_name_in_prompt = "Instagram"
            self.graph_type = "following"
            self.node_type = "user"
        elif self.args.dataset in ['citeseer']:
            self.category_list = category_citeseer
            self.dataset_name_in_prompt = "CiteSeer"
            self.graph_type = "academic citation"
            self.node_type = "paper"
        elif self.args.dataset in ['reddit']:
            self.category_list = category_reddit
            self.dataset_name_in_prompt = "Reddit"
            self.graph_type = "following"
            self.node_type = "user"
        elif self.args.dataset in ['products']:
            self.category_list = category_products
            self.dataset_name_in_prompt = "Amazon Product"
            self.graph_type = "co-purchase"
            self.node_type = "product"
        else:
            raise NotImplementedError

        self.Specific_template = '''You are given a single target <NODE_TYPE_TOKEN> and a group of background <NODE_TYPE_TOKEN>s from the same community within a <GRAPH_TYPE_TOKEN> graph. Each <NODE_TYPE_TOKEN> is represented by its title and abstract. The background <NODE_TYPE_TOKEN>s, which are connected to the target <NODE_TYPE_TOKEN>, may share a common semantic focus or research topic with the target <NODE_TYPE_TOKEN>.\n\nTarget <NODE_TYPE_TOKEN>:\n<TARGET_NODE_INFO>\n\nBackground <NODE_TYPE_TOKEN>s:\n<BACKGROUND_NODE_INFO>\n\n\nTask: Identify and summarize the content of the background <NODE_TYPE_TOKEN>s that are semantically aligned with the target <NODE_TYPE_TOKEN>, focusing on shared research disciplinary, domains, directions, or topics. Ignore background <NODE_TYPE_TOKEN>s that are not semantically aligned with the target <NODE_TYPE_TOKEN>. Avoid summarizing each <NODE_TYPE_TOKEN> individually.\n\nPlease present the semantically aligned content in 3 to 4 well-structured sentences (no more than 5), written in a cohesive and formal academic style, without headings, task instructions, author names, <GRAPH_TYPE_TOKEN>s, or meta-commentary.\n\nAnswer:'''
        self.Common_Shared_template = '''You are given a group of target <NODE_TYPE_TOKEN>s and a single background <NODE_TYPE_TOKEN> from the same community within a <GRAPH_TYPE_TOKEN> graph. Each <NODE_TYPE_TOKEN> is represented by its title and abstract. The background <NODE_TYPE_TOKEN>, which is connected to the target <NODE_TYPE_TOKEN>s, may share a common semantic focus or research topic with the target <NODE_TYPE_TOKEN>s.\n\nTarget <NODE_TYPE_TOKEN>s:\n<TARGET_NODE_INFO>\n\nBackground <NODE_TYPE_TOKEN>:\n<BACKGROUND_NODE_INFO>\n\n\nTask: Identify and summarize the content of the background <NODE_TYPE_TOKEN> that is semantically aligned with the group of target <NODE_TYPE_TOKEN>s, focusing on shared research disciplinary, domains, directions, or topics that make the background <NODE_TYPE_TOKEN> meaningfully connected to the different target <NODE_TYPE_TOKEN>s. Avoid summarizing each <NODE_TYPE_TOKEN> individually.\n\nPlease present the semantically aligned content in 3 to 4 well-structured sentences (no more than 5), written in a cohesive and formal academic style, without headings, task instructions, author names, <GRAPH_TYPE_TOKEN>s, or meta-commentary.\n\nAnswer:'''
        self.Mixed_Community_template = '''You are given a set of target <NODE_TYPE_TOKEN>s and a group of background <NODE_TYPE_TOKEN>s from the same community within a <GRAPH_TYPE_TOKEN> graph. Each <NODE_TYPE_TOKEN> is represented by its title and abstract. The background <NODE_TYPE_TOKEN>s, which are connected to the target <NODE_TYPE_TOKEN>s, may share a common semantic focus or research topic with the target <NODE_TYPE_TOKEN>s.\n\nTarget <NODE_TYPE_TOKEN>s:\n<TARGET_NODE_INFO>\n\nBackground <NODE_TYPE_TOKEN>s:\n<BACKGROUND_NODE_INFO>\n\n\nTask: Identify and summarize the content of the background <NODE_TYPE_TOKEN>s that are semantically aligned with the group of target <NODE_TYPE_TOKEN>s, focusing on shared research disciplinary, domains, directions, or topics. Ignore background <NODE_TYPE_TOKEN>s that are not semantically aligned with the target <NODE_TYPE_TOKEN>s. Avoid summarizing each <NODE_TYPE_TOKEN> individually.\n\nPlease present the semantically aligned content in 3 to 4 well-structured sentences (no more than 5), written in a cohesive and formal academic style, without headings, task instructions, author names, <GRAPH_TYPE_TOKEN>s, or meta-commentary.\n\nAnswer:'''
        self.summary_template = """The following list records <NODE_TYPE_TOKEN>s related to the current one, with the relationship being '<GRAPH_TYPE_TOKEN>': <SUMMARY_CORPUS>.\nPlease summarize the information above with a short paragraph, and find some common points that can reflect the category of the <NODE_TYPE_TOKEN>."""
        DEFAULT_GRAPH_TYPE_TOKEN = "<GRAPH_TYPE_TOKEN>"
        DEFAULT_NODE_TYPE_TOKEN = "<NODE_TYPE_TOKEN>"
        self.Specific_template = self.Specific_template.replace(DEFAULT_GRAPH_TYPE_TOKEN, self.graph_type)
        self.Specific_template = self.Specific_template.replace(DEFAULT_NODE_TYPE_TOKEN, self.node_type)
        self.Common_Shared_template = self.Common_Shared_template.replace(DEFAULT_GRAPH_TYPE_TOKEN, self.graph_type)
        self.Common_Shared_template = self.Common_Shared_template.replace(DEFAULT_NODE_TYPE_TOKEN, self.node_type)
        self.Mixed_Community_template = self.Mixed_Community_template.replace(DEFAULT_GRAPH_TYPE_TOKEN, self.graph_type)
        self.Mixed_Community_template = self.Mixed_Community_template.replace(DEFAULT_NODE_TYPE_TOKEN, self.node_type)
        self.summary_template = self.summary_template.replace(DEFAULT_GRAPH_TYPE_TOKEN, self.graph_type)
        self.summary_template = self.summary_template.replace(DEFAULT_NODE_TYPE_TOKEN, self.node_type)

        template_1 = """\nGiven a <GRAPH_TYPE_TOKEN> graph, the 0th node (target <NODE_TYPE_TOKEN>) has the following information:\n<CENTER_RAW_TEXT>\n\n\nQuestion: Based on the features of the target <NODE_TYPE_TOKEN>, please determine the most appropriate <DATASET_TOKEN> sub-category for the target <NODE_TYPE_TOKEN>.\n\nCategories: <category_list>.\n\n\nPlease think about the categorization in a step by step manner, and only output the single most relevant category of the target <NODE_TYPE_TOKEN>. Do not give any reasoning or extra text for your answer.\n\nAnswer:"""
        graph_tree_template = """\nGiven a <GRAPH_TYPE_TOKEN> graph, the 0th node (target <NODE_TYPE_TOKEN>) has the following information:\n<CENTER_RAW_TEXT>\n\nThe target <NODE_TYPE_TOKEN> is connected to the following <NODE_TYPE_TOKEN>s:\n<NEIGHBOR_INFO>\n\n\nQuestion: Based on the features of the target <NODE_TYPE_TOKEN> and its <GRAPH_TYPE_TOKEN> network, please determine the most appropriate <DATASET_TOKEN> sub-category for the target <NODE_TYPE_TOKEN>.\n\nCategories: <category_list>.\n\n\nPlease think about the categorization of the target <NODE_TYPE_TOKEN> in a structured manner, and only output the single most relevant category of the target <NODE_TYPE_TOKEN>. Do not give any reasoning or extra text for your answer.\n\nAnswer:"""
        template_1 = template_1.replace(DEFAULT_DATASET_NAME_TOKEN, self.dataset_name_in_prompt)
        template_1 = template_1.replace(DEFAULT_GRAPH_TYPE_TOKEN, self.graph_type)
        template_1 = template_1.replace(DEFAULT_NODE_TYPE_TOKEN, self.node_type)
        graph_tree_template = graph_tree_template.replace(DEFAULT_DATASET_NAME_TOKEN, self.dataset_name_in_prompt)
        graph_tree_template = graph_tree_template.replace(DEFAULT_GRAPH_TYPE_TOKEN, self.graph_type)
        graph_tree_template = graph_tree_template.replace(DEFAULT_NODE_TYPE_TOKEN, self.node_type)
        self.use_raw = True
        self.center_prompt = template_1
        self.graph_tree_prompt = graph_tree_template

        self.load_metadata()
        self.data_reconstruction_process()

        self.re_id = self.re_id_construction()
        self.match_ori_node_idx()
        self.rag_neighbor_prompt()

    def load_metadata(self):
        start_time = datetime.now()
        print(f"INFO: preprocessing graph {self.args.dataset} ...")
        self.experiment_name = f'dataset/{self.args.dataset}'
        os.makedirs(f'{self.experiment_name}', exist_ok=True)
        self.ori_dataset = torch.load(f'dataset/{self.args.dataset}/{self.args.dataset}_data.pt')
        all_node_idx = torch.arange(self.ori_dataset.x.shape[0])
        self.sample_node_idx = all_node_idx[self.ori_dataset.test_mask]
        end_time = datetime.now()
        print(end_time - start_time)
        print("INFO: Loading metadata files done...")

    def find_target(self):
        subset, sub_edge_index, _, _ = k_hop_subgraph(
            node_idx=self.sample_node_idx,
            num_hops=1,
            edge_index=self.ori_dataset.edge_index,
            relabel_nodes=True
        )
        sub_x = self.ori_dataset.x[subset]
        sub_y = self.ori_dataset.y[subset]
        sub_test_mask = self.ori_dataset.test_mask[subset]
        sub_node_idx_list = subset.tolist()
        if len(self.ori_dataset.label_abbr) != 0:
            sub_label_abbr = [self.ori_dataset.label_abbr[i] for i in sub_node_idx_list]
        else:
            sub_label_abbr = []
        if len(self.ori_dataset.label_whol) != 0:
            sub_label_whol = [self.ori_dataset.label_whol[i] for i in sub_node_idx_list]
        else:
            sub_label_whol = []
        if len(self.ori_dataset.raw_texts) != 0:
            sub_raw_texts = [self.ori_dataset.raw_texts[i] for i in sub_node_idx_list]
        else:
            sub_raw_texts = []
        target_data = Data(x=sub_x, edge_index=sub_edge_index, y=sub_y, test_mask=sub_test_mask, label_abbr=sub_label_abbr, label_whol=sub_label_whol, raw_texts=sub_raw_texts, ori_node_idx=subset)
        return target_data

    def to_nx(self, data):
        set_random_seed(self.args.project_seed)
        edge_index = data.knn_edge_index
        if edge_index.shape[0] == 2:
            edge_index = edge_index.t()
        G = nx.Graph()
        G.add_nodes_from(range(data.num_nodes))
        edges = list(map(tuple, edge_index.tolist()))
        G.add_edges_from(edges)
        return G

    def get_node_embedding(self, dataset, batch_size=10):
        bert_model = AutoModel.from_pretrained(f"{self.args.model_path}/Bert-base-uncased").to(self.args.device)
        bert_tokenizer = AutoTokenizer.from_pretrained(f"{self.args.model_path}/Bert-base-uncased")
        bert_model.eval()

        sub_raw_texts = dataset.raw_texts
        node_embeddings = []

        for i in tqdm(range(0, len(sub_raw_texts), batch_size)):
            batch_texts = sub_raw_texts[i:i + batch_size]
            inputs = bert_tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(self.args.device)

            with torch.no_grad():
                outputs = bert_model(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]  # 取 CLS 向量
                node_embeddings.append(cls_embeddings.cpu())
        node_embeddings = torch.cat(node_embeddings, dim=0)  # shape: [N, 768]

        return node_embeddings
    
    def add_knn_pyg(self, k, node_embed, edge_index, device):
        num_nodes = node_embed.shape[0]
        connected_nodes = torch.unique(edge_index)
        all_nodes = torch.arange(num_nodes, device=self.args.device)
        iso_nodes = torch.tensor(
            list(set(all_nodes.tolist()) - set(connected_nodes.tolist()))
        ).to(self.args.device)
        if len(iso_nodes) == 0:
            return edge_index.t() if edge_index.shape[0] != 2 else edge_index
        node_embed = node_embed.to(self.args.device)
        edge_index = edge_index.to(self.args.device)
        knn_edge_index = knn_graph(node_embed, k=k+1, loop=False, cosine=True)
        mask = (torch.isin(knn_edge_index[0], iso_nodes) | torch.isin(knn_edge_index[1], iso_nodes))
        knn_edge_index_filtered = knn_edge_index[:, mask]  # shape: [2, E_filtered]
        knn_edge_index_filtered = torch.cat(
            [knn_edge_index_filtered, knn_edge_index_filtered[[1, 0], :]],
            dim=1
        )
        if edge_index.shape[0] == 2:
            edge_index = edge_index
        else:
            edge_index = edge_index.t()
        edge_index_combined = torch.cat([edge_index, knn_edge_index_filtered], dim=1)  # shape: [2, E]
        edge_index_combined = torch.unique(edge_index_combined, dim=1)
        return edge_index_combined  # [2, E]

    def calc_e1(self, adj: torch.Tensor):
        adj = adj - torch.diag_embed(torch.diag(adj))
        degree = adj.sum(dim=1)
        vol = adj.sum()
        idx = degree.nonzero().reshape(-1)
        g = degree[idx]
        return -((g / vol) * torch.log2(g / vol)).sum()

    def get_adj_matrix(self, node_num, edge_index, weight) -> torch.Tensor:
        adj_matrix = torch.zeros((node_num, node_num))
        adj_matrix[edge_index.t()[0], edge_index.t()[1]] = weight
        adj_matrix = adj_matrix - torch.diag_embed(torch.diag(adj_matrix))  # 去除对角线
        return adj_matrix

    def get_weight(self, node_embedding, edge_index):
        node_num = node_embedding.shape[0]
        edge_index = edge_index.to(node_embedding.device)
        links = node_embedding[edge_index]
        weight = []
        vec1 = links[:, 0, :]
        vec2 = links[:, 1, :]
        weight = F.cosine_similarity(vec1, vec2, dim=1)  # shape: [E]
        weight = weight.clone().detach() + 1
        weight[torch.isnan(weight)] = 0
        M = weight.mean() / (2 * node_num)
        weight = weight + M
        return weight

    def knn_maxE1(self, edge_index: torch.Tensor, node_embedding: torch.Tensor):
        old_e1 = 0
        node_num = node_embedding.shape[0]
        k = 1
        while k < 50:
            edge_index_k = self.add_knn_pyg(k, node_embedding, edge_index, self.args.device)
            weight = self.get_weight(node_embedding, edge_index_k)
            adj = self.get_adj_matrix(node_num, edge_index_k, weight)
            e1 = self.calc_e1(adj)
            if e1 - old_e1 > 0.1:
                k += 5
            elif e1 - old_e1 > 0.01:
                k += 3
            elif e1 - old_e1 > 0.001:
                k += 1
            else:
                break
            old_e1 = e1
        return k

    def get_knn_target_graph(self):
        target_data = self.find_target()
        node_embeddings = self.get_node_embedding(target_data, batch_size=self.args.node_embedding_batch_size)
        best_k = self.knn_maxE1(target_data.edge_index, node_embeddings)
        best_edge_index = self.add_knn_pyg(best_k, node_embeddings, target_data.edge_index, self.args.device)
        target_data.knn_edge_index = best_edge_index
        return target_data

    def packet_process(self, data):
        set_random_seed(self.args.project_seed)
        G_nx = self.to_nx(data)
        G_ig = ig.Graph(edges=list(G_nx.edges()), directed=False)
        G_ig.vs["name"] = list(range(data.num_nodes))
        if self.args.dataset in ['reddit']:
            self.resolution = 1.2
        else:
            self.resolution = 1.0
        part = leidenalg.find_partition(G_ig, leidenalg.RBConfigurationVertexPartition, resolution_parameter=self.resolution,seed=self.args.project_seed)
        return {i: list(comm) for i, comm in enumerate(part)}

    def extract_cluster(self, edge_index, cluster_nodes):
        cluster_set = set(cluster_nodes.tolist())
        src, dst = edge_index
        mask = [(s.item() in cluster_set and d.item() in cluster_set) for s, d in zip(src, dst)]
        mask = torch.tensor(mask, dtype=torch.bool)
        sub_edge_index = edge_index[:, mask]
        return sub_edge_index

    def condensation_llm_answer_generation(self, recon_data):
        set_random_seed(self.args.condensation_seed)
        condensation_mask = recon_data.reconstruct_condensation_mask
        raw_texts = recon_data.reconstruct_raw_texts
        new_raw_texts = []
        for num, query in tqdm(enumerate(raw_texts)):
            if not condensation_mask[num]:
                new_raw_texts.append(query)
                continue
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                tokens = self.tokenizer(
                    query,
                    padding=True,
                    return_tensors="pt",
                    truncation=True
                ).to(self.args.device)
                output = self.model.generate(
                    tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                    max_new_tokens=self.args.max_token,
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,
                    temperature=1.0,
                    top_k=50,
                    top_p=0.95,
                    do_sample=True
                )
                generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                answer = generated_text.split('\n\nAnswer:', 1)[-1].strip()
                new_raw_texts.append(answer)
            except torch.cuda.OutOfMemoryError:
                new_raw_texts.append(DEFAULT_CONDENSATION_FAILURE_TOKEN + query)
                for var in ['input_ids', 'output']:
                    if var in locals():
                        del locals()[var]
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.ipc_collect()
        recon_data.reconstruct_condense_raw_texts = new_raw_texts
        return recon_data
    
    def data_reconstruction_process(self):
        set_random_seed(self.args.project_seed)
        all_x, all_y, all_raw_text, all_target_mask = [], [], [], []
        all_query_texts, all_label_abbr, all_label_whol, all_ori_node_idx = [], [], [], []
        all_new_edge_index, node_idx_mapping = set(), {}
        next_start_new_idx = 0
        self.dataset = self.get_knn_target_graph()
        all_edge_index = self.dataset.knn_edge_index

        def build_new_dict(coding_tree, labels):
            new_dict = {}
            for k, v in coding_tree.tree_node.items():
                content = v.__dict__
                content['node_label'] = int(labels[k]) if k in coding_tree.leaves else -1
                new_dict[k] = content
            return new_dict

        def process_cluster(cluster_data, ori_node_idx, sample_mask):
            adj_matrix = to_dense_adj(cluster_data.edge_index).squeeze(0).numpy()
            coding_tree = PartitionTree(adj_matrix=adj_matrix)
            coding_tree.build_encoding_tree(k=2)
            new_dict = build_new_dict(coding_tree, cluster_data.y.numpy())
            l1_nodes = coding_tree.get_l1_community_nodes()

            _, type_node_dict, type_node_detail = self.log_coding_tree(new_dict, sample_mask, l1_nodes, None)
            cluster_data.l1_community_mapping = type_node_detail
            prompts, _ = self.build_condensation_prompt_from_type_node_dict(cluster_data, type_node_dict, new_dict)
            recon_data = self.reconstruct_condensation_data(prompts, cluster_data, type_node_detail)
            cond_data = self.condensation_llm_answer_generation(recon_data)
            return cond_data

        if self.args.dataset not in ['citeseer']:
            sample_node_idx = torch.arange(self.dataset.x.shape[0])[self.dataset.test_mask]
            clusters = self.packet_process(self.dataset)

            for cluster_id, cluster_node_idx in clusters.items():
                cluster_node_idx = torch.tensor(cluster_node_idx)
                node_id_map = {nid.item(): idx for idx, nid in enumerate(cluster_node_idx)}
                sample_mask = torch.tensor([nid.item() in sample_node_idx for nid in cluster_node_idx], dtype=torch.bool)
                raw_edge_index = self.extract_cluster(all_edge_index, cluster_node_idx)
                relabelled_edge_index = torch.tensor([
                    [node_id_map[s.item()] for s in raw_edge_index[0]],
                    [node_id_map[d.item()] for d in raw_edge_index[1]]
                ], dtype=torch.long)
                def safe_get(attr):
                    return [getattr(self.dataset, attr)[i] for i in cluster_node_idx.tolist()] if len(
                        getattr(self.dataset, attr)) else []
                cluster_data = Data(
                    x=self.dataset.x[cluster_node_idx],
                    edge_index=relabelled_edge_index,
                    y=self.dataset.y[cluster_node_idx],
                    label_abbr=safe_get('label_abbr'),
                    label_whol=safe_get('label_whol'),
                    raw_texts=safe_get('raw_texts'),
                    in_cluster_test_mask=sample_mask,
                )
                cond_data = process_cluster(cluster_data, cluster_node_idx, sample_mask)
                cluster_node_mapping = cond_data.community_cluster_node_mapping
                condensation_mapping = cond_data.reconstruct_condensation_node_mapping
                local_to_global = {int(local): int(global_id) for global_id, local in cluster_node_mapping.items()}
                for local_id, condensed_id in condensation_mapping.items():
                    if condensed_id is not None:
                        global_idx = local_to_global[local_id]
                        node_idx_mapping[global_idx] = condensed_id + next_start_new_idx
                valid_new_ids = [v for v in condensation_mapping.values() if v is not None]
                if not valid_new_ids:
                    continue
                next_start_new_idx += max(valid_new_ids) + 1
                all_x.append(cond_data.reconstruct_x)
                all_y.append(cond_data.reconstruct_y)
                all_raw_text.extend(cond_data.reconstruct_condense_raw_texts)
                all_target_mask.append(cond_data.reconstruct_target_mask)
                all_query_texts.extend(cond_data.reconstruct_raw_texts)
                all_label_abbr.extend(cond_data.reconstruct_all_label_abbr)
                all_label_whol.extend(cond_data.reconstruct_all_label_whol)
                all_ori_node_idx.extend(cond_data.reconstruct_ori_node_idx)

        else:
            all_node_idx = torch.arange(self.dataset.x.shape[0])
            cluster_data = Data(
                x=self.dataset.x,
                edge_index=all_edge_index,
                y=self.dataset.y,
                label_abbr=self.dataset.label_abbr,
                label_whol=self.dataset.label_whol,
                raw_texts=self.dataset.raw_texts,
                in_cluster_test_mask=self.dataset.test_mask
            )
            cond_data = process_cluster(cluster_data, all_node_idx, self.dataset.test_mask)
            cluster_node_mapping = {i: i for i in range(cond_data.community_cluster_x.shape[0])}
            condensation_mapping = cond_data.reconstruct_condensation_node_mapping
            local_to_global = {int(local): int(global_id) for global_id, local in cluster_node_mapping.items()}

            for local_id, condensed_id in condensation_mapping.items():
                if condensed_id is not None:
                    global_idx = local_to_global[local_id]
                    node_idx_mapping[global_idx] = condensed_id + next_start_new_idx
            next_start_new_idx += max(condensation_mapping.values()) + 1

            all_x.append(cond_data.reconstruct_x)
            all_y.append(cond_data.reconstruct_y)
            all_raw_text.extend(cond_data.reconstruct_condense_raw_texts)
            all_target_mask.append(cond_data.reconstruct_target_mask)
            all_label_abbr.extend(cond_data.reconstruct_all_label_abbr)
            all_label_whol.extend(cond_data.reconstruct_all_label_whol)

        for src, tgt in all_edge_index.t().tolist():
            if src in node_idx_mapping and tgt in node_idx_mapping:
                new_src, new_tgt = node_idx_mapping[src], node_idx_mapping[tgt]
                if new_src != new_tgt:
                    edge = (new_src, new_tgt)
                    reverse_edge = (new_tgt, new_src)
                    if edge not in all_new_edge_index and reverse_edge not in all_new_edge_index:
                        all_new_edge_index.add(edge)

        edge_index_tensor = torch.tensor(list(all_new_edge_index), dtype=torch.long).t().contiguous()
        x = torch.cat(all_x, dim=0)
        y = torch.cat(all_y, dim=0)
        condensation_mask = torch.cat(all_target_mask, dim=0)
        target_mask = ~condensation_mask
        assert x.shape[0] == y.shape[0] == condensation_mask.shape[0] == target_mask.shape[0] == len(all_raw_text)
        new_data = Data(x=x, edge_index=edge_index_tensor, y=y, raw_texts=all_raw_text, target_mask=condensation_mask, label_abbr=all_label_abbr, label_whol=all_label_whol)
        return new_data

    def count_L1_community_types(self, ordered_keys, new_dict, l1_community_node_idx, test_mask):
        type_counts = defaultdict(int)
        type_node_dict = {}
        type_node_dict_detail = {}
        for key in ordered_keys:
            type_node_dict[key] = []
            type_node_dict_detail[key] = {}
        global_targets = []
        global_backgrounds = []
        test_mask_set = set(torch.nonzero(test_mask, as_tuple=False).view(-1).tolist())
        for n in range(len(test_mask)):
            if test_mask[n]:
                global_targets.append(n)
            else:
                global_backgrounds.append(n)
        for cid in l1_community_node_idx:
            children = list(new_dict[cid].get('children', []))
            targets = [n for n in children if n in test_mask_set]
            backgrounds = [n for n in children if n not in test_mask_set]
            if len(targets) == 1 and len(backgrounds) == 0:
                comm_type = "A: Pure Target"
            elif len(targets) == 0 and len(backgrounds) == 1:
                comm_type = "B: Pure Background"
            elif len(targets) == 0 and len(backgrounds) > 1:
                comm_type = "E: Background"
            elif len(targets) == 1 and len(backgrounds) >= 1:
                comm_type = "F: Specific"
            elif len(targets) > 1 and len(backgrounds) == 1:
                comm_type = "G: Common Shared"
            elif len(targets) > 1 and len(backgrounds) == 0:
                comm_type = "H: Unknown"
            elif len(targets) > 1 and len(backgrounds) > 1:
                comm_type = "K: Mixed Community"
            else:
                comm_type = "Z: Invalid"
            type_counts[comm_type] += 1
            type_node_dict[comm_type].append(cid)
            type_node_dict_detail[comm_type][int(cid)] = [int(node_idx) for node_idx in children]
        return dict(type_counts), type_node_dict, type_node_dict_detail

    def log_coding_tree(self, new_dict, test_mask, l1_community_node_idx, logger):
        ordered_keys = [
            "A: Pure Target",
            "B: Pure Background",
            "E: Background",
            "F: Specific",
            "G: Common Shared",
            "H: Unknown",
            "K: Mixed Community",
            "Z: Invalid"
        ]
        type_counts, type_node_dict, type_node_dict_detail = self.count_L1_community_types(ordered_keys, new_dict, l1_community_node_idx, test_mask)
        return type_counts, type_node_dict, type_node_dict_detail

    def random_sample_com_node(self, background_nodes, cluster_data, last_tokens_count):
        set_random_seed(self.args.condensation_seed)
        sampled_nodes = []
        background_info = ""
        token_budget = last_tokens_count
        shuffled_nodes = list(background_nodes)
        random.shuffle(shuffled_nodes)
        for i, nid in enumerate(shuffled_nodes):
            node_text = f"Node {i + 1}: {cluster_data.raw_texts[nid]}"
            temp_info = background_info + ("\n" if background_info else "") + node_text
            tokens = self.tokenizer(temp_info, return_tensors="pt").to(self.args.device)
            token_count = tokens["input_ids"].shape[1]
            if token_count > token_budget:
                break
            sampled_nodes.append(nid)
            background_info = temp_info
        return sampled_nodes, background_info

    def grouping_sample_com_node(self, background_nodes, cluster_data, last_tokens_count):
        set_random_seed(self.args.condensation_seed)
        shuffled_nodes = list(background_nodes)
        random.shuffle(shuffled_nodes)
        group_info_list, group_node_list, group_embed_list = [], [], []
        current_node_list, current_info_list = [], []
        current_text = ""
        for idx, nid in enumerate(shuffled_nodes):
            node_text = f"Node {len(current_node_list) + 1}: {cluster_data.raw_texts[nid]}"
            test_text = current_text + ("\n" if current_text else "") + node_text
            tokens = self.tokenizer(test_text, return_tensors="pt").to(self.args.device)
            token_len = tokens["input_ids"].shape[1]
            if token_len > last_tokens_count and current_node_list:
                group_info_list.append(current_text)
                group_node_list.append(current_node_list)
                bg_embeds = torch.stack([cluster_data.x[n] for n in current_node_list])
                group_embed_list.append(bg_embeds.mean(dim=0))
                current_node_list = [nid]
                current_info_list = [f"Node 1: {cluster_data.raw_texts[nid]}"]
                current_text = current_info_list[0]
            else:
                current_node_list.append(nid)
                current_info_list.append(node_text)
                current_text = "\n".join(current_info_list)
        if current_node_list:
            group_info_list.append(current_text)
            group_node_list.append(current_node_list)
            bg_embeds = torch.stack([cluster_data.x[n] for n in current_node_list])
            group_embed_list.append(bg_embeds.mean(dim=0))
        return group_node_list, group_info_list, group_embed_list

    def build_condensation_prompt_from_type_node_dict(self, cluster_data, type_node_dict, new_dict):
        set_random_seed(self.args.condensation_seed)
        templates = {
            "F: Specific": self.Specific_template,
            "G: Common Shared": self.Common_Shared_template,
            "K: Mixed Community": self.Mixed_Community_template
        }
        prompts = []
        test_mask = cluster_data.in_cluster_test_mask
        test_mask_set = set(torch.nonzero(test_mask, as_tuple=False).view(-1).tolist())
        all_nodes_set = set(range(len(test_mask)))
        background_nodes_set = all_nodes_set - test_mask_set
        for community_type, community_node_idx_list in type_node_dict.items():
            template = templates.get(community_type)
            if template is None:
                continue
            for community_idx in community_node_idx_list:
                leaves_idx_list = list(new_dict[community_idx]['children'])
                target_nodes = [nid for nid in leaves_idx_list if nid in test_mask_set]
                background_nodes = [nid for nid in leaves_idx_list if nid in background_nodes_set]
                prompt_template = copy.deepcopy(template)

                def build_embedding_and_prompt(target_info, background_info):
                    prompt = prompt_template.replace(TARGET_INFO_DEFAULT_TOKEN, target_info)
                    prompt = prompt.replace(BACKGROUND_INFO_DEFAULT_TOKEN, background_info)
                    bg_embeddings = torch.stack([cluster_data.x[nid] for nid in background_nodes])
                    condensed_embedding = bg_embeddings.mean(dim=0)
                    return prompt, condensed_embedding

                if community_type == "F: Specific" and target_nodes and background_nodes:
                    target = target_nodes[0]
                    target_info = cluster_data.raw_texts[target]
                    background_info = "\n".join(
                        [f"Node {i + 1}: {cluster_data.raw_texts[nid]}" for i, nid in enumerate(background_nodes)])
                    tokens_needed = self.tokenizer(background_info, return_tensors="pt").to(self.args.device)["input_ids"].shape[1]
                    remaining_tokens = self.args.llm_length - self.tokenizer(prompt_template.replace(BACKGROUND_INFO_DEFAULT_TOKEN, ''), return_tensors="pt").to(self.args.device)["input_ids"].shape[1] - 1
                    if tokens_needed > remaining_tokens:
                        groups, texts, embeddings = self.grouping_sample_com_node(background_nodes, cluster_data, remaining_tokens)
                        for g_nodes, g_text, g_embed in zip(groups, texts, embeddings):
                            prompt = prompt_template.replace(TARGET_INFO_DEFAULT_TOKEN, target_info).replace(BACKGROUND_INFO_DEFAULT_TOKEN, g_text)
                            prompts.append(
                                {"type": community_type,
                                 "community_idx": int(community_idx),
                                 "target_nodes": [int(target)],
                                 "background_nodes": list(map(int, g_nodes)),
                                 "condensed_bg_embedding": g_embed,
                                 "prompt": prompt}
                            )
                    else:
                        prompt, embed = build_embedding_and_prompt(target_info, background_info)
                        prompts.append(
                            {"type": community_type,
                             "community_node_idx": int(community_idx),
                             "target_nodes": [int(target)],
                             "background_nodes": list(map(int, background_nodes)),
                             "condensed_bg_embedding": embed,
                             "prompt": prompt}
                        )
                elif community_type == "G: Common Shared" and target_nodes and background_nodes:
                    background = background_nodes[0]
                    background_info = cluster_data.raw_texts[background]
                    target_info = "\n".join([f"Node {i + 1}: {cluster_data.raw_texts[nid]}" for i, nid in enumerate(target_nodes)])
                    tokens_needed = self.tokenizer(target_info, return_tensors="pt").to(self.args.device)["input_ids"].shape[1]
                    remaining_tokens = self.args.llm_length - self.tokenizer(prompt_template.replace(TARGET_INFO_DEFAULT_TOKEN, ''), return_tensors="pt").to(self.args.device)["input_ids"].shape[1] - 1
                    if tokens_needed > remaining_tokens:
                        sampled_nodes, sampled_text = self.random_sample_com_node(target_nodes, cluster_data, remaining_tokens)
                        target_info = sampled_text
                    prompt, embed = build_embedding_and_prompt(target_info, background_info)
                    prompts.append(
                        {"type": community_type,
                         "community_node_idx": int(community_idx),
                         "target_nodes": list(map(int, target_nodes)),
                         "background_nodes": list(map(int, background_nodes)),
                         "condensed_bg_embedding": embed,
                         "prompt": prompt}
                    )
                elif community_type == "K: Mixed Community" and target_nodes and background_nodes:
                    target_info = "\n".join([f"Node {i + 1}: {cluster_data.raw_texts[nid]}" for i, nid in enumerate(target_nodes)])
                    background_info = "\n".join([f"Node {i + 1}: {cluster_data.raw_texts[nid]}" for i, nid in enumerate(background_nodes)])
                    full_prompt = prompt_template.replace(TARGET_INFO_DEFAULT_TOKEN, target_info).replace(BACKGROUND_INFO_DEFAULT_TOKEN, background_info)
                    token_count = self.tokenizer(full_prompt, return_tensors="pt").to(self.args.device)["input_ids"].shape[1]
                    if token_count <= self.args.llm_length:
                        embed = torch.stack([cluster_data.x[nid] for nid in background_nodes]).mean(dim=0)
                        prompts.append(
                            {"type": community_type,
                             "community_node_idx": int(community_idx),
                             "target_nodes": list(map(int, target_nodes)),
                             "background_nodes": list(map(int, background_nodes)),
                             "condensed_bg_embedding": embed,
                             "prompt": full_prompt}
                        )
        return prompts

    def reconstruct_condensation_data(self, prompts_list, cluster_data, type_node_dict_detail):
        set_random_seed(self.args.condensation_seed)
        ori_x = cluster_data.x
        ori_y = cluster_data.y
        ori_edge_index = cluster_data.edge_index
        ori_label_abbr = cluster_data.label_abbr
        ori_label_whol = cluster_data.label_whol
        embedding_dim = ori_x.shape[1]
        test_mask = cluster_data.in_cluster_test_mask
        test_mask_set = set(torch.nonzero(test_mask, as_tuple=False).view(-1).tolist())
        all_nodes_set = set(range(len(test_mask)))
        background_nodes_set = all_nodes_set - test_mask_set
        node_idx_mapping = {}
        new_embeddings, new_raw_texts, all_y, new_ys, all_label_abbr, all_label_whol = [], [], [], [], [], []
        num = 0
        for node_idx in range(ori_x.shape[0]):
            if node_idx not in background_nodes_set:
                node_idx_mapping[node_idx] = num
                new_embeddings.append(ori_x[node_idx])
                if len(ori_label_abbr) == 0:
                    all_label_abbr.append('')
                else:
                    all_label_abbr.append(ori_label_abbr[node_idx])
                if len(ori_label_whol) == 0:
                    all_label_whol.append('')
                else:
                    all_label_whol.append(ori_label_whol[node_idx])
                all_y.append(ori_y[node_idx])
                num += 1
        for prompt in prompts_list:
            bg_nodes = prompt['background_nodes']
            bg_embedding = prompt.get('condensed_bg_embedding', None)
            temp_y = []
            temp_label_abbr = []
            temp_label_whol = []
            for old_idx in bg_nodes:
                node_idx_mapping[old_idx] = num
                temp_y.append(ori_y[old_idx])
                if len(ori_label_abbr) == 0:
                    temp_label_abbr.append('')
                else:
                    temp_label_abbr.append(ori_label_abbr[old_idx])
                if len(ori_label_whol) == 0:
                    temp_label_whol.append('')
                else:
                    temp_label_whol.append(ori_label_whol[old_idx])
            new_embeddings.append(bg_embedding)
            new_raw_texts.append(prompt['prompt'])
            all_y.append(temp_y)
            all_label_abbr.append(temp_label_abbr)
            all_label_whol.append(temp_label_whol)
            num += 1
        edge_set = set()
        for src, dst in ori_edge_index.t().tolist():
            if src not in node_idx_mapping or dst not in node_idx_mapping:
                continue
            new_src = node_idx_mapping[src]
            new_dst = node_idx_mapping[dst]
            if new_src != new_dst:
                edge = (new_src, new_dst)
                edge_set.add(edge)
        if edge_set:
            new_edge_index = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous()
        else:
            new_edge_index = torch.empty((2, 0), dtype=torch.long)
        new_sample_mask = torch.zeros(num, dtype=torch.bool)
        for old_idx in test_mask_set:
            if old_idx in node_idx_mapping:
                new_sample_mask[node_idx_mapping[old_idx]] = True
        condensation_mask = torch.zeros(num, dtype=torch.bool)
        offset = ori_x.shape[0] - len(background_nodes_set)
        for i in range(len(prompts_list)):
            condensation_mask[offset + i] = True
        reconstruct_x = torch.stack(new_embeddings) if new_embeddings else torch.empty((0, embedding_dim))
        reconstruct_y = torch.stack(new_ys) if new_ys else torch.empty((0,), dtype=torch.long)
        reconstructed_data = Data(community_cluster_x=cluster_data.x, reconstruct_x=reconstruct_x, reconstruct_edge_index=new_edge_index, reconstruct_y=reconstruct_y, reconstruct_raw_texts=new_raw_texts, reconstruct_target_mask=new_sample_mask, reconstruct_condensation_node_mapping=node_idx_mapping, reconstruct_all_label_abbr=all_label_abbr, reconstruct_all_label_whol=all_label_whol, reconstruct_condensation_mask=condensation_mask)

        return reconstructed_data

    def find_neighbors(self, hop, node_idx, edge_index):
        if isinstance(node_idx, int) or isinstance(node_idx, list):
            node_idx = torch.tensor([node_idx], dtype=torch.long)
        is_connected = (edge_index[0] == node_idx).any() or (edge_index[1] == node_idx).any()
        if not is_connected:
            return []
        subset_hop, _, _, _ = k_hop_subgraph(node_idx.view(-1), hop, edge_index, relabel_nodes=False)
        neighbors = subset_hop.flatten().unique()
        neighbors = neighbors[neighbors != node_idx]
        return neighbors.tolist()

    def re_id_construction(self):
        re_id = {}
        for node_idx in self.sample_node_idx:
            neighbor_set = self.find_neighbors(self.args.hop, node_idx, self.dataset.edge_index)
            re_id[int(node_idx)] = neighbor_set
        return re_id

    def match_ori_node_idx(self):
        target_mask = self.dataset.target_mask
        condense_raw_texts = self.dataset.raw_texts
        all_raw_texts = self.ori_dataset.raw_texts
        text_to_idx = {text: idx for idx, text in enumerate(all_raw_texts)}
        all_condensation_idx = torch.arange(self.dataset.x.shape[0])
        all_global_idx_list = []
        for node_idx in all_condensation_idx:
            if target_mask[node_idx]:
                text = condense_raw_texts[int(node_idx)]
                if text not in text_to_idx:
                    all_global_idx_list.append(-999)
                    continue
                else:
                    global_idx = text_to_idx.get(text, -1)
                    all_global_idx_list.append(global_idx)
            else:
                all_global_idx_list.append(-1)
        self.dataset.match_ori_node_idx = all_global_idx_list

    def random_neighbor_prompt(self):
        start_time = datetime.now()
        set_random_seed(self.args.llm_classification_seed)
        one_hop_generated_data = []
        raw_texts = self.dataset.raw_texts
        labels = self.dataset.y
        label_categories_whol = self.dataset.label_whol
        if len(self.dataset.label_abbr) == 0:
            label_categories_abbr = self.dataset.label_whol
        else:
            label_categories_abbr = self.dataset.label_abbr

        for node_idx in self.sample_node_idx:
            raw_text = raw_texts[int(node_idx)]
            node_label = labels[int(node_idx)]
            ori_node_idx = self.dataset.match_ori_node_idx[int(node_idx)]
            label_abbr = label_categories_abbr[int(node_idx)]
            label_whol = label_categories_whol[int(node_idx)]
            neighbor_set = self.re_id[int(node_idx)]
            if not neighbor_set:
                source_text_0 = copy.deepcopy(self.center_prompt)
                source_text_0 = source_text_0.replace(DEFAULT_CATEGORY_TOKEN, self.category_list)
                source_text_0 = source_text_0.replace(DEFAULT_CENTER_RAW_TEXT_TOKEN, raw_text)
                final_text = copy.deepcopy(source_text_0)
                one_hop_generated_data.append({
                    "node_id": int(node_idx),
                    "ori_node_id": int(ori_node_idx),
                    "query": final_text,
                    "node_list": [],
                    "all_neighbor": [],
                    "label_abbr": label_abbr,
                    "label_whol": label_whol
                })
                continue
            source_text = copy.deepcopy(self.graph_tree_prompt)
            source_text = source_text.replace(DEFAULT_CATEGORY_TOKEN, self.category_list)
            source_text = source_text.replace(DEFAULT_CENTER_RAW_TEXT_TOKEN, raw_text)
            temp_source_text = copy.deepcopy(source_text)
            already_tokens = self.tokenizer(temp_source_text.replace(DEFAULT_G_NEIGHBOR_INFO_TOKEN, ''), return_tensors="pt").to(self.args.device)
            already_tokens_count = already_tokens["input_ids"].shape[1]
            last_tokens_count = self.args.llm_length - already_tokens_count - 1
            neighbor_texts = []
            already_sampled = set()
            count = 0
            while count <= len(neighbor_set):
                remaining_neighbors = list(set(neighbor_set) - already_sampled)
                if not remaining_neighbors:
                    break
                sampled_neighbor = int(np.random.choice(remaining_neighbors, 1, replace=False)[0])
                node_info = raw_texts[sampled_neighbor].strip()
                temp_neighbor_texts = neighbor_texts + [f"(Node {count + 1}): {node_info}"]
                temp_graph_text = '\n'.join(temp_neighbor_texts)
                if len(self.tokenizer(temp_graph_text, add_special_tokens=False)['input_ids']) < last_tokens_count:
                    neighbor_texts.append(f"(Node {count + 1}): {node_info}")
                    already_sampled.add(sampled_neighbor)
                    count += 1
                else:
                    break
            final_text = source_text.replace(DEFAULT_G_NEIGHBOR_INFO_TOKEN, '\n'.join(neighbor_texts))
            sampled_neighbor_idx = list(already_sampled)
            one_hop_generated_data.append({
                "node_id": int(node_idx),
                "ori_node_id": int(ori_node_idx),
                "query": final_text,
                "node_list": [int(n) for n in sampled_neighbor_idx],
                "all_neighbor": [int(n) for n in neighbor_set],
                "label_abbr": label_abbr,
                "label_whol": label_whol
            })
        end_time = datetime.now()
        print(end_time - start_time)
        random_experiment_path = f'{self.experiment_name}'
        os.makedirs(random_experiment_path, exist_ok=True)
        output_file = f"{random_experiment_path}/one_hop_generated_data_{self.args.new_tips}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(one_hop_generated_data, f, indent=4, ensure_ascii=False)
        print(f"INFO: Saved {output_file}")

    def rag_neighbor_prompt(self):
        start_time = datetime.now()
        set_random_seed(self.args.llm_classification_seed)
        one_hop_generated_data = []
        raw_texts = self.dataset.raw_texts
        labels = self.dataset.y
        label_categories_whol = self.dataset.label_whol
        if len(self.dataset.label_abbr) == 0:
            label_categories_abbr = self.dataset.label_whol
        else:
            label_categories_abbr = self.dataset.label_abbr
        emb_token = AutoTokenizer.from_pretrained(f"{self.args.model_path}/Bert-base-uncased")
        emb_model = AutoModel.from_pretrained(f"{self.args.model_path}/Bert-base-uncased").to(self.args.device)
        emb_model.eval()
        def get_embedding(text):
            inputs = emb_token(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = emb_model(**inputs)
            return outputs.last_hidden_state[:, 0, :]
        for node_idx in self.sample_node_idx:
            raw_text = raw_texts[int(node_idx)]
            ori_node_idx = self.dataset.match_ori_node_idx[int(node_idx)]
            label_abbr = label_categories_abbr[int(node_idx)]
            label_whol = label_categories_whol[int(node_idx)]
            neighbor_set = self.re_id[int(node_idx)]
            if not neighbor_set:
                source_text_0 = copy.deepcopy(self.center_prompt)
                source_text_0 = source_text_0.replace(DEFAULT_CATEGORY_TOKEN, self.category_list)
                source_text_0 = source_text_0.replace(DEFAULT_CENTER_RAW_TEXT_TOKEN, raw_text)
                one_hop_generated_data.append({
                    "node_id": int(node_idx),
                    "ori_node_id": int(ori_node_idx),
                    "query": source_text_0,
                    "node_list": [],
                    "all_neighbor": [],
                    "label_abbr": label_abbr,
                    "label_whol": label_whol
                })
                continue
            source_text = copy.deepcopy(self.graph_tree_prompt)
            source_text = source_text.replace(DEFAULT_CATEGORY_TOKEN, self.category_list)
            source_text = source_text.replace(DEFAULT_CENTER_RAW_TEXT_TOKEN, raw_text)
            temp_source_text = copy.deepcopy(source_text)
            already_tokens = self.tokenizer(temp_source_text.replace(DEFAULT_G_NEIGHBOR_INFO_TOKEN, ''), return_tensors="pt").to(self.args.device)
            already_tokens_count = already_tokens["input_ids"].shape[1]
            last_tokens_count = self.args.llm_length - already_tokens_count - 1
            center_text = f"{raw_text}"
            center_embedding = get_embedding(center_text).cpu().numpy()
            neighbor_texts = []
            neighbor_embeddings = []
            for neighbor in neighbor_set:
                neighbor_id = int(neighbor)
                neighbor_info = raw_texts[neighbor_id].strip()
                neighbor_texts.append((neighbor_id, neighbor_info))
                neighbor_embeddings.append(get_embedding(neighbor_info).cpu().numpy())
            neighbor_embeddings = np.vstack(neighbor_embeddings)
            d = neighbor_embeddings.shape[1]
            index = faiss.IndexFlatIP(d)
            index.add(neighbor_embeddings)
            distances, indices = index.search(center_embedding, len(neighbor_set))
            selected_neighbors = [neighbor_texts[int(idx)] for idx in indices[0]]
            graph_text_list = []
            already_sampled = set()
            count = 0
            for neighbor_id, neighbor_info in selected_neighbors:
                if count > len(neighbor_set):
                    break
                new_text = f"(Node {count + 1}): {neighbor_info}"
                temp_graph_text_list = graph_text_list + [new_text]
                temp_graph_text = '\n'.join(temp_graph_text_list)
                if len(self.tokenizer(temp_graph_text, add_special_tokens=False)['input_ids']) < last_tokens_count:
                    graph_text_list.append(new_text)
                    already_sampled.add(neighbor_id)
                    count += 1
                else:
                    break
            final_text = source_text.replace(DEFAULT_G_NEIGHBOR_INFO_TOKEN, '\n'.join(graph_text_list))
            sampled_neighbor_idx = list(already_sampled)
            one_hop_generated_data.append({
                "node_id": int(node_idx),
                "ori_node_id": int(ori_node_idx),
                "query": final_text,
                "node_list": [int(n) for n in sampled_neighbor_idx],
                "all_neighbor": [int(n) for n in neighbor_set],
                "label_abbr": label_abbr,
                "label_whol": label_whol
            })
        end_time = datetime.now()
        print(end_time - start_time)
        rag_experiment_path = f'{self.experiment_name}'
        os.makedirs(rag_experiment_path, exist_ok=True)
        output_file = f"{rag_experiment_path}/one_hop_generated_data_{self.args.new_tips}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(one_hop_generated_data, f, indent=4, ensure_ascii=False)
        print(f"INFO: Saved {output_file}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--new_date', type=str, default='20250606')
    argparser.add_argument('--task', type=str, default="classification")
    argparser.add_argument('--dataset', type=str, default="ogb_arxiv")
    argparser.add_argument('--model_path', type=str, default="/root/autodl-tmp/A_base_model")
    argparser.add_argument('--model_name', type=str, default="Llama-3.1-8b-Instruct")
    argparser.add_argument('--llm_length', type=int, default=4000)
    argparser.add_argument('--max_token', type=int, default=100)
    argparser.add_argument('--sample_neighbor_seed', type=int, default=0)
    argparser.add_argument('--new_tips', type=str, default='t0')
    argparser.add_argument('--round', type=str, default='r0')
    argparser.add_argument('--project_seed', type=int, default=0)
    argparser.add_argument('--hop', type=int, default=1)
    argparser.add_argument('--tree_height', type=int, default=2)
    argparser.add_argument('--node_embedding_batch_size', type=int, default=10)
    argparser.add_argument('--batch_size', type=int, default=10)
    argparser.add_argument('--condensation_seed', type=int, default=0)
    argparser.add_argument('--condensation_max_token', type=int, default=100)
    argparser.add_argument('--llm_classification_seed', type=int, default=0)
    argparser.add_argument('--reconstruct_full_data_seed', type=int, default=0)
    args = argparser.parse_args()
    graph_base = graph_dataset(args, args.dataset)
