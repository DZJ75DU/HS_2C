import numpy as np
import torch
import os
import sys
from pathlib import Path
from torch.utils.data import Dataset
from torch_geometric.utils import k_hop_subgraph
import faiss
import json
from datetime import datetime
import gc
import copy
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

project_root = Path(__file__).parent.parent.parent.resolve()
hs2c_path = project_root / '7_HS2C_pipeline'
sys.path.insert(0, str(hs2c_path))
from data_provider.constants import *
from utils.utils import set_random_seed


class SubgraphSampler:
    def __init__(self, config):
        self.config = config
        self.method = config['method']
        self.device = config['device']
        self.llm_length = config['llm_length']
        self.model_name = config['model_name']

        template_1_2 = """\nGiven a <GRAPH_TYPE_TOKEN> graph, the 0th node (target <NODE_TYPE_TOKEN>) has the following information:\n<CENTER_RAW_TEXT>\n\n\nQuestion: Based on the features of the target <NODE_TYPE_TOKEN>, please determine the most appropriate <DATASET_TOKEN> sub-category for the target <NODE_TYPE_TOKEN>.\n\nCategories: <category_list>.\n\n\nPlease think about the categorization in a step by step manner, and only output the single most relevant category of the target <NODE_TYPE_TOKEN>. Do not give any reasoning or extra text for your answer.\n\nAnswer:"""

        graph_tree_template_1_2 = """\nGiven a <GRAPH_TYPE_TOKEN> graph, the 0th node (target <NODE_TYPE_TOKEN>) has the following information:\n<CENTER_RAW_TEXT>\n\nThe target <NODE_TYPE_TOKEN> is connected to the following <NODE_TYPE_TOKEN>s:\n<NEIGHBOR_INFO>\n\n\nQuestion: Based on the features of the target <NODE_TYPE_TOKEN> and its <GRAPH_TYPE_TOKEN> network, please determine the most appropriate <DATASET_TOKEN> sub-category for the target <NODE_TYPE_TOKEN>.\n\nCategories: <category_list>.\n\n\nPlease think about the categorization of the target <NODE_TYPE_TOKEN> in a structured manner, and only output the single most relevant category of the target <NODE_TYPE_TOKEN>. Do not give any reasoning or extra text for your answer.\n\nAnswer:"""

        self.summary_template = """The following list records <NODE_TYPE_TOKEN>s related to the current one, with the relationship being '<GRAPH_TYPE_TOKEN>': <SUMMARY_CORPUS>.\nPlease summarize the information above with a short paragraph, and find some common points that can reflect the category of the <NODE_TYPE_TOKEN>."""

        graph_summary_template_2 = """\nGiven a <GRAPH_TYPE_TOKEN> graph, the 0th node (target <NODE_TYPE_TOKEN>) has the following information:\n<CENTER_RAW_TEXT>\n\nHere I give you the content of the node itself and the summary information of its 1-hop neighbors. The relation between the node and its neighbors is '<GRAPH_TYPE_TOKEN>':\n<SUMMARY_RAW_TEXT>\n\n\nQuestion: Based on the features of the target <NODE_TYPE_TOKEN> and its <GRAPH_TYPE_TOKEN> network, please determine the most appropriate <DATASET_TOKEN> sub-category for the target <NODE_TYPE_TOKEN>.\n\nCategories: <category_list>.\n\n\nPlease think about the categorization of the target <NODE_TYPE_TOKEN> in a structured manner, and only output the single most relevant category of the target <NODE_TYPE_TOKEN>. Do not give any reasoning or extra text for your answer.\n\nAnswer:"""

        template_1_2 = template_1_2.replace(DEFAULT_DATASET_NAME_TOKEN, self.config['dataset_name_in_prompt'])
        template_1_2 = template_1_2.replace(DEFAULT_GRAPH_TYPE_TOKEN, self.config['graph_type'])
        self.template_1_2 = template_1_2.replace(DEFAULT_NODE_TYPE_TOKEN, self.config['node_type'])

        # -----

        graph_tree_template_1_2 = graph_tree_template_1_2.replace(DEFAULT_DATASET_NAME_TOKEN, self.config['dataset_name_in_prompt'])
        graph_tree_template_1_2 = graph_tree_template_1_2.replace(DEFAULT_GRAPH_TYPE_TOKEN, self.config['graph_type'])
        self.graph_tree_template_1_2 = graph_tree_template_1_2.replace(DEFAULT_NODE_TYPE_TOKEN, self.config['node_type'])

        # -----

        self.summary_template = self.summary_template.replace(DEFAULT_GRAPH_TYPE_TOKEN, self.config['graph_type'])
        self.summary_template = self.summary_template.replace(DEFAULT_NODE_TYPE_TOKEN, self.config['node_type'])

        # -----

        graph_summary_template_2 = graph_summary_template_2.replace(DEFAULT_DATASET_NAME_TOKEN, self.config['dataset_name_in_prompt'])
        graph_summary_template_2 = graph_summary_template_2.replace(DEFAULT_GRAPH_TYPE_TOKEN, self.config['graph_type'])
        self.graph_summary_template_2 = graph_summary_template_2.replace(DEFAULT_NODE_TYPE_TOKEN, self.config['node_type'])

        experiment_data_path = 'generated_data'
        experiment_data_name = f"{self.config['new_date']}_{self.config['dataset']}"
        experiment_name = f"cluster@{self.config['cluster_name']}_seed@{self.config['cluster_seed']}_resolution@{self.config['resolution']}"
        self.experiment_path = f'{experiment_data_path}/{experiment_data_name}/{experiment_name}'

        self.neighbor_path = f'{self.experiment_path}/cluster_split'
        os.makedirs(self.neighbor_path, exist_ok=True)
        self.neighbor_tree_height_path = f"{self.experiment_path}/cluster_split/tree_height@{self.config['tree_height']}"
        os.makedirs(self.neighbor_tree_height_path, exist_ok=True)
        self.llm_seed_path = f"{self.neighbor_tree_height_path}/llm_classification_seed@{self.config['llm_classification_seed']}"
        os.makedirs(self.llm_seed_path, exist_ok=True)

        self.use_raw = True
        self.center_prompt = template_1_2
        self.graph_tree_prompt = graph_tree_template_1_2
        self.graph_summary_prompt = graph_summary_template_2

        self.one_hop_generated_data = {}

    def sample(self, base_data, recons_data, llm_engine):
        if self.config['point'] <= 3:
            print("[Sampler] Loading sampled data...")
            self.base_data = base_data
            self.dataset = recons_data
            self.tokenizer = llm_engine.tokenizer
            self.model = llm_engine.model
            all_node_idx = torch.tensor([i for i in range(self.dataset.x.shape[0])])
            self.sample_node_idx = all_node_idx[self.dataset.target_mask]

            self.re_id = self._re_id_construction()
            self._match_ori_node_idx()

            sample_data = self.rag_sample_neighbor()
        else:
            print("[Sampler] Sampling done.")
            rag_experiment_path = f'{self.llm_seed_path}/rag'
            output_file = f"{rag_experiment_path}/{self.model_name}_rag_sample_node_one_hop_generated_data_{self.config['new_tips']}.json"
            with open(output_file, "r", encoding="utf-8") as f:
                sample_data = json.load(f)
        return sample_data


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

    def _re_id_construction(self):
        re_id = {}
        for node_idx in self.sample_node_idx:
            neighbor_set = self.find_neighbors(self.config['hop'], node_idx, self.dataset.edge_index)
            re_id[int(node_idx)] = neighbor_set
        return re_id

    def _match_ori_node_idx(self):
        target_mask = self.dataset.target_mask
        all_target_node_num = sum(target_mask)
        condense_raw_texts = self.dataset.raw_texts
        all_raw_texts = self.base_data.raw_texts
        text_to_idx = {text: idx for idx, text in enumerate(all_raw_texts)}
        all_condensation_idx = torch.arange(self.dataset.x.shape[0])
        all_global_idx_list = []
        for node_idx in all_condensation_idx:
            if target_mask[node_idx]:
                text = condense_raw_texts[int(node_idx)]
                if text not in text_to_idx:
                    print(f"Warning: text not found in original dataset: {text}")
                    all_global_idx_list.append(-999)
                    continue
                else:
                    global_idx = text_to_idx.get(text, -1)
                    all_global_idx_list.append(global_idx)
            else:
                all_global_idx_list.append(-1)

        self.dataset.match_ori_node_idx = all_global_idx_list
        torch.save(self.dataset, f"{self.neighbor_path}/cluster_data_tree_height@{self.config['tree_height']}_reconstructed_full_complement_match_ori_node_idx_data.pt")

    def rag_sample_neighbor(self):
        start_time = datetime.now()
        set_random_seed(self.config['llm_classification_seed'])

        self.one_hop_generated_data["RAG"] = []

        raw_texts = self.dataset.raw_texts
        labels = self.dataset.y
        label_names = self.dataset.label_whol
        label_abbr_names = self.dataset.label_abbr
        if not label_abbr_names:
            label_abbr_names = label_names

        print("&"*30)
        print(label_names)
        print(label_abbr_names)

        emb_token = AutoTokenizer.from_pretrained(f"{self.config['model_path']}/Bert-base-uncased")
        emb_model = AutoModel.from_pretrained(f"{self.config['model_path']}/Bert-base-uncased").to(self.device)
        emb_model.eval()

        def get_embedding(text):
            """获取 BERT [CLS] embedding"""
            inputs = emb_token(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()} 

            with torch.no_grad():
                outputs = emb_model(**inputs)
            return outputs.last_hidden_state[:, 0, :]

        for node_idx in self.sample_node_idx:
            
            raw_text = raw_texts[int(node_idx)]
            node_label = labels[int(node_idx)]
            if int(node_label) == -1:
                continue
            ori_node_idx = self.dataset.match_ori_node_idx[int(node_idx)]
            node_label_num = int(node_label)
            print(node_label_num)
            print("&"*30)
            label_abbr = label_abbr_names[node_label_num]
            label_whol = label_names[node_label_num]
            neighbor_set = self.re_id[int(node_idx)]

            if not neighbor_set:
                source_text_0 = copy.deepcopy(self.center_prompt)
                source_text_0 = source_text_0.replace(DEFAULT_CATEGORY_TOKEN, self.config['category_list'])
                source_text_0 = source_text_0.replace(DEFAULT_CENTER_RAW_TEXT_TOKEN, raw_text)

                self.one_hop_generated_data["RAG"].append({
                    "node_id": int(node_idx),
                    "ori_node_id": int(ori_node_idx),
                    "query": source_text_0,
                    "node_list": [],
                    "all_neighbor": [],
                    "node_label": node_label_num,
                    "label_abbr": label_abbr,
                    "label_whol": label_whol
                })
                continue

            source_text = copy.deepcopy(self.graph_tree_prompt)
            source_text = source_text.replace(DEFAULT_CATEGORY_TOKEN, self.config['category_list'])
            source_text = source_text.replace(DEFAULT_CENTER_RAW_TEXT_TOKEN, raw_text)

            temp_source_text = copy.deepcopy(source_text)
            already_tokens = self.tokenizer(temp_source_text.replace(DEFAULT_G_NEIGHBOR_INFO_TOKEN, ''), return_tensors="pt").to(self.device)
            already_tokens_count = already_tokens["input_ids"].shape[1]
            last_tokens_count = self.llm_length - already_tokens_count - 1

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

            # 创建 FAISS 索引
            d = neighbor_embeddings.shape[1]
            index = faiss.IndexFlatIP(d)
            index.add(neighbor_embeddings)

            k = min(len(neighbor_set), self.config['k_hop_neighbors'][0])
            distances, indices = index.search(center_embedding, k)
            selected_neighbors = [neighbor_texts[int(idx)] for idx in indices[0]]

            graph_text_list = []
            already_sampled = set()
            max_count = k
            count = 0

            for neighbor_id, neighbor_info in selected_neighbors:
                if count > max_count:
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

            self.one_hop_generated_data["RAG"].append({
                "node_id": int(node_idx),
                "ori_node_id": int(ori_node_idx),
                "query": final_text,
                "node_list": [int(n) for n in sampled_neighbor_idx],
                "all_neighbor": [int(n) for n in neighbor_set],
                "node_label": node_label_num,
                "label_abbr": label_abbr,
                "label_whol": label_whol
            })

        end_time = datetime.now()
        print(end_time - start_time)
        print(f"INFO: RAG Sample Done!")
        rag_experiment_path = f'{self.llm_seed_path}/rag'
        os.makedirs(rag_experiment_path, exist_ok=True)
        output_file = f"{rag_experiment_path}/{self.model_name}_rag_sample_node_one_hop_generated_data_{self.config['new_tips']}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.one_hop_generated_data["RAG"], f, indent=4, ensure_ascii=False)
        print(f"INFO: Saved {output_file}")
        print(f"INFO: RAG Sample Saved!")

        return self.one_hop_generated_data["RAG"]
