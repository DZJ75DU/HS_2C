import re
import os
import sys
from pathlib import Path
import torch
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data
from .coding_tree_build import PartitionTree
from collections import defaultdict
import random
import copy
import numpy as np
from tqdm import tqdm
import gc
import json

project_root = Path(__file__).parent.parent.parent.resolve()
hs2c_path = project_root / '7_HS2C_pipeline'
sys.path.insert(0, str(hs2c_path))
from data_provider.constants import *
from utils.utils import set_random_seed

class GraphReconstructor:
    def __init__(self, config, llm_engine):
        self.config = config
        self.dataset_name = config['dataset']
        self.tree_height = config['tree_height']
        self.tree_seed = config['cluster_seed']
        self.device = config['device']
        self.tokenizer = llm_engine.tokenizer
        self.model = llm_engine.model
        self.llm_length = config['llm_length']
        self.max_token = config['max_token']
        self.model_name = config['model_name']
        self.enhanced_data = None

        self.Specific_template = '''You are given a single target <NODE_TYPE_TOKEN> and a group of background <NODE_TYPE_TOKEN>s from the same community within a <GRAPH_TYPE_TOKEN> graph. Each <NODE_TYPE_TOKEN> is represented by its title and abstract. The background <NODE_TYPE_TOKEN>s, which are connected to the target <NODE_TYPE_TOKEN>, may share a common semantic focus or research topic with the target <NODE_TYPE_TOKEN>.\n\nTarget <NODE_TYPE_TOKEN>:\n<TARGET_NODE_INFO>\n\nBackground <NODE_TYPE_TOKEN>s:\n<BACKGROUND_NODE_INFO>\n\n\nTask: Identify and summarize the content of the background <NODE_TYPE_TOKEN>s that are semantically aligned with the target <NODE_TYPE_TOKEN>, focusing on shared research disciplinary, domains, directions, or topics. Ignore background <NODE_TYPE_TOKEN>s that are not semantically aligned with the target <NODE_TYPE_TOKEN>. Avoid summarizing each <NODE_TYPE_TOKEN> individually.\n\nPlease present the semantically aligned content in 3 to 4 well-structured sentences (no more than 5), written in a cohesive and formal academic style, without headings, task instructions, author names, <GRAPH_TYPE_TOKEN>s, or meta-commentary.\n\nAnswer:'''

        self.Common_Shared_template = '''You are given a group of target <NODE_TYPE_TOKEN>s and a single background <NODE_TYPE_TOKEN> from the same community within a <GRAPH_TYPE_TOKEN> graph. Each <NODE_TYPE_TOKEN> is represented by its title and abstract. The background <NODE_TYPE_TOKEN>, which is connected to the target <NODE_TYPE_TOKEN>s, may share a common semantic focus or research topic with the target <NODE_TYPE_TOKEN>s.\n\nTarget <NODE_TYPE_TOKEN>s:\n<TARGET_NODE_INFO>\n\nBackground <NODE_TYPE_TOKEN>:\n<BACKGROUND_NODE_INFO>\n\n\nTask: Identify and summarize the content of the background <NODE_TYPE_TOKEN> that is semantically aligned with the group of target <NODE_TYPE_TOKEN>s, focusing on shared research disciplinary, domains, directions, or topics that make the background <NODE_TYPE_TOKEN> meaningfully connected to the different target <NODE_TYPE_TOKEN>s. Avoid summarizing each <NODE_TYPE_TOKEN> individually.\n\nPlease present the semantically aligned content in 3 to 4 well-structured sentences (no more than 5), written in a cohesive and formal academic style, without headings, task instructions, author names, <GRAPH_TYPE_TOKEN>s, or meta-commentary.\n\nAnswer:'''

        self.Mixed_Community_template = '''You are given a set of target <NODE_TYPE_TOKEN>s and a group of background <NODE_TYPE_TOKEN>s from the same community within a <GRAPH_TYPE_TOKEN> graph. Each <NODE_TYPE_TOKEN> is represented by its title and abstract. The background <NODE_TYPE_TOKEN>s, which are connected to the target <NODE_TYPE_TOKEN>s, may share a common semantic focus or research topic with the target <NODE_TYPE_TOKEN>s.\n\nTarget <NODE_TYPE_TOKEN>s:\n<TARGET_NODE_INFO>\n\nBackground <NODE_TYPE_TOKEN>s:\n<BACKGROUND_NODE_INFO>\n\n\nTask: Identify and summarize the content of the background <NODE_TYPE_TOKEN>s that are semantically aligned with the group of target <NODE_TYPE_TOKEN>s, focusing on shared research disciplinary, domains, directions, or topics. Ignore background <NODE_TYPE_TOKEN>s that are not semantically aligned with the target <NODE_TYPE_TOKEN>s. Avoid summarizing each <NODE_TYPE_TOKEN> individually.\n\nPlease present the semantically aligned content in 3 to 4 well-structured sentences (no more than 5), written in a cohesive and formal academic style, without headings, task instructions, author names, <GRAPH_TYPE_TOKEN>s, or meta-commentary.\n\nAnswer:'''

        self.summary_template = """The following list records <NODE_TYPE_TOKEN>s related to the current one, with the relationship being '<GRAPH_TYPE_TOKEN>': <SUMMARY_CORPUS>.\nPlease summarize the information above with a short paragraph, and find some common points that can reflect the category of the <NODE_TYPE_TOKEN>."""

        self.Specific_template = self.Specific_template.replace(DEFAULT_GRAPH_TYPE_TOKEN, self.config['graph_type'])
        self.Specific_template = self.Specific_template.replace(DEFAULT_NODE_TYPE_TOKEN, self.config['node_type'])
        # -----

        self.Common_Shared_template = self.Common_Shared_template.replace(DEFAULT_GRAPH_TYPE_TOKEN, self.config['graph_type'])
        self.Common_Shared_template = self.Common_Shared_template.replace(DEFAULT_NODE_TYPE_TOKEN, self.config['node_type'])
        # -----

        self.Mixed_Community_template = self.Mixed_Community_template.replace(DEFAULT_GRAPH_TYPE_TOKEN, self.config['graph_type'])
        self.Mixed_Community_template = self.Mixed_Community_template.replace(DEFAULT_NODE_TYPE_TOKEN, self.config['node_type'])
        # -----

        self.summary_template = self.summary_template.replace(DEFAULT_GRAPH_TYPE_TOKEN, self.config['graph_type'])
        self.summary_template = self.summary_template.replace(DEFAULT_NODE_TYPE_TOKEN, self.config['node_type'])

        self.llm_length = self.config['llm_length']

    def convert_json_serializable(self, obj):
        if isinstance(obj, dict):
            return {k: self.convert_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_json_serializable(item) for item in obj]
        elif isinstance(obj, set):
            return [self.convert_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, torch.Tensor):
            if obj.numel() == 1:
                return obj.item()
            return obj.tolist()
        else:
            return obj

    def reconstruct(self):
        dataset_path = f'{hs2c_path}/generated_data'
        dataset_name = f"{self.config['new_date']}_{self.dataset_name}"
        experiment_name = f"cluster@{self.config['cluster_name']}_seed@{self.config['cluster_seed']}_resolution@{self.config['resolution']}"
        experiment_path = f'{dataset_path}/{dataset_name}/{experiment_name}'
        
        enhanced_graph_path = f"{dataset_path}/{dataset_name}/{self.config['new_date']}_{self.config['dataset']}_knn_target_graph.pt"
        if os.path.exists(enhanced_graph_path):
            self.enhanced_data = torch.load(enhanced_graph_path, weights_only=False)
        else:
            print(f"[Warn] Enhanced graph not found at {enhanced_graph_path}")

        self.save_path = f'{experiment_path}/cluster_split'
        self.cluster_data_path = f'{self.save_path}/cluster_data'
        if self.config['point'] <= 2:
            print(f"[Reconstruct] Partition Tree Reconstruction (height={self.tree_height})...")
            set_random_seed(self.tree_seed)
            self.cluster_data_coding_tree_new_dict_list = []
            self.cluster_data_coding_tree_list = []

            clusters_files = self._sample_cluster()
            self.reconstruct_data = self._reconstruct_full_graph_from_clusters_baseline(clusters_files)
            self.reconstruct_data = self._check_cluster_condensation(self.reconstruct_data)
        else:
            print("[Reconstruct] Reconstruction done.")
            self.reconstruct_data = torch.load(f'{self.save_path}/{self.model_name}_cluster_data_tree_height@{self.tree_height}_reconstructed_full_complement_data.pt', weights_only=False)
        return self.reconstruct_data

    def _sample_cluster(self):
        pattern = re.compile(r'cluster@(\d+)$')
        cluster_files = []

        for dirname in os.listdir(self.cluster_data_path):
            match = pattern.match(dirname)
            if match:
                num = int(match.group(1))
                cluster_files.append((num, dirname))
        cluster_files.sort(key=lambda x: x[0])
        return [dirname for _, dirname in cluster_files]

    def _reconstruct_full_graph_from_clusters_baseline(self, clusters_files):
        set_random_seed(self.tree_seed)
        all_x, all_y, all_raw_text, all_target_mask, all_query_texts, all_ori_node_idx = [], [], [], [], [], []
        all_new_edge_index = set()
        node_idx_mapping = {}

        next_cluster_start_new_idx = 0

        all_edge_index = self.enhanced_data.knn_edge_index
        tree_height = self.tree_height

        for num, cluster_name in enumerate(clusters_files):
            cluster_coding_tree_path = f'{self.cluster_data_path}/{cluster_name}/{self.model_name}_coding_tree_height@{tree_height}'
            if self.max_token != 100:
                condensation_data_path = f'{cluster_coding_tree_path}/3_condensation_cluster_data_max_token@{self.max_token}.pt'
            else:
                condensation_data_path = f'{cluster_coding_tree_path}/3_condensation_cluster_data.pt'
            condensation_data = torch.load(condensation_data_path, weights_only=False)
            if num == 0:
                all_label_whol = condensation_data.label_names
                all_label_abbr = condensation_data.label_abbr_names

            cluster_node_mapping = condensation_data.community_cluster_node_mapping  # global → local
            condensation_mapping = condensation_data.reconstruct_condensation_node_mapping  # local → new idx (or None)

            local_to_global = {int(local): int(global_id) for global_id, local in cluster_node_mapping.items()}

            for local_cluster_id, condensed_id in condensation_mapping.items():
                if condensed_id is not None:
                    global_idx = local_to_global[int(local_cluster_id)]
                    node_idx_mapping[int(global_idx)] = int(condensed_id + next_cluster_start_new_idx)

            valid_new_ids = [v for v in condensation_mapping.values() if v is not None]
            if not valid_new_ids:
                continue
            condensation_length = max(valid_new_ids)
            next_cluster_start_new_idx += condensation_length + 1

            all_x.append(condensation_data.reconstruct_x)
            all_y.append(condensation_data.reconstruct_y)
            all_raw_text.extend(condensation_data.reconstruct_condense_raw_texts)
            all_target_mask.append(condensation_data.reconstruct_target_mask)
            all_query_texts.extend(condensation_data.reconstruct_raw_texts)
            all_ori_node_idx.extend(condensation_data.reconstruct_ori_node_idx)

        for src_global, tgt_global in all_edge_index.t().tolist():
            if src_global in node_idx_mapping and tgt_global in node_idx_mapping:
                new_src = node_idx_mapping[src_global]
                new_tgt = node_idx_mapping[tgt_global]

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

        new_data = Data(
            x=x,
            edge_index=edge_index_tensor,
            y=y,
            raw_texts=all_raw_text,
            condensation_mask=target_mask,
            target_mask=condensation_mask,
            query_texts=all_query_texts,
            label_abbr=all_label_abbr,
            label_whol=all_label_whol,
            ori_node_idx=all_ori_node_idx,
        )

        valid_label_mask = new_data.y != -1
        new_data.target_mask = new_data.target_mask & valid_label_mask

        return new_data

    def _check_cluster_condensation(self, reconstruct_data):
        set_random_seed(self.tree_seed)

        raw_texts = reconstruct_data.raw_texts

        temp_json_path = f'{self.save_path}/{self.model_name}_cluster_data_tree_height@{self.tree_height}_reconstructed_full_data.json'

        all_failure_text_list = []
        all_failure_text_complement_list = []

        for fail_num, text in enumerate(raw_texts):
            if DEFAULT_CONDENSATION_FAILURE_TOKEN in text:
                print(fail_num)
                all_failure_text_list.append(text)
        print(f'INFO: All CONDENSATION_FAILURE texts -- {len(all_failure_text_list)}')

        if len(all_failure_text_list) == 0:
            torch.save(reconstruct_data,f'{self.save_path}/{self.model_name}_cluster_data_tree_height@{self.tree_height}_reconstructed_full_complement_data.pt')
            return reconstruct_data

        else:
            count = 0
            for num, raw_text in enumerate(all_failure_text_list):
                print(f'INFO: NO.{count + 1} CONDENSATION_FAILURE text complement')
                count += 1

                full_query = raw_text.split(DEFAULT_CONDENSATION_FAILURE_TOKEN, 1)[1].strip()
                print(full_query)

                separator = "\n\n\nTask: Identify and summarize the content of the background <NODE_TYPE_TOKEN>s that are semantically aligned with the group of target <NODE_TYPE_TOKEN>s, focusing on shared research disciplinary, domains, directions, or topics. Ignore background <NODE_TYPE_TOKEN>s that are not semantically aligned with the target <NODE_TYPE_TOKEN>s. Avoid summarizing each <NODE_TYPE_TOKEN> individually.\n\nPlease present the semantically aligned content in 3 to 4 well-structured sentences (no more than 5), written in a cohesive and formal academic style, without headings, task instructions, author names, <GRAPH_TYPE_TOKEN>s, or meta-commentary.\n\nAnswer:"
                separator2 = "\nPlease summarize the information above with a short paragraph, and find some common points that can reflect the category of the paper."
                separator3 = "\n\nAnswer:"

                separator = separator.replace(DEFAULT_GRAPH_TYPE_TOKEN, self.config['graph_type'])
                separator = separator.replace(DEFAULT_NODE_TYPE_TOKEN, self.config['node_type'])

                separator2 = separator2.replace(DEFAULT_GRAPH_TYPE_TOKEN, self.config['graph_type'])
                separator2 = separator2.replace(DEFAULT_NODE_TYPE_TOKEN, self.config['node_type'])

                if separator in full_query:
                    content, instruction = full_query.split(separator, 1)
                    instruction = separator + instruction
                elif separator2 in full_query:
                    content, instruction = full_query.split(separator2, 1)
                    instruction = separator + instruction
                elif separator3 in full_query:
                    content, instruction = full_query.split(separator3, 1)
                    instruction = separator3 + instruction
                else:
                    raise ValueError

                content_ids = self.tokenizer.encode(content, truncation=False)
                instruction_ids = self.tokenizer.encode(instruction, truncation=False)

                if len(content_ids) + len(instruction_ids) > self.max_token:
                    available = self.max_token - len(instruction_ids)
                    content_ids = content_ids[:available]
                    content = self.tokenizer.decode(content_ids, skip_special_tokens=True)

                query = content + instruction
                print(query)

                try:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    tokens = self.tokenizer(query, padding=True, return_tensors="pt", truncation=True).to(
                        self.device)

                    input_ids = tokens["input_ids"]
                    attention_mask = tokens["attention_mask"]

                    output = self.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.max_token,
                        num_return_sequences=1,
                        no_repeat_ngram_size=2,
                        temperature=1.0,
                        top_k=50,
                        top_p=0.95, 
                        do_sample=True,
                    )

                    generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

                    answer = generated_text.split(
                        separator3,
                        1)[1].strip()

                    all_failure_text_complement_list.append(answer)

                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    gc.collect()
                    torch.cuda.ipc_collect()
                    all_failure_text_complement_list.append(raw_text)

                if count % 100 == 0:
                    with open(temp_json_path, 'w', encoding='utf-8') as f_1:
                        json.dump(all_failure_text_complement_list, f_1, indent=2, ensure_ascii=False)

            with open(temp_json_path, 'w', encoding='utf-8') as f_2:
                json.dump(all_failure_text_complement_list, f_2, indent=2, ensure_ascii=False)

            failure_num = 0
            new_raw_texts = []
            for num, text in enumerate(raw_texts):
                if DEFAULT_CONDENSATION_FAILURE_TOKEN in text:
                    new_raw_texts.append(all_failure_text_complement_list[failure_num])
                    failure_num += 1
                else:
                    new_raw_texts.append(text)

            reconstruct_data.raw_texts = new_raw_texts
            torch.save(reconstruct_data, f'{self.save_path}/{self.model_name}_cluster_data_tree_height@{self.tree_height}_reconstructed_full_complement_data.pt')

            return reconstruct_data
