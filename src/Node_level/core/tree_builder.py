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

class TreeBuilder:
    def __init__(self, config, llm_engine):
        self.config = config
        self.dataset = config['dataset']
        self.tree_height = config['tree_height']
        self.tree_seed = config['cluster_seed']
        self.device = config['device']
        self.tokenizer = llm_engine.tokenizer
        self.model = llm_engine.model
        self.llm_length = config['llm_length']
        self.max_token = config['max_token']
        self.model_name = config['model_name']

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

    def build(self, data, clusters):
        """
        clusters: dict {cluster_id: [node_list]} from CommunityDetector
        """
        dataset_path = f'{hs2c_path}/generated_data'
        dataset_name = f"{self.config['new_date']}_{self.dataset}"
        experiment_name = f"cluster@{self.config['cluster_name']}_seed@{self.config['cluster_seed']}_resolution@{self.config['resolution']}"
        experiment_path = f'{dataset_path}/{dataset_name}/{experiment_name}'
        self.save_path = f'{experiment_path}/cluster_split'
        self.cluster_data_path = f'{self.save_path}/cluster_data'
        if self.config['point'] <= 1:
            set_random_seed(self.tree_seed)
            print(f"[TreeBuilder] Building Partition Tree (height={self.tree_height})...")

            self.cluster_data_coding_tree_new_dict_list = []
            self.cluster_data_coding_tree_list = []

            clusters_files = self._sample_cluster()
            self._get_coding_tree_dict(clusters_files)
            return True
        else:
            print("[TreeBuilder] Tree Building done.")
            return True

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

    def _get_coding_tree_dict(self, cluster_files):
        for num, cluster_name in enumerate(cluster_files):
            cluster_path = f'{self.cluster_data_path}/{cluster_name}'
            cluster_data_path = os.path.join(cluster_path, f'{cluster_name}_data.pt')
            cluster_data = torch.load(cluster_data_path, weights_only=False)
            cluster_coding_tree_path = f'{self.cluster_data_path}/{cluster_name}/{self.model_name}_coding_tree_height@{self.tree_height}'
            os.makedirs(cluster_coding_tree_path, exist_ok=True)

            adj_matrix = to_dense_adj(cluster_data.edge_index)
            print(type(adj_matrix))
            if self.dataset in ['citeseer']:
                undirected_adj = adj_matrix.squeeze(0).detach().cpu().numpy()
            else:
                undirected_adj = adj_matrix.squeeze(0).numpy()

            coding_tree = PartitionTree(adj_matrix=undirected_adj)
            coding_tree.build_encoding_tree(k=self.tree_height)

            new_dict = {}
            leaves_ids = coding_tree.leaves
            node_label_list = cluster_data.y.numpy().tolist()
            for k, v in coding_tree.tree_node.items():
                content = v.__dict__
                if k in leaves_ids:
                    content['node_label'] = int(node_label_list[k])
                else:
                    content['node_label'] = int(-1)
                new_dict[k] = content

            l1_community_node_idx = coding_tree.get_l1_community_nodes()

            test_mask = cluster_data.in_cluster_test_mask

            _, type_node_dict, type_node_dict_detail = self._count_L1_community_types(
                new_dict, 
                test_mask=test_mask, 
                l1_community_node_idx=l1_community_node_idx
            )
            cluster_data.l1_community_mapping = type_node_dict_detail

            if self.config['save_original_cluster_data']:
                torch.save(cluster_data, f'{cluster_coding_tree_path}/1_original_cluster_data.pt')

            prompts, result = self._build_condensation_prompt_from_type_node_dict(cluster_data, type_node_dict, new_dict)

            reconstruct_cluster_data = self.reconstruct_condensation_cluster_data(prompts, cluster_data, type_node_dict_detail)
            if self.config['save_reconstructed_cluster_data']:
                torch.save(reconstruct_cluster_data, f'{cluster_coding_tree_path}/2_reconstruct_cluster_data.pt')

            condensation_cluster_data = self._cluster_condensation_llm_answer_generation(reconstruct_cluster_data)
            if self.config['save_condensation_cluster_data']:
                if self.max_token != 100:
                    torch.save(condensation_cluster_data, f'{cluster_coding_tree_path}/3_condensation_cluster_data_max_token@{self.max_token}.pt')
                else:
                    torch.save(condensation_cluster_data, f'{cluster_coding_tree_path}/3_condensation_cluster_data.pt')

    def _count_L1_community_types(self, new_dict, test_mask, l1_community_node_idx):
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

    def _build_condensation_prompt_from_type_node_dict(self, cluster_data, type_node_dict, new_dict):
        set_random_seed(self.tree_seed)

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

        before_total = 0
        after_total = 0
        detail = {}

        for community_type, community_node_idx_list in type_node_dict.items():
            before, after = 0, 0

            template = templates.get(community_type)

            for community_idx in community_node_idx_list:
                temp_template = copy.deepcopy(template)
                leaves_idx_list = list(new_dict[community_idx]['children'])

                if community_type == "B: Pure Background":

                    before += 1
                    after += 0

                elif community_type == "E: Background":
                    background_nodes = [nid for nid in leaves_idx_list if nid in background_nodes_set]
                    before += len(background_nodes)
                    after += 0

                elif community_type == "F: Specific":
                    target_nodes = [nid for nid in leaves_idx_list if nid in test_mask_set]
                    background_nodes = [nid for nid in leaves_idx_list if nid in background_nodes_set]
                    before += 1 + len(background_nodes)

                    target = target_nodes[0]

                    target_info = f"{cluster_data.raw_texts[target]}"
                    prompt_template = temp_template.replace(TARGET_INFO_DEFAULT_TOKEN, target_info)

                    temp_temp_template = copy.deepcopy(prompt_template)
                    already_tokens = self.tokenizer(temp_temp_template.replace(BACKGROUND_INFO_DEFAULT_TOKEN, ''), return_tensors="pt").to(self.device)

                    already_tokens_count = already_tokens["input_ids"].shape[1]
                    last_tokens_count = self.llm_length - already_tokens_count - 1

                    background_info = "\n".join([
                        f"Node {i + 1}: {cluster_data.raw_texts[nid]}"
                        for i, nid in enumerate(background_nodes)
                    ])

                    tokens = self.tokenizer(background_info, return_tensors="pt").to(self.device)
                    token_count = tokens["input_ids"].shape[1]

                    if token_count > last_tokens_count:
                        group_node_list, group_info_list, group_embed_list = self._grouping_sample_com_node(background_nodes, cluster_data, last_tokens_count)
                        after += 1 + len(group_node_list)

                        for group_num, group_node in enumerate(group_node_list):
                            group_prompt_template = copy.deepcopy(prompt_template)

                            final_info = group_info_list[group_num]
                            condensed_embedding = group_embed_list[group_num]

                            prompt = group_prompt_template.replace(BACKGROUND_INFO_DEFAULT_TOKEN, final_info)

                            prompts.append({
                                "type": community_type,
                                "community_idx": int(community_idx),
                                "target_nodes": [int(target)],
                                "background_nodes": list(int(node_idx) for node_idx in group_node),
                                "condensed_bg_embedding": condensed_embedding,
                                "prompt": prompt
                            })

                    else:
                        after += 1 + 1

                        final_info = background_info

                        prompt = prompt_template.replace(BACKGROUND_INFO_DEFAULT_TOKEN, final_info)

                        bg_nodes_embeddings = torch.stack([cluster_data.x[node_idx] for node_idx in background_nodes])
                        condensed_embedding = bg_nodes_embeddings.mean(dim=0)  # 或者使用 max, sum 等方式

                        prompts.append({
                            "type": community_type,
                            "community_node_idx": int(community_idx),
                            "target_nodes": [int(target)],
                            "background_nodes": list(int(node_idx) for node_idx in background_nodes),
                            "condensed_bg_embedding": condensed_embedding,
                            "prompt": prompt
                        })

                elif community_type == "G: Common Shared":
                    target_nodes = [nid for nid in leaves_idx_list if nid in test_mask_set]
                    background_nodes = [nid for nid in leaves_idx_list if nid in background_nodes_set]

                    before += len(target_nodes) + 1
                    after += len(target_nodes) + 1

                    background = background_nodes[0]
                    background_info = f"{cluster_data.raw_texts[background]}"
                    prompt_template = copy.deepcopy(temp_template)
                    prompt = prompt_template.replace(BACKGROUND_INFO_DEFAULT_TOKEN, background_info)

                    temp_temp_template = copy.deepcopy(prompt)
                    already_tokens = self.tokenizer(temp_temp_template.replace(TARGET_INFO_DEFAULT_TOKEN, ''), return_tensors="pt").to(self.device)

                    already_tokens_count = already_tokens["input_ids"].shape[1]
                    last_tokens_count = self.llm_length - already_tokens_count - 1

                    target_info = "\n".join([
                        f"Node {i + 1}: {cluster_data.raw_texts[nid]}"
                        for i, nid in enumerate(target_nodes)
                    ])

                    tokens = self.tokenizer(target_info, return_tensors="pt").to(self.device)
                    token_count = tokens["input_ids"].shape[1]

                    if token_count > last_tokens_count:
                        sampled_nodes, final_info = self._random_sample_com_node(target_nodes, cluster_data, last_tokens_count)
                    else:
                        final_info = target_info

                    prompt = prompt.replace(TARGET_INFO_DEFAULT_TOKEN, final_info)

                    bg_nodes_embeddings = torch.stack([cluster_data.x[node_idx] for node_idx in background_nodes])
                    condensed_embedding = bg_nodes_embeddings.mean(dim=0)

                    prompts.append({
                        "type": community_type,
                        "community_node_idx": int(community_idx),
                        "target_nodes": list(int(node_idx) for node_idx in target_nodes),
                        "background_nodes": list(int(node_idx) for node_idx in background_nodes),
                        "condensed_bg_embedding": condensed_embedding,
                        "prompt": prompt
                    })

                elif community_type == "K: Mixed Community":
                    target_nodes = [nid for nid in leaves_idx_list if nid in test_mask_set]
                    background_nodes = [nid for nid in leaves_idx_list if nid in background_nodes_set]
                    before += len(target_nodes) + len(background_nodes)
                    base_prompt_template_1 = copy.deepcopy(temp_template)
                    target_info = "\n".join([
                        f"Node {i + 1}: {cluster_data.raw_texts[nid]}"
                        for i, nid in enumerate(target_nodes)
                    ])

                    background_info = "\n".join([
                        f"Node {i + 1}: {cluster_data.raw_texts[nid]}"
                        for i, nid in enumerate(background_nodes)
                    ])

                    base_prompt_template_1 = base_prompt_template_1.replace(TARGET_INFO_DEFAULT_TOKEN, target_info)
                    base_prompt_template_1 = base_prompt_template_1.replace(BACKGROUND_INFO_DEFAULT_TOKEN, background_info)
                    already_tokens = self.tokenizer(base_prompt_template_1, return_tensors="pt").to(self.device)
                    already_tokens_count = already_tokens["input_ids"].shape[1]

                    if already_tokens_count < self.llm_length:
                        after += len(target_nodes) + 1
                        bg_nodes_embeddings = torch.stack([cluster_data.x[node_idx] for node_idx in background_nodes])
                        condensed_embedding = bg_nodes_embeddings.mean(dim=0)

                        prompts.append({
                            "type": community_type,
                            "community_node_idx": int(community_idx),
                            "target_nodes": list(int(node_idx) for node_idx in target_nodes),
                            "background_nodes": list(int(node_idx) for node_idx in background_nodes),
                            "condensed_bg_embedding": condensed_embedding,
                            "prompt": base_prompt_template_1
                        })

                    else:
                        target_n = 3
                        if len(target_nodes) <= target_n:
                            base_prompt_template_2 = copy.deepcopy(temp_template)

                            temp_base_prompt_template_2 = copy.deepcopy(base_prompt_template_2)
                            temp_base_prompt_template_2 = temp_base_prompt_template_2.replace(TARGET_INFO_DEFAULT_TOKEN, target_info)

                            already_tokens = self.tokenizer(temp_base_prompt_template_2.replace(BACKGROUND_INFO_DEFAULT_TOKEN, ''), return_tensors="pt").to(self.device)
                            already_tokens_count = already_tokens["input_ids"].shape[1]
                            last_tokens_count = self.llm_length - already_tokens_count - 1

                            if last_tokens_count <= 2000:
                                temp_target_n = 1
                                current_target_node = random.sample(target_nodes, temp_target_n)[0]
                                temp_target_info = f"{cluster_data.raw_texts[current_target_node]}"
                                base_prompt_template_2 = base_prompt_template_2.replace(TARGET_INFO_DEFAULT_TOKEN, temp_target_info)
                                target_node_used = [current_target_node]
                            else:
                                base_prompt_template_2 = temp_base_prompt_template_2
                                target_node_used = target_nodes

                            group_node_list, group_info_list, group_embed_list = self._grouping_sample_com_node(background_nodes, cluster_data, last_tokens_count)
                            after += len(target_nodes) + len(group_node_list)

                            for group_num, group_node in enumerate(group_node_list):
                                group_prompt_template = copy.deepcopy(base_prompt_template_2)

                                final_info = group_info_list[group_num]
                                condensed_embedding = group_embed_list[group_num]

                                prompt = group_prompt_template.replace(BACKGROUND_INFO_DEFAULT_TOKEN, final_info)

                                prompts.append({
                                    "type": community_type,
                                    "community_idx": int(community_idx),
                                    "target_nodes": list(int(node_idx) for node_idx in target_nodes),
                                    "background_nodes": list(int(node_idx) for node_idx in group_node),
                                    "condensed_bg_embedding": condensed_embedding,
                                    "prompt": prompt
                                })
                        else:
                            sampled_bg_nodes = set()
                            round_num = 0
                            after += len(target_nodes)
                            already_sample_target_nodes = set(target_nodes)

                            fail_times = 0

                            while len(sampled_bg_nodes) < len(background_nodes):
                                if fail_times >= 3:
                                    break
                                after += 1
                                round_num += 1
                                if self.dataset == 'wikics':
                                    current_target_nodes = random.sample(already_sample_target_nodes, 1)
                                    already_sample_target_nodes = already_sample_target_nodes - set(current_target_nodes)
                                else:
                                    current_target_nodes = random.sample(target_nodes, target_n)
                                target_info = "\n".join([
                                    f"Node {i + 1}: {cluster_data.raw_texts[nid]}"
                                    for i, nid in enumerate(current_target_nodes)
                                ])

                                base_prompt_template_3 = copy.deepcopy(temp_template)

                                temp_base_prompt_template_3 = copy.deepcopy(base_prompt_template_3)
                                temp_base_prompt_template_3 = temp_base_prompt_template_3.replace(TARGET_INFO_DEFAULT_TOKEN, target_info)

                                already_tokens = self.tokenizer(temp_base_prompt_template_3.replace(BACKGROUND_INFO_DEFAULT_TOKEN, ''), return_tensors="pt").to(self.device)
                                already_tokens_count = already_tokens["input_ids"].shape[1]
                                last_tokens_count = self.llm_length - already_tokens_count - 1

                                if last_tokens_count <= 2000:
                                    temp_target_n = 1
                                    current_target_node = random.sample(target_nodes, temp_target_n)[0]
                                    temp_target_info = f"{cluster_data.raw_texts[current_target_node]}"
                                    base_prompt_template_3 = base_prompt_template_3.replace(TARGET_INFO_DEFAULT_TOKEN, temp_target_info)
                                    target_node_used = [current_target_node]
                                elif last_tokens_count < 0:
                                    current_target_nodes = random.sample(target_nodes, target_n - 1)
                                    target_info = "\n".join([
                                        f"Node {i + 1}: {cluster_data.raw_texts[nid]}"
                                        for i, nid in enumerate(current_target_nodes)
                                    ])
                                    base_prompt_template_3 = copy.deepcopy(base_prompt_template_3)
                                    base_prompt_template_3 = base_prompt_template_3.replace(TARGET_INFO_DEFAULT_TOKEN, target_info)
                                else:
                                    base_prompt_template_3 = temp_base_prompt_template_3
                                    target_node_used = current_target_nodes

                                temp_base_prompt_template_3 = copy.deepcopy(base_prompt_template_3)

                                already_tokens = self.tokenizer(temp_base_prompt_template_3.replace(BACKGROUND_INFO_DEFAULT_TOKEN, ''), return_tensors="pt").to(self.device)
                                already_tokens_count = already_tokens["input_ids"].shape[1]
                                last_tokens_count = self.llm_length - already_tokens_count - 1

                                remaining_bg_nodes = list(set(background_nodes) - sampled_bg_nodes)
                                print(last_tokens_count)
                                sampled_nodes, final_info = self._random_sample_com_node(remaining_bg_nodes, cluster_data, last_tokens_count)

                                print(sampled_nodes)
                                print(cluster_data.x.shape)

                                prompt = base_prompt_template_3.replace(BACKGROUND_INFO_DEFAULT_TOKEN, final_info)

                                sampled_bg_nodes.update(sampled_nodes)

                                if len(sampled_nodes) != 0:
                                    bg_nodes_embeddings = torch.stack([cluster_data.x[nid] for nid in sampled_nodes])
                                    condensed_embedding = bg_nodes_embeddings.mean(dim=0)

                                    prompts.append({
                                        "type": community_type,
                                        "community_idx": int(community_idx),
                                        "target_nodes": list(int(node_idx) for node_idx in target_nodes),
                                        "background_nodes": list(int(node_idx) for node_idx in sampled_nodes),
                                        "condensed_bg_embedding": condensed_embedding,
                                        "prompt": prompt
                                    })
                                else:
                                    fail_times += 1
                                    bg_nodes_embeddings = torch.tensor(-1)
                                    condensed_embedding = bg_nodes_embeddings

                            if fail_times >= 3:
                                base_prompt_template_3 = copy.deepcopy(self.summary_template)
                                already_tokens = self.tokenizer(base_prompt_template_3.replace(DEFAULT_SUMMARY_CORPUS_TOKEN, ''), return_tensors="pt").to(self.device)
                                already_tokens_count = already_tokens["input_ids"].shape[1]
                                last_tokens_count = self.llm_length - already_tokens_count - 1

                                group_node_list, group_info_list, group_embed_list = self._grouping_sample_com_node(background_nodes, cluster_data, last_tokens_count)
                                after += len(group_node_list)
                                for group_num, group_node in enumerate(group_node_list):
                                    group_prompt_template = copy.deepcopy(base_prompt_template_3)

                                    final_info = group_info_list[group_num]
                                    condensed_embedding = group_embed_list[group_num]

                                    prompt = group_prompt_template.replace(DEFAULT_SUMMARY_CORPUS_TOKEN, final_info)

                                    prompts.append({
                                        "type": community_type,
                                        "community_idx": int(community_idx),
                                        "target_nodes": list(int(node_idx) for node_idx in target_nodes),
                                        "background_nodes": list(int(node_idx) for node_idx in group_node),
                                        "condensed_bg_embedding": condensed_embedding,
                                        "prompt": prompt
                                    })

                else:
                    continue

            before_total += before
            after_total += after
            detail[community_type] = {"before": before, "after": after, "reduced": before - after}

        result = {
            "total_before": before_total,
            "total_after": after_total,
            "total_reduced": before_total - after_total,
            "per_type": detail
        }

        return prompts, result


    def _random_sample_com_node(self, background_nodes, cluster_data, last_tokens_count):
        set_random_seed(self.tree_seed)
        sampled_nodes = []
        background_info = ""
        token_budget = last_tokens_count

        shuffled_nodes = list(background_nodes)
        random.shuffle(shuffled_nodes)

        for i, nid in enumerate(shuffled_nodes):
            node_text = f"Node {i + 1}: {cluster_data.raw_texts[nid]}"
            temp_info = background_info + ("\n" if background_info else "") + node_text

            tokens = self.tokenizer(temp_info, return_tensors="pt").to(self.device)
            token_count = tokens["input_ids"].shape[1]

            if token_count > token_budget:
                break

            sampled_nodes.append(nid)
            background_info = temp_info

        return sampled_nodes, background_info

    def _grouping_sample_com_node(self, background_nodes, cluster_data, last_tokens_count):
        set_random_seed(self.tree_seed)

        shuffled_nodes = list(background_nodes)
        random.shuffle(shuffled_nodes)

        group_info_list = []
        group_node_list = []
        group_embed_list = []

        current_node_list = []
        current_info_list = []
        current_text = ""

        for idx, nid in enumerate(shuffled_nodes):
            node_text = f"Node {len(current_node_list) + 1}: {cluster_data.raw_texts[nid]}"
            test_text = current_text + ("\n" if current_text else "") + node_text

            tokens = self.tokenizer(test_text, return_tensors="pt").to(self.device)
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

    def reconstruct_condensation_cluster_data(self, prompts_list, cluster_data, type_node_dict_detail):
        set_random_seed(self.tree_seed)
        ori_x = cluster_data.x
        ori_y = cluster_data.y
        ori_edge_index = cluster_data.edge_index
        label_names = cluster_data.label_names
        label_abbr_names = cluster_data.label_abbr_names
        ori_node_idx = cluster_data.all_cluster_ori_node_idx
        ori_all_raw_texts = cluster_data.raw_texts
        l1_community_mapping = cluster_data.l1_community_mapping
        embedding_dim = ori_x.shape[1]

        test_mask = cluster_data.in_cluster_test_mask
        test_mask_set = set(torch.nonzero(test_mask, as_tuple=False).view(-1).tolist())
        all_nodes_set = set(range(len(test_mask)))
        background_nodes_set = all_nodes_set - test_mask_set

        node_idx_mapping = {}

        new_embeddings = []
        new_raw_texts = []
        new_ys = []
        all_y = []
        new_ori_node_idx = []
        all_ori_node_idx = []

        num = 0
        for node_idx in range(ori_x.shape[0]):
            if node_idx not in background_nodes_set:
                node_idx_mapping[node_idx] = num
                new_embeddings.append(ori_x[node_idx])

                new_ori_node_idx.append(ori_node_idx[node_idx])
                all_ori_node_idx.append(ori_node_idx[node_idx])

                new_raw_texts.append(f'{ori_all_raw_texts[node_idx]}')
                all_y.append(ori_y[node_idx])
                new_ys.append(ori_y[node_idx])

                num += 1

        for prompt in prompts_list:
            bg_nodes = prompt['background_nodes']
            bg_embedding = prompt.get('condensed_bg_embedding', None)

            temp_y = []
            temp_node_idx = []
            for old_idx in bg_nodes:
                node_idx_mapping[old_idx] = num
                temp_y.append(ori_y[old_idx])
                temp_node_idx.append(ori_node_idx[old_idx])

            new_embeddings.append(bg_embedding)
            new_raw_texts.append(prompt['prompt'])
            all_y.append(temp_y)
            all_ori_node_idx.append(temp_node_idx)

            new_ys.append(torch.tensor(-1))
            new_ori_node_idx.append(-1)
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

        new_type_node_dict = {}
        reversed_node_mapping = {v: k for k, v in node_idx_mapping.items()}

        for comm_type, communities in type_node_dict_detail.items():
            new_type_node_dict[comm_type] = {}
            for parent_node, leaf_nodes in communities.items():
                mapped_leaf_nodes = set()
                for node in leaf_nodes:
                    if node in node_idx_mapping:
                        mapped_leaf_nodes.add(int(node_idx_mapping[node]))
                new_type_node_dict[comm_type][parent_node] = list(mapped_leaf_nodes)

        reconstruct_x = torch.stack(new_embeddings) if new_embeddings else torch.empty((0, embedding_dim))
        reconstruct_y = torch.stack(new_ys) if new_ys else torch.empty((0,), dtype=torch.long)


        reconstructed_data = Data(community_cluster_x=cluster_data.x,
                                  community_cluster_edge_index=cluster_data.edge_index,
                                  community_cluster_y=cluster_data.y,
                                  community_cluster_raw_texts=cluster_data.raw_texts,
                                  community_target_mask=cluster_data.ori_test_mask,
                                  graph_target_mask=cluster_data.ori_test_mask,
                                  community_in_cluster_target_mask=cluster_data.in_cluster_test_mask,
                                  community_cluster_com_all_cluster_ori_node_idx=cluster_data.all_cluster_ori_node_idx,
                                  community_in_cluster_target_ori_node_idx=cluster_data.target_ori_node_idx,
                                  community_cluster_l1_community_mapping=l1_community_mapping,
                                  community_cluster_node_mapping=cluster_data.cluster_node_mapping,
                                  reconstruct_x=reconstruct_x,
                                  reconstruct_edge_index=new_edge_index,
                                  reconstruct_y=reconstruct_y,
                                  reconstruct_ori_node_idx=new_ori_node_idx,
                                  reconstruct_raw_texts=new_raw_texts,
                                  reconstruct_target_mask=new_sample_mask,
                                  reconstruct_l1_community_mapping=new_type_node_dict,
                                  reconstruct_condensation_node_mapping=node_idx_mapping,
                                  reconstruct_all_y=all_y,
                                  label_names=label_names,
                                  label_abbr_names=label_abbr_names,
                                  reconstruct_all_ori_node_idx=all_ori_node_idx,
                                  reconstruct_condensation_mask=condensation_mask)

        return reconstructed_data

    def _cluster_condensation_llm_answer_generation(self, re_cluster_data):
        set_random_seed(self.tree_seed)

        condensation_mask = re_cluster_data.reconstruct_condensation_mask
        raw_texts = re_cluster_data.reconstruct_raw_texts
        new_raw_texts = []

        for num, raw_text in tqdm(enumerate(raw_texts)):
            query = raw_text

            if condensation_mask[num]:
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    tokens = self.tokenizer(query, padding=True, return_tensors="pt", truncation=True).to(self.device)
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
                    if "The following list records papers related to the current one, with the relationship being 'citation'" in generated_text:
                        answer = generated_text.split("Please summarize the information above with a short paragraph, and find some common points that can reflect the category of the paper.", 1)[1].strip()
                    else:
                        answer = generated_text.split('\n\nAnswer:', 1)[1].strip()
                    new_raw_texts.append(answer)
                except torch.cuda.OutOfMemoryError:
                    new_raw_texts.append(DEFAULT_CONDENSATION_FAILURE_TOKEN + query)
                    if 'input_ids' in locals():
                        del input_ids
                    if 'output' in locals():
                        del output
                    torch.cuda.empty_cache()
                    gc.collect()
                    torch.cuda.ipc_collect()
                    continue
            else:
                new_raw_texts.append(query)
        re_cluster_data.reconstruct_condense_raw_texts = new_raw_texts
        return re_cluster_data
