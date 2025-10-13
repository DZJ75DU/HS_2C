import torch
import numpy as np
import os
import json
import argparse
from datetime import datetime
from tqdm import tqdm
import copy
from collections import defaultdict

from torch_geometric.data import Data, DataLoader
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_networkx, from_networkx
from torch_cluster import knn_graph
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import networkx as nx
from ogb.graphproppred import PygGraphPropPredDataset

import sys
from dataset import get_dataset
from utils import set_random_seed
from coding_tree_build import PartitionTree
from sklearn.metrics.pairwise import cosine_similarity


class OGBGMolhivHS2C:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("=" * 80)
        print("Loading OGBG-Molhiv dataset...")
        
        dataset = PygGraphPropPredDataset(root=args.data_dir, name='ogbg-molhiv')
        split_idx = dataset.get_idx_split()
        
        self.train_dataset = dataset[split_idx["train"]]
        self.val_dataset = dataset[split_idx["valid"]]
        self.test_dataset = dataset[split_idx["test"]]
        self.num_classes = 2
        self.num_features = dataset.num_features
        
        print(f"✓ Dataset loaded")
        print(f"  - Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
        print(f"  - Num classes: {self.num_classes}, Num features: {self.num_features}")
        print("=" * 80)
        
        self.sentence_tokens = None
        
        self.atom_decoder = {
            1: 'H',   2: 'He',  5: 'B',   6: 'C',   7: 'N',
            8: 'O',   9: 'F',   15: 'P',  16: 'S',  17: 'Cl',
            35: 'Br', 53: 'I',  14: 'Si', 11: 'Na', 12: 'Mg',
            13: 'Al', 19: 'K',  20: 'Ca', 26: 'Fe', 29: 'Cu',
            30: 'Zn', 34: 'Se', 33: 'As'
        }
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            f"{args.model_path}/{args.model_name}"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.model = AutoModelForCausalLM.from_pretrained(
            f"{args.model_path}/{args.model_name}",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        print("✓ LLM loaded")
        
        if args.use_knn_enhancement:
            print("Loading embedding model for KNN enhancement...")
            self.emb_tokenizer = AutoTokenizer.from_pretrained(
                f"{args.model_path}/Bert-base-uncased"
            )
            self.emb_model = AutoModel.from_pretrained(
                f"{args.model_path}/Bert-base-uncased"
            ).to(self.device)
            self.emb_model.eval()
            print("✓ Embedding model loaded")
    
    def get_atom_symbol(self, node_feature):
        index = int(node_feature[0].item())
        actual_atomic_num = index + 1
        return self.atom_decoder.get(actual_atomic_num, f'Z{actual_atomic_num}')
    
    def get_atom_text(self, node_id, data):
        return self.get_atom_symbol(data.x[node_id])
    
    def get_graph_id(self, data, default_id=0):
        for attr in ['name', 'idx', 'original_graph_id', 'graph_idx']:
            if hasattr(data, attr):
                v = getattr(data, attr)
                try:
                    if torch.is_tensor(v):
                        return int(v.item()) if v.dim() == 0 else int(v[0].item())
                    return int(v)
                except:
                    continue
        return int(default_id)
    
    # ==================== Step 1: KNN Enhancement ====================
    
    def knn_enhancement(self, data, atom_symbols):
        if not self.args.use_knn_enhancement:
            return data
        
        print(f"  [KNN] Enhancing graph with structure-entropy based k selection")
        node_embeddings = self._get_node_embeddings(data, atom_symbols)
        
        best_k = self._knn_maxE1_fullgraph(data.edge_index, node_embeddings, data.num_nodes)
        
        enhanced_edge_index = self._add_knn_edges_fullgraph(
            k=best_k,
            node_embeddings=node_embeddings,
            edge_index=data.edge_index,
            num_nodes=data.num_nodes
        )
        
        enhanced_data = Data(
            x=data.x,
            edge_index=enhanced_edge_index,
            y=data.y,
            num_nodes=data.num_nodes
        )
        
        num_new_edges = (enhanced_edge_index.size(1) - data.edge_index.size(1)) // 2
        print(f"  [KNN] Selected k={best_k}, added {num_new_edges} edges")
        
        return enhanced_data

    def _get_node_embeddings(self, data, atom_symbols):
        node_embeddings = []
        for node_id in range(data.num_nodes):
            atom_symbol = atom_symbols[node_id]
            emb = self._get_node_embedding(atom_symbol)
            node_embeddings.append(emb)
        
        node_embeddings = torch.tensor(
            np.vstack(node_embeddings), 
            dtype=torch.float32
        ).to(self.device)
        
        return node_embeddings

    def _get_node_embedding(self, text):
        inputs = self.emb_tokenizer(
            str(text),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.emb_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()


    def _add_knn_edges_fullgraph(self, k, node_embeddings, edge_index, num_nodes):
        edge_index = edge_index.to(self.device)
        try:            
            knn_edge_index = knn_graph(
                node_embeddings, 
                k=k,
                loop=False,
                cosine=True
            )
            
            print(f"    Generated {knn_edge_index.size(1)} directed KNN edges")
            
        except ImportError:
            print("  [KNN] Warning: torch_cluster not installed, using sklearn fallback")
            return self._add_knn_edges_sklearn(k, node_embeddings, edge_index, num_nodes)
        
        knn_edge_index_undirected = torch.cat([
            knn_edge_index,
            knn_edge_index[[1, 0], :]
        ], dim=1)
        
        edge_index_combined = torch.cat([edge_index, knn_edge_index_undirected], dim=1)
        
        edge_index_combined = torch.unique(edge_index_combined, dim=1)
        
        print(f"    Original edges: {edge_index.size(1)}, After KNN: {edge_index_combined.size(1)}")
        
        return edge_index_combined


    def _add_knn_edges_sklearn(self, k, node_embeddings, edge_index, num_nodes):

        edge_index = edge_index.to(self.device)
        
        node_embeddings_np = node_embeddings.cpu().numpy()
        similarity_matrix = cosine_similarity(node_embeddings_np)
        
        new_edges = []
        for node_id in range(num_nodes):
            similarities = similarity_matrix[node_id]
            similarities[node_id] = -1 
            
            top_k_indices = np.argsort(similarities)[-k:]
            
            for neighbor in top_k_indices:
                if neighbor != node_id:
                    new_edges.append([node_id, neighbor])
                    new_edges.append([neighbor, node_id])
        
        if len(new_edges) > 0:
            new_edge_index = torch.tensor(new_edges, dtype=torch.long, device=self.device).t()
            edge_index_combined = torch.cat([edge_index, new_edge_index], dim=1)
            edge_index_combined = torch.unique(edge_index_combined, dim=1)
            return edge_index_combined
        else:
            return edge_index


    def _calc_e1(self, adj):
        adj = adj - torch.diag_embed(torch.diag(adj))

        degree = adj.sum(dim=1)
        vol = adj.sum()
        
        if vol == 0:
            return 0.0
        

        idx = degree.nonzero().reshape(-1)
        g = degree[idx]
        
        e1 = -((g / vol) * torch.log2(g / vol)).sum()
        
        return e1.item()


    def _get_adj_matrix(self, num_nodes, edge_index, weight):
        adj_matrix = torch.zeros((num_nodes, num_nodes), device=self.device)
        adj_matrix[edge_index[0], edge_index[1]] = weight
        
        adj_matrix = adj_matrix - torch.diag_embed(torch.diag(adj_matrix))
        
        return adj_matrix


    def _get_edge_weight(self, node_embeddings, edge_index):        
        num_nodes = node_embeddings.shape[0]
        
        vec1 = node_embeddings[edge_index[0]]
        vec2 = node_embeddings[edge_index[1]]
        
        weight = F.cosine_similarity(vec1, vec2, dim=1)
        
        weight = weight + 1
        
        weight[torch.isnan(weight)] = 0
        
        M = weight.mean() / (2 * num_nodes)
        weight = weight + M
        
        return weight


    def _knn_maxE1_fullgraph(self, edge_index, node_embeddings, num_nodes):
        print("  [KNN] Finding optimal k by maximizing structural entropy (E1) - 全图模式")
        
        old_e1 = 0
        k = 1
        best_k = 1
        max_e1 = 0
        
        weight_original = self._get_edge_weight(node_embeddings, edge_index)
        adj_original = self._get_adj_matrix(num_nodes, edge_index, weight_original)
        e1_original = self._calc_e1(adj_original)
        print(f"    Original graph E1: {e1_original:.4f}")
        
        old_e1 = e1_original
        max_e1 = e1_original
        
        while k < 20:
            try:
                edge_index_k = self._add_knn_edges_fullgraph(
                    k, node_embeddings, edge_index, num_nodes
                )
                
                weight = self._get_edge_weight(node_embeddings, edge_index_k)
                
                adj = self._get_adj_matrix(num_nodes, edge_index_k, weight)
                
                e1 = self._calc_e1(adj)
                
                delta_e1 = e1 - old_e1
                
                print(f"    k={k:2d}, E1={e1:.4f}, ΔE1={delta_e1:+.4f}")
                
                if e1 > max_e1:
                    max_e1 = e1
                    best_k = k
                
                if delta_e1 > 0.1:
                    k += 3
                elif delta_e1 > 0.01:
                    k += 2
                elif delta_e1 > 0.001:
                    k += 1
                else:
                    print(f"    E1 growth stopped (ΔE1 < 0.001)")
                    break
                
                old_e1 = e1
                
            except Exception as e:
                print(f"    Error at k={k}: {e}")
                break
        
        print(f"  [KNN] Selected optimal k={best_k} (max E1={max_e1:.4f}, gain={max_e1-e1_original:.4f})")
        
        return best_k
    
    # ==================== Step 2: Coding Tree Construction ====================
    
    def build_coding_tree(self, data):
        print(f"  [Tree] Building coding tree (height={self.args.tree_height})")
        adj_matrix = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
        undirected_adj = adj_matrix.cpu().numpy()
        
        coding_tree = PartitionTree(adj_matrix=undirected_adj)
        
        coding_tree.build_encoding_tree(k=self.args.tree_height)
        
        hierarchical_communities = self._extract_hierarchical_communities(coding_tree)
        
        print(f"  [Tree] Found {len(hierarchical_communities)} levels")
        for level, comms in hierarchical_communities.items():
            print(f"  [Tree] Level {level}: {len(comms)} communities")
        
        return coding_tree, hierarchical_communities
    
    def _extract_hierarchical_communities(self, coding_tree):
        hierarchical_communities = {}
        tree_node_dict = coding_tree.tree_node
        
        hierarchical_communities[0] = {}
        for leaf_id in coding_tree.leaves:
            node = tree_node_dict[leaf_id]
            if hasattr(node, 'partition') and node.partition:
                hierarchical_communities[0][leaf_id] = list(node.partition)
        
        l1_community_node_idx = coding_tree.get_l1_community_nodes()
        hierarchical_communities[1] = {}
        
        for comm_idx, comm_id in enumerate(l1_community_node_idx):
            node = tree_node_dict[comm_id]
            if hasattr(node, 'partition') and node.partition:
                hierarchical_communities[1][comm_idx] = list(node.partition)
        
        print(f"  [Tree] Extracted Level 0: {len(hierarchical_communities[0])} leaves")
        print(f"  [Tree] Extracted Level 1: {len(hierarchical_communities[1])} communities")
        
        return hierarchical_communities
    
    # ==================== Step 3: Community Condensation ====================
    
    def community_condensation(self, data, atom_symbols, hierarchical_communities):
        print(f"  [Condense] Condensing Level 1 communities using LLM")
        
        condensed_info = {}
        target_level = 1
        
        if target_level not in hierarchical_communities:
            print(f"  [Condense] Warning: Level {target_level} not found")
            return condensed_info
        
        condensed_info[target_level] = {}
        communities = hierarchical_communities[target_level]
        
        print(f"  [Condense] Processing {len(communities)} Level 1 communities")
        
        for comm_id, node_list in communities.items():
            comm_atoms = []
            for node_id in node_list:
                if node_id < len(atom_symbols):
                    comm_atoms.append(atom_symbols[node_id])
            
            if len(comm_atoms) <= 2:
                condensed_desc = ', '.join(comm_atoms)
                print(f"    Community {comm_id}: {len(comm_atoms)} atoms → kept as-is")
            else:
                condensation_prompt = self._create_condensation_prompt(
                    comm_atoms, target_level, comm_id
                )
                condensed_desc = self._llm_condense(condensation_prompt)
                print(f"    Community {comm_id}: {len(comm_atoms)} atoms → condensed")
            
            condensed_info[target_level][comm_id] = {
                'nodes': node_list,
                'atoms': comm_atoms,
                'condensed_desc': condensed_desc
            }
        
        print(f"  [Condense] Condensed {len(communities)} Level 1 communities")
        
        return condensed_info
    
    def _create_condensation_prompt(self, atoms, level, comm_id):
        prompt = (
            f"Analyze the following atoms in a molecular substructure:\n"
            f"{', '.join(atoms)}\n\n"
            f"Task: Describe this substructure in ONE concise phrase (5-8 words maximum).\n"
            f"Focus on:\n"
            f"1. The functional group or structural motif\n"
            f"2. Any notable chemical properties\n"
            f"3. Relationships between the atoms\n\n"
            f"Output a short descriptive phrase:"
        )
        return prompt
    
    def _llm_condense(self, prompt):
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.args.condensation_max_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        condensed = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "Output a short descriptive phrase:" in condensed:
            condensed = condensed.split("Output a short descriptive phrase:")[-1].strip()
        elif "Output:" in condensed:
            condensed = condensed.split("Output:")[-1].strip()
        else:
            lines = [l.strip() for l in condensed.split('\n') if l.strip()]
            if lines:
                condensed = lines[-1]

        condensed = condensed.lstrip(':').lstrip('.').lstrip('"').rstrip('"').strip()

        condensed = ' '.join(condensed.split()[:10])
        
        return condensed
    
    # ==================== Step 4: Graph Reconstruction ====================
    
    def graph_reconstruction(self, data, coding_tree, hierarchical_communities, condensed_info):
        print(f"  [Reconstruct] Reconstructing graph from Level 1 communities")
        
        target_level = 1
        
        if target_level not in hierarchical_communities:
            print(f"  [Reconstruct] Warning: Level {target_level} not found")
            return {'super_nodes': {}, 'super_edges': {}, 'level': target_level}
        
        communities = hierarchical_communities[target_level]
        
        community_connections = defaultdict(set)
        edge_index = data.edge_index.cpu()
        
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            
            src_comm = self._find_community(src, communities)
            dst_comm = self._find_community(dst, communities)
            
            if src_comm is not None and dst_comm is not None and src_comm != dst_comm:
                community_connections[src_comm].add(dst_comm)
        
        reconstructed_graph = {
            'level': target_level,
            'super_nodes': {},
            'super_edges': {},
            'num_communities': len(communities),
            'num_original_nodes': data.num_nodes,
            'num_original_edges': data.edge_index.size(1) // 2
        }
        
        if target_level in condensed_info:
            for comm_id in communities.keys():
                if comm_id in condensed_info[target_level]:
                    reconstructed_graph['super_nodes'][comm_id] = \
                        condensed_info[target_level][comm_id]['condensed_desc']
                    reconstructed_graph['super_edges'][comm_id] = \
                        list(community_connections.get(comm_id, []))
        
        print(f"  [Reconstruct] Created {len(reconstructed_graph['super_nodes'])} super-nodes from Level 1")
        
        return reconstructed_graph
    
    def _find_community(self, node_id, communities):
        for comm_id, node_list in communities.items():
            if node_id in node_list:
                return comm_id
        return None
    
    # ==================== Step 5: LLM Classification ====================
    
    def create_classification_prompt(self, data, atom_symbols, reconstructed_graph, condensed_info):
        original_atoms = ', '.join(atom_symbols)
        
        super_nodes = reconstructed_graph['super_nodes']
        super_edges = reconstructed_graph['super_edges']
        
        hierarchical_desc_parts = []
        for comm_id in sorted(super_nodes.keys()):
            hierarchical_desc_parts.append(super_nodes[comm_id])
        hierarchical_desc = ' → '.join(hierarchical_desc_parts)
        
        prompt = (
            "You are a molecular property prediction expert.\n"
            "Analyze the following molecule using its hierarchical structure:\n\n"
            f"[Molecule Composition]\n{original_atoms}\n\n"
            f"[Hierarchical Structure]\n{hierarchical_desc}\n\n"
            "[Structural Connections]\n"
        )
        
        for comm_id in sorted(super_nodes.keys()):
            connections = super_edges.get(comm_id, [])
            if connections:
                for dst_comm in sorted(connections):
                    src_text = super_nodes[comm_id]
                    dst_text = super_nodes[dst_comm]
                    prompt += f'"{src_text}" connected to "{dst_text}"\n'
        
        if not any(super_edges.values()):
            prompt += "(No cross-group connections - independent structural units)\n"
        
        prompt += (
            "\n[Question] Based on the molecule structure and its hierarchical organization, "
            "does this molecule have HIV inhibition activity?\n"
            "Output EXACTLY one word:\n"
            "1. Active\n"
            "2. Inactive\n\n"
            "[Answer]"
        )
        
        return prompt
    
    def classify_graph(self, prompt):
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.args.max_length
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "[Answer]" in full_output:
            answer = full_output.split("[Answer]")[-1].strip()
        else:
            answer = full_output.strip()
        
        answer_lower = answer.lower()
        
        if answer_lower.strip() in ['active', 'inactive']:
            prediction = 1 if answer_lower.strip() == 'active' else 0
        elif ('active' in answer_lower and 
            ('inactive' not in answer_lower or answer_lower.find('active') < answer_lower.find('inactive'))):
            prediction = 1
        elif ('inactive' in answer_lower and 
            ('active' not in answer_lower or answer_lower.find('inactive') < answer_lower.find('active'))):
            prediction = 0
        elif '1' in answer_lower and '2' not in answer_lower:
            prediction = 1
        elif '2' in answer_lower and '1' not in answer_lower:
            prediction = 0
        else:
            prediction = -1
        
        return prediction, answer
    
    # ==================== Main Pipeline ====================
    
    def process_single_graph(self, data, graph_idx):
        atom_symbols = []
        for node_id in range(data.num_nodes):
            atom_symbol = self.get_atom_symbol(data.x[node_id])
            atom_symbols.append(atom_symbol)
        
        true_label = data.y.item()
        
        print(f"\n{'='*80}")
        print(f"Processing Molecule {graph_idx} (atoms={data.num_nodes}, bonds={data.edge_index.size(1)//2})")
        print(f"{'='*80}")
        
        enhanced_data = self.knn_enhancement(data, atom_symbols)
        
        coding_tree, hierarchical_communities = self.build_coding_tree(enhanced_data)
        
        condensed_info = self.community_condensation(
            enhanced_data, atom_symbols, hierarchical_communities
        )
        
        reconstructed_graph = self.graph_reconstruction(
            enhanced_data, coding_tree, hierarchical_communities, condensed_info
        )
        
        prompt = self.create_classification_prompt(
            enhanced_data, atom_symbols, reconstructed_graph, condensed_info
        )
        
        prediction, raw_answer = self.classify_graph(prompt)
        
        print(f"\n  [Result] True: {true_label}, Pred: {prediction}, Answer: {raw_answer}")
        
        return {
            'graph_idx': graph_idx,
            'true_label': true_label,
            'prediction': prediction,
            'raw_answer': raw_answer,
            'num_super_nodes': len(reconstructed_graph['super_nodes']),
            'prompt': prompt
        }
    
    def evaluate(self):
        test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False)
        
        results = []
        correct = 0
        total = 0
        unparseable = 0

        actual_test_size = min(len(self.test_dataset), self.args.max_test_samples) \len(self.test_dataset)
        
        for idx, data in enumerate(tqdm(test_loader, desc="Evaluating", total=actual_test_size)):
            
            try:
                if isinstance(data, list) and len(data) == 1:
                    data = data[0]
                    
                result = self.process_single_graph(data, idx)
                
                total += 1
                if result['prediction'] == -1:
                    unparseable += 1
                elif result['prediction'] == result['true_label']:
                    correct += 1
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing graph {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        accuracy = correct / total if total > 0 else 0
        
        print(f"\n{'='*80}")
        print(f"Final Results:")
        print(f"  Total: {total}")
        print(f"  Correct: {correct}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Unparseable: {unparseable}")
        print(f"{'='*80}")
        
        return results, accuracy


def main():
    parser = argparse.ArgumentParser(
        description='Graph-Level HS2C Classification for OGBG-Molhiv'
    )
    parser.add_argument('--dataset', type=str, default='ogbg-molhiv')
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--data_seed', type=int, default=42)
    parser.add_argument('--use_knn_enhancement', action='store_true')
    parser.add_argument('--tree_height', type=int, default=2)
    parser.add_argument('--condensation_max_tokens', type=int, default=15)
    parser.add_argument('--model_name', type=str, default='Llama-3.2-3b-Instruct')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--save_dir', type=str, default='./results/hs2c_ogbg_molhiv')
    
    args = parser.parse_args()
    classifier = OGBGMolhivHS2C(args)
    
    results, accuracy = classifier.evaluate()
    
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(
        args.save_dir,
        f"hs2c_tree{args.tree_height}_{'knn' if args.use_knn_enhancement else 'noknn'}.json"
    )
    
    with open(save_path, 'w') as f:
        json.dump({
            'accuracy': accuracy,
            'results': results,
            'args': vars(args)
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {save_path}")
    
    return accuracy


if __name__ == "__main__":
    main()
