import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModel
from torch_geometric.utils import k_hop_subgraph, to_undirected, add_self_loops
from torch_cluster import knn_graph
import networkx as nx
import community as community_louvain
import igraph as ig
import leidenalg
from collections import defaultdict
import faiss
import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent.parent.resolve()
hs2c_path = project_root / '7_HS2C_pipeline'
sys.path.insert(0, str(hs2c_path))
from utils.utils import set_random_seed

class GraphEnhancer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))
        self.model_path = config.get('model_path', '')
        self.node_embed_batch = config.get('node_embedding_batch_size', 10)
        self.cluster_seed = config.get('cluster_seed', 0)
        self.resolution = config.get('resolution', 1.0)
        self.model_name = config["model_name"]

    def enhance(self, data):
        self.experiment_pre_name = f"{hs2c_path}/generated_data/{self.config['new_date']}_{self.config['dataset']}"
        os.makedirs(f'{self.experiment_pre_name}', exist_ok=True)
        self.experiment_name = f"cluster@{self.config['cluster_name']}_seed@{self.config['cluster_seed']}_resolution@{self.config['resolution']}"
        self.experiment_path = f'{self.experiment_pre_name}/{self.experiment_name}'
        os.makedirs(self.experiment_path, exist_ok=True)
        self.cluster_save_path = f'{self.experiment_path}/cluster_split'
        os.makedirs(self.cluster_save_path, exist_ok=True)
        all_node_idx = torch.arange(data.x.shape[0])
        self.sample_node_idx = all_node_idx[data.test_mask]
        print("sample_node_idx", len(self.sample_node_idx))

        if self.config['point'] <= 0:
            print("[GraphEnhancer] Enhancing graph with adaptive KNN...")
            target_data = self._find_target_graph(data)
            node_embeddings = self._get_node_embeddings(target_data)
            if node_embeddings is None:
                print("[Warn] No embeddings available; skip enhancement.")
                return data

            best_k = self._knn_maxE1(target_data.edge_index, node_embeddings)
            best_k = best_k

            print(f"DEBUG: target_data.edge_index max: {target_data.edge_index.max()}, shape: {target_data.edge_index.shape}")
            enhanced_edge_index = self._add_knn_pyg(best_k, node_embeddings, target_data.edge_index)
            print(f"DEBUG: enhanced_edge_index max: {enhanced_edge_index.max()}, shape: {enhanced_edge_index.shape}")
            target_data.knn_edge_index = enhanced_edge_index

            if self.config['save_target_graph']:
                torch.save(target_data, f"{self.experiment_pre_name}/{self.config['new_date']}_{self.config['dataset']}_knn_target_graph.pt")

            print(f"✓ Graph enhanced. Edges: {target_data.edge_index.shape[1] if target_data.edge_index is not None else 'unknown'} (k={best_k})")
            self.data = target_data
            self.cluster_detection()
        else:
            print("[GraphEnhancer] Loading enhanced graph...")
            target_data = torch.load(f"{self.experiment_pre_name}/{self.config['new_date']}_{self.config['dataset']}_knn_target_graph.pt", weights_only=False)

        return target_data
    
    # ----------------------------------------------------------------------------------------------
    def _get_node_embeddings(self, data):
        """Return node embeddings. Prefer BERT over raw_texts; fallback to data.x."""
        if hasattr(data, 'raw_texts') and data.raw_texts is not None and len(data.raw_texts) > 0:
            return self._encode_with_bert(data.raw_texts)

        return None

    def _encode_with_bert(self, texts):
        if not self.model_path:
            print("[Warn] model_path missing; cannot build BERT embeddings.")
            return None

        bert_model = AutoModel.from_pretrained(f"{self.model_path}/Bert-base-uncased").to(self.device)
        bert_tokenizer = AutoTokenizer.from_pretrained(f"{self.model_path}/Bert-base-uncased")
        bert_model.eval()

        embeddings = []
        batch_size = max(1, int(self.node_embed_batch))
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                inputs = bert_tokenizer(
                    batch_texts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.config['device'])
                outputs = bert_model(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                embeddings.append(cls_embeddings.cpu())

        return torch.cat(embeddings, dim=0)

    def _add_knn_pyg(self, k, node_embed, edge_index):
        node_embed = node_embed.to(self.device)
        edge_index = edge_index.to(self.device)

        num_nodes = node_embed.shape[0]
        connected_nodes = torch.unique(edge_index)
        all_nodes = torch.arange(num_nodes, device=self.device)
        isolated_nodes = torch.tensor(list(set(all_nodes.tolist()) - set(connected_nodes.tolist())), device=self.device)

        if len(isolated_nodes) == 0:
            print("[GraphEnhancer] No isolated nodes found; keep original edges.")
            return edge_index if edge_index.shape[0] == 2 else edge_index.t()

        embeddings_np = F.normalize(node_embed, p=2, dim=1).cpu().detach().numpy()
        d_dim = embeddings_np.shape[1]
        index = faiss.IndexFlatIP(d_dim) 
        index.add(embeddings_np)
        D, I = index.search(embeddings_np, k + 1)
        src_idx = np.repeat(np.arange(num_nodes), k + 1)
        dst_idx = I.flatten()
        knn_row = torch.from_numpy(src_idx).to(self.device)
        knn_col = torch.from_numpy(dst_idx).to(self.device).long()
        mask_self = knn_row != knn_col
        knn_edge_index = torch.stack([knn_row[mask_self], knn_col[mask_self]], dim=0)

        mask = (torch.isin(knn_edge_index[0], isolated_nodes) | torch.isin(knn_edge_index[1], isolated_nodes))
        knn_edge_index_filtered = knn_edge_index[:, mask]

        knn_edge_index_filtered = torch.cat(
            [knn_edge_index_filtered, knn_edge_index_filtered[[1, 0], :]], dim=1
        )

        base_edge_index = edge_index if edge_index.shape[0] == 2 else edge_index.t()
        print("edge_index", edge_index.shape)
        print("knn_edge_index_filtered", knn_edge_index_filtered.shape)
        edge_index_combined = torch.cat([base_edge_index, knn_edge_index_filtered], dim=1)
        edge_index_combined = torch.unique(edge_index_combined, dim=1)
        print("edge_index_combined", edge_index_combined.shape)
        return edge_index_combined

    def _calc_e1(self, adj: torch.Tensor):
        adj = adj - torch.diag_embed(torch.diag(adj))
        degree = adj.sum(dim=1)
        vol = adj.sum()
        idx = degree.nonzero().reshape(-1)
        g = degree[idx]
        return -((g / vol) * torch.log2(g / vol)).sum()

    def _get_adj_matrix(self, node_num, edge_index, weight) -> torch.Tensor:
        adj_matrix = torch.zeros((node_num, node_num), device=edge_index.device)
        weight = weight.to(edge_index.device)
        adj_matrix[edge_index[0], edge_index[1]] = weight
        adj_matrix = adj_matrix - torch.diag_embed(torch.diag(adj_matrix))
        return adj_matrix

    def _get_weight(self, node_embedding, edge_index):
        edge_index = edge_index.to(node_embedding.device)
        links = node_embedding[edge_index]
        vec1 = links[0]
        vec2 = links[1]
        weight = F.cosine_similarity(vec1, vec2, dim=1)  # shape: [E]
        weight = weight.clone().detach() + 1
        weight[torch.isnan(weight)] = 0
        M = weight.mean() / (2 * node_embedding.shape[0])
        weight = weight + M
        return weight

    def _knn_maxE1(self, edge_index: torch.Tensor, node_embedding: torch.Tensor):
        old_e1 = 0.0
        node_num = node_embedding.shape[0]
        k = 1

        while k < 50:
            edge_index_k = self._add_knn_pyg(k, node_embedding, edge_index)
            weight = self._get_weight(node_embedding, edge_index_k)
            adj = self._get_adj_matrix(node_num, edge_index_k, weight)
            e1 = self._calc_e1(adj)

            if e1 - old_e1 > 0.1:
                k += 5
            elif e1 - old_e1 > 0.01:
                k += 3
            elif e1 - old_e1 > 0.001:
                k += 1
            else:
                break
            old_e1 = e1

        print(f"[GraphEnhancer] max1SE k: {k}")
        return k

    def _to_nx(self, data):
        set_random_seed(self.cluster_seed)
        edge_index = getattr(data, 'knn_edge_index', None)
        if edge_index is None:
            edge_index = data.edge_index
        if edge_index.shape[0] == 2:
            edge_index = edge_index.t()
        G = nx.Graph()
        G.add_nodes_from(range(data.num_nodes))
        edges = list(map(tuple, edge_index.tolist()))
        G.add_edges_from(edges)
        return G

    def _run_louvain(self, data):
        set_random_seed(self.cluster_seed)
        G = self._to_nx(data)
        partition = community_louvain.best_partition(G)
        communities = defaultdict(list)
        for node, cid in partition.items():
            communities[cid].append(node)
        return dict(communities)

    def _run_leiden(self, data):
        set_random_seed(self.cluster_seed)
        G_nx = self._to_nx(data)
        G_ig = ig.Graph(edges=list(G_nx.edges()), directed=False)
        G_ig.vs["name"] = list(range(data.num_nodes))
        partition = leidenalg.find_partition(
            G_ig,
            leidenalg.ModularityVertexPartition,
            seed=self.cluster_seed
        )
        return {i: list(comm) for i, comm in enumerate(partition)}

    def _run_leiden_with_resolution(self, data):
        set_random_seed(self.cluster_seed)
        G_nx = self._to_nx(data)
        G_ig = ig.Graph(edges=list(G_nx.edges()), directed=False)
        G_ig.vs["name"] = list(range(data.num_nodes))
        part = leidenalg.find_partition(
            G_ig,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=self.resolution,
            seed=self.cluster_seed
        )
        communities = {i: list(comm) for i, comm in enumerate(part)}
        print({i: len(nodes) for i, nodes in communities.items()})
        return communities

    def _extract_cluster_subgraph(self, edge_index, cluster_nodes):
        cluster_set = set(cluster_nodes.tolist())
        src, dst = edge_index
        mask = [(s.item() in cluster_set and d.item() in cluster_set) for s, d in zip(src, dst)]
        mask = torch.tensor(mask, dtype=torch.bool)
        sub_edge_index = edge_index[:, mask]
        return sub_edge_index

    def _find_target_graph(self, data=None):
        subset, sub_edge_index, _, _ = k_hop_subgraph(
            node_idx=self.sample_node_idx,
            num_hops=1,
            edge_index=data.edge_index,
            relabel_nodes=True
        )

        sub_x = data.x[subset]
        sub_y = data.y[subset]
        sub_test_mask = data.test_mask[subset]
        sub_node_idx_list = subset.tolist()

        if len(data.raw_texts) != 0:
            sub_raw_texts = [data.raw_texts[i] for i in sub_node_idx_list]
        else:
            sub_raw_texts = []

        if hasattr(data, 'label_names'):
            label_names = data.label_names
            if 'arxiv' in self.config['dataset'].lower():
                label_abbr_names = self.config['label_abbr_dict']
            else:
                label_abbr_names = label_names

        else:
            label_names = self.config['label_whol_dict']
            label_abbr_names = self.config['label_abbr_dict']

        sub_data = Data(x=sub_x, edge_index=sub_edge_index, y=sub_y, test_mask=sub_test_mask, raw_texts=sub_raw_texts, ori_node_idx=subset, label_names=label_names, label_abbr_names=label_abbr_names)
        print(sub_data)

        return sub_data

    def cluster_detection(self):
        set_random_seed(self.cluster_seed)
        if self.resolution:
            clusters = self._run_leiden_with_resolution(self.data)
            cluster_data_path = f'{self.cluster_save_path}/cluster_data'
            os.makedirs(cluster_data_path, exist_ok=True)

            cluster_id_list = list(clusters.keys())
            for num, cluster_id in enumerate(cluster_id_list):
                cluster_num_path = f'{cluster_data_path}/cluster@{num}'
                print(f'cluster@{num}--{len(clusters[cluster_id])}')
                os.makedirs(cluster_num_path, exist_ok=True)
                cluster_node_idx = clusters[cluster_id]

                sample_mask = torch.zeros(len(cluster_node_idx), dtype=torch.bool)
                sample_ori_node_idx_list = []

                for i, node_idx in enumerate(cluster_node_idx):
                    if node_idx in self.sample_node_idx:
                        sample_mask[i] = True
                        sample_ori_node_idx_list.append(node_idx)
                
                cluster_node_idx = torch.tensor(cluster_node_idx)
                node_id_map = {nid.item(): idx for idx, nid in enumerate(cluster_node_idx)}

                cluster_x = self.data.x[cluster_node_idx]
                cluster_y = self.data.y[cluster_node_idx]

                ori_ori_node_idx_list = [self.data.ori_node_idx[i] for i in cluster_node_idx.tolist()]
                
                if len(self.data.test_mask) == 0:
                    cluster_test_mask = []
                else:
                    cluster_test_mask = self.data.test_mask[cluster_node_idx]

                if len(self.data.raw_texts) == 0:
                    cluster_raw_text_list = []
                else:
                    cluster_raw_text_list = [self.data.raw_texts[i] for i in cluster_node_idx.tolist()]

                raw_edge_index = self._extract_cluster_subgraph(self.data.knn_edge_index, cluster_node_idx)
                relabelled_edge_index = torch.tensor(
                    [[node_id_map[s.item()] for s in raw_edge_index[0]],
                     [node_id_map[d.item()] for d in raw_edge_index[1]]],
                    dtype=torch.long
                )

                if self.data.label_names:
                    label_names = self.data.label_names
                    label_abbr_names = self.data.label_abbr_names
                else:
                    label_names = self.config['label_whol_dict']
                    label_abbr_names = self.config['label_abbr_dict']

                cluster_data = Data(
                    x=cluster_x,
                    edge_index=relabelled_edge_index,
                    y=cluster_y,
                    label_names=label_names,
                    label_abbr_names=label_abbr_names,
                    raw_texts=cluster_raw_text_list,
                    ori_test_mask=self.data.test_mask,
                    in_cluster_test_mask=sample_mask,
                    all_cluster_ori_node_idx=cluster_node_idx,
                    ori_ori_target_node_idx=ori_ori_node_idx_list,
                    target_ori_node_idx=sample_ori_node_idx_list,
                    knn_target_ori_node_idx=sample_ori_node_idx_list,
                    cluster_node_mapping=node_id_map
                )

                if self.config['save_cluster']:
                    torch.save(cluster_data, f'{cluster_num_path}/cluster@{num}_data.pt')
        else:
            cluster_data_path = f'{self.cluster_save_path}/cluster_data'
            os.makedirs(cluster_data_path, exist_ok=True)
            cluster_num_path = f'{cluster_data_path}/cluster@{0}'
            os.makedirs(cluster_num_path, exist_ok=True)
            all_node_idx = torch.arange(self.data.x.shape[0])
            target_node_idx = all_node_idx[self.data.test_mask]
            ori_ori_node_idx_list = [self.data.ori_node_idx[i] for i in target_node_idx.tolist()]
            if self.data.label_names:
                label_names = self.data.label_names
                label_abbr_names = self.data.label_abbr_names
            else:
                label_names = self.config['label_whol_dict']
                label_abbr_names = self.config['label_abbr_dict']

            cluster_data = Data(x=self.data.x,
                                edge_index=self.data.knn_edge_index,
                                y=self.data.y,
                                label_names=label_names,
                                label_abbr_names=label_abbr_names,
                                raw_texts=self.data.raw_texts,
                                ori_test_mask=self.data.test_mask,
                                in_cluster_test_mask=self.data.test_mask,
                                all_cluster_ori_node_idx=all_node_idx,
                                ori_ori_target_node_idx=target_node_idx,
                                target_ori_node_idx=target_node_idx,
                                knn_target_ori_node_idx=target_node_idx,
                                cluster_node_mapping=[])

            if self.config['save_cluster']:
                torch.save(cluster_data, f'{cluster_num_path}/cluster@{0}_data.pt')
