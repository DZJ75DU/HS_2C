#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import math
import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

from data_handler import DataHandler
from feature_adapter import UnifiedAdapter
from structure_learner import StructureLearner
from llm_engine import LLMEngine

class HS2CPipeline:
    def __init__(self, args):
        self.args = args

        self.data_handler = DataHandler(args)
        self.adapter = self.data_handler.get_adapter()
        self.structure_learner = StructureLearner(args, self.data_handler.device)
        self.llm_engine = LLMEngine(args, self.data_handler.device)
        
        name_lower = args.dataset.lower()
        if any(x in name_lower for x in ['graph-sst', 'graph-twitter']):
            self.dataset_type = "sentiment"
        elif any(x in name_lower for x in ['bbbp', 'clintox', 'muv', 'sider', 'pcba', 'toxcast']):
            self.dataset_type = "molecule"

    def run(self):
        test_loader = DataLoader(self.data_handler.test_dataset, batch_size=1, shuffle=False)
        dataset_desc = self.data_handler.dataset_desc
        
        results = []
        correct = 0
        total = 0
        total_tasks = 0
        correct_tasks = 0
        
        limit = self.args.max_test_samples or len(test_loader)
        compression_ratios = []

        for idx, data in enumerate(tqdm(test_loader, total=limit)):
            if idx >= limit: break
            
            try:
                if isinstance(data, list): data = data[0]
                
                result = self.process_single_graph(data, idx)
                if 'compression_ratio' in result:
                    compression_ratios.append(result['compression_ratio'])
                
                if result.get('is_multi_task', False):
                    num_tasks = len(result['predictions'])
                    task_correct = sum([1 for t, p in zip(result['true_labels'], result['predictions']) if t == p])
                    total_tasks += num_tasks
                    correct_tasks += task_correct
                    total += 1
                else:
                    total += 1
                    if result['prediction'] == result['true_label']:
                        correct += 1
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing graph {idx}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*50}")
        print(f"Evaluation Complete")

        avg_compression_ratio = sum(compression_ratios) / len(compression_ratios) if compression_ratios else 0.0

        dataset_info = self._get_dataset_info(self.args.dataset)
        task_type = dataset_info.get('task_type', 'binary_classification')
        
        final_accuracy = 0.0
        final_auc = 0.0
        final_precision = 0.0
        final_recall = 0.0
        final_f1 = 0.0
        
        if total_tasks > 0:
            final_accuracy = correct_tasks / total_tasks if total_tasks > 0 else 0.0
            y_true = []
            y_pred = []
            y_prob = []
            
            for r in results:
                if r.get('is_multi_task', False):
                    for t, p in zip(r['true_labels'], r['predictions']):
                        if p != -1:
                            y_true.append(t)
                            y_pred.append(p)
                    
                    if 'probs' in r and len(r['probs']) > 0:
                        for i, prob in enumerate(r['probs']):
                            if i < len(r['predictions']) and r['predictions'][i] != -1:
                                if isinstance(prob, (int, float)):
                                    y_prob.append(prob)
                                else:
                                    y_prob.append(0.5)
            
            if len(y_true) > 0 and len(set(y_true)) > 1:
                final_precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
                final_recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
                final_f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
                try:
                    if len(y_prob) > 0 and len(set(y_true)) > 1:
                        final_auc = roc_auc_score(y_true, y_prob)
                    else:
                        final_auc = 0.0
                except Exception as e:
                    print(f"[Warn] Multi-task AUC calc failed: {e}")
                    final_auc = 0.0
            else:
                final_precision = final_recall = final_f1 = final_auc = 0.0
                tn = fp = fn = tp = 0
            
            print(f"Task-Level Accuracy: {correct_tasks}/{total_tasks} = {100*final_accuracy:.2f}%")
            print(f"Precision: {final_precision:.4f}")
            print(f"Recall: {final_recall:.4f}")
            print(f"F1-Score: {final_f1:.4f}")
            print(f"ROC-AUC: {final_auc:.4f}")
            print(f"Graphs Processed: {total}")
            print(f"Avg Compression Ratio: {avg_compression_ratio:.4f}")
            if len(y_true) > 0:
                print(f"\nConfusion Matrix:")
                print(f"  True Negative:  {tn}")
                print(f"  False Positive: {fp}")
                print(f"  False Negative: {fn}")
                print(f"  True Positive:  {tp}")
        else:
            if total > 0:
                final_accuracy = correct / total
                y_true_raw = []
                for r in results:
                    if 'true_labels' in r:
                        true_labels_val = r['true_labels']
                        if isinstance(true_labels_val, list):
                            y_true_raw.append(true_labels_val)
                        else:
                            y_true_raw.append([true_labels_val])
                    elif 'true_label' in r:
                        true_label_val = r['true_label']
                        if isinstance(true_label_val, list):
                            y_true_raw.append(true_label_val)
                        else:
                            y_true_raw.append([true_label_val])
                    else:
                        y_true_raw.append([0])
                y_pred = [r['prediction'] for r in results]
                is_multi_label_format = any(isinstance(yt, list) and len(yt) > 1 for yt in y_true_raw)
                
                if is_multi_label_format and task_type == 'multi_label_classification':
                    y_true = y_true_raw
                else:
                    y_true = [yt[0] if isinstance(yt, list) and len(yt) > 0 else yt for yt in y_true_raw]
                
                label_dict = dataset_info.get('label_dict', {0: 'No', 1: 'Yes'})
                y_prob = []
                for r in results:
                    if len(label_dict) == 2:
                        y_prob.append(r.get('prob_class_1', 0.5))
                    else:
                        probs = r.get('probs', [0.0] * len(label_dict))
                        y_prob.append(probs)
                
                if is_multi_label_format and task_type == 'multi_label_classification':
                    y_true_flat = [yt[0] if isinstance(yt, list) and len(yt) > 0 else yt for yt in y_true]
                    unique_labels = set(y_true_flat)
                    avg_method = 'macro'
                else:
                    unique_labels = set(y_true)
                    is_multiclass = (
                        task_type in ['multi_class_classification'] or 
                        len(unique_labels) > 2 or 
                        any(l > 1 for l in unique_labels)
                    )
                    
                    if is_multiclass:
                        avg_method = 'weighted'
                    else:
                        avg_method = 'binary'
                
                if len(y_true) > 0:
                    if is_multi_label_format and task_type == 'multi_label_classification':
                        y_true_for_metrics = y_true_flat
                        y_pred_for_metrics = y_pred
                    else:
                        y_true_for_metrics = y_true
                        y_pred_for_metrics = y_pred
                    
                    final_precision = precision_score(y_true_for_metrics, y_pred_for_metrics, average=avg_method, zero_division=0)
                    final_recall = recall_score(y_true_for_metrics, y_pred_for_metrics, average=avg_method, zero_division=0)
                    final_f1 = f1_score(y_true_for_metrics, y_pred_for_metrics, average=avg_method, zero_division=0)
                    
                    if task_type == 'binary_classification':
                        y_true_cm = y_true_for_metrics if is_multi_label_format else y_true
                        cm = confusion_matrix(y_true_cm, y_pred, labels=[0, 1])
                        tn, fp, fn, tp = cm.ravel()
                    else:
                        tn = fp = fn = tp = 0
                    
                    try:
                        if is_multi_label_format and task_type == 'multi_label_classification':
                            y_true_check = y_true_flat if 'y_true_flat' in locals() else y_true
                            has_multiple_classes = len(set(y_true_check)) > 1 if len(y_true_check) > 0 else False
                        else:
                            has_multiple_classes = len(set(y_true)) > 1 if len(y_true) > 0 else False
                        
                        if len(y_true) > 0 and has_multiple_classes:
                            if is_multi_label_format and task_type == 'multi_label_classification':
                                y_true_auc = y_true
                            else:
                                y_true_auc = y_true_flat if (is_multi_label_format and 'y_true_flat' in locals()) else y_true
                            
                            if task_type == 'binary_classification':
                                final_auc = roc_auc_score(y_true_auc, y_prob)
                            elif task_type == 'multi_class_classification':
                                y_prob_array = np.array(y_prob)
                                final_auc = roc_auc_score(y_true_auc, y_prob_array, multi_class='ovr')
                            elif task_type == 'multi_label_classification':
                                if is_multi_label_format:
                                    num_labels = len(y_true[0]) if len(y_true) > 0 and isinstance(y_true[0], list) else 0
                                    num_classes = len(y_prob[0]) if len(y_prob) > 0 and isinstance(y_prob[0], list) else 0
                                    
                                    if num_labels > 0 and num_classes > 0:
                                        y_true_array = np.array(y_true)
                                        
                                        if y_true_array.ndim == 1:
                                            y_true_auc_fallback = y_true_flat if (is_multi_label_format and 'y_true_flat' in locals()) else y_true
                                            y_prob_array = np.array(y_prob)
                                            if y_prob_array.ndim == 1:
                                                final_auc = roc_auc_score(y_true_auc_fallback, y_prob_array)
                                            else:
                                                final_auc = roc_auc_score(y_true_auc_fallback, y_prob_array, multi_class='ovr')
                                        elif y_true_array.ndim == 2:
                                            label_aucs = []
                                            for label_idx in range(num_labels):
                                                y_true_label = y_true_array[:, label_idx]
                                                if len(set(y_true_label)) < 2:
                                                    continue
                                                if label_idx < num_classes:
                                                    y_prob_label = [prob[label_idx] if label_idx < len(prob) else 0.5 
                                                                   for prob in y_prob]
                                                    
                                                    try:
                                                        label_auc = roc_auc_score(y_true_label, y_prob_label)
                                                        label_aucs.append(label_auc)
                                                    except Exception as e:
                                                        print(f"[Warn] AUC calc failed for label {label_idx}: {e}")
                                            
                                            if len(label_aucs) > 0:
                                                final_auc = np.mean(label_aucs)
                                            else:
                                                final_auc = 0.0
                                                print("[Warn] No valid labels for multi-label AUC calculation")
                                    else:
                                        final_auc = 0.0
                                        print("[Warn] Invalid format for multi-label AUC calculation")
                                else:
                                    y_prob_array = np.array(y_prob)
                                    final_auc = roc_auc_score(y_true, y_prob_array, multi_class='ovr')
                            else:
                                y_true_auc = y_true_flat if (is_multi_label_format and 'y_true_flat' in locals()) else y_true
                                
                                if len(label_dict) == 2:
                                    final_auc = roc_auc_score(y_true_auc, y_prob)
                                else:
                                    y_prob_array = np.array(y_prob)
                                    final_auc = roc_auc_score(y_true_auc, y_prob_array, multi_class='ovr')
                        else:
                            final_auc = 0.0
                    except Exception as e:
                        print(f"[Warn] AUC calc failed: {e}")
                else:
                    final_precision = final_recall = final_f1 = final_auc = 0.0
                    tn = fp = fn = tp = 0
                
                print(f"Accuracy: {correct}/{total} = {100*final_accuracy:.2f}%")
                print(f"Precision: {final_precision:.4f}")
                print(f"Recall: {final_recall:.4f}")
                print(f"F1-Score: {final_f1:.4f}")
                print(f"ROC-AUC: {final_auc:.4f}")
                print(f"Avg Compression Ratio: {avg_compression_ratio:.4f}")
                if task_type == 'binary_classification' and len(y_true) > 0:
                    print(f"\nConfusion Matrix:")
                    print(f"  True Negative:  {tn}")
                    print(f"  False Positive: {fp}")
                    print(f"  False Negative: {fn}")
                    print(f"  True Positive:  {tp}")
        
        print(f"{'='*50}\n")
        self._save_results(results, final_accuracy, final_auc, final_precision, final_recall, final_f1, avg_compression_ratio)
        
        return results

    def process_single_graph(self, data, graph_idx):
        node_texts = []
        for i in range(data.num_nodes):
            node_texts.append(self.adapter.get_node_text(i, data, graph_id=graph_idx))
        node_embeddings = self.adapter.get_node_embeddings(
            data, self.llm_engine, graph_id=graph_idx
        )
        enhanced_data = self.structure_learner.knn_enhancement(data, node_embeddings)
        coding_tree, communities = self.structure_learner.build_coding_tree(enhanced_data)
        condensed_info = self._condense_communities(communities, node_texts)
        reconstructed_graph = self._reconstruct_graph(enhanced_data, communities, condensed_info)
        original_nodes = data.num_nodes
        coarsened_nodes = len(reconstructed_graph['super_nodes'])
        compression_ratio = coarsened_nodes / original_nodes if original_nodes > 0 else 1.0
        original_text = ', '.join(node_texts) if self.dataset_type == "molecule" else ' '.join(node_texts)
        dataset_info = self._get_dataset_info(self.args.dataset)
        is_multi_task = (dataset_info and dataset_info.get('task_type') == 'multi_task_binary_classification')
        if is_multi_task:
            if 'task_order' in dataset_info:
                task_ids = dataset_info['task_order']
            else:
                task_ids = list(dataset_info['tasks'].keys())
            
            num_tasks = len(task_ids)
            predictions = []
            true_labels = []
            raw_answers = []
            all_scores = []
            probs = []
            
            print(f"  [Multi-Task] Processing {num_tasks} tasks")
            for task_idx, task_id in enumerate(task_ids):
                if torch.isnan(data.y[0, task_idx]):
                    continue
                prediction, raw_answer, scores = self.llm_engine.predict_graph(
                    original_text, reconstructed_graph, self.dataset_type, 
                    self.args.dataset, dataset_info, task_id
                )
                predictions.append(prediction)
                raw_answers.append(raw_answer)
                all_scores.append(scores)
                true_label = int(data.y[0, task_idx].item())
                true_labels.append(true_label)
                prob_class_1 = 0.5
                if scores and prediction != -1:
                    try:
                        task_info = dataset_info['tasks'][task_id]
                        label_dict = task_info['label_dict']
                        if len(label_dict) == 2:
                            name_0 = label_dict.get(0, 'No')
                            name_1 = label_dict.get(1, 'Yes')
                            loss_0 = scores.get(name_0, float('inf'))
                            loss_1 = scores.get(name_1, float('inf'))
                            prob_class_1 = 1 / (1 + math.exp(loss_1 - loss_0))
                    except Exception as e:
                        print(f"  [Warn] Prob calc failed for task {task_id}: {e}")
                
                probs.append(prob_class_1)
                
                print(f"    Task {task_idx} ({task_id}): True={true_label}, Pred={prediction}, Prob(1)={prob_class_1:.4f}")
            
            num_valid_tasks = len(predictions)
            if num_valid_tasks > 0:
                correct = sum([1 for t, p in zip(true_labels, predictions) if p != -1 and t == p])
                accuracy = correct / num_valid_tasks
            else:
                accuracy = 0.0
            
            return {
                'graph_idx': graph_idx,
                'true_labels': true_labels,
                'predictions': predictions,
                'raw_answers': raw_answers,
                'scores': all_scores,
                'probs': probs,
                'task_ids': task_ids,
                'accuracy': accuracy,
                'is_multi_task': True,
                'compression_ratio': compression_ratio
            }
        else:
            prediction, raw_answer, scores = self.llm_engine.predict_graph(
                original_text, reconstructed_graph, self.dataset_type, 
                self.args.dataset, dataset_info, None
            )
            probs = []
            prob_class_1 = 0.5
            if scores:
                try:
                    label_dict = dataset_info.get('label_dict', {0: 'No', 1: 'Yes'})
                    losses = []
                    sorted_class_ids = sorted(label_dict.keys())
                    for cls_id in sorted_class_ids:
                        label_str = label_dict[cls_id]
                        losses.append(scores.get(label_str, float('inf')))
                    losses_tensor = torch.tensor(losses, dtype=torch.float)
                    probs_tensor = torch.nn.functional.softmax(-losses_tensor, dim=0)
                    probs = probs_tensor.tolist()
                    if len(probs) == 2:
                        prob_class_1 = probs[1]
                    else:
                        prob_class_1 = 0.0
                except Exception as e:
                    print(f"  [Warn] Prob calc failed: {e}")
                    probs = [0.0] * len(label_dict)
                    prob_class_1 = 0.5
            else:
                label_dict = dataset_info.get('label_dict', {0: 'No', 1: 'Yes'})
                probs = [0.0] * len(label_dict)
                prob_class_1 = 0.5
            if data.y.dim() == 0:
                true_label = data.y.item()
                true_labels = [true_label]
            elif data.y.dim() == 1:
                if data.y.size(0) == 1:
                    true_label = data.y.item()
                    true_labels = [true_label]
                else:
                    true_label = data.y.tolist()
                    true_labels = true_label
            else:
                true_label = data.y[0].item() if data.y.size(0) == 1 else data.y[0].tolist()
                true_labels = true_label if isinstance(true_label, list) else [true_label]
            
            display_label = true_labels[0] if isinstance(true_labels, list) and len(true_labels) > 0 else true_label
            print(f"  [Result] True: {display_label}, Pred: {prediction}, Prob(1): {prob_class_1:.4f}")
            
            return {
                'graph_idx': graph_idx,
                'true_label': true_label,
                'true_labels': true_labels,
                'prediction': prediction,
                'prob_class_1': prob_class_1,
                'probs': probs,
                'raw_answer': raw_answer,
                'scores': scores,
                'is_multi_task': False,
                'compression_ratio': compression_ratio
            }
    
    def _get_dataset_info(self, dataset_name):
        from dataset.info import MOLECULENET_DESC, GRAPH_DESC
        name_lower = dataset_name.lower()
        if name_lower.startswith('graph-'):
            lookup_name = name_lower
        else:
            lookup_name = name_lower
        if lookup_name in MOLECULENET_DESC:
            return MOLECULENET_DESC[lookup_name]
        elif lookup_name in GRAPH_DESC:
            return GRAPH_DESC[lookup_name]
        else:
            print(f"Warning: No metadata found for {dataset_name}, using defaults")
            return {
                'dataset_desc': f"Dataset: {dataset_name}",
                'task_type': 'binary_classification',
                'label_dict': {0: 'No', 1: 'Yes'},
                'label_desc': {
                    0: 'No: negative class',
                    1: 'Yes: positive class'
                }
            }

    def _condense_communities(self, communities, node_texts):
        condensed_info = {1: {}}
        if 1 not in communities: return condensed_info
        
        print(f"  [Condense] Processing {len(communities[1])} communities")
        for comm_id, node_list in communities[1].items():
            words = [node_texts[i] for i in node_list if i < len(node_texts)]
            
            if len(words) <= 2:
                desc = ', '.join(words)
            else:
                desc = self.llm_engine.condense_community(words, 1, comm_id, self.dataset_type)
            
            condensed_info[1][comm_id] = {'condensed_desc': desc}
        return condensed_info

    def _reconstruct_graph(self, data, communities, condensed_info):
        target_level = 1
        if target_level not in communities: return {'super_nodes': {}, 'super_edges': {}}
        comms = communities[target_level]
        edge_index = data.edge_index.cpu()
        node_to_comm = {}
        for cid, nodes in comms.items():
            for n in nodes: node_to_comm[n] = cid
        super_edges = {}
        for i in range(edge_index.size(1)):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            if u in node_to_comm and v in node_to_comm:
                cu, cv = node_to_comm[u], node_to_comm[v]
                if cu != cv:
                    if cu not in super_edges: super_edges[cu] = set()
                    super_edges[cu].add(cv)
        for k in super_edges: super_edges[k] = list(super_edges[k])
        
        super_nodes = {}
        if target_level in condensed_info:
            for cid in comms:
                if cid in condensed_info[target_level]:
                    super_nodes[cid] = condensed_info[target_level][cid]['condensed_desc']
                    
        return {'super_nodes': super_nodes, 'super_edges': super_edges}

    def _save_results(self, results, accuracy, auc=0.0, precision=0.0, recall=0.0, f1=0.0, avg_compression=0.0):
        tips = getattr(self.args, 'tips', None)
        output_dir = f"generated_data/{self.args.dataset}/seed@{self.args.data_seed}"
        os.makedirs(output_dir, exist_ok=True)

        if tips:
            filename = f"{self.args.model_name}_hs2c_eval_{tips}.json"
        else:
            filename = f"{self.args.model_name}_hs2c_eval.json"
        path = os.path.join(output_dir, filename)
        
        with open(path, 'w') as f:
            json.dump({
                'dataset': self.args.dataset, 
                'data_seed': self.args.data_seed, 
                'llm_model': self.args.model_name, 
                'accuracy': accuracy, 
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'avg_compression_ratio': avg_compression,
                'results': results, 
                'args': vars(self.args)
            }, f, indent=2)
        print(f"\nResults saved to {path}")
