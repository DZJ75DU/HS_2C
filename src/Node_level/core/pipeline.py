from .graph_enhancer import GraphEnhancer
from .tree_builder import TreeBuilder
from .sampler import SubgraphSampler
from .inference import LLMClassifier
from .reconstruct import GraphReconstructor

import sys
import json
import os
from tqdm import tqdm
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.resolve()
hs2c_path = project_root / '7_HS2C_pipeline'
sys.path.insert(0, str(hs2c_path))
from data_provider.dataset_pro import HS2CDataset
from llm_engine.base_engine import LLMEngine


class HS2CPipeline:
    def __init__(self, config):
        self.config = config
        self.base_dataset = HS2CDataset(config)
        self.llm_engine = LLMEngine(config)
        
        self.enhancer = GraphEnhancer(config)
        self.tree_builder = TreeBuilder(config, self.llm_engine)
        self.reconstructor = GraphReconstructor(config, self.llm_engine)
        self.sampler = SubgraphSampler(config)
        self.classifier = LLMClassifier(config, self.llm_engine)
        
        self.data = None
        self.partition = None
        self.tree = None
        self.summaries = None
        
    def run(self):
        self.base_data = self.base_dataset.get_data()
        if self.base_data is None:
            print("[Error] No data loaded.")
            return
        print("base data loaded.", self.base_data)
        
        self.enhanced_data = self.enhancer.enhance(self.base_data)
        print("enhanced data ready.", self.enhanced_data)
        
        self.partition_tree = self.tree_builder.build(self.enhanced_data, self.partition)

        self.reconstruct_data = self.reconstructor.reconstruct()
        print("reconstructed data ready.", self.reconstruct_data)

        self.sample_data = self.sampler.sample(self.base_data, self.reconstruct_data, self.llm_engine)
        
        results = []
        correct_count = 0
        total_count = 0
        
        for sample_item in tqdm(self.sample_data, desc="Inference"):

            node_idx = sample_item['node_id']
            ori_node_idx = sample_item['ori_node_id']
            prompt = sample_item['query']
            node_label = sample_item['node_label']
            label_whol = sample_item['label_whol']
            label_abbr = sample_item['label_abbr']
            sampled_neighbor_idx = sample_item['node_list']
            neighbor_set = sample_item['all_neighbor']
            pred_result = self.classifier.predict(prompt)
            pred = pred_result['predicted_label']
            scores = pred_result['scores']
            probabilities = pred_result['probabilities']

            is_correct = False
            if str(label_whol).lower().strip() == str(pred).lower().strip():
                is_correct = True
                correct_count += 1
            
            total_count += 1
            
            results.append({
                "node_id": node_idx,
                "ori_node_id": ori_node_idx,
                "query": prompt,
                "node_list": sampled_neighbor_idx,
                "all_neighbor": neighbor_set,
                "prediction": pred,
                "ground_truth_label": node_label,
                "ground_truth": label_whol,
                "ground_truth_abbr": label_abbr,
                "probabilities": probabilities,
                "is_correct": is_correct
            })
            
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        print(f"[Pipeline] Inference Completed. Accuracy: {accuracy:.4f} ({correct_count}/{total_count})")

        save_dir = self.config['result_save_dir']
        os.makedirs(save_dir, exist_ok=True)
        model_name = self.config['model_name']
        llm_seed = self.config['llm_classification_seed']
        os.makedirs(f"{hs2c_path}/{save_dir}/20250606_{self.config['dataset']}/llm_classification_seed@{llm_seed}", exist_ok=True)
        
        save_path = f"{hs2c_path}/{save_dir}/20250606_{self.config['dataset']}/llm_classification_seed@{llm_seed}/{model_name}_hs2c_results.json"
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        save_acc_path = f"{hs2c_path}/{save_dir}/20250606_{self.config['dataset']}/llm_classification_seed@{llm_seed}/{model_name}_hs2c_acc_results.json"
        with open(save_acc_path, 'w', encoding='utf-8') as f:
            json.dump(f"[Pipeline] Inference Completed. Accuracy: {accuracy:.4f} ({correct_count}/{total_count})", f, indent=4, ensure_ascii=False)
            
        print(f"[Pipeline] Results saved to {save_path}")
