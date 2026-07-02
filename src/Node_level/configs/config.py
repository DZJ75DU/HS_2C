import sys
import torch
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.resolve()
hs2c_path = project_root / '7_HS2C_pipeline'
data_root = project_root / '0_baseline' / 'generated_data'
sys.path.insert(0, str(hs2c_path))

from data_provider.constants import DEFAULT_DATASET_PATHS, K_HOP_NEIGHBOR_LIMITS, RESOLUTION, CATEGORY_LISTS, LABEL_WHOL_DICT, LABEL_ABBR_DICT

class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Baseline Pipeline Configuration")
        self._add_arguments()
        self.args = self.parser.parse_args()
        self.args.data_root = str(data_root)
        self._post_process_args()
        self.get_device()

    def _add_arguments(self):
        # Experiment Info
        self.parser.add_argument('--new_date', type=str, default='20250606')
        self.parser.add_argument('--task', type=str, default="classification")
        self.parser.add_argument('--dataset', type=str, default="ogb_arxiv", choices=["ogb_arxiv", "arxiv_2023", "citeseer", "instagram", "reddit", "products", "book_hete", "book_homo", "dblp_hete", "dblp_homo"])
        self.parser.add_argument('--data_preprocess_seed', type=int, default=0)
        self.parser.add_argument('--sample_neighbor_seed', type=int, default=0)
        self.parser.add_argument('--method', default="rag", choices=["random", "number", "degree", "rag"])
        self.parser.add_argument('--neighbor_sample_method', type=str, default='rag')
        
        # Model Info
        self.parser.add_argument('--model_path', type=str, default="/A_base_model")
        self.parser.add_argument('--model_name', type=str, default="Llama-3.1-8b-Instruct")
        self.parser.add_argument('--embedding_model_name', type=str, default="Bert-base-uncased")
        self.parser.add_argument('--llm_length', type=int, default=4000)
        self.parser.add_argument('--max_token', type=int, default=200)
        
        # Sampling Config
        self.parser.add_argument('--hop', type=int, default=1)
        self.parser.add_argument('--tips', type=str, default='t0')
        self.parser.add_argument('--new_tips', type=str, default='t0')

        # Enhancement Config
        self.parser.add_argument('--save_target_graph', action='store_true')
        self.parser.add_argument('--save_original_cluster_data', action='store_true')
        self.parser.add_argument('--node_embedding_batch_size', type=int, default=10)
        self.parser.add_argument('--cluster_name', type=str, default='leiden')
        self.parser.add_argument('--resolution', type=float, default=None)
        self.parser.add_argument('--save_cluster', action='store_true')
        self.parser.add_argument('--cluster_seed', type=int, default=0)
        
        # Coding Tree Config
        self.parser.add_argument('--tree_height', type=int, default=2)

        # LLM Condensation Config
        self.parser.add_argument('--condensation_seed', type=int, default=0)

        # Reconstruction Config
        self.parser.add_argument('--save_reconstructed_cluster_data', action='store_true')
        self.parser.add_argument('--save_condensation_cluster_data', action='store_true')

        # Plug_in Config
        self.parser.add_argument('--plug_in_output_file', type=str)
        
        # Classification
        self.parser.add_argument('--run_classification', action='store_true')
        self.parser.add_argument('--llm_classification_seed', type=int, default=0)
        self.parser.add_argument('--llm_classification_answer_seed', type=int, default=0)
        self.parser.add_argument('--round', type=str, default='r0')
        self.parser.add_argument('--point', type=int, default=0)
        self.parser.add_argument('--result_save_dir', type=str, default='hs2c_results')

        # Device Config
        self.parser.add_argument('--cuda', type=int, default=None)


    def _post_process_args(self):
        self.args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.args.baseline_root = DEFAULT_DATASET_PATHS[self.args.dataset]
        self.args.k_hop_neighbors = K_HOP_NEIGHBOR_LIMITS['default']
        self.args.resolution = RESOLUTION[self.args.dataset]

        # Determine Dataset Properties
        if 'arxiv' in self.args.dataset:
            self.args.category_list = CATEGORY_LISTS['arxiv']
            self.args.dataset_name_in_prompt = "ArXiv CS"
            self.args.graph_type = "citation"
            self.args.node_type = "paper"
            self.args.label_whol_dict = LABEL_WHOL_DICT['arxiv']
            self.args.label_abbr_dict = LABEL_ABBR_DICT['arxiv']
        elif self.args.dataset in ['instagram']:
            self.args.category_list = CATEGORY_LISTS['instagram']
            self.args.dataset_name_in_prompt = "Instagram"
            self.args.graph_type = "following"
            self.args.node_type = "user"
            self.args.label_whol_dict = LABEL_WHOL_DICT['instagram']
            self.args.label_abbr_dict = LABEL_ABBR_DICT['instagram']
        elif self.args.dataset in ['citeseer']:
            self.args.category_list = CATEGORY_LISTS['citeseer']
            self.args.dataset_name_in_prompt = "CiteSeer"
            self.args.graph_type = "citation"
            self.args.node_type = "paper"
            self.args.label_whol_dict = LABEL_WHOL_DICT['citeseer']
            self.args.label_abbr_dict = LABEL_ABBR_DICT['citeseer']
        elif self.args.dataset in ['reddit']:
            self.args.category_list = CATEGORY_LISTS['reddit']
            self.args.dataset_name_in_prompt = "Reddit"
            self.args.graph_type = "following"
            self.args.node_type = "user"
            self.args.label_whol_dict = LABEL_WHOL_DICT['reddit']
            self.args.label_abbr_dict = LABEL_ABBR_DICT['reddit']
        elif self.args.dataset in ['products']:
            self.args.category_list = CATEGORY_LISTS['products']
            self.args.dataset_name_in_prompt = "Amazon Product"
            self.args.graph_type = "co-purchase"
            self.args.node_type = "product"
            self.args.label_whol_dict = LABEL_WHOL_DICT['products']
            self.args.label_abbr_dict = LABEL_ABBR_DICT['products']
        elif 'book' in self.args.dataset:
            self.args.category_list = CATEGORY_LISTS['book']
            self.args.dataset_name_in_prompt = "GoodReads Book"
            self.args.graph_type = "literature"
            self.args.node_type = "book"
            self.args.label_whol_dict = LABEL_WHOL_DICT['book']
            self.args.label_abbr_dict = LABEL_ABBR_DICT['book']
        elif 'dblp' in self.args.dataset:
            self.args.category_list = CATEGORY_LISTS['dblp']
            self.args.dataset_name_in_prompt = "DBLP CS"
            self.args.graph_type = "citation"
            self.args.node_type = "paper"
            self.args.label_whol_dict = LABEL_WHOL_DICT['dblp']
            self.args.label_abbr_dict = LABEL_ABBR_DICT['dblp']
        else:
            # Fallback or raise error
            self.args.category_list = ""
            self.args.dataset_name_in_prompt = "Graph"
            self.args.graph_type = "generic"
            self.args.node_type = "node"
            self.args.label_whol_dict = {}
            self.args.label_abbr_dict = {}

    def get_device(self):
        if self.args.cuda is not None:
            if self.args.cuda < 0:
                print("[Main] Info: Using CPU because cuda flag < 0.")
                self.args.device = torch.device('cpu')
                self.args.cuda = -1
                return

            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                if self.args.cuda >= num_gpus:
                    print(f"[Main] Warning: Requested cuda:{self.args.cuda} but only {num_gpus} GPUs available. Using cuda:0.")
                    self.args.cuda = 0
                self.args.device = torch.device(f'cuda:{self.args.cuda}')
            else:
                print("[Main] Warning: CUDA not available, using CPU.")
                self.args.cuda = -1
                self.args.device = torch.device('cpu')
        
        else:
            if torch.cuda.is_available():
                self.args.cuda = 0
                self.args.device = torch.device('cuda:0')
            else:
                self.args.cuda = -1
                self.args.device = torch.device('cpu')

    def to_dict(self):
        return vars(self.args)
