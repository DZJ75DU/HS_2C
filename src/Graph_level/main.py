import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import HS2CPipeline
from utils import set_random_seed

def main():
    parser = argparse.ArgumentParser(description='Unified HS2C Graph-Level Classification Pipeline')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='Graph-SST2', choices=['BBBP', 'ClinTox', 'SIDER', 'MUV', 'Graph-SST2', 'Graph-SST5', 'Graph-Twitter'])
    parser.add_argument('--data_dir', type=str, default='./dataset', help='Data directory')
    parser.add_argument('--data_seed', type=int, default=0, help='Random seed')
    
    # Model
    parser.add_argument('--model_name', type=str, default='Llama-3.2-3b-Instruct', help='LLM model name')
    parser.add_argument('--model_path', type=str, default='/A_base_model', help='Path to LLM')
    parser.add_argument('--max_length', type=int, default=4000, help='Max context length')
    
    # Structure Learning
    parser.add_argument('--use_knn_enhancement', action='store_true', help='Enable KNN enhancement')
    parser.add_argument('--tree_height', type=int, default=2, help='Coding tree height')
    
    # Condensation
    parser.add_argument('--condensation_max_tokens', type=int, default=15, help='Max tokens for condensation')
    
    # Evaluation
    parser.add_argument('--max_test_samples', type=int, default=None, help='Limit test samples (debug)')
    parser.add_argument('--ori_text_in_classification', action='store_true', help='Use original text for classification')
    parser.add_argument('--save_dir', type=str, default='./hs2c_results', help='Result save directory')
    parser.add_argument('--tips', type=str, default=None)

    parser.add_argument('--cuda', type=int, default=5, help='CUDA device index')
    
    args = parser.parse_args()
    
    set_random_seed(args.data_seed)
    print(f"Starting HS2C Pipeline for {args.dataset}")
    
    pipeline = HS2CPipeline(args)
    pipeline.run()

if __name__ == "__main__":
    main()
