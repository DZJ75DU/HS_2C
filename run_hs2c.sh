CUDA_VISIBLE_DEVICES=0 python data_preprocess.py --new_date 20250606 --dataset ogb_arxiv --model_name Llama-3.1-8b-Instruct
CUDA_VISIBLE_DEVICES=0 python data_preprocess.py --new_date 20250606 --dataset arxiv_2023 --model_name Llama-3.1-8b-Instruct
CUDA_VISIBLE_DEVICES=0 python data_preprocess.py --new_date 20250606 --dataset instagram --model_name Llama-3.1-8b-Instruct
CUDA_VISIBLE_DEVICES=0 python data_preprocess.py --new_date 20250606 --dataset citeseer --model_name Llama-3.1-8b-Instruct
CUDA_VISIBLE_DEVICES=0 python data_preprocess.py --new_date 20250606 --dataset reddit --model_name Llama-3.1-8b-Instruct
CUDA_VISIBLE_DEVICES=0 python data_preprocess.py --new_date 20250606 --dataset products --model_name Llama-3.1-8b-Instruct

CUDA_VISIBLE_DEVICES=0 python reconstruct_classification.py --new_date 20250606 --dataset ogb_arxiv --model_name Llama-3.1-8b-Instruct
CUDA_VISIBLE_DEVICES=0 python reconstruct_classification.py --new_date 20250606 --dataset arxiv_2023 --model_name Llama-3.1-8b-Instruct
CUDA_VISIBLE_DEVICES=0 python reconstruct_classification.py --new_date 20250606 --dataset instagram --model_name Llama-3.1-8b-Instruct
CUDA_VISIBLE_DEVICES=0 python reconstruct_classification.py --new_date 20250606 --dataset citeseer --model_name Llama-3.1-8b-Instruct
CUDA_VISIBLE_DEVICES=0 python reconstruct_classification.py --new_date 20250606 --dataset reddit --model_name Llama-3.1-8b-Instruct
CUDA_VISIBLE_DEVICES=0 python reconstruct_classification.py --new_date 20250606 --dataset products --model_name Llama-3.1-8b-Instruct
