# $\text{H}\text{S}_2\text{C}$
Homophily-aware Structural and Semantic Compression for Graph Reasoning with LLM

## Project Overview
`Abstract` Large language models (LLMs) have demonstrated promising capabilities in text-attributed graph (TAG) understanding. Recent studies focus on verbalizing the graph structures via handcrafted prompts for LLMs, typically incorporating the target node information and its neighborhood context. However, constrained by the context window, existing methods mainly resort to random sampling, which inevitably introduces noise and redundancy, thereby causing reasoning instability. A stra-ightforward solution is <b>graph compression</b>, dropping some nodes/edges according to node-level importance criteria (e.g., degree, semantic similarity). Nevertheless, these methods overlook the <i>global structure</i> and <i>semantic information</i> from a local perspective, thereby leading to suboptimal performance. To solve this problem, we propose a <u>H</u>omophily-aware <u>S</u>tructural and <u>S</u>emantic <u>C</u>ompression (HS<sub>2</sub>C) for LLMs, discovering the community-level graph homophily for accurate LLM reasoning. Specifically, we first conduct a structural entropy minimization guided global hierarchical partition of TAG, which detects groups of structurally strongly coupled nodes and organizes them into homophilic communities utilizing the coding tree. Secondly, based on predefined community types, we apply differentiated summary of background nodes within each community, selectively preserving semantically homophilic information aligned with the target nodes. Finally, we reconstruct the compressed graph for downstream LLM inference. Extensive experiments on 10 node-level benchmarks across LLMs of varying sizes and families show that, by feeding LLMs with structurally and semantically compressed inputs, HS<sub>2</sub>C reduces graph storage cost and improves reasoning performance, validating its superiority and scalability. On OGBN-ArXiv, it achieves a 94.98% graph scale compression while improving accuracy by 3.06%–4.92%. Extensions to 7 diverse graph-level benchmarks further consolidate HS<sub>2</sub>C's task generalizability.

![overall framework](figures/overall_framework.png)

## Dependency
PYTHON 3.8, PyTorch 2.0.0, PyTorch Geometric 2.5.3, transformers 4.46.3

## Datasets
You can make directory `src/dataset`, and download the Node-level datasets through the [Google drive](https://drive.google.com/drive/folders/1w_XtfoaOqNjgdzYLRIW0kA1uZLB3CCDp).
Graph-Level datasets are available on [Google drive](https://drive.google.com/drive/folders/1CeA7LYE1Pp7ZS5_qDrYoaJGtKWtVPM0t).

## Model
You can make directory `src/A_base_model`, and download the [LLaMA 3.1 8B Instruct model](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), [LLaMA 3.2 3B Instruct model](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct), [Vicuna 7B model](https://huggingface.co/lmsys/vicuna-7b-v1.5) from Hugging Face.
