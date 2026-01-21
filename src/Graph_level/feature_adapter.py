import torch
import numpy as np

class UnifiedAdapter:
    def __init__(self, device):
        self.device = device

    def get_node_text(self, node_id, data, **kwargs):
        if hasattr(data, 'raw_texts') and data.raw_texts is not None:
            raw_texts = data.raw_texts
            if isinstance(raw_texts, list) and len(raw_texts) == 1 and isinstance(raw_texts[0], list):
                raw_texts = raw_texts[0]
            
            if node_id < len(raw_texts):
                return str(raw_texts[node_id])
        return f"node_{node_id}"

    def get_node_embeddings(self, data, llm_engine, **kwargs):
        node_texts = []
        if hasattr(data, 'raw_texts') and data.raw_texts is not None:
            raw_texts = data.raw_texts
            if isinstance(raw_texts, list) and len(raw_texts) == 1 and isinstance(raw_texts[0], list):
                raw_texts = raw_texts[0]
            node_texts = [str(t) for t in raw_texts]
        else:
            node_texts = [f"node_{i}" for i in range(data.num_nodes)]
            
        if len(node_texts) > data.num_nodes:
            node_texts = node_texts[:data.num_nodes]
        elif len(node_texts) < data.num_nodes:
            node_texts.extend([f"node_{i}" for i in range(len(node_texts), data.num_nodes)])
        embeddings = []
        for text in node_texts:
            emb = llm_engine.get_text_embedding(text)
            embeddings.append(emb)
            
        return torch.tensor(np.vstack(embeddings), dtype=torch.float32).to(self.device)
