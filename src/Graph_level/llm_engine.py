import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from dataset.info import MOLECULENET_DESC

class LLMEngine:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        
        print(f"Loading LLM: {args.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(f"{args.model_path}/{args.model_name}")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        device_map = "auto"
        if isinstance(self.device, torch.device) and self.device.type == 'cuda':
            device_map = {"": self.device.index if self.device.index is not None else 0}
            
        self.model = AutoModelForCausalLM.from_pretrained(
            f"{args.model_path}/{args.model_name}",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device_map
        )
        self.emb_tokenizer = AutoTokenizer.from_pretrained(f"{args.model_path}/Bert-base-uncased")
        self.emb_model = AutoModel.from_pretrained(f"{args.model_path}/Bert-base-uncased").to(self.device)
        self.emb_model.eval()
        self.embedding_dim = 768

    def get_text_embedding(self, text):
        inputs = self.emb_tokenizer(
            str(text), return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.device)
        with torch.no_grad():
            outputs = self.emb_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()

    def condense_community(self, words, level, comm_id, dataset_type="molecule"):
        prompt = self._create_condensation_prompt(words, dataset_type)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.args.condensation_max_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        condensed = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._parse_llm_output(condensed, "Output only the condensed phrase:")

    def _parse_llm_output(self, text, split_str):
        if split_str in text:
            return text.split(split_str)[-1].strip()
        return text

    def _create_condensation_prompt(self, words, dataset_type):
        items = ', '.join(words)
        if dataset_type == "molecule":
            return (
                f"Analyze the following atoms in a molecular substructure:\n{items}\n\n"
                f"Task: Describe this substructure in a concise phrase (5-8 words).\n"
                f"Focus on: \n"
                f"1. The functional group or structural motif.\n"
                f"2. Any notable chemical properties.\n\n"
                f"Output only the condensed phrase:"
            )
        else:
            return (
                f"Analyze the following words from a sentence:\n{items}\n\n"
                f"Task: Summarize the semantic meaning and sentiment tendency of these words in a concise phrase (5-8 words maximum).\n"
                f"Focus on: \n"
                f"1. The collective semantic meaning. \n"
                f"2. Any overall sentiment tendency (positive/negative). \n\n"
                f"Output only the condensed phrase:"
            )

    def predict_graph(self, original_text, reconstructed_graph, dataset_type="molecule", dataset_name="", dataset_info=None, task_id=None):
        prompt = self._create_classification_prompt(
            original_text, reconstructed_graph, dataset_type, dataset_name, dataset_info, task_id
        )
        
        if dataset_info and 'task_type' in dataset_info:
            if dataset_info['task_type'] == 'multi_task_binary_classification' and task_id is not None:
                label_dict = dataset_info['tasks'][task_id]['label_dict']
            else:
                label_dict = dataset_info.get('label_dict', {})
        else:
            label_dict = {0: "No", 1: "Yes"}
            
        candidate_labels = list(label_dict.values())
        
        result = self.compute_label_likelihood(prompt, candidate_labels)
        predicted_label_str = result['predicted_label']
        
        predicted_id = -1
        for key, val in label_dict.items():
            if val == predicted_label_str:
                predicted_id = key
                break
                
        return predicted_id, predicted_label_str, result['scores']

    def compute_label_likelihood(self, prompt: str, candidate_labels: list) -> dict:
        if not prompt.strip().endswith("Answer:"):
            prompt = prompt.strip() + "\n\nAnswer:"
        
        scores = {}
        
        prompt_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.args.max_length
        ).to(self.device)
        
        prompt_length = prompt_inputs.input_ids.shape[1]
        
        for label in candidate_labels:
            full_text = prompt + " " + label
            
            inputs = self.tokenizer(
                full_text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=self.args.max_length
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            label_len = inputs.input_ids.shape[1] - prompt_length
            
            if label_len <= 0:
                scores[label] = float('inf')
                continue

            shift_logits = logits[:, prompt_length-1 : -1, :].contiguous()
            shift_labels = inputs.input_ids[:, prompt_length:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            scores[label] = loss.item()
        
        predicted_label = min(scores, key=scores.get)
        
        return {
            'predicted_label': predicted_label,
            'scores': scores
        }

    def _create_classification_prompt(self, original_text, graph, dataset_type, dataset_name, dataset_info=None, task_id=None):
        super_nodes = graph['super_nodes']
        super_edges = graph['super_edges']
        
        hierarchical_desc = ' → '.join([super_nodes[k] for k in sorted(super_nodes.keys())])
        
        structure_desc = ""
        for comm_id in sorted(super_nodes.keys()):
            connections = super_edges.get(comm_id, [])
            for dst in sorted(connections):
                structure_desc += f'"{super_nodes[comm_id]}" connected to "{super_nodes[dst]}"\n'
        
        if not structure_desc: 
            structure_desc = "(No cross-group connections)\n"

        if dataset_info and 'task_type' in dataset_info:
            if dataset_info['task_type'] == 'multi_task_binary_classification' and task_id is not None:
                task_info = dataset_info['tasks'][task_id]
                task_desc = task_info['dataset_desc']
                label_dict = task_info['label_dict']
                label_desc = task_info.get('label_desc', {})
            else:
                task_desc = dataset_info.get('dataset_desc', '')
                label_dict = dataset_info.get('label_dict', {})
                label_desc = dataset_info.get('label_desc', {})
        else:
            task_desc = f"Classify the {dataset_type}."
            label_dict = {0: "class_0", 1: "class_1"}
            label_desc = {}

            options_list = [f"{label_dict[k]}" for k in sorted(label_dict.keys())]
        options_str = " / ".join(options_list)
        
        label_explanations = ""
        if label_desc:
            label_explanations = "\n[Label Descriptions]\n"
            for k in sorted(label_desc.keys()):
                label_explanations += f"- {label_desc[k]}\n"

        return (
            f"You are an expert in {dataset_type} analysis.\n"
            f"[Task] {task_desc}\n\n"
            f"[Hierarchical Structure]\n{hierarchical_desc}\n\n"
            f"[Connections]\n{structure_desc}"
            f"{label_explanations}\n"
            f"[Question] Based on the above information, what is the classification?\n"
            f"Output EXACTLY one word from: {options_str}\n\n"
            f"Answer:"
        )

