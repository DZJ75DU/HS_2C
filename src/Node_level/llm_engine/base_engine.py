import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

project_root = Path(__file__).parent.parent.parent.resolve()
hs2c_path = project_root / '7_HS2C_pipeline'
sys.path.insert(0, str(hs2c_path))


class LLMEngine:
    def __init__(self, config):
        self.config = config
        self.model_path = config['model_path']
        self.model_name = config['model_name']
        self.device = config['device']
        
        self._load_model()
        self._load_classify_model()
        
    def _load_model(self):
        print(f"[LLMEngine] Loading model: {self.model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(f"{self.model_path}/{self.model_name}")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.padding_side = "right"
                
            self.model = AutoModelForCausalLM.from_pretrained(
                f"{self.model_path}/{self.model_name}",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )
            print("✓ Model loaded.")
        except Exception as e:
            print(f"[Error] Failed to load model: {e}")

    def _load_classify_model(self):
        try:
            self.classify_tokenizer = AutoTokenizer.from_pretrained(f"{self.model_path}/{self.model_name}")
            if self.classify_tokenizer.pad_token is None:
                self.classify_tokenizer.pad_token = self.classify_tokenizer.eos_token
                self.classify_tokenizer.pad_token_id = self.classify_tokenizer.eos_token_id
            self.classify_tokenizer.padding_side = "left"
                
        except Exception as e:
            print(f"[Error] Failed to load model: {e}")
            
    def generate(self, prompt, max_new_tokens=100):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0
            )
            
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if prompt in response:
            response = response.replace(prompt, "").strip()
            
        return response

    def compute_label_likelihood(self, prompt, candidate_labels):
        """
        Computes P(Label | Prompt) for each label in candidate_labels.
        Returns a dict with scores (NLL) and probabilities.
        """
        self.model.eval()
        
        if not prompt.strip().endswith("Answer:"):
            prompt = prompt.strip() + "\n\nAnswer:"
        
        scores = {}
        
        with torch.no_grad():
            prompt_inputs = self.classify_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config['llm_length']
            ).to(self.device)
            
            prompt_length = prompt_inputs.input_ids.shape[1]
            
            for label in candidate_labels:
                full_text = prompt + " " + label
                
                inputs = self.classify_tokenizer(
                    full_text, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=self.config['llm_length'] + 20
                ).to(self.device)

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

        neg_scores = {k: -v for k, v in scores.items()}
        score_values = torch.tensor(list(neg_scores.values()))
        probs = torch.nn.functional.softmax(score_values, dim=0).cpu().numpy()
        
        probabilities = {label: float(prob) for label, prob in zip(neg_scores.keys(), probs)}
        
        return {
            'predicted_label': predicted_label,
            'scores': scores,
            'probabilities': probabilities
        }
