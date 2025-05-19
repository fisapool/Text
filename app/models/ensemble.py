from typing import List, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from app.models.paraphraser import Paraphraser
import os
from dotenv import load_dotenv

load_dotenv()

class ParaphraseEnsemble:
    def __init__(self):
        self.models = []
        self.tokenizers = []
        self.weights = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load models from environment variable
        model_names = os.getenv("ENSEMBLE_MODELS", "mistralai/Mistral-7B-Instruct-v0.2").split(",")
        model_weights = os.getenv("ENSEMBLE_WEIGHTS", "1.0").split(",")
        
        for name, weight in zip(model_names, model_weights):
            model = AutoModelForCausalLM.from_pretrained(
                name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(name)
            
            self.models.append(model)
            self.tokenizers.append(tokenizer)
            self.weights.append(float(weight))
        
        # Normalize weights
        self.weights = np.array(self.weights) / sum(self.weights)
        
        # Initialize base paraphraser for fallback
        self.base_paraphraser = Paraphraser()

    def _create_prompt(self, text: str, style: str) -> str:
        style_prompts = {
            "neutral": "Paraphrase the following text while maintaining its original meaning:",
            "formal": "Paraphrase the following text in a formal and professional tone:",
            "creative": "Paraphrase the following text in a creative and engaging way:"
        }
        return f"""<s>[INST] {style_prompts.get(style, style_prompts['neutral'])}
        
{text} [/INST]"""

    @torch.no_grad()
    def paraphrase(
        self,
        text: str,
        style: str = "neutral",
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 3
    ) -> Dict[str, Any]:
        """Generate paraphrases using ensemble of models"""
        prompt = self._create_prompt(text, style)
        paraphrases = []
        scores = []

        for model, tokenizer, weight in zip(self.models, self.tokenizers, self.weights):
            try:
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length
                ).to(self.device)

                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=num_return_sequences,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

                for output in outputs:
                    paraphrased = tokenizer.decode(output, skip_special_tokens=True)
                    paraphrased = paraphrased.replace(prompt, "").strip()
                    
                    # Calculate confidence score
                    with torch.no_grad():
                        logits = model(**inputs).logits
                        score = torch.softmax(logits, dim=-1).max().item()
                    
                    paraphrases.append(paraphrased)
                    scores.append(score * weight)

            except Exception as e:
                print(f"Error in model {model.name_or_path}: {str(e)}")
                continue

        if not paraphrases:
            # Fallback to base paraphraser
            return {
                "paraphrased": self.base_paraphraser.paraphrase(text, style, max_length),
                "confidence": 0.5,
                "ensemble_used": False
            }

        # Select best paraphrase based on weighted scores
        best_idx = np.argmax(scores)
        return {
            "paraphrased": paraphrases[best_idx],
            "confidence": float(scores[best_idx]),
            "ensemble_used": True,
            "all_paraphrases": paraphrases,
            "all_scores": [float(s) for s in scores]
        } 