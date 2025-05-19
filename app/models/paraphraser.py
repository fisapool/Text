from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

class Paraphraser:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        
        # Load fine-tuned weights if available
        if os.path.exists("models/fine_tuned"):
            self.model.load_state_dict(torch.load("models/fine_tuned/pytorch_model.bin"))
        
        self.model.eval()
    
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
        max_length: Optional[int] = 512
    ) -> str:
        # Create the prompt
        prompt = self._create_prompt(text, style)
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        ).to(self.device)
        
        # Generate paraphrase
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode and clean up the output
        paraphrased_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        # Remove the prompt from the output
        paraphrased_text = paraphrased_text.replace(prompt, "").strip()
        
        return paraphrased_text 