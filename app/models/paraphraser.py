from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Optional, List
import os
from dotenv import load_dotenv
import re
from nltk.tokenize import sent_tokenize
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

load_dotenv()

class Paraphraser:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = os.getenv("MODEL_NAME", "facebook/bart-large-cnn")
        
        # Configurable parameters
        self.generation_params = {
            "temperature": float(os.getenv("GENERATION_TEMPERATURE", "0.8")),
            "top_p": float(os.getenv("GENERATION_TOP_P", "0.95")),
            "num_sequences": int(os.getenv("GENERATION_NUM_SEQUENCES", "3")),
            "no_repeat_ngram_size": int(os.getenv("GENERATION_NO_REPEAT", "3")),
            "max_length": int(os.getenv("GENERATION_MAX_LENGTH", "512"))
        }
        
        # Style-specific parameters
        self.style_params = {
            "formal": {
                "temperature": 0.6,  # More conservative
                "top_p": 0.8,       # More focused
                "no_repeat_ngram_size": 4  # Stricter repetition control
            },
            "creative": {
                "temperature": 0.9,  # More creative
                "top_p": 0.98,      # More diverse
                "no_repeat_ngram_size": 2  # Allow some repetition
            },
            "neutral": {
                "temperature": 0.8,  # Balanced
                "top_p": 0.95,      # Balanced diversity
                "no_repeat_ngram_size": 3  # Standard repetition control
            }
        }
        
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
    
    def _create_enhanced_prompt(self, text: str, style: str) -> str:
        style_prompts = {
            "neutral": """Paraphrase the following text while maintaining its original meaning. 
            Use varied vocabulary and sentence structure while keeping the core message intact.
            Make it sound natural and fluent:""",
            
            "formal": """Paraphrase the following text in a formal and professional tone.
            Use sophisticated vocabulary and complex sentence structures.
            Maintain a serious and authoritative voice:""",
            
            "creative": """Paraphrase the following text in a creative and engaging way.
            Use vivid language and varied sentence structures.
            Make it more expressive and interesting while keeping the main idea:"""
        }
        
        # Add style-specific instructions
        style_instructions = {
            "neutral": "Focus on clarity and natural flow.",
            "formal": "Use academic language and professional terminology.",
            "creative": "Add literary devices and engaging expressions."
        }
        
        return f"""
{style_prompts.get(style, style_prompts['neutral'])}

{text}

Additional instructions: {style_instructions.get(style, '')}
"""
    
    def _post_process(self, text: str, original_text: str) -> str:
        # Clean up the output
        text = text.strip()
        
        # Remove any remaining prompt text
        text = re.sub(r'\[INST\].*?\[/INST\]', '', text, flags=re.DOTALL)
        
        # Ensure proper sentence structure
        sentences = sent_tokenize(text)
        text = ' '.join(sentences)
        
        # Maintain similar length to original
        if len(text.split()) > len(original_text.split()) * 1.5:
            text = ' '.join(text.split()[:len(original_text.split())])
        
        return text
    
    def _get_generation_params(self, style: str) -> dict:
        # Get base parameters
        params = self.generation_params.copy()
        
        # Override with style-specific parameters
        if style in self.style_params:
            params.update(self.style_params[style])
        
        return params
    
    def _generate_multiple(self, prompt: str, style: str = "neutral") -> List[str]:
        # Get generation parameters for the style
        params = self._get_generation_params(style)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=params["max_length"]
        ).to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_length=params["max_length"],
            num_return_sequences=params["num_sequences"],
            temperature=params["temperature"],
            top_p=params["top_p"],
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=params["no_repeat_ngram_size"]
        )
        
        return [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    
    def _select_best_output(self, outputs: List[str], original_text: str) -> str:
        # Simple selection based on length and similarity
        best_output = outputs[0]
        min_diff = float('inf')
        
        for output in outputs:
            # Calculate length difference
            len_diff = abs(len(output.split()) - len(original_text.split()))
            
            # Prefer outputs closer to original length
            if len_diff < min_diff:
                min_diff = len_diff
                best_output = output
        
        return best_output
    
    @torch.no_grad()
    def paraphrase(
        self,
        text: str,
        style: str = "neutral",
        max_length: Optional[int] = None
    ) -> str:
        # Override max_length if provided
        if max_length is not None:
            self.generation_params["max_length"] = max_length
        
        # Create the enhanced prompt
        prompt = self._create_enhanced_prompt(text, style)
        
        # Generate multiple variations
        outputs = self._generate_multiple(prompt, style)
        
        # Select the best output
        best_output = self._select_best_output(outputs, text)
        
        # Post-process the output
        paraphrased_text = self._post_process(best_output, text)
        
        return paraphrased_text