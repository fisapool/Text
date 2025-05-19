import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict, Any
from app.models.ensemble import ParaphraseEnsemble
from app.models.quality_scorer import QualityScorer
from app.models.post_processor import PostProcessor
import logging

app = FastAPI()

MODELS = {
    "en": "Vamsi/T5_Paraphrase_Paws",
    "fr": "plguillou/t5-base-fr-sum-cnndm",
    "de": "mrm8488/bert2bert_shared-german-finetuned-summarization",
    # Add more as needed
}

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

loaded_models = {}
loaded_tokenizers = {}

def get_model_and_tokenizer(lang):
    if lang not in loaded_models:
        model_name = MODELS.get(lang, MODELS["en"])
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            model.to(device) # Move model to determined device
            loaded_tokenizers[lang] = tokenizer
            loaded_models[lang] = model
            print(f"Successfully loaded model '{model_name}' for language '{lang}' on device: {device}")
        except Exception as e:
            print(f"Error loading model '{model_name}' for language '{lang}': {e}")
            raise HTTPException(status_code=500, detail=f"Error loading model for language {lang}")
    return loaded_tokenizers[lang], loaded_models[lang]

class ParaphraseRequest(BaseModel):
    text: str # Keep single text for now, will add batch later
    lang: str = "en"
    style: str = "default" # Add style parameter for future use

class BatchParaphraseRequest(BaseModel):
    texts: list[str]
    lang: str = "en"
    style: str = "default" # Add style parameter for future use

class ParaphraseResponse(BaseModel):
    paraphrase: str

class BatchParaphraseResponse(BaseModel):
    paraphrases: list[str]

# Pre-load English model on startup (optional but recommended for commercial APIs)
# try:
#     get_model_and_tokenizer("en")
# except Exception as e:
#     print(f"Failed to pre-load English model: {e}")

@app.post("/paraphrase", response_model=ParaphraseResponse)
def paraphrase(req: ParaphraseRequest):
    tokenizer, model = get_model_and_tokenizer(req.lang)
    input_text = f"paraphrase: {req.text} </s>"
    encoding = tokenizer.encode_plus(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    input_ids = encoding["input_ids"].to(device) # Move input to device
    attention_mask = encoding["attention_mask"].to(device) # Move attention mask to device

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=256, # Can be a parameter
        do_sample=True, # Can be a parameter
        top_k=120,      # Can be a parameter
        top_p=0.98,     # Can be a parameter
        early_stopping=True,
        num_return_sequences=1, # For batching, this might be > 1 or handled differently
        temperature=0.7, # Added temperature
        repetition_penalty=1.2 # Added repetition penalty
    )
    paraphrased = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"paraphrase": paraphrased}

@app.post("/batch_paraphrase", response_model=BatchParaphraseResponse)
def batch_paraphrase(req: BatchParaphraseRequest):
    tokenizer, model = get_model_and_tokenizer(req.lang)
    
    # Prepare texts for batch processing
    input_texts = [f"paraphrase: {text} </s>" for text in req.texts]
    encoding = tokenizer.batch_encode_plus(
        input_texts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=512
    )
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=256, # Can be a parameter
        do_sample=True, # Can be a parameter
        top_k=120,      # Can be a parameter
        top_p=0.98,     # Can be a parameter
        early_stopping=True,
        num_return_sequences=1, # Generate one paraphrase per input text in the batch
        temperature=0.7,
        repetition_penalty=1.2
    )
    
    # Decode batch outputs
    paraphrased_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    
    return {"paraphrases": paraphrased_texts}

logger = logging.getLogger(__name__)

class ParaphraseService:
    def __init__(self):
        self.ensemble = ParaphraseEnsemble()
        self.quality_scorer = QualityScorer()
        self.post_processor = PostProcessor()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initialized ParaphraseService with device: {self.device}")

    async def paraphrase(
        self,
        text: str,
        style: str = "neutral",
        num_variations: int = 3,
        post_process: bool = True
    ) -> Dict[str, Any]:
        """Generate paraphrases with quality scoring and post-processing"""
        try:
            # Generate initial paraphrases
            paraphrases = await self.ensemble.generate(
                text,
                num_variations=num_variations
            )
            
            # Score paraphrases
            scored_paraphrases = []
            for para in paraphrases:
                scores = self.quality_scorer.score(text, para)
                scored_paraphrases.append({
                    "text": para,
                    "scores": scores
                })
            
            # Rank paraphrases
            ranked_paraphrases = self.quality_scorer.rank(scored_paraphrases)
            
            # Post-process if requested
            if post_process:
                processed_paraphrases = []
                for para in ranked_paraphrases:
                    result = self.post_processor.process(
                        para["text"],
                        style=style,
                        options={
                            "fix_grammar": True,
                            "adjust_style": True,
                            "remove_repetitions": True,
                            "normalize_whitespace": True
                        }
                    )
                    processed_paraphrases.append({
                        "text": result["processed_text"],
                        "scores": para["scores"],
                        "changes": result["changes_made"]
                    })
                ranked_paraphrases = processed_paraphrases
            
            return {
                "paraphrases": ranked_paraphrases,
                "original_text": text,
                "style": style
            }
            
        except Exception as e:
            logger.error(f"Error in paraphrase generation: {str(e)}")
            raise

    async def batch_paraphrase(
        self,
        texts: List[str],
        style: str = "neutral",
        num_variations: int = 3,
        post_process: bool = True
    ) -> List[Dict[str, Any]]:
        """Process multiple texts in batch"""
        results = []
        for text in texts:
            try:
                result = await self.paraphrase(
                    text,
                    style=style,
                    num_variations=num_variations,
                    post_process=post_process
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing text in batch: {str(e)}")
                results.append({
                    "error": str(e),
                    "original_text": text
                })
        return results 