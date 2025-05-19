from typing import Dict, Any, List
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import nltk
import os
from dotenv import load_dotenv

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

load_dotenv()

class QualityScorer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load semantic similarity model
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load fluency model (using a sentiment model as a proxy for fluency)
        self.fluency_model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        ).to(self.device)
        self.fluency_tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )
        
        # Load style classifier if available
        style_model_path = os.getenv("STYLE_CLASSIFIER_PATH")
        if style_model_path and os.path.exists(style_model_path):
            self.style_model = AutoModelForSequenceClassification.from_pretrained(
                style_model_path
            ).to(self.device)
            self.style_tokenizer = AutoTokenizer.from_pretrained(style_model_path)
        else:
            self.style_model = None
            self.style_tokenizer = None

    def calculate_semantic_similarity(self, original: str, paraphrase: str) -> float:
        """Calculate semantic similarity using sentence transformers"""
        embeddings1 = self.similarity_model.encode([original])[0]
        embeddings2 = self.similarity_model.encode([paraphrase])[0]
        similarity = np.dot(embeddings1, embeddings2) / (
            np.linalg.norm(embeddings1) * np.linalg.norm(embeddings2)
        )
        return float(similarity)

    def calculate_fluency_score(self, text: str) -> float:
        """Calculate fluency score using a sentiment model as proxy"""
        inputs = self.fluency_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.fluency_model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1)
            # Use the positive sentiment score as a proxy for fluency
            return float(scores[0][1])

    def calculate_style_score(self, text: str, target_style: str) -> float:
        """Calculate style adherence score"""
        if not self.style_model:
            return 1.0  # Return neutral score if no style model available
            
        inputs = self.style_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.style_model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1)
            # Assuming style classes are ordered: [neutral, formal, creative]
            style_idx = {"neutral": 0, "formal": 1, "creative": 2}.get(target_style, 0)
            return float(scores[0][style_idx])

    def calculate_bleu_score(self, original: str, paraphrase: str) -> float:
        """Calculate BLEU score for lexical overlap"""
        reference = word_tokenize(original.lower())
        candidate = word_tokenize(paraphrase.lower())
        return sentence_bleu([reference], candidate)

    def score_paraphrase(
        self,
        original: str,
        paraphrase: str,
        style: str,
        weights: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """Calculate comprehensive quality scores for a paraphrase"""
        if weights is None:
            weights = {
                "semantic_similarity": 0.4,
                "fluency": 0.3,
                "style": 0.2,
                "bleu": 0.1
            }
        
        # Calculate individual scores
        semantic_score = self.calculate_semantic_similarity(original, paraphrase)
        fluency_score = self.calculate_fluency_score(paraphrase)
        style_score = self.calculate_style_score(paraphrase, style)
        bleu_score = self.calculate_bleu_score(original, paraphrase)
        
        # Calculate weighted score
        weighted_score = (
            weights["semantic_similarity"] * semantic_score +
            weights["fluency"] * fluency_score +
            weights["style"] * style_score +
            weights["bleu"] * bleu_score
        )
        
        return {
            "overall_score": float(weighted_score),
            "semantic_similarity": float(semantic_score),
            "fluency": float(fluency_score),
            "style_adherence": float(style_score),
            "lexical_overlap": float(bleu_score),
            "weights_used": weights
        }

    def rank_paraphrases(
        self,
        original: str,
        paraphrases: List[str],
        style: str,
        weights: Dict[str, float] = None
    ) -> List[Dict[str, Any]]:
        """Rank multiple paraphrases by quality"""
        scored_paraphrases = []
        for paraphrase in paraphrases:
            scores = self.score_paraphrase(original, paraphrase, style, weights)
            scored_paraphrases.append({
                "paraphrase": paraphrase,
                "scores": scores
            })
        
        # Sort by overall score
        scored_paraphrases.sort(key=lambda x: x["scores"]["overall_score"], reverse=True)
        return scored_paraphrases 