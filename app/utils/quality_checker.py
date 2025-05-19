from typing import Dict, Any, List, Tuple
import language_tool_python
import nltk
from nltk.tokenize import sent_tokenize
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from textstat import textstat
import re

class QualityChecker:
    def __init__(self):
        self.language_tool = language_tool_python.LanguageTool('en-US')
        nltk.download('punkt', quiet=True)
        
    def check_grammar(self, text: str) -> Dict[str, Any]:
        """Check grammar and spelling"""
        matches = self.language_tool.check(text)
        return {
            "error_count": len(matches),
            "errors": [
                {
                    "message": match.message,
                    "offset": match.offset,
                    "length": match.errorLength,
                    "context": match.context
                }
                for match in matches
            ]
        }
    
    def check_readability(self, text: str) -> Dict[str, float]:
        """Calculate various readability scores"""
        return {
            "flesch_reading_ease": textstat.flesch_reading_ease(text),
            "smog_index": textstat.smog_index(text),
            "coleman_liau_index": textstat.coleman_liau_index(text),
            "automated_readability_index": textstat.automated_readability_index(text),
            "dale_chall_readability_score": textstat.dale_chall_readability_score(text)
        }
    
    def check_semantic_similarity(self, original: str, paraphrased: str) -> float:
        """Calculate semantic similarity between original and paraphrased text"""
        original_sentences = sent_tokenize(original)
        paraphrased_sentences = sent_tokenize(paraphrased)
        
        # Calculate BLEU score
        bleu_score = sentence_bleu(
            [original_sentences],
            paraphrased_sentences,
            weights=(0.25, 0.25, 0.25, 0.25)
        )
        
        return bleu_score
    
    def check_style_consistency(self, text: str, style: str) -> Dict[str, Any]:
        """Check if the text maintains the requested style"""
        style_indicators = {
            "formal": {
                "contractions": False,
                "passive_voice": True,
                "first_person": False,
                "sentence_length": "long"
            },
            "casual": {
                "contractions": True,
                "passive_voice": False,
                "first_person": True,
                "sentence_length": "short"
            },
            "academic": {
                "contractions": False,
                "passive_voice": True,
                "first_person": False,
                "sentence_length": "long"
            }
        }
        
        indicators = style_indicators.get(style, style_indicators["neutral"])
        
        # Check contractions
        has_contractions = bool(re.search(r'\b\w+\'[a-z]+\b', text))
        
        # Check passive voice
        passive_pattern = r'\b(am|is|are|was|were|be|been|being)\s+\w+ed\b'
        has_passive_voice = bool(re.search(passive_pattern, text))
        
        # Check first person
        first_person_pattern = r'\b(I|we|me|us|my|our)\b'
        has_first_person = bool(re.search(first_person_pattern, text))
        
        # Check sentence length
        sentences = sent_tokenize(text)
        avg_sentence_length = np.mean([len(s.split()) for s in sentences])
        is_long_sentences = avg_sentence_length > 15
        
        return {
            "style_match": {
                "contractions": has_contractions == indicators["contractions"],
                "passive_voice": has_passive_voice == indicators["passive_voice"],
                "first_person": has_first_person == indicators["first_person"],
                "sentence_length": is_long_sentences == (indicators["sentence_length"] == "long")
            },
            "metrics": {
                "avg_sentence_length": avg_sentence_length,
                "has_contractions": has_contractions,
                "has_passive_voice": has_passive_voice,
                "has_first_person": has_first_person
            }
        }
    
    def check_quality(self, original: str, paraphrased: str, style: str) -> Dict[str, Any]:
        """Perform comprehensive quality check"""
        grammar_check = self.check_grammar(paraphrased)
        readability_scores = self.check_readability(paraphrased)
        semantic_similarity = self.check_semantic_similarity(original, paraphrased)
        style_consistency = self.check_style_consistency(paraphrased, style)
        
        # Calculate overall quality score
        quality_score = (
            (1 - min(grammar_check["error_count"] / 10, 1)) * 0.3 +
            (semantic_similarity) * 0.3 +
            (sum(style_consistency["style_match"].values()) / 4) * 0.2 +
            (min(readability_scores["flesch_reading_ease"] / 100, 1)) * 0.2
        )
        
        return {
            "quality_score": quality_score,
            "grammar_check": grammar_check,
            "readability_scores": readability_scores,
            "semantic_similarity": semantic_similarity,
            "style_consistency": style_consistency,
            "passes_quality_threshold": quality_score >= 0.7
        } 