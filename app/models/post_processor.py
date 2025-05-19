from typing import Dict, Any, List
import re
import language_tool_python
import os
from dotenv import load_dotenv

load_dotenv()

class PostProcessor:
    def __init__(self):
        # Initialize language tool for grammar checking
        self.language_tool = language_tool_python.LanguageTool('en-US')
        
        # Load style-specific rules
        self.style_rules = {
            "formal": {
                "contractions": False,
                "passive_voice": True,
                "min_sentence_length": 10,
                "max_sentence_length": 30
            },
            "creative": {
                "contractions": True,
                "passive_voice": False,
                "min_sentence_length": 5,
                "max_sentence_length": 50
            },
            "neutral": {
                "contractions": True,
                "passive_voice": True,
                "min_sentence_length": 8,
                "max_sentence_length": 40
            }
        }

    def fix_grammar(self, text: str) -> str:
        """Fix grammar and spelling errors"""
        matches = self.language_tool.check(text)
        corrected_text = text
        for match in reversed(matches):
            if match.replacements:
                corrected_text = (
                    corrected_text[:match.offset] +
                    match.replacements[0] +
                    corrected_text[match.offset + match.errorLength:]
                )
        return corrected_text

    def adjust_style(self, text: str, style: str) -> str:
        """Adjust text according to style rules"""
        rules = self.style_rules.get(style, self.style_rules["neutral"])
        
        # Handle contractions
        if not rules["contractions"]:
            text = text.replace("'s", " is").replace("'re", " are").replace("'t", " not")
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        processed_sentences = []
        
        for sentence in sentences:
            # Adjust sentence length
            if len(sentence.split()) < rules["min_sentence_length"]:
                # Combine with next sentence if too short
                if processed_sentences:
                    processed_sentences[-1] += " " + sentence
                else:
                    processed_sentences.append(sentence)
            elif len(sentence.split()) > rules["max_sentence_length"]:
                # Split long sentences
                words = sentence.split()
                mid = len(words) // 2
                processed_sentences.extend([
                    " ".join(words[:mid]),
                    " ".join(words[mid:])
                ])
            else:
                processed_sentences.append(sentence)
        
        return " ".join(processed_sentences)

    def remove_repetitions(self, text: str) -> str:
        """Remove repetitive phrases and words"""
        # Remove consecutive repeated words
        text = re.sub(r'\b(\w+)(\s+\1\b)+', r'\1', text)
        
        # Remove repetitive phrases (3+ words)
        words = text.split()
        for i in range(len(words) - 2):
            phrase = " ".join(words[i:i+3])
            if text.count(phrase) > 1:
                text = text.replace(phrase, "", text.count(phrase) - 1)
        
        return text

    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace and punctuation"""
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        text = re.sub(r'([.,!?])\s+', r'\1 ', text)
        
        # Ensure proper spacing after sentences
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        return text.strip()

    def process(
        self,
        text: str,
        style: str = "neutral",
        options: Dict[str, bool] = None
    ) -> Dict[str, Any]:
        """Apply all post-processing steps"""
        if options is None:
            options = {
                "fix_grammar": True,
                "adjust_style": True,
                "remove_repetitions": True,
                "normalize_whitespace": True
            }
        
        original = text
        changes = []
        
        if options["fix_grammar"]:
            text = self.fix_grammar(text)
            if text != original:
                changes.append("grammar_fixes")
        
        if options["adjust_style"]:
            text = self.adjust_style(text, style)
            if text != original:
                changes.append("style_adjustments")
        
        if options["remove_repetitions"]:
            text = self.remove_repetitions(text)
            if text != original:
                changes.append("repetition_removal")
        
        if options["normalize_whitespace"]:
            text = self.normalize_whitespace(text)
            if text != original:
                changes.append("whitespace_normalization")
        
        return {
            "processed_text": text,
            "changes_made": changes,
            "original_text": original
        } 