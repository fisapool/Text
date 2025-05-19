from dataclasses import dataclass
from typing import Optional, List

@dataclass
class TrainingConfig:
    # Model Configuration
    base_model: str = "facebook/bart-large-cnn"
    output_dir: str = "models/premium"
    
    # Training Parameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    
    # LoRA Configuration (Parameter-Efficient Fine-Tuning)
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Dataset Configuration
    train_file: str = "data/train.json"
    validation_file: str = "data/validation.json"
    max_source_length: int = 512
    max_target_length: int = 512
    
    # Premium Features
    style_categories: List[str] = [
        "academic",
        "business",
        "creative",
        "technical",
        "formal",
        "casual"
    ]
    
    # Quality Thresholds
    min_bleu_score: float = 0.85
    min_rouge_score: float = 0.80
    min_semantic_similarity: float = 0.90

@dataclass
class PremiumFeatures:
    # Advanced Paraphrasing
    style_transfer: bool = True
    tone_adjustment: bool = True
    complexity_control: bool = True
    
    # Quality Enhancements
    grammar_check: bool = True
    plagiarism_detection: bool = True
    readability_optimization: bool = True
    
    # Batch Processing
    batch_size: int = 32
    concurrent_requests: int = 10
    
    # Output Options
    multiple_variants: bool = True
    max_variants: int = 5
    confidence_scores: bool = True 