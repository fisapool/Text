import os
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
import evaluate
import numpy as np
from typing import Dict, Any
from .config import TrainingConfig, PremiumFeatures

class PremiumParaphraserTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model and tokenizer
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        
        # Setup LoRA if enabled
        if config.use_lora:
            self._setup_lora()
            
        # Initialize metrics
        self.bleu = evaluate.load("bleu")
        self.rouge = evaluate.load("rouge")
        self.semantic_similarity = evaluate.load("sentence_transformers")
        
    def _setup_lora(self):
        """Configure LoRA for parameter-efficient fine-tuning"""
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=["q_proj", "v_proj"]
        )
        self.model = get_peft_model(self.model, peft_config)
        
    def preprocess_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess the dataset for training"""
        inputs = examples["source"]
        targets = examples["target"]
        
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.config.max_source_length,
            truncation=True,
            padding="max_length"
        )
        
        labels = self.tokenizer(
            targets,
            max_length=self.config.max_target_length,
            truncation=True,
            padding="max_length"
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def compute_metrics(self, eval_preds):
        """Compute evaluation metrics"""
        preds, labels = eval_preds
        
        # Decode predictions and labels
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Calculate metrics
        bleu_score = self.bleu.compute(predictions=decoded_preds, references=decoded_labels)
        rouge_score = self.rouge.compute(predictions=decoded_preds, references=decoded_labels)
        
        # Calculate semantic similarity
        semantic_scores = []
        for pred, label in zip(decoded_preds, decoded_labels):
            similarity = self.semantic_similarity.compute(
                predictions=[pred],
                references=[label],
                model_name="all-MiniLM-L6-v2"
            )
            semantic_scores.append(similarity["score"])
        
        return {
            "bleu": bleu_score["bleu"],
            "rouge": rouge_score["rouge1"],
            "semantic_similarity": np.mean(semantic_scores)
        }
    
    def train(self):
        """Main training function"""
        # Load dataset
        dataset = load_dataset(
            "json",
            data_files={
                "train": self.config.train_file,
                "validation": self.config.validation_file
            }
        )
        
        # Preprocess dataset
        tokenized_datasets = dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        
        # Setup training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="semantic_similarity"
        )
        
        # Initialize trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForSeq2Seq(self.tokenizer),
            compute_metrics=self.compute_metrics
        )
        
        # Train the model
        trainer.train()
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        return trainer

if __name__ == "__main__":
    config = TrainingConfig()
    trainer = PremiumParaphraserTrainer(config)
    trainer.train() 