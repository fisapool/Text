import json
import os
from typing import List, Dict, Any
from datasets import load_dataset
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

class DataPreparator:
    def __init__(self, output_dir: str = "data"):
        self.output_dir = output_dir
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        os.makedirs(output_dir, exist_ok=True)
        
    def prepare_premium_dataset(self, 
                              source_texts: List[str],
                              target_texts: List[str],
                              styles: List[str] = None,
                              min_similarity: float = 0.7,
                              test_size: float = 0.2):
        """
        Prepare a premium dataset for fine-tuning
        
        Args:
            source_texts: List of original texts
            target_texts: List of paraphrased texts
            styles: List of style categories for each pair
            min_similarity: Minimum semantic similarity threshold
            test_size: Proportion of data to use for validation
        """
        # Validate input lengths
        assert len(source_texts) == len(target_texts), "Source and target lists must have same length"
        if styles:
            assert len(styles) == len(source_texts), "Styles list must match source/target length"
        
        # Calculate semantic similarities
        similarities = []
        for src, tgt in zip(source_texts, target_texts):
            similarity = self._calculate_similarity(src, tgt)
            similarities.append(similarity)
        
        # Filter based on similarity threshold
        valid_indices = [i for i, sim in enumerate(similarities) if sim >= min_similarity]
        
        filtered_sources = [source_texts[i] for i in valid_indices]
        filtered_targets = [target_texts[i] for i in valid_indices]
        filtered_styles = [styles[i] if styles else "neutral" for i in valid_indices]
        
        # Split into train and validation sets
        train_sources, val_sources, train_targets, val_targets, train_styles, val_styles = train_test_split(
            filtered_sources, filtered_targets, filtered_styles,
            test_size=test_size,
            random_state=42
        )
        
        # Prepare datasets
        train_data = self._prepare_dataset(train_sources, train_targets, train_styles)
        val_data = self._prepare_dataset(val_sources, val_targets, val_styles)
        
        # Save datasets
        self._save_dataset(train_data, "train.json")
        self._save_dataset(val_data, "validation.json")
        
        return {
            "train_size": len(train_data),
            "val_size": len(val_data),
            "avg_similarity": np.mean(similarities),
            "min_similarity": np.min(similarities),
            "max_similarity": np.max(similarities)
        }
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        embeddings1 = self.similarity_model.encode([text1])
        embeddings2 = self.similarity_model.encode([text2])
        similarity = np.dot(embeddings1, embeddings2.T)[0][0]
        return float(similarity)
    
    def _prepare_dataset(self, 
                        sources: List[str],
                        targets: List[str],
                        styles: List[str]) -> List[Dict[str, Any]]:
        """Prepare dataset in the required format"""
        return [
            {
                "source": src,
                "target": tgt,
                "style": style,
                "metadata": {
                    "length_ratio": len(tgt) / len(src),
                    "word_count": len(tgt.split())
                }
            }
            for src, tgt, style in zip(sources, targets, styles)
        ]
    
    def _save_dataset(self, data: List[Dict[str, Any]], filename: str):
        """Save dataset to JSON file"""
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

def prepare_from_csv(csv_path: str, output_dir: str = "data"):
    """Prepare dataset from a CSV file"""
    preparator = DataPreparator(output_dir)
    
    # Load data from CSV
    dataset = load_dataset('csv', data_files=csv_path)
    
    # Extract texts and styles
    source_texts = dataset['train']['original_text']
    target_texts = dataset['train']['paraphrased_text']
    styles = dataset['train'].get('style', ['neutral'] * len(source_texts))
    
    # Prepare dataset
    stats = preparator.prepare_premium_dataset(
        source_texts=source_texts,
        target_texts=target_texts,
        styles=styles,
        min_similarity=0.7
    )
    
    print("Dataset preparation completed:")
    print(f"Training samples: {stats['train_size']}")
    print(f"Validation samples: {stats['val_size']}")
    print(f"Average similarity: {stats['avg_similarity']:.3f}")
    print(f"Min similarity: {stats['min_similarity']:.3f}")
    print(f"Max similarity: {stats['max_similarity']:.3f}")

if __name__ == "__main__":
    # Example usage with CSV file
    prepare_from_csv("rapidapi.csv") 