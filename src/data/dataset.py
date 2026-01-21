import json
import os
import pickle

from tqdm import tqdm
from datasets import load_dataset, Dataset as HFDataset
from sklearn.preprocessing import LabelEncoder
from transformers import PreTrainedTokenizer


class ThreatDataset:
    """
    Handles loading, encoding, and processing of the threat dataset.
    """
    def __init__(self, data_path: str):
        self.data_path = data_path
        # Use load_dataset for memory efficiency (memory mapping)
        print(f"Loading dataset from {data_path} using memory mapping...")
        self.dataset = load_dataset('json', data_files=data_path, split='train')
        self.encoders = {
            "threat": LabelEncoder(),
            "category": LabelEncoder(),
            "subcategory": LabelEncoder()
        }
        self._fit_encoders()

    def _fit_encoders(self):
        """Fits label encoders on the dataset."""
        # Accessing columns triggers memory mapping reads, which is efficient
        print("Fitting encoders...")
        
        # We need to extract lists to fit encoders. this loads columns into RAM temporarily
        # but one column at a time is better than the whole dataset.
        
        def safe_get(batch, col):
            return [x if x is not None else 'unknown' for x in batch[col]]

        # Using batches or simple list comprehension if dataset fits in memory-mapped virt mem
        # For simplicity and robustness, we can just access columns if not massive.
        # If massive, we should iterate. Assuming it fits for column extraction.
        
        try:
            threats = [str(self.dataset[i]['is_threat']).lower() for i in range(len(self.dataset))]
        except KeyError:
             # Fallback or different column name handling
             threats = ['false'] * len(self.dataset)

        # Helper to get column with default
        def get_col_list(col_name, default='unknown'):
            if col_name in self.dataset.column_names:
                return [x if x is not None else default for x in self.dataset[col_name]]
            return [default] * len(self.dataset)

        categories = get_col_list('category')
        subcats = get_col_list('sub-category')

        self.encoders['threat'].fit(threats + ['true', 'false'])
        self.encoders['category'].fit(categories)
        self.encoders['subcategory'].fit(subcats)

    def save_encoders(self, output_dir: str):
        """Saves encoders for inference."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(os.path.join(output_dir, "encoders.pkl"), "wb") as f:
            pickle.dump(self.encoders, f)

    def get_hf_dataset(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        batch_size: int = 1000
    ) -> HFDataset:
        """Converts raw data to a tokenized Hugging Face Dataset."""
        
        def process_fn(batch):
            # Tokenize
            tokenized = tokenizer(
                batch['prompt'],
                truncation=True,
                padding="max_length",
                max_length=max_length
            )
            
            # Encode labels
            threats = [str(x).lower() for x in batch['is_threat']]
            labels_threat = self.encoders['threat'].transform(threats)
            
            # Handle categories (list of vals)
            cats = batch.get('category', [])
            cats = [x if x is not None else 'unknown' for x in cats]
            labels_category = self.encoders['category'].transform(cats)
            
            # Handle subcategories
            subs = batch.get('sub-category', [])
            subs = [x if x is not None else 'unknown' for x in subs]
            labels_subcategory = self.encoders['subcategory'].transform(subs)
            
            return {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "labels_threat": labels_threat,
                "labels_category": labels_category,
                "labels_subcategory": labels_subcategory
            }

        # Apply mapping
        processed = self.dataset.map(
            process_fn,
            batched=True,
            batch_size=batch_size,
            remove_columns=self.dataset.column_names,
            desc="Tokenizing and processing dataset"
        )
        
        return processed.with_format("torch")
