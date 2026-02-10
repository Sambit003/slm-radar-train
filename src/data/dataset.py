import os
import pickle
from typing import Optional

from datasets import load_dataset, Dataset as HFDataset
from sklearn.preprocessing import LabelEncoder
from transformers import PreTrainedTokenizer


class ThreatDataset:
    """
    Handles loading, encoding, and processing of the threat dataset.
    """
    def __init__(self, train_file: str, validation_file: Optional[str] = None,
                 test_file: Optional[str] = None):
        self.data_files = {"train": train_file}
        if validation_file:
            self.data_files["validation"] = validation_file
        if test_file:
            self.data_files["test"] = test_file

        # Use load_dataset for memory efficiency (memory mapping)
        print(f"Loading datasets from {self.data_files} using memory mapping")
        self.dataset = load_dataset('json', data_files=self.data_files)

        self.encoders = {
            "threat": LabelEncoder(),
            "category": LabelEncoder(),
            "subcategory": LabelEncoder()
        }
        self._fit_encoders()

    def _fit_encoders(self):
        """Fits label encoders on the training dataset."""
        print("Fitting encoders on training set...")
        train_ds = self.dataset['train']

        try:
            threats = [str(train_ds[i]['is_threat'])
                       .lower() for i in range(len(train_ds))]
        except KeyError:
            threats = ['false'] * len(train_ds)

        # Helper to get column with default
        def get_col_list(dataset, col_name, default='unknown'):
            if col_name in dataset.column_names:
                return [x if x is not None
                        else default for x in dataset[col_name]]
            return [default] * len(dataset)

        categories = get_col_list(train_ds, 'category')
        subcats = get_col_list(train_ds, 'sub-category')

        # Fit encoders
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

        # Apply mapping to all splits
        processed = self.dataset.map(
            process_fn,
            batched=True,
            batch_size=batch_size,
            remove_columns=self.dataset['train'].column_names,
            desc="Tokenizing and processing dataset"
        )

        return processed.with_format("torch")
