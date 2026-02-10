import os
import pickle
from typing import Optional, Tuple

import numpy as np
from datasets import load_dataset, DatasetDict
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
    ) -> DatasetDict:
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

    def get_class_weights(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute class weights from the training set."""
        train_ds = self.dataset["train"]

        threats = [str(x).lower() for x in train_ds["is_threat"]]
        threat_labels = self.encoders["threat"].transform(threats)

        if "category" in train_ds.column_names:
            cats = train_ds["category"]
        else:
            cats = ["unknown"] * len(train_ds)
        cats = [x if x is not None else "unknown" for x in cats]
        category_labels = self.encoders["category"].transform(cats)

        if "sub-category" in train_ds.column_names:
            subs = train_ds["sub-category"]
        else:
            subs = ["unknown"] * len(train_ds)
        subs = [x if x is not None else "unknown" for x in subs]
        subcategory_labels = self.encoders["subcategory"].transform(subs)

        def compute_weights(
            labels: np.ndarray,
            num_classes: int
        ) -> np.ndarray:
            counts = np.bincount(
                labels,
                minlength=num_classes
            ).astype(np.float32)
            total = counts.sum()
            raw = np.where(
                counts > 0,
                total / (num_classes * counts),
                1.0
            )
            # Sqrt-dampen to avoid extreme weights
            weights = np.sqrt(raw).astype(np.float32)
            # Normalize so mean weight = 1.0
            weights = weights / weights.mean()
            return weights

        threat_weights = compute_weights(
            threat_labels,
            len(self.encoders["threat"].classes_)
        )
        category_weights = compute_weights(
            category_labels,
            len(self.encoders["category"].classes_)
        )
        subcategory_weights = compute_weights(
            subcategory_labels,
            len(self.encoders["subcategory"].classes_)
        )

        return threat_weights, category_weights, subcategory_weights
