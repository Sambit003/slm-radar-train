import json
import os
import pickle

from tqdm import tqdm
from datasets import Dataset as HFDataset
from sklearn.preprocessing import LabelEncoder
from transformers import PreTrainedTokenizer


class ThreatDataset:
    """
    Handles loading, encoding, and processing of the threat dataset.
    """
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.raw_data = self._load_data()
        self.encoders = {
            "threat": LabelEncoder(),
            "category": LabelEncoder(),
            "subcategory": LabelEncoder()
        }
        self._fit_encoders()

    def _load_data(self):
        """Loads JSONL data."""
        data = []
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                total_lines = sum(1 for _ in f)
        except Exception:
            total_lines = None

        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=total_lines, desc=f"Loading {os.path.basename(self.data_path)}"):
                if line.strip():
                    data.append(json.loads(line))
        return data

    def _fit_encoders(self):
        """Fits label encoders on the dataset."""
        threats = [str(x['is_threat']).lower() for x in self.raw_data]
        categories = [x.get('category', 'unknown') for x in self.raw_data]
        subcats = [x.get('sub-category', 'unknown') for x in self.raw_data]

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
        max_length: int
    ) -> HFDataset:
        """Converts raw data to a tokenized Hugging Face Dataset."""
        print("Extracting prompts and labels...")
        prompts = [item['prompt'] for item in self.raw_data]
        threats = [str(item['is_threat']).lower() for item in self.raw_data]
        categories = [item.get('category', 'unknown') for item in self.raw_data]
        subcats = [item.get('sub-category', 'unknown') for item in self.raw_data]

        print("Batch tokenizing prompts...")
        tokenized = tokenizer(
            prompts,
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

        print("Batch encoding labels...")
        labels_threat = self.encoders['threat'].transform(threats)
        labels_category = self.encoders['category'].transform(categories)
        labels_subcategory = self.encoders['subcategory'].transform(subcats)

        processed_data = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels_threat": labels_threat.tolist(),
            "labels_category": labels_category.tolist(),
            "labels_subcategory": labels_subcategory.tolist()
        }

        print("Creating HuggingFace Dataset...")
        return HFDataset.from_dict(processed_data).with_format("torch")
