import json
import os
import pickle

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
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
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
        processed_data = {
            "input_ids": [],
            "attention_mask": [],
            "labels_threat": [],
            "labels_category": [],
            "labels_subcategory": []
        }

        for item in self.raw_data:
            # Tokenize
            encodings = tokenizer(
                item['prompt'],
                truncation=True,
                padding="max_length",
                max_length=max_length
            )
            
            t_label = str(item['is_threat']).lower()
            c_label = item.get('category', 'unknown')
            s_label = item.get('sub-category', 'unknown')

            processed_data["input_ids"].append(encodings["input_ids"])
            processed_data["attention_mask"].append(
                encodings["attention_mask"]
            )
            processed_data["labels_threat"].append(
                self.encoders['threat'].transform([t_label])[0]
            )
            processed_data["labels_category"].append(
                self.encoders['category'].transform([c_label])[0]
            )
            processed_data["labels_subcategory"].append(
                self.encoders['subcategory'].transform([s_label])[0]
            )

        return HFDataset.from_dict(processed_data).with_format("torch")
