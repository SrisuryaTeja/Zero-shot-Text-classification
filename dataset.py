import json
import torch
from collections import Counter
from torch.utils.data import Dataset


class ZeroShotDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            self.data = json.load(f)

        # Normalize label casing
        for item in self.data:
            item["labels"] = [l.lower().strip() for l in item["labels"]]
            
        label_set = set()
        for item in self.data:
            for l in item["labels"]:
                label_set.add(l)

        self.all_labels = sorted(list(label_set))
        self.label_to_index = {l: i for i, l in enumerate(self.all_labels)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch, label_to_index, num_labels):

    texts = [item["text"] for item in batch]

    B = len(batch)
    M = num_labels

    positive_mask = torch.zeros(B, M)

    for i, item in enumerate(batch):
        for l in item["labels"]:
            idx = label_to_index[l]
            positive_mask[i, idx] = 1

    return texts, positive_mask