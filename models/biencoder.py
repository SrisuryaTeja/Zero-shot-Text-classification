import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class BiEncoderModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Learnable temperature
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(0.05)))

    def encode(self, texts, device):
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        outputs = self.encoder(**inputs)
        last_hidden = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"].unsqueeze(-1)

        # Mask-aware mean pooling
        embeddings = (last_hidden * attention_mask).sum(1) / attention_mask.sum(1)

        # Normalize (important for contrastive)
        embeddings = F.normalize(embeddings, dim=-1)

        return embeddings

    def forward(self, texts, all_labels, device):
        """
        texts: List[str] (B)
        labels: List[str] (M unique labels in batch)
        """

        text_emb = self.encode(texts, device)      # [B, D]
        label_emb = self.encode(all_labels, device)    # [221, D]
        
        temperature = torch.exp(self.log_temperature)
        similarity = torch.matmul(text_emb, label_emb.T)

        
        similarity = similarity / temperature

        return similarity