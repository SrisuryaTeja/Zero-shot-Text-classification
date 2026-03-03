import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class PolyEncoderModel(nn.Module):
    def __init__(self, model_name, num_poly_codes=16):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.num_poly_codes = num_poly_codes
        hidden_size = self.encoder.config.hidden_size


        self.poly_codes = nn.Parameter(
            torch.randn(num_poly_codes, hidden_size)
        )

        self.log_temperature = nn.Parameter(torch.log(torch.tensor(0.07)))

    def encode_tokens(self, texts, device):
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        outputs = self.encoder(**inputs)
        token_embeddings = outputs.last_hidden_state  # [B, T, D]
        attention_mask = inputs["attention_mask"].unsqueeze(-1)

        return token_embeddings, attention_mask

    def forward(self, texts, labels, device):

        # Encode text tokens
        token_emb, mask = self.encode_tokens(texts, device)
        B, T, D = token_emb.shape

        # Expand poly codes for batch
        poly_codes = self.poly_codes.unsqueeze(0).expand(B, -1, -1)  # [B, M, D]

        # Attention: poly codes attend to token embeddings
        attention_scores = torch.matmul(poly_codes, token_emb.transpose(1, 2))  # [B, M, T]
        attention_scores = attention_scores.masked_fill(
            mask.transpose(1, 2) == 0, -1e9
        )

        attention_weights = torch.softmax(attention_scores, dim=-1)

        poly_repr = torch.matmul(attention_weights, token_emb)  # [B, M, D]
        poly_repr = F.normalize(poly_repr, dim=-1)

        # Encode labels
        label_emb = self.encode_labels(labels, device)  # [L, D]

        # Interaction: labels attend over poly codes
        similarity = torch.einsum("bmd,ld->bml", poly_repr, label_emb)
        similarity = similarity.max(dim=1).values  # [B, L]

        temperature = torch.exp(self.log_temperature)
        similarity = similarity / temperature

        return similarity

    def encode_labels(self, labels, device):
        inputs = self.tokenizer(
            labels,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        outputs = self.encoder(**inputs)
        last_hidden = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"].unsqueeze(-1)

        embeddings = (last_hidden * attention_mask).sum(1) / attention_mask.sum(1)
        embeddings = F.normalize(embeddings, dim=-1)

        return embeddings