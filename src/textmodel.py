"""A small but REAL transformer text classifier.

Embedding -> 2 TransformerEncoder layers -> mean pooling -> linear head.
Big enough that optimization is measurable, small enough to train on any
laptop CPU in about a minute.
"""
import torch
import torch.nn as nn


class TinySentimentTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        max_len: int = 32,
        num_classes: int = 2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True,   # inputs are (batch, seq, features)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (batch, seq) of token ids
        seq_len = input_ids.shape[1]
        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        x = self.encoder(x)          # (batch, seq, d_model)
        x = x.mean(dim=1)            # mean pooling over the sequence
        return self.head(x)          # (batch, num_classes) logits