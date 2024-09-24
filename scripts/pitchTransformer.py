import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(
            torch.ones(features)
        )  # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features))  # bias is a learnable parameter

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        std = x.std(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class audioPitchTransformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, output_dim, dropout=0.1):
        super(audioPitchTransformer, self).__init__()
        self.layerNorm = LayerNormalization(input_size)
        self.projection = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(128, d_model),
        )
        self.positional_encoding = PositionalEncoding(
            d_model=d_model, dropout=dropout, max_len=400
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        x = self.layerNorm(x)
        x = self.projection(x)
        x = self.positional_encoding(x)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, d_model)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, d_model)
        x = self.fc(x)
        return x
