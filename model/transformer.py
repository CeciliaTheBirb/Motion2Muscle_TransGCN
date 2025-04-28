import torch
import torch.nn as nn
import math
from torch.nn import TransformerEncoderLayer, TransformerEncoder


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch, d_model]
        x = x + self.pe[:x.size(0)]
        return x


class AdaLayerNorm(nn.Module):
    def __init__(self, hidden_dim, demo_dim):
        super().__init__()
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.gamma = nn.Linear(demo_dim, hidden_dim)
        self.beta = nn.Linear(demo_dim, hidden_dim)

    def forward(self, x, demo):
        """
        x: [B, T, H]
        demo: [B, D]
        """
        normed = self.layernorm(x)  # [B, T, H]
        gamma = self.gamma(demo).unsqueeze(1)  # [B, 1, H]
        beta = self.beta(demo).unsqueeze(1)    # [B, 1, H]
        return normed * (1 + gamma) + beta


class AdaTransformerBlock(nn.Module):
    def __init__(self, hidden_dim, nhead, demo_dim):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=nhead, batch_first=True)
        self.norm1 = AdaLayerNorm(hidden_dim, demo_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm2 = AdaLayerNorm(hidden_dim, demo_dim)

    def forward(self, x, demo, mask=None):
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = x + attn_output
        x = self.norm1(x, demo)
        ff_output = self.ffn(x)
        x = x + ff_output
        x = self.norm2(x, demo)
        return x


class Transformer(nn.Module):
    def __init__(self, num_joints=20, demo_dim=3, hidden_dim=128, nhead=8, num_layers=2, output_dim=402):
        super().__init__()
        self.input_width = num_joints * 3
        self.hidden_dim = hidden_dim
        self.demo_dim = demo_dim

        self.input_proj = nn.Sequential(
            nn.Conv1d(self.input_width, hidden_dim, kernel_size=1),
            nn.ReLU()
        )

        self.pos_encoder = PositionalEncoding(d_model=hidden_dim)

        self.transformer_blocks = nn.ModuleList([
            AdaTransformerBlock(hidden_dim=hidden_dim, nhead=nhead, demo_dim=demo_dim)
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)

    def forward(self, x, demo):
        """
        x: [B, T, J, 3]
        demo: [B, demo_dim]
        """
        B, T, J, C = x.shape
        assert C == 3
        x = x.view(B, T, J * C).permute(0, 2, 1)    
        x = self.input_proj(x)                      
        x = x.permute(0, 2, 1)                     

        x = self.pos_encoder(x.transpose(0, 1))    
        x = x.transpose(0, 1)                     

        for block in self.transformer_blocks:
            x = block(x, demo)                   

        x = x.permute(0, 2, 1)                    
        x = self.output_proj(x)            
        x = x.permute(0, 2, 1) 
        return x
