import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.key_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value_gen = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        q = self.query_gen(embedded)
        k = self.key_gen(embedded)
        v = self.value_gen(embedded)
        attn_output = F.scaled_dot_product_attention( q, k, v, is_causal=True,dropout_p=0.0) #HAHAHA much easier
        return torch.round(attn_output * 10000) / 10000