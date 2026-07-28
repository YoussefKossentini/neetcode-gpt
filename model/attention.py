import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.attention_dim = attention_dim
        self.key_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value_gen = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        k = self.key_gen(embedded)
        q = self.query_gen(embedded)
        v = self.value_gen(embedded)

        score = (q @ torch.transpose(k, 1, 2)) / (self.attention_dim ** 0.5)

        mask = torch.tril(torch.ones(k.shape[1], k.shape[1]))
        mask = mask.to(score.device)
        score = score.masked_fill(mask == 0, float('-inf'))

        score = nn.functional.softmax(score, dim=2)

        return torch.round(score @ v, decimals=4)