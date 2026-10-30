import torch
import torch.nn as nn
from .ast import AdaptiveSparseSA
from .semnet import SEFN
from .mona import Mona

__all__ = ['DynamicTanh', 'ASTE']

class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"


class ASTE(nn.Module):
    """Adaptive Sparse Transformer Encoder"""
    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0, act=nn.GELU(), normalize_before=False):
        super().__init__()
        self.assa = AdaptiveSparseSA(c1, num_heads=num_heads, sparseAtt=True)
        self.ffn = SEFN(c1, 2.0, False)
        self.norm1 = DynamicTanh(c1, channels_last=False)
        self.norm2 = DynamicTanh(c1, channels_last=False)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.mona1 = Mona(c1)
        self.mona2 = Mona(c1)
        self.act = act
        self.normalize_before = normalize_before

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        BS, C, H, W = src.size()
        src_ = src
        src2 = self.assa(src).permute(0, 2, 1).view([-1, C, H, W]).contiguous()
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = self.mona1(src)
        src2 = self.ffn(src, src_)
        src = src + self.dropout2(src2)
        return self.mona2(self.norm2(src))

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)
