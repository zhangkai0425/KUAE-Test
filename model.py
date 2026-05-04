import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_musa
except ImportError:
    torch_musa = None


def sync_device(device):
    if device.type == "musa":
        torch.musa.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


# 🔥 自己实现的 MultiHeadAttention（避免官方算子）
class SimpleMultiheadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        assert d_model % nhead == 0

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.nhead, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.out_proj(out)

        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device, x.dtype)


class StabilizerEmbedder(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()

        self.event_linear = nn.Linear(1, d_model)

        self.resnet = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, 3, padding=1),
            nn.ReLU(),
        )

        self.pos = PositionalEncoding(d_model)

    def forward(self, event):
        x = self.event_linear(event)

        x = x.permute(0, 2, 1)
        x = x + self.resnet(x)
        x = x.permute(0, 2, 1)

        return self.pos(x)


class SyndromeTransformerLayer(nn.Module):
    def __init__(self, d, d_model=128, nhead=4, dropout=0.1):
        super().__init__()

        self.d = d

        self.self_attn = SimpleMultiheadAttention(d_model, nhead, dropout)

        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        B, L, D = x.shape

        # attention
        attn = self.self_attn(x)
        x = self.norm1(x + attn)

        # FFN
        ff = self.linear2(F.relu(self.linear1(x)))
        x = self.norm2(x + ff)

        # 2D reshape
        H = self.d + 1
        x2d = x.view(B, H, H, D).permute(0, 3, 1, 2)

        x2d = self.conv(x2d)

        x2d = x2d.permute(0, 2, 3, 1).reshape(B, L, D)

        return x + x2d


class RNNCore(nn.Module):
    def __init__(self, d, d_model=128, nhead=4):
        super().__init__()

        self.proj = nn.Linear(d_model, d_model)

        self.layers = nn.ModuleList(
            [SyndromeTransformerLayer(d, d_model, nhead) for _ in range(3)]
        )

    def forward(self, state, embed):
        x = (state + embed) * 0.707
        x = self.proj(x)

        for layer in self.layers:
            x = layer(x)

        return x


class Readout(nn.Module):
    def __init__(self, d, d_model=128):
        super().__init__()

        self.d = d
        self.linear = nn.Linear(d_model * d, 1)

        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, 2),
            nn.ReLU(),
        )

    def forward(self, x):
        B, L, D = x.shape

        H = self.d + 1

        x = x.view(B, H, H, D).permute(0, 3, 1, 2)
        x = self.conv(x)

        x = x.mean(dim=3)
        x = x.reshape(B, -1)

        return torch.sigmoid(self.linear(x))


class MaskedBlockDecoder(nn.Module):
    def __init__(self, d, core_size, buffer_size, d_model=192, nhead=4):
        super().__init__()

        self.d = d
        self.d_model = d_model

        self.embed = StabilizerEmbedder(d_model)
        self.core = RNNCore(d, d_model, nhead)
        self.readout = Readout(d, d_model)

        self.latency_us = 0.0

    def forward(self, x):
        B, N, H, W = x.shape
        L = H * W

        x = x.view(B, N, L, 1)

        device = x.device

        state = torch.zeros(B, L, self.d_model, device=device, dtype=x.dtype)

        sync_device(device)
        start = time.perf_counter_ns()

        for t in range(N):
            emb = self.embed(x[:, t])
            state = self.core(state, emb)

        out = self.readout(state)

        sync_device(device)
        end = time.perf_counter_ns()

        self.latency_us = (end - start) / 1000.0

        return out