import torch.nn as nn

def get_activation(name):
    if name.lower() == "relu":
        return nn.ReLU()
    elif name.lower() == "gelu":
        return nn.GELU()
    elif name.lower() == "silu":
        return nn.SiLU()
    else:
        raise ValueError(f"Unknown activation function: {name}")

def get_norm(name, dim):
    if name.lower() == "layer":
        return nn.LayerNorm(dim)
    elif name.lower() == "batch":
        return nn.BatchNorm1d(dim)
    elif name.lower() == "group":
        return nn.GroupNorm(32, dim)
    elif name.lower() == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown normalization function: {name}")

class ResConv(nn.Module):
    def __init__(
        self,
        dim_channels,
        kernel_size,
        bias=True,
        activation="gelu",
        norm="none",
        dropout=0.1,
    ):
        super(ResConv, self).__init__()
        self.layers = nn.Sequential(
            get_norm(norm, dim_channels),
            get_activation(activation),
            nn.Conv1d(dim_channels, dim_channels, kernel_size, padding=kernel_size // 2, bias=bias),
            get_norm(norm, dim_channels),
            get_activation(activation),
            nn.Conv1d(dim_channels, dim_channels, kernel_size, padding=kernel_size // 2, bias=bias),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        # x: [bsz, nframes, nchannels]
        return x + self.layers(x)