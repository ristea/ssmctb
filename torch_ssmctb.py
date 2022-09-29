import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class ChannelWiseTransformerBlock(nn.Module):
    def __init__(self, num_patches, patch_dim=1, dim=64, heads=5, dim_head=64, dropout=0.):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(patch_dim)
        self.projection = nn.Linear(patch_dim ** 2, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))

        self.mha = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        x = self.avg_pool(z)
        x = x.flatten(-2)

        x = self.projection(x)
        x += self.pos_embedding

        x = self.mha(x)
        x = x.mean(-1).unsqueeze(-1).unsqueeze(-1)
        x = self.sigmoid(x)

        return z * x


# SSMCTB implementation
class SSMCTB(nn.Module):
    def __init__(self, channels, kernel_dim=1, dilation=1):
        '''
            channels: The number of filter at the output (usually the same with the number of filter from the input)
            kernel_dim: The dimension of the sub-kernels ' k' ' from the paper
            dilation: The dilation dimension 'd' from the paper
            reduction_ratio: The reduction ratio for the SE block ('r' from the paper)
        '''
        super(SSMCTB, self).__init__()
        self.pad = kernel_dim + dilation
        self.border_input = kernel_dim + 2 * dilation + 1

        self.relu = nn.ReLU()
        self.transformer = ChannelWiseTransformerBlock(num_patches=channels, patch_dim=1)

        self.conv1 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_dim)
        self.conv2 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_dim)
        self.conv3 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_dim)
        self.conv4 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_dim)

    def forward(self, x_in):
        x = F.pad(x_in, (self.pad, self.pad, self.pad, self.pad), "constant", 0)

        x1 = self.conv1(x[:, :, :-self.border_input, :-self.border_input])
        x2 = self.conv2(x[:, :, self.border_input:, :-self.border_input])
        x3 = self.conv3(x[:, :, :-self.border_input, self.border_input:])
        x4 = self.conv4(x[:, :, self.border_input:, self.border_input:])
        x = self.relu(x1 + x2 + x3 + x4)

        x = self.transformer(x)

        return x, torch.mean((x - x_in) ** 2)  # output, loss

# model = SSMCTB(32)
# model(torch.zeros((3, 32, 64, 64)))
