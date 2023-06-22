"""
Implementation of ViT model with standard linear modules replaced with their RPQ counterparts.

This is a modified version of the ViT model from the https://github.com/lucidrains/vit-pytorch repository.
"""
import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn.modules.utils import _pair
from rpq.nn import RPQLinear
import math


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, num_codebooks, rpq_codebooks: nn.ParameterList, dropout = 0., expand_heads=False):
        super().__init__()

        if expand_heads:
            expansion = (hidden_dim//dim)
        else:
            expansion = 1
        self.fc1 = RPQLinear(dim, hidden_dim, num_codebooks*expansion, shared_codebooks=True)
        self.fc1.codebooks = rpq_codebooks[2]
        self.fc2 = RPQLinear(hidden_dim, dim, num_codebooks*expansion, shared_codebooks=True)
        self.fc2.codebooks = rpq_codebooks[3]

        self.net = nn.Sequential(
            self.fc1,
            nn.GELU(),
            nn.Dropout(dropout),
            self.fc2,
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, rpq_codebooks: nn.ParameterList, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = RPQLinear(dim, inner_dim * 3, heads, bias = False, shared_codebooks=True)
        self.to_qkv.codebooks = rpq_codebooks[0]

        if project_out:
            self.out_proj = RPQLinear(inner_dim, dim, heads, shared_codebooks=True)
            self.out_proj.codebooks = rpq_codebooks[1]
            self.to_out = nn.Sequential(
                self.out_proj,
                nn.Dropout(dropout)
            )
        else:
            self.to_out = nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., expand_heads=False):
        super().__init__()
        if expand_heads:
            expansion = (mlp_dim//dim)
        else:
            expansion = 1
        self.rpq_codebooks = nn.ParameterList(
            [torch.empty(heads, 256, dim // heads),
            torch.empty(heads, 256, dim // heads),
            torch.empty(heads*expansion, 256, mlp_dim // heads),
            torch.empty(heads*expansion, 256, mlp_dim // heads)]
        )
        # init all codebooks using kaiming_uniform_
        for codebook in self.rpq_codebooks:
            nn.init.kaiming_uniform_(codebook, a=math.sqrt(5))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, self.rpq_codebooks, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, heads*(mlp_dim//dim), self.rpq_codebooks, dropout = dropout, expand_heads=expand_heads))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class RPQViT(nn.Module):
    def __init__(self, 
                 *, 
                 image_size, 
                 patch_size, 
                 num_classes, 
                 dim, 
                 depth, 
                 heads, 
                 mlp_dim, 
                 pool = 'cls', 
                 channels = 3, 
                 dim_head = 64, 
                 dropout = 0., 
                 emb_dropout = 0.,
                 expand_heads = False
                 ):
        super().__init__()
        image_height, image_width = _pair(image_size)
        patch_height, patch_width = _pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, expand_heads)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)






