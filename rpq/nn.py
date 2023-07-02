import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor
from torch.nn import Parameter
from torch.nn import functional as F
from typing import Optional
import math
import torch.nn.init as init


class RPQEmbedding(nn.Module):
    """A simple lookup table that stores embeddings of a fixed dictionary and size using a rpq weight matrix.
    
       For padding_idx functionality, the padded embeddings are not initialized to 0.
    
    """
    __constants__ = ['num_embeddings', 'embedding_dim', 'codebook_dim', 'max_norm',
                     'norm_type', 'scale_grad_by_freq', 'sparse', 'num_codebooks']
    num_embeddings: int
    embedding_dim: int
    num_codebooks: int
    padding_idx: Optional[float]
    max_norm: Optional[float]
    norm_type: float
    scale_grad_by_freq: bool
    weight: Tensor
    sparse: bool
    num_codebooks: int
    
    def __init__(self, num_embeddings: int, embedding_dim: int, num_codebooks: int, use_subset=True,
                 padding_idx: Optional[int] = None, max_norm: Optional[float] = None, norm_type: float = 2., 
                 scale_grad_by_freq: bool = False, sparse: bool = False, 
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        
        self.num_codebooks = num_codebooks
        assert self.embedding_dim % num_codebooks == 0, 'embedding_dim should be divisible by num_codebooks'
        self.codebook_dim = self.embedding_dim//self.num_codebooks
        
        self.use_subset = use_subset
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        self.rpqweight = RPQWeight(self.num_codebooks, self.codebook_dim, self.num_embeddings,
                                   device=factory_kwargs['device'], dtype=factory_kwargs['dtype'])
                                         
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.normal_(self.rpqweight.codebooks)

    def forward(self, input: Tensor) -> Tensor:
        if self.use_subset:
            return self.rpqweight(subset=input.flatten()).view(*input.shape, self.embedding_dim)
        else:
            return F.embedding(
                input, self.rpqweight(), self.padding_idx, self.max_norm, 
                self.norm_type, self.scale_grad_by_freq, sparse=self.sparse)

    def extra_repr(self) -> str:
        s = '{num_embeddings}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        return s.format(**self.__dict__)


class RPQLinear(nn.Module):
    """Applies linear transformation to the incoming data."""
    __constants__ = ['in_features', 'out_features', 'num_codebooks']
    in_features: int
    out_features: int
    num_codebooks: int
    codebooks: Tensor
    
    def __init__(self, in_features: int, out_features: int, num_codebooks: int, shared_codebooks=False, 
                 split='column', bias:bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_codebooks = num_codebooks
        self.split = split
        self.shared_codebooks = shared_codebooks
        
        if self.split == 'row':
            assert self.out_features % num_codebooks == 0, 'out_features should be divisible by num_codebooks'
            self.codebook_dim = self.out_features//self.num_codebooks
            self.register_buffer("codes",
                                torch.randint(high=256, size=(self.num_codebooks, self.in_features), 
                                            dtype=torch.uint8, device=factory_kwargs['device']))
        elif self.split == 'column':
            assert self.in_features % num_codebooks == 0, 'in_features should be divisible by num_codebooks'
            self.codebook_dim = self.in_features//self.num_codebooks
            self.register_buffer("codes",
                                torch.randint(high=256, size=(self.num_codebooks, self.out_features), 
                                            dtype=torch.uint8, device=factory_kwargs['device']))
        if not self.shared_codebooks:
            self.codebooks = Parameter(torch.empty(self.num_codebooks, 256, self.codebook_dim, **factory_kwargs))
        
        if bias:
            self.bias = Parameter(torch.empty(self.out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if not self.shared_codebooks:
            init.kaiming_uniform_(self.codebooks, a=math.sqrt(5))
        if self.bias is not None:
            # manually get fan_in
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
            
    def expand(self, codes, codebooks):
        codes_expand = repeat(codes, 'h c -> h c d', d = self.codebook_dim)
        return codebooks.gather(dim=1, index=codes_expand.long())
        
    def get_weight(self) -> Tensor:
        return rearrange(self.expand(self.codes, self.codebooks), 
                         'h c d -> c (h d)')

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.get_weight(), self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, num_codebooks={}, split={}'.format(
            self.in_features, self.out_features, self.num_codebooks, self.split
        )
    

class RPQWeight(nn.Module):
    """A standalone layer that initializes an RPQ weight matrix from a given full weight matrix."""
    __constants__ = ['num_codebooks', 'codebook_dim', 'num_vectors']
    num_codebooks: int
    codebook_dim: int
    num_vectors: int
    codebooks: Tensor
    
    def __init__(self, num_codebooks: int, codebook_dim: int, num_vectors: int,  shared_codebooks=False, 
                 split: str = 'column', device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_dim = codebook_dim
        self.num_vectors = num_vectors
        self.shared_codebooks = shared_codebooks
        self.split = split
        if (self.split != 'row') and (self.split != 'column'):
            raise ValueError(f"split must be either 'row' or 'column', got {self.split}")
        
        self.register_buffer("codes",
                             torch.randint(high=256, size=(self.num_codebooks, self.num_vectors), 
                                           dtype=torch.uint8, device=factory_kwargs['device']))
        if not self.shared_codebooks:
            self.codebooks = Parameter(torch.empty(self.num_codebooks, 256, 
                                                self.codebook_dim, **factory_kwargs))            
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if not self.shared_codebooks:
            init.kaiming_normal_(self.codebooks)

    def expand(self, codes, codebooks, subset):
        codes_expand = repeat(codes[:, subset], 'h c -> h c d', d=self.codebook_dim)
        weight = codebooks.gather(dim=1, index=codes_expand.long())
        weight = rearrange(weight, 'h c d -> c (h d)')
        return weight

    def forward(self, subset=slice(None)) -> Tensor:
        return self.expand(self.codes, self.codebooks, subset)

    def extra_repr(self) -> str:
        s = 'num_codebooks={}, codebook_dim={}, num_vectors={}'.format(
            self.num_codebooks, self.codebook_dim, self.num_vectors)
        return s
