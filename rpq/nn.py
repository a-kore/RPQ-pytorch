import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor
from torch.nn import Parameter
from torch.nn import functional as F
from typing import Optional, List
import math
from torch.nn.modules.utils import _single, _pair
import torch.nn.init as init 


class RPQWeight(nn.Module):
    """A standalone layer that initializes an RPQ weight matrix from a given shape."""
    __constants__ = ['num_codebooks', 'codebook_dim', 'num_vectors']
    num_codebooks: int
    codebook_dim: int
    num_vectors: int
    codebooks: Tensor
    
    def __init__(self, num_codebooks: int, codebook_dim: int, num_vectors: int, 
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_dim = codebook_dim
        self.num_vectors = num_vectors
        
        self.register_buffer("indices",
                             torch.randint(high=256, size=(self.num_codebooks, self.num_vectors), 
                                           dtype=torch.uint8, device=factory_kwargs['device']))
        self.codebooks = Parameter(torch.empty(self.num_codebooks, 256, 
                                               self.codebook_dim, **factory_kwargs))            
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.normal_(self.codebooks)

    def expand(self, indices, codebooks):
        dim = codebooks.shape[-1]
        indices_expand = repeat(indices, 'h c -> h c d', d = dim)
        return codebooks.gather(dim=1, index=indices_expand.long())

    def forward(self) -> Tensor:
        return self.expand(self.indices, self.codebooks)

    def extra_repr(self) -> str:
        s = 'num_codebooks={}, codebook_dim={}, num_vectors={}'.format(
            self.num_codebooks, self.codebook_dim, self.num_vectors)
        return s


class RPQEmbedding(nn.Module):
    """A simple lookup table that stores embeddings of a fixed dictionary and size.
    
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
    
    def __init__(self, num_embeddings: int, embedding_dim: int, num_codebooks: int, 
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
        
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        
        self.register_buffer("indices",
                             torch.randint(high=256, size=(self.num_codebooks, self.num_embeddings), 
                                           dtype=torch.uint8, device=factory_kwargs['device']))
        self.codebooks = Parameter(torch.empty(self.num_codebooks, 256, 
                                               self.codebook_dim, **factory_kwargs))            
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.normal_(self.codebooks)

    def expand(self, indices, codebooks):
        dim = codebooks.shape[-1]
        indices_expand = repeat(indices, 'h c -> h c d', d = dim)
        return codebooks.gather(dim=1, index=indices_expand.long())
        
    def get_weight(self) -> Tensor:
        return rearrange(self.expand(self.indices, self.codebooks), 
                         'h c d -> c (h d )')

    def forward(self, input: Tensor) -> Tensor:
        return F.embedding(
            input, self.get_weight(), self.padding_idx, self.max_norm, 
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
    
    def __init__(self, in_features: int, out_features: int, num_codebooks: int, 
                 bias:bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_codebooks = num_codebooks
        assert self.out_features % num_codebooks == 0, 'out_features should be divisible by num_codebooks'
        self.codebook_dim = self.out_features//self.num_codebooks
        
        self.register_buffer("indices",
                             torch.randint(high=256, size=(self.num_codebooks, self.in_features), 
                                           dtype=torch.uint8, device=factory_kwargs['device']))
        self.codebooks = Parameter(torch.empty(self.num_codebooks, 256, self.codebook_dim, **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(self.out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.codebooks, a=math.sqrt(5))
        if self.bias is not None:
            # manually get fan_in
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
            
    def expand(self, indices, codebooks):
        dim = codebooks.shape[-1]
        indices_expand = repeat(indices, 'h c -> h c d', d = dim)
        return codebooks.gather(dim=1, index=indices_expand.long())
        
    def get_weight(self) -> Tensor:
        return rearrange(self.expand(self.indices, self.codebooks), 
                         'h c d -> (h d ) c')

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.get_weight(), self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


