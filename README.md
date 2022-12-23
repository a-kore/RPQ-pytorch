# RPQ-pytorch
Reverse Product Quantization (RPQ) of weights to reduce static memory usage.

![](https://github.com/a-kore/RPQ-pytorch/raw/main/assets/rpq_diagram.gif)

<!-- Go into how the method works. -->

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Benchmarks](#benchmarks)

## Introduction

[Product quantization](https://www.pinecone.io/learn/product-quantization/) is a method for reducing the memory requirements for vector similarity search.  It reduces the memory footprint by chunking the vectors into subvectors that are each compressed into a set of codebooks with 256 codes each.  This allows us to have a set of codes that can be represented by uint8 indices instead of the full vector representation.

If we reverse this process, we can dynamically spawn a larger set of vectors from a much smaller set of codebooks containing sub-vectors and a set of randomized uint8 indices, rather than having to persistently hold a much larger set of vectors.  This can be used during the forward pass to expand/compile the weight just-in-time in order to perform the operations on the input.  

This creates a state for a model where the weights are "dormant" and expanded to their active state just before use.  This plays very well with methods like **gradient checkpointing** (and inference, similarly) where we can unpack the weights again rather then storing them.  In other words, the weights are part of the dynamic computational graph and can be forgotten/unpacked whenever they are needed.

However, this doesn't come for free, the indices inherit from a set of shared codebooks, so the larger the weights, the more likelihood that vectors generated will share sub-vectors.  This can be prevented by increasing the number of codebooks, but requires more testing to see what the minimum number of codebooks required for each implementation should be.

For instance, in the [Usage](#usage) section we define an RPQOPT model(OPT variant w/ RPQ weights) where the number of codebooks is set to the number of heads.  This is chosen abitrarily, but works well since the `hidden_dim` must be divisible by `num_codebooks`. 

The effect of having a set of entangled vectors is unknown and would require rigorous testing with standard benchmarks for comparison.  Intuitively, this would have a different outcome depending on the way the final weight structure is used.  For a vector quantization module, it could be advantageous to have codes be entangled to avoid the issue of "dead codes" and increase codebook utilization.


## Installation

```bash
pip install rpq-pytorch
```

## Usage

#### Standalone Weights

A standalone module `RPQWeight` is available as an `nn.Module` wrapper that intializes a set of dynamic PQ weight and returns the expanded set of weight vectors.

```python
from rpq.nn import RPQWeight

w = RPQWeight(num_codebooks=72, codebook_dim=128, num_vectors=9216) 

print(w.codebooks.shape, w.indices.shape) # torch.Size([72, 256, 128]) torch.Size([72, 9216])

print(w().shape) # torch.Size([72, 9216, 128])
```

#### Layers

A set of common layers are re-implemented with quantized weights.  It follows the same usage as `torch.nn` modules with an extra argument for the `num_codebooks` for each layer.  For each layer, the `out_features`/`num_embeddings` must be divisible by the `num_codebooks`.

```python
from rpq.nn import RPQLinear

layer = RPQLinear(in_features=1024, out_features=4096, num_codebooks=16)

x = torch.randn(1, 1, 1024) # (b, n, d)
y = layer(x) # (1, 1, 4096)

```

Layers implemented:

- [x] `RPQLinear`
- [x] `RPQEmbedding`*
- [ ] `RPQConv1d`
- [ ] `RPQConv2d`
- [ ] `RPQConvTranspose2d`
- [ ] `RPQConv1d`
- [ ] `RPQConvTranspose1d`
- [ ] `RPQConv3d`
- [ ] `RPQConvTranspose3d`
- [ ] `RPQBilinear`

*Note: `Embedding` layers are a lookup table and therefore very fast, as such the operation to expand the weights for `RPQEmbedding` adds a lot of time to the operation especially for a small number of tokens (10s of $\mu s$ -> 10s of ms).

#### Models

Using the layer implementations, we can implement models via drop-in replacement of their static weight counterparts.

##### RPQViT (ViT Giant)

```python
from vit_pytorch import ViT
from rpq.models.rpqvit import RPQViT
from rpq.utils import model_size

# vit_giant_patch14_336
model = ViT(
    image_size=336,
    patch_size=14,
    num_classes=1000,
    dim=1280,
    depth=32,
    heads=16,
    mlp_dim=5120,
    dropout=0.1,
    emb_dropout=0.1
)

# rpqvit_giant_patch14_336
rpq_model = RPQViT(
    image_size=336,
    patch_size=14,
    num_classes=1000,
    dim=1280,
    depth=32,
    heads=16,
    mlp_dim=5120,
    dropout=0.1,
    emb_dropout=0.1
)

model_size(model)
model_size(rpq_model)
```
```
model size: 2252.157MB
model size: 361.429MB  
```
Approximately ~6x reduction in model size.

##### RPQOPT (opt-66b)

```python

import torch
from transformers.models.opt.modeling_opt import OPTConfig
from transformers import GPT2Tokenizer
from rpq.models.rpqopt import RPQOPTModel
from rpq.utils import model_size


tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-66b")
conf = OPTConfig.from_pretrained("facebook/opt-66b")
rpq_model = RPQOPTModel(conf) # randomly initialized model

inputs = tokenizer("Hello, my dog is cute.", return_tensors="pt")

with torch.no_grad():
    outputs = rpq_model(**inputs)

model_size(rpq_model)
```
```
model size: 5885.707MB 
```
This is an RPQOPT-66b initialized at float32 precision, a static weight version (standard OPT-66b) would be **264 GB** in size. This amount to approximately ~44x reduction in size.


## Benchmarks

Due to the entanglement of the weight matrix arising as result of the inheritance from a shared set of codebooks, testing the RPQ model variants against the original methods would be important to characterize issues/tradeoffs with training stability, especially at scale.  Those tests will be displayed in the table below:

| Model | Config | Model Size | Dataset | Validation Accuracy | Epochs |
| --- | --- | --- | --- | --- | -- |
| ViT | vit_base_patch16_224 | 330MB | MNIST | 93.6 | 10 |
| RPQViT | vit_base_patch16_224 | 88MB | MINST | 96.4 | 10 |
| ViT | vit_base_patch16_224 | 330MB | CIFAR10 | TBD | 90 |
| RPQViT | vit_base_patch16_224 | 88MB | CIFAR10 | TBD | 90 |
| ViT | vit_base_patch16_224 | 330MB | Imagenet | TBD | 90 |
| RPQViT | vit_base_patch16_224 | 88MB | Imagenet | TBD | 90 |


## TODO

- [ ] Implement `RPQConv1d` layer
- [ ] Implement `RPQConv2d` layer
- [ ] Implement `RPQConv3d` layer
- [ ] Implement `RPQConvTranspose1d` layer
- [ ] Implement `RPQConvTranspose2d` layer
- [ ] Implement `RPQConvTranspose3d` layer
- [ ] Implement `RPQBilinear` layer
- [ ] Perform More benchmarks with LLMs (BERT, OPT, etc.,)
- [ ] Explore methods of conversion from pre-trained static weights to dynamic RPQ weights




