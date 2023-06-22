import math

def model_size(model):
    """Returns the size of the model in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))

def calc_head_expansion(vocab_size, hidden_dim, num_heads, min_subvector_size=32):
    expansion = 2**round(math.log(vocab_size/hidden_dim, 2))
    subvector_size = max(min_subvector_size, hidden_dim//(expansion*num_heads))
    return hidden_dim//subvector_size