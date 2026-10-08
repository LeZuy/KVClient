import torch
import numpy as np
from transformers.cache_utils import DynamicCache

KVShape =  list[dict[str, torch.Size]]

def flatten_tensor(_nd_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    return _nd_tensor.reshape(-1), _nd_tensor.shape

def unflatten_tensor(_1d_tensor: torch.Tensor, 
                    shape: torch.Size, 
                    offset: int ) -> torch.Tensor:
    numel = shape.numel()
    tensor = _1d_tensor[offset:offset + numel].reshape(shape)
    return tensor, offset + numel

def flatten_dynacache(kv_cache: DynamicCache) -> tuple[torch.Tensor, KVShape]:
    tensors: list[torch.Tensor] = []
    shapes: list[dict[str, torch.Size]] = []

    for layer in kv_cache.layers:
        keys, k_shape = flatten_tensor(layer.keys)
        values, v_shape = flatten_tensor(layer.values)
        tensors.extend([keys, values])

        shapes.append({"key": k_shape,
                       "value": v_shape})

    return torch.cat(tensors), shapes

def unflatten_dynacache(_1d_tensor: torch.Tensor,
                       shapes: list[dict[str, torch.Size]]) -> DynamicCache:
    cache = DynamicCache()
    offset = 0
    for i, shape in enumerate(shapes):
        K, offset = unflatten_tensor(_1d_tensor, shape["key"], offset)
        V, offset = unflatten_tensor(_1d_tensor, shape["value"], offset)
        cache.update(K, V, layer_idx=i)
    if offset != _1d_tensor.numel():
        raise ValueError(f"Consumed {offset}/{_1d_tensor.numel()} elements.")
    return cache

def save_flatKV(_1d_tensor: torch.Tensor, file):
    np.savetxt("kv_cache.txt",_1d_tensor.detach().cpu().float().numpy())

if __name__ == "__main__":
    pass
