import os
import torch
import struct
import hashlib
from dataclasses import dataclass
from transformers.cache_utils import DynamicCache

@dataclass
class KVSegment:
    level: int
    index: int
    interval: tuple[int, int]
    tokens: list[int]
    K: torch.Tensor
    V: torch.Tensor

def dyadic_decomposition(n: int):
    if n <= 0:
        return []
    intervals = []
    offset = 0
    for level in range(n.bit_length()-1, -1, -1):
        size = 1 << level
        if n & size:
            start = offset + 1
            end = offset + size
            index = offset // size
            intervals.append((level, index, start, end))
            offset += size
    return intervals

def decompose_query(tokens):
    result = []
    for level, index, start, end in dyadic_decomposition(len(tokens)):
        segment = tokens[start - 1:end]
        result.append({"level": level, "index": index,
                    "interval": (start, end), "tokens": segment})
    return result

def hash_interval(h_prefix: bytes, l: int, r: int,
                  model_id: str, p: str, tokens: list[int]) -> bytes:
    h = hashlib.sha256()
    h.update(h_prefix)
    h.update(struct.pack(">QQ", l, r))
    h.update(model_id.encode())
    h.update(b"\x00")
    h.update(p.encode())
    h.update(b"\x00")

    for token in tokens:
        h.update(struct.pack(">I", token))

    return h.digest()

def slice_layer_kv(k, v, interval):
    start, end = interval
    return (
        # [batch_size, num_heads, seq_len, head_dim]
        k[:, :, start - 1:end, :].contiguous(),
        v[:, :, start - 1:end, :].contiguous(),
    )

def slice_kv_cache(past_key_values, interval):
    segments = []
    for layer in past_key_values.layers:
        k_seg, v_seg = slice_layer_kv(layer.keys, layer.values, interval)
        segments.append((k_seg, v_seg))
    return segments

def concat_kv_cache(kv_segs):
    keys, values = zip(*kv_segs)
    return torch.cat(keys, dim=2), torch.cat(values, dim=2)

def map_to_column(h: bytes, n_columns: int) -> int:
    return int.from_bytes(h, "big") % n_columns

def all_dyintervals(n: int):
    intervals = []
    for end in range(1, n + 1):
        size = end & -end
        level = size.bit_length() - 1
        start = end - size + 1
        index = (start - 1) // size
        intervals.append((level, index, start, end))
    return intervals

def save(tokens, past_key_values, output_dir="./kv_storages"):
    os.makedirs(output_dir, exist_ok=True)
    saved = []
    for level, index, start, end in all_dyintervals(len(tokens)):
        kv = slice_kv_cache(past_key_values, (start, end))
        path = output_dir + f"/kv_cache_lv{level}_idx{index}.pt"
        torch.save(kv, path)
        saved.append({"level": level,
                      "index": index,
                      "interval": (start, end),
                      "path": path
                    })
        print(f"Saved [{start},{end}] level={level}, index={index}")
    return saved

def find(tokens):
    cache = DynamicCache()
    kv_segs = []
    for item in decompose_query(tokens):
        path = f"./kv_storages/kv_cache_lv{item['level']}_idx{item['index']}.pt"
        print(f"Finding file: {path}")
        kv_segs.append(torch.load(path))

    for layer_idx in range(len(kv_segs[0])):
        layer_segments = [segment[layer_idx] for segment in kv_segs]
        keys, values = concat_kv_cache(layer_segments)
        cache.update(keys,values,layer_idx)
        
    return cache