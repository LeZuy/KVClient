import os
import torch
import struct
import hashlib
from dataclasses import dataclass
from transformers.cache_utils import DynamicCache

H0 = bytes(32)
KV_STORAGE_DIR = "./kv_storages"

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
    intervals = []
    for level, index, start, end in dyadic_decomposition(len(tokens)):
        segment = tokens[start - 1:end]
        intervals.append({"level": level, "index": index,
                          "interval": (start, end), "tokens": segment})
    return intervals

def hash_interval(h_prefix: bytes, l: int, i: int,
                  model_id: str, p: str, tokens: list[int]) -> bytes:
    profile_digest = bytes.fromhex(p)
    if len(h_prefix) != 32:
        raise ValueError("h_prefix is not 32 bytes")
    elif len(profile_digest) != 32:
        raise ValueError("profile fingerprint is not 32 bytes")
    model_bytes = model_id.encode("utf-8")
    h = hashlib.sha256()
    h.update(h_prefix)
    h.update(struct.pack(">QQ", l, i))
    h.update(struct.pack(">I", len(model_bytes)))
    h.update(model_bytes)
    h.update(profile_digest)
    h.update(struct.pack(">Q", len(tokens)))
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

def save(Q: list, m: str, p: str, KV: DynamicCache) -> list[dict]:
    os.makedirs(KV_STORAGE_DIR, exist_ok=True)
    saved = []
    h_prefix = {0: H0}
    for l, i, start, end in all_dyintervals(len(Q)):
        h_j = hash_interval(h_prefix[start - 1], l, i, m, p, Q[start - 1:end])
        h_prefix[end] = h_j
        kv = slice_kv_cache(KV, (start, end))
        path = KV_STORAGE_DIR + f"/{h_j.hex()}.pt"
        torch.save(kv, path)
        saved.append({"level": l,
                      "index": i,
                      "interval": (start, end),
                      "path": path
                    })
        print(f"Saved [{start},{end}] level={l}, index={i}")
    return saved

def find(Q: list, m: str, p: str) -> DynamicCache:
    cache = DynamicCache()
    kv_segs = []
    h_j = H0
    for l, i, start, end in dyadic_decomposition(len(Q)):
        h_j = hash_interval(h_j, l, i, m, p, Q[start - 1:end])
        path = KV_STORAGE_DIR + f"/{h_j.hex()}.pt"
        print(f"Finding file: {path}")
        kv_segs.append(torch.load(path))

    for layer_idx in range(len(kv_segs[0])):
        layer_segments = [segment[layer_idx] for segment in kv_segs]
        keys, values = concat_kv_cache(layer_segments)
        cache.update(keys,values,layer_idx)
        
    return cache