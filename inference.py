import copy
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from profile import extract_profile
from kvclient import decompose_query, save, find

MODEL = "Qwen/Qwen3-8B"
PROMPT1 = "Explain how KV cache works in an LLM model"
PROMPT2 = "Explain how KV cache works"

def forward_with_cache(model, new_input_ids, cache):
    cached_len = cache.get_seq_length()
    n_new = new_input_ids.shape[1]
    cache_position = torch.arange(cached_len, cached_len + n_new, device=new_input_ids.device)
    with torch.inference_mode():
        outputs = model(
            input_ids=new_input_ids,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )
    return outputs

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, local_files_only=True)
    model.eval()
    model_profile = extract_profile(model)

    inputs = tokenizer(PROMPT1, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs1 = model(**inputs, use_cache=True, return_dict=True)
    print(f"Input:{inputs}")
    print(f"Decomposed query: {decompose_query(inputs['input_ids'][0].tolist())}")
    save(inputs['input_ids'][0].tolist(), MODEL, model_profile,outputs1.past_key_values)
    
    inputs = tokenizer(PROMPT2, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        base_outputs = model(**inputs, use_cache=True, return_dict=True)
    print(f"Input:{inputs}")
    print(f"Decomposed query: {decompose_query(inputs['input_ids'][0].tolist())}")

    prefix_tokens = inputs['input_ids'][0, :-1].tolist()
    last_token = inputs['input_ids'][:, -1:]
    cache = copy.deepcopy(find(prefix_tokens, MODEL, model_profile))
    prefix_len = len(prefix_tokens)
    src = outputs1.past_key_values          # cache gốc của PROMPT1, trước khi save
    ref = base_outputs.past_key_values 
    for l in range(len(ref.layers)):
        k = cache.layers[l].keys
        d_src = (k.float() - src.layers[l].keys[:, :, :prefix_len].float()).abs().max().item()
        d_ref = (k.float() - ref.layers[l].keys[:, :, :prefix_len].float()).abs().max().item()
        print(f"layer {l:2d}  vs_src={d_src:.3e}  vs_ref={d_ref:.3e}")

    ref_cache = copy.deepcopy(src)
    ref_cache.crop(prefix_len)
    ref_out = forward_with_cache(model, last_token, ref_cache)
    kv_out  = forward_with_cache(model, last_token, copy.deepcopy(cache))
    assert torch.equal(ref_out.logits, kv_out.logits)   # phải bit-exact
            # 3) So với full prefill: chỉ kiểm tra "đủ gần" theo ý nghĩa
    a = base_outputs.logits[0, -1].float()
    b = kv_out.logits[0, -1].float()
    print("max |Δlogit|:", (a - b).abs().max().item())
    assert a.argmax() == b.argmax()
    assert torch.equal(a.topk(5).indices, b.topk(5).indices)