import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
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

    inputs = tokenizer(PROMPT1, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs1 = model(**inputs, use_cache=True, return_dict=True)
    print(f"Input:{inputs}")
    print(f"Decomposed query: {decompose_query(inputs['input_ids'][0].tolist())}")
    save(inputs['input_ids'][0].tolist(), outputs1.past_key_values)
    
    inputs = tokenizer(PROMPT2, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        base_outputs = model(**inputs, use_cache=True, return_dict=True)
    print(f"Input:{inputs}")
    print(f"Decomposed query: {decompose_query(inputs['input_ids'][0].tolist())}")

    prefix_tokens = inputs['input_ids'][0, :-1].tolist()
    last_token = inputs['input_ids'][:, -1:]
    cache = find(prefix_tokens)
    prefix_len = len(prefix_tokens)
    assert cache.get_seq_length() == prefix_len, (cache.get_seq_length(), prefix_len)
    print("Before:", cache.get_seq_length())
    kvclient_outputs = forward_with_cache(model, last_token, cache)
    print("After:", kvclient_outputs.past_key_values.get_seq_length())
    torch.testing.assert_close(
        base_outputs.logits[:, -1, :].float(),
        kvclient_outputs.logits[:, -1, :].float(),
        rtol=1e-2,
        atol=1e-2,
    )
    print("\nPASS: baseline and reused-cache logits are close.")
        