import torch
from model import SmolLM2
from torchinfo import summary

VOCAB_SIZE = 49152
HIDDEN_DIM = 576
NUM_LAYERS = 30
NUM_Q_HEADS = 9
NUM_KV_HEADS = 3
HEAD_DIM = HIDDEN_DIM // NUM_Q_HEADS

B = 2   #batch size
T = 8   #sequence length

torch.manual_seed(0)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    model = SmolLM2().to(device).to(dtype)
    model.eval()

    #forward pass (no cache)
    input_ids = torch.randint(0, VOCAB_SIZE, (B, T), device=device)

    print("\nModel summary:")
    summary(
        model,
        input_data=input_ids,
        verbose=2,
        col_names=("input_size", "output_size", "num_params"),
    )
    model.eval()

    with torch.no_grad():
        logits = model(input_ids)

    print("✓ Forward (no cache)")

    assert logits.shape == (B, T, VOCAB_SIZE), \
        f"Expected logits {(B, T, VOCAB_SIZE)}, got {logits.shape}"

    # prefill with KV cache
    with torch.no_grad():
        logits_prefill, kv_cache = model(input_ids, use_cache=True)

    print("✓ Prefill with KV cache")

    assert logits_prefill.shape == (B, T, VOCAB_SIZE)
    assert isinstance(kv_cache, list)
    assert len(kv_cache) == NUM_LAYERS

    for layer_cache in kv_cache:
        k, v = layer_cache["k"], layer_cache["v"]
        assert k.shape == v.shape
        assert k.shape == (B, NUM_KV_HEADS, T, HEAD_DIM)

    #decode one token with cache
    next_token = torch.randint(0, VOCAB_SIZE, (B, 1), device=device)

    with torch.no_grad():
        logits_decode, kv_cache_2 = model(
            next_token,
            kv_cache=kv_cache,
            use_cache=True
        )

    print("✓ Single-token decode with KV cache")

    assert logits_decode.shape == (B, 1, VOCAB_SIZE)
    assert len(kv_cache_2) == NUM_LAYERS

    for old, new in zip(kv_cache, kv_cache_2):
        assert new["k"].shape[2] == old["k"].shape[2] + 1
        assert new["v"].shape[2] == old["v"].shape[2] + 1

    #cache vs no-cache equivalence
    full_input = torch.cat([input_ids, next_token], dim=1)

    with torch.no_grad():
        logits_full = model(full_input)
        logits_cached, _ = model(next_token, kv_cache=kv_cache, use_cache=True)

    last_logits_full = logits_full[:, -1, :]
    last_logits_cached = logits_cached[:, -1, :]

    max_diff = (last_logits_full - last_logits_cached).abs().max().item()

    print(f"✓ Cache equivalence check (max diff = {max_diff:.6e})")

    assert torch.allclose(
        last_logits_full,
        last_logits_cached,
        atol=1e-5,
        rtol=1e-5,
    ), "Cached and non-cached logits do not match"

    print("\nAll shape and cache checks passed ✅")


if __name__ == "__main__":
    main()
