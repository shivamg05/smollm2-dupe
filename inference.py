import torch
from model import SmolLM2
from transformers import AutoTokenizer
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_200000.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading tokenizer...")
    tokenizer = load_tokenizer()

    print(f"Loading model from {args.checkpoint}...")
    model = load_model(device, args.checkpoint)

    print(f"Generating text...")
    output = generate(
        model=model, 
        tokenizer=tokenizer, 
        prompt=args.prompt, 
        max_new_tokens=args.max_new_tokens, 
        device=device
    )

    print(f"\nGenerated text:\n{output}")


@torch.no_grad()
def generate(model: SmolLM2, tokenizer, prompt: str, max_new_tokens: int = 200, device: str="cpu") -> str:
    
    #tokenize prompt
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = encoded["input_ids"].to(device)

    #prefill to build kv cache
    logits, kv_cache = model(input_ids, use_cache=True)

    #first token
    next_token_logits = logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    generated = torch.cat([input_ids, next_token], dim=1)

    for _ in range(max_new_tokens - 1):
        logits, kv_cache = decode_step(model, next_token, kv_cache)

        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=1)

        if tokenizer.eos_token_id is not None:
            if (next_token == tokenizer.eos_token_id).all():
                break
        
    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)

    return output_text


@torch.no_grad()
def decode_step(model: SmolLM2, input_ids: torch.Tensor, kv_cache):
    logits, new_kv_cache = model(input_ids, kv_cache=kv_cache, use_cache=True)
    return logits, new_kv_cache


def load_model(device: str, checkpoint_path: str) -> SmolLM2:
    model = SmolLM2().to(device).to(torch.bfloat16)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def load_tokenizer() -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", use_fast=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


if __name__ == "__main__":
    main()