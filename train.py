import argparse
import math
import os
from typing import List
from model import SmolLM2, HIDDEN_DIM, VOCAB_SIZE
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from inference import load_tokenizer
from data import load_data
from scheduler import LinearWarmupDecay
from itertools import cycle

SEQ_LEN = 2048
SEED = 8
DTYPE = torch.bfloat16


def tokenize_and_chunk(texts, tokenizer, seq_len: int) -> List[torch.Tensor]:
    token_stream = []
    for text in texts:
        enc = tokenizer(text, return_tensors="pt")
        token_stream.append(enc["input_ids"].squeeze(0))

    token_stream = torch.cat(token_stream, dim=0)
    total_len = (token_stream.size(0) // seq_len) * seq_len
    if token_stream.size(0) == 0:
        return []

    token_stream = token_stream[:total_len]
    chunks = token_stream.view(-1, seq_len)
    return [chunk for chunk in chunks]


def get_loaders(x: List[torch.Tensor], micro_batch_size=8, split=0.8):
    if not x:
        raise ValueError("No token chunks available for DataLoader.")

    x = torch.stack(x, dim=0)
    n = x.size(0)
    #ensure at least 1 sample for train
    split_idx = max(1, math.floor(n * split))

    x_train, x_val = x[:split_idx], x[split_idx:]
    
    #adjust batch size if dataset is smaller
    actual_train_batch_size = min(micro_batch_size, len(x_train))
    actual_val_batch_size = min(micro_batch_size, max(1, len(x_val)))
    
    print(f"Train samples: {len(x_train)}, Val samples: {len(x_val)}")
    print(f"Train batch size: {actual_train_batch_size}, Val batch size: {actual_val_batch_size}")

    train_ds = TensorDataset(x_train)
    val_ds = TensorDataset(x_val)

    #don't drop last batch for small datasets
    drop_last = len(x_train) > micro_batch_size
    train_loader = DataLoader(train_ds, batch_size=actual_train_batch_size, shuffle=True, drop_last=drop_last)
    val_loader = DataLoader(val_ds, batch_size=actual_val_batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader

#executes full forward + backward pass on a single input batch
def train_step(
        model: SmolLM2,
        batch,
        loss_fn,
        optimizer: torch.optim,
        scheduler,
        device: str,
        clip_grad: float):
    
    model.train()
    (input_ids,) = batch
    input_ids = input_ids.to(device)
    inputs = input_ids[:, :-1]
    labels = input_ids[:, 1:]

    optimizer.zero_grad(set_to_none=True)
    pred = model(inputs)

    loss = loss_fn(
        pred.reshape(-1, pred.size(-1)),
        labels.reshape(-1),
    )

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    return loss.item()


@torch.no_grad()
def eval_step(
        model: SmolLM2,
        loader: DataLoader,
        loss_fn,
        device: str):
    
    model.eval()
    total = 0.0
    total_tokens = 0

    for batch in loader:
        (input_ids,) = batch
        input_ids = input_ids.to(device)
        inputs = input_ids[:, :-1]
        labels = input_ids[:, 1:]

        pred = model(inputs)
        loss = loss_fn(
            pred.reshape(-1, pred.size(-1)),
            labels.reshape(-1),
        )

        total += loss.item() * labels.numel()
        total_tokens += labels.numel()

    return total / max(total_tokens, 1)


def save_checkpoint(
        step: int,
        model: SmolLM2,
        optimizer: torch.optim.Optimizer,
        scheduler: LinearWarmupDecay,
        checkpoints_path: str) -> None:
    
    os.makedirs(checkpoints_path, exist_ok=True)
    checkpoint = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": {
            "step_num": scheduler.step_num if scheduler is not None else 0,
        },
    }
    path = os.path.join(checkpoints_path, f"checkpoint_{step:06d}.pt")
    torch.save(checkpoint, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=128, help="Sequence length (reduced default for testing)")
    parser.add_argument("--train_steps", type=int, default=100, help="Number of training steps")
    args = parser.parse_args()

    micro_batch_size = 2  # Reduced for small datasets
    train_steps = args.train_steps  # Changed to use argument with sensible default
    lr = 3e-3
    weight_decay = 0.01
    lr_warmup_steps = 10  # Reduced for testing
    lr_decay_start_step = 80  # Reduced for testing
    lr_decay_steps = 20  # Reduced for testing
    min_decay_lr = 0.0
    betas = (0.9, 0.95)
    eps = 1e-8
    clip_grad = 1.0
    ckpt_interval = 50  # Reduced for testing
    val_check_interval = 10  # Reduced for testing
    checkpoints_path = "checkpoints"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(SEED)
    if device == "cuda":
        torch.cuda.manual_seed_all(SEED)

    print("Loading tokenizer...")
    tokenizer = load_tokenizer()

    #load, tokenize data
    print("Loading raw data...")
    data = load_data()
    print(f"Tokenizing {len(data)} samples...")
    data = tokenize_and_chunk(data, tokenizer, seq_len=args.seq_len)
    if not data:
        raise ValueError("Tokenized data is empty; add more samples or reduce SEQ_LEN.")
    print(f"Built {len(data)} token chunks at seq_len={args.seq_len}.")

    print("Building dataloaders...")
    train_loader, val_loader = get_loaders(data, micro_batch_size=micro_batch_size)

    print("Initializing model...")
    model = SmolLM2().to(device).to(DTYPE)
    loss_fn = nn.CrossEntropyLoss()

    print(f"Starting training on {device} (steps={train_steps}, micro_batch_size={micro_batch_size})...")
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        betas=betas, 
        eps=eps, 
        weight_decay=weight_decay, 
        fused=True if device == "cuda" else False
    )
    
    scheduler = LinearWarmupDecay(
        optimizer,
        base_lr=lr,
        warmup_steps=lr_warmup_steps,
        decay_start_step=lr_decay_start_step,
        decay_steps=lr_decay_steps,
        min_lr=min_decay_lr
    )

    #use itertools.cycle to infinitely repeat the dataloader
    train_loader_infinite = cycle(train_loader)
    
    for step in range(train_steps):
        batch = next(train_loader_infinite)
        
        train_loss = train_step(model, batch, loss_fn, optimizer, scheduler, device, clip_grad)
        out = f"step: {step:06d} | train {train_loss:.4f}"
        
        if step % val_check_interval == 0 and step != 0:
            val_loss = eval_step(model, val_loader, loss_fn, device)
            out = f"step: {step:06d} | train {train_loss:.4f} | val {val_loss:.4f}"

        if step % ckpt_interval == 0 and step != 0:
            save_checkpoint(step, model, optimizer, scheduler, checkpoints_path)
    
        print(out)
    
    print(f"\nTraining complete! Final step: {step}")

    save_checkpoint(step, model, optimizer, scheduler, checkpoints_path)


if __name__ == "__main__":
    main()