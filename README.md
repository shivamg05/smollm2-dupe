# SmolLM2-135M Implementation

A from-scratch PyTorch implementation of [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M).

## Features

- Decoder-only transformer with 30 layers, 135M parameters
- Grouped-Query Attention (9 query heads, 3 KV heads)
- SwiGLU feed-forward networks
- RoPE positional embeddings
- KV caching for efficient inference

## Installation

```bash
pip install torch torchinfo transformers
```

## Usage

### Test

```bash
python test_shapes.py
```

### Train

```bash
# Quick test with dummy data
python train.py --seq_len 128 --train_steps 100 --micro_batch_size 2

# Full training (requires real dataset)
python train.py
```

### Generate

```bash
python inference.py --prompt "Neural networks learn" --checkpoint checkpoints/<your final checkpoint>.pt
```

## Architecture

| Component | Value |
|-----------|-------|
| Layers | 30 |
| Hidden Dim | 576 |
| Intermediate Dim | 1536 |
| Vocab Size | 49,152 |
| Q Heads | 9 |
| KV Heads | 3 |
| Max Seq Length | 2048 |

## Files

- `model.py` - Model architecture
- `train.py` - Training script
- `inference.py` - Text generation
- `scheduler.py` - Learning rate scheduler
- `data.py` - Sample dataset
- `test_shapes.py` - Validation tests