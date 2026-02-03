SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "A short line of text for language modeling.",
    "SmolLM2 is a compact transformer model.",
    "Tokenization splits text into pieces.",
    "Training uses causal prediction over sequences.",
    "Batching improves throughput on GPUs.",
    "Attention lets models mix contextual signals.",
    "RMSNorm stabilizes the transformer layers.",
    "SwiGLU provides gated nonlinearity.",
    "Rotary embeddings encode positions in attention.",
    "Causal masks prevent peeking into the future.",
    "Learning rate schedules need warmup and decay.",
    "Weight decay regularizes large models.",
    "Checkpointing saves progress during training.",
    "Validation helps track generalization.",
]


def load_data():
    return SAMPLE_TEXTS
