import torch
import torch.nn as nn
import math
from typing import Dict, List, Optional, Tuple, Union

HIDDEN_DIM = 576
INTERMEDIATE_DIM = 1536
VOCAB_SIZE = 49152

class SmolLM2(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.norm = nn.RMSNorm(HIDDEN_DIM, eps=1e-5)
        self.layers = nn.ModuleList([TransformerBlock() for _ in range(30)])
    
    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[List[Dict[str, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]]:
        x = self.emb(x)

        new_kv_cache = [] if use_cache else None

        for i, block in enumerate(self.layers):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, updated_kv = block(x, kv_cache=layer_cache)

            if use_cache:
                new_kv_cache.append(updated_kv)


        x = self.norm(x)
        logits = x @ self.emb.weight.T

        if use_cache:
            return logits, new_kv_cache
        else:
            return logits


class TransformerBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.attn = GroupedQueryAttention()
        self.ffn = SwiGLU()
        self.norm1 = nn.RMSNorm(HIDDEN_DIM, eps=1e-5)
        self.norm2 = nn.RMSNorm(HIDDEN_DIM, eps=1e-5)
        
    
    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        #input: 576-dim vector

        #normalize, pass through GQA, residual connection
        attn_out, updated_kv = self.attn(self.norm1(x), kv_cache=kv_cache)
        x = x + attn_out
        #normalize again, pass through SwiGLU-based ffn, residual connection
        x = x + self.ffn(self.norm2(x))
        #output: 576-dim vector
        return x, updated_kv


class GroupedQueryAttention(nn.Module):

    def __init__(self, dropout: float = 0.05):
        super().__init__()
        self.num_q_heads = 9
        self.num_kv_heads = 3
        self.head_dim = HIDDEN_DIM // self.num_q_heads

        self.q_proj = nn.Linear(HIDDEN_DIM, self.head_dim * self.num_q_heads, bias=False)
        self.k_proj = nn.Linear(HIDDEN_DIM, self.head_dim * self.num_kv_heads, bias=False)
        self.v_proj = nn.Linear(HIDDEN_DIM, self.head_dim * self.num_kv_heads, bias=False)
        self.o_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.rotary_emb = RotaryEmbedding(dim=self.head_dim)
        self.dropout = nn.Dropout(dropout)
    

    def _apply_rotary(self, x, cos, sin):
        # x: (batch, heads, seq_len, head_dim)
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        x_rot = torch.stack((-x2, x1), dim=-1).reshape_as(x)
        return (x * cos) + (x_rot * sin)
    

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        #input: 576-dim vector
        B, T, _ = x.shape

        #determine starting pos for rotary embeddings
        if kv_cache is not None:
            start_pos = kv_cache['k'].shape[2]
        else:
            start_pos = 0

        #project into Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        #reshape from (B, T, D) --> (B, heads, T, D_head)
        q = q.view(B, T, self.num_q_heads, self.head_dim).transpose(1,2)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1,2)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1,2)

        #apply rotary embedding to Q, K
        cos, sin = self.rotary_emb(seq_len=start_pos + T)
        cos = cos[:, :, start_pos:start_pos + T, :]
        sin = sin[:, :, start_pos:start_pos + T, :]
        q = self._apply_rotary(q, cos, sin)
        k = self._apply_rotary(k, cos, sin)

        #concat kv cache if exists
        if kv_cache is not None:
            k = torch.cat([kv_cache['k'], k], dim=2)
            v = torch.cat([kv_cache['v'], v], dim=2)

        #expansion --> turn k,v to (B, q_heads, T, head_dim)
        repeat_factor = self.num_q_heads // self.num_kv_heads
        k_expanded = k.repeat_interleave(repeat_factor, dim=1)
        v_expanded = v.repeat_interleave(repeat_factor, dim=1)

        #calculate attn scores
        attn_scores = torch.matmul(q, k_expanded.transpose(-2,-1))
        attn_scores /= math.sqrt(self.head_dim)

        full_seq_len = k.shape[2]
        mask = torch.tril(torch.ones(full_seq_len, full_seq_len, device=x.device)).bool()
        mask = mask[-T:, :]

        attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        c = torch.matmul(attn_probs, v_expanded)

        #merge heads
        c = c.transpose(1, 2).contiguous()
        c = c.view(B, T, HIDDEN_DIM)

        #output projection: 576-dim vector
        out = self.o_proj(c)

        #return output and updated cache
        updated_cache = {'k': k, 'v': v}

        return out, updated_cache


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 100000.0,
        dtype=torch.float32,
    ):
        super().__init__()

        if dim % 2 != 0:
            raise ValueError("RoPE head dimension must be even")

        self.dim = dim
        self.base = base
        self.max_seq_len_cached = max_position_embeddings

        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2, dtype=dtype) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # initialize cache
        self._build_cache(self.max_seq_len_cached)


    def _build_cache(self, seq_len: int):

        self.max_seq_len_cached = seq_len

        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)

        emb = torch.cat([freqs, freqs], dim=-1)

        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)


    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        returns: cos, sin of shape (1, 1, seq_len, head_dim)
        """

        if seq_len > self.max_seq_len_cached:
            self._build_cache(seq_len)

        return (
            self.cos_cached[:, :, :seq_len, :],
            self.sin_cached[:, :, :seq_len, :]
        )
    

class SwiGLU(nn.Module):

    def __init__(self, dtype=torch.bfloat16, init_std: float = 0.041666666666666664, dropout: float = 0.05):
        super().__init__()
        self.activation = nn.SiLU()
        self.gate_proj = nn.Linear(HIDDEN_DIM, INTERMEDIATE_DIM, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(HIDDEN_DIM, INTERMEDIATE_DIM, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(INTERMEDIATE_DIM, HIDDEN_DIM, bias=False, dtype=dtype)
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters(init_std)


    def reset_parameters(self, std: float):
        for layer in (self.gate_proj, self.up_proj, self.down_proj):
            nn.init.normal_(layer.weight, mean=0.0, std=std)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #input: 576-dim vector

        #project into 2, 1536-dim vectors
        gate = self.activation(self.gate_proj(x)) #also apply silu activation to gate projection
        up = self.up_proj(x)
        #multiply gate by up, re-project to 576-dim vector
        x = gate * up
        x = self.dropout(x)
        x = self.down_proj(x)
        #output: 576-dim vector
        return x
