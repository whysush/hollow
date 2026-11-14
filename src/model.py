"""HOLLOW language model: a GPT-2-style decoder-only transformer written from scratch.

Architecture (matches Hugging Face GPT2LMHeadModel exactly, so weights can be converted in Phase 7):
  * learned token embeddings  wte: (vocab_size, n_embd)
  * learned position embeddings wpe: (block_size, n_embd)
  * n_layer blocks, each pre-LayerNorm:
        x = x + attn(ln_1(x))      causal multi-head self-attention
        x = x + mlp(ln_2(x))       Linear(n_embd -> 4*n_embd), GELU (tanh approximation, the
                                   "gelu_new" used by GPT-2), Linear(4*n_embd -> n_embd)
  * final LayerNorm ln_f, then the output projection, which is TIED to wte (same tensor).
Parameter names follow GPT-2 ("transformer.h.0.attn.c_attn.weight", ...). One difference: we use
nn.Linear, whose weight is stored (out_features, in_features); HF GPT-2 uses Conv1D, whose weight
is stored (in_features, out_features). The Phase 7 converter transposes those matrices.

Default config: n_layer=8, n_head=8, n_embd=512, block_size=512, vocab_size=16384
  parameters: wte 16384*512 = 8,388,608; wpe 512*512 = 262,144;
              per block 12*512^2 + 13*512 (weights + biases + 2 LayerNorms) = 3,152,384;
              8 blocks = 25,219,072; ln_f 1,024  => total 33,870,848 (~34M, embeddings counted once
              because the output head is tied).
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 16384   # BPE vocabulary size (Phase 4 tokenizer, includes 6 special tokens)
    block_size: int = 512     # maximum context length in tokens (= number of position embeddings)
    n_layer: int = 8          # transformer blocks
    n_head: int = 8           # attention heads; head dim = n_embd / n_head = 64
    n_embd: int = 512         # residual stream width
    dropout: float = 0.0      # 0 for pretraining (we see each token ~once, no overfitting risk)
    layer_norm_epsilon: float = 1e-5  # same epsilon as HF GPT-2 (layer_norm_epsilon=1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        # One fused projection producing q, k and v: (B, T, n_embd) -> (B, T, 3*n_embd).
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=True)
        # Output projection back into the residual stream: (B, T, n_embd) -> (B, T, n_embd).
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=True)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape                      # batch, time (tokens), channels (n_embd)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)   # each (B, T, C)
        hd = C // self.n_head                  # head dimension, 64 for the default config
        # (B, T, C) -> (B, n_head, T, hd) so attention runs independently per head.
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        # PyTorch scaled-dot-product attention: softmax(q k^T / sqrt(hd)) v with a causal mask,
        # dispatched to FlashAttention / memory-efficient kernels on the GPU.
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)      # back to (B, T, C)
        return self.resid_drop(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        # Expansion factor 4 (GPT-2 standard): 512 -> 2048 -> 512.
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=True)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=True)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        # GELU with the tanh approximation, identical to HF "gelu_new" (GPT-2's activation).
        return self.drop(self.c_proj(F.gelu(self.c_fc(x), approximate="tanh")))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))   # pre-LayerNorm residual attention
        x = x + self.mlp(self.ln_2(x))    # pre-LayerNorm residual MLP
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
            drop=nn.Dropout(cfg.dropout),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f=nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # Weight tying: the output projection reuses the token embedding matrix (vocab, n_embd).
        self.lm_head.weight = self.transformer.wte.weight
        self.apply(self._init_weights)
        # GPT-2 scaled init: residual output projections get std 0.02 / sqrt(2 * n_layer), because
        # each block adds two residual contributions and the sum's variance should stay ~constant.
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)   # GPT-2 init std 0.02
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self):
        # Tied weights are one tensor, so parameters() already counts the embedding once.
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, loss_mask=None):
        """idx: (B, T) int64 token ids with T <= block_size.
        targets: optional (B, T) int64 next-token ids.
        loss_mask: optional (B, T) float/bool; 1 where the target counts toward the loss (used in
                   supervised fine-tuning to train only on tokens after <|sty|>).
        Returns (logits (B, T, vocab), loss scalar or None)."""
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)                      # (T,)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)     # (B, T, n_embd)
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)                                      # (B, T, vocab)
        loss = None
        if targets is not None:
            # Cross-entropy in float32 for numerical stability under bf16 autocast.
            per_tok = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1),
                                      reduction="none", ignore_index=-1)
            if loss_mask is not None:
                m = loss_mask.view(-1).float()
                loss = (per_tok * m).sum() / m.sum().clamp(min=1.0)
            else:
                valid = (targets.view(-1) != -1).float()
                loss = (per_tok * valid).sum() / valid.sum().clamp(min=1.0)
        return logits, loss

    def configure_optimizer(self, weight_decay, lr, betas, device_type):
        # Weight decay applies to 2-D matrices (linear weights, embeddings) only; biases and
        # LayerNorm gains are 1-D and are not decayed (standard GPT-2 / nanoGPT practice).
        decay = [p for p in self.parameters() if p.dim() >= 2]
        no_decay = [p for p in self.parameters() if p.dim() < 2]
        groups = [{"params": decay, "weight_decay": weight_decay},
                  {"params": no_decay, "weight_decay": 0.0}]
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=(device_type == "cuda"))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_p=0.9, stop_id=None):
        """Sample up to max_new_tokens tokens. idx: (B, T). Returns (B, T + n) and per-step
        log-probs of the chosen tokens (B, n), used for mean-NLL candidate ranking."""
        logps = []
        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.block_size:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :].float() / max(temperature, 1e-5)
            probs = F.softmax(logits, dim=-1)
            # Nucleus (top-p) sampling: keep the smallest set of tokens whose probability mass
            # reaches top_p, renormalise, and sample from it.
            sp, si = torch.sort(probs, descending=True)
            cum = torch.cumsum(sp, dim=-1)
            sp[(cum - sp) > top_p] = 0.0
            sp = sp / sp.sum(dim=-1, keepdim=True)
            choice = torch.multinomial(sp, 1)
            nxt = si.gather(-1, choice)                                   # (B, 1)
            logps.append(torch.log(probs.gather(-1, nxt) + 1e-12))
            idx = torch.cat([idx, nxt], dim=1)
            if stop_id is not None:
                finished |= nxt.squeeze(1) == stop_id
                if finished.all():
                    break
        return idx, torch.cat(logps, dim=1)
