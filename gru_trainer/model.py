"""Model definitions for the compact GRU agent."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn


class SmallGRU(nn.Module):
    """A lightweight yet expressive two-layer GRU language model.

    The architecture emphasises parameter efficiency without sacrificing modelling
    capacity. Key design choices:

    * Character-level inputs with a shared embedding/decoder matrix (weight tying).
    * Two stacked GRU layers with LayerNorm stabilisation between them.
    * Dropout applied to embeddings and hidden states for regularisation.

    Parameters
    ----------
    vocab_size:
        Number of tokens (including padding) in the vocabulary.
    embed_dim:
        Dimensionality of the shared embedding/hidden state. Defaults to 384 which
        balances expressiveness with a modest parameter count.
    dropout:
        Dropout probability applied to embeddings and between GRU layers.
    """

    def __init__(
        self,
        vocab_size: int,
        *,
        embed_dim: int = 384,
        num_layers: int = 2,
        dropout: float = 0.1,
        layer_norm: bool = True,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        if num_layers != 2:
            raise ValueError("SmallGRU is intentionally constrained to two layers")

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            embed_dim,
            embed_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim) if layer_norm else nn.Identity()
        self.output_dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(embed_dim, vocab_size, bias=False)

        # Tie weights for efficiency.
        self.decoder.weight = self.embedding.weight

        self._init_parameters()

    def _init_parameters(self) -> None:
        init_range = 1.0 / math.sqrt(self.embed_dim)
        nn.init.uniform_(self.embedding.weight, -init_range, init_range)
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.zeros_(self.decoder.bias) if self.decoder.bias is not None else None

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the GRU forward pass.

        Parameters
        ----------
        input_ids:
            ``LongTensor`` of shape ``(batch, seq_len)`` containing token indices.
        hidden:
            Optional initial hidden state of shape ``(num_layers, batch, embed_dim)``.

        Returns
        -------
        logits:
            Tensor of shape ``(batch, seq_len, vocab_size)`` representing unnormalised
            log probabilities for the next token at each position.
        hidden:
            Final hidden state tuple from the GRU.
        """

        embeddings = self.embedding_dropout(self.embedding(input_ids))
        output, hidden_state = self.gru(embeddings, hidden)
        output = self.norm(output)
        output = self.output_dropout(output)
        logits = self.decoder(output)
        return logits, hidden_state

    def generate(
        self,
        start_tokens: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Autoregressively sample from the model for qualitative inspection."""

        self.eval()
        hidden: Optional[torch.Tensor] = None
        tokens = [start_tokens]

        for _ in range(max_length):
            logits, hidden = self.forward(tokens[-1].unsqueeze(0), hidden)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens.append(next_token.squeeze(0))

        return torch.cat(tokens, dim=0)

