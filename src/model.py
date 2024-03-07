"""
This module implements a transformer to translate from protein (amino acid sequence) into GO terms.
It is built on the PyTorch / Lightning architecture.
Written by: Artur Jaroszewicz (@beyondtheproof)
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# import lightning as L

BATCH_SIZE: int = 8  # B

AA_MAX_LEN: int = 36000
AA_VOCAB_SIZE: int = 24 + 1  # 24 amino acids and <EOS>

GO_MAX_LEN: int = 256  # T
NUM_EMBED: int = 16  # C
GO_VOCAB_SIZE: int = 18789 + 1  # 18789 amino acids and <EOS>

HEAD_SIZE: int = NUM_EMBED


class EncoderHead(nn.Module):
    def __init__(
        self,
        num_embed: int = NUM_EMBED,
        head_size: int = HEAD_SIZE,
    ):
        super().__init__()
        self.num_embed = num_embed
        self.head_size = head_size

        # Add self-attention heads
        self.query = nn.Linear(self.num_embed, self.head_size, bias=False)
        self.key = nn.Linear(self.num_embed, self.head_size, bias=False)
        self.value = nn.Linear(self.num_embed, self.head_size, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Dot product the queries and the keys, scale by the number of channels
        # q.shape: (B, T, head_size)
        # weights.shape: (B, T, T)
        weights = q @ k.transpose(-2, -1) * (C**-0.5)
        weights = weights.softmax(dim=-1)
        # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        out = weights @ v
        return out


class DecoderHead(nn.Module):
    def __init__(
        self,
        num_embed: int = NUM_EMBED,
        head_size: int = HEAD_SIZE,
        max_len: int = GO_MAX_LEN,
    ):
        super().__init__()
        self.num_embed = num_embed
        self.head_size = head_size
        self.max_len = max_len

        # Add self-attention heads
        self.query = nn.Linear(self.num_embed, self.head_size, bias=False)
        self.key = nn.Linear(self.num_embed, self.head_size, bias=False)
        self.value = nn.Linear(self.num_embed, self.head_size, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(max_len, max_len)))

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Dot product the queries and the keys, scale by the number of channels
        # q.shape: (B, T, head_size)
        # weights.shape: (B, T, T)
        weights = q @ k.transpose(-2, -1) * (C**-0.5)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = weights.softmax(dim=-1)

        # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        out = weights @ v
        return out


class AAEncoder(nn.Module):
    def register_positional_embedding(self):
        # Positions of tokens
        position = torch.arange(self.max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.num_embed, 2) * (-math.log(10000.0) / self.num_embed)
        )
        # Positional embedding has the same dimension as token embedding
        pe = torch.zeros(self.max_len, 1, self.num_embed)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def __init__(
        self,
        vocab_size: int = AA_VOCAB_SIZE,
        num_embed: int = NUM_EMBED,
        head_size: int = HEAD_SIZE,
        max_len: int = AA_MAX_LEN,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.num_embed = num_embed
        self.head_size = head_size
        self.max_len = max_len
        self.num_heads = 1

        self.embedding_table = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.num_embed
        )
        self.register_positional_embedding()
        self.self_attention = nn.Sequential(
            *[
                EncoderHead(
                    num_embed=self.num_embed,
                    head_size=self.head_size,
                )
                for _ in range(self.num_heads)
            ]
        )

    def forward(self, x: torch.Tensor):
        """
        Returns a positional, self-attention embedding of input tokens which are amino acid residues
        """
        # Takes an input of shape (batch, time [max_len], channel [num_embed])

        # Truncate the input if needed
        x = x[-self.max_len :]
        x = self.embedding_table(x)  # embed
        x = x + self.pe[: x.size(0)]  # add positional embedding
        x = x + self.self_attention(x)  # add self-attention

        return x


class GODecoder(nn.Module):
    def register_positional_embedding(self):
        # Positions of tokens
        position = torch.arange(self.max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.num_embed, 2) * (-math.log(10000.0) / self.num_embed)
        )
        # Positional embedding has the same dimension as token embedding
        pe = torch.zeros(self.max_len, 1, self.num_embed)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def __init__(
        self,
        vocab_size: int = GO_VOCAB_SIZE,
        num_embed: int = NUM_EMBED,
        head_size: int = HEAD_SIZE,
        max_len: int = GO_MAX_LEN,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.num_embed = num_embed
        self.head_size = head_size
        self.max_len = max_len
        self.num_heads = 1

        self.embedding_table = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.num_embed
        )
        self.register_positional_embedding()
        self.self_attention = nn.Sequential(
            *[
                DecoderHead(
                    num_embed=self.num_embed,
                    head_size=self.head_size,
                    max_len=self.max_len,
                )
                for _ in range(self.num_heads)
            ]
        )

    def forward(self, x: torch.Tensor):
        """
        Returns a positional, self-attention embedding of input tokens which are amino acid residues
        """
        # Takes an input of shape (batch, time [max_len], channel [num_embed])

        # Truncate the input if needed
        x = x[-self.max_len :]

        x = self.embedding_table(x)  # embed
        x = x + self.pe[: x.size(0)]  # add positional embedding
        x = x + self.self_attention(x)  # add self-attention

        return x


class EncoderDecoderHead(nn.Module):
    def __init__(self, n_embed: int = NUM_EMBED):
        super().__init__()
        self.n_embed = n_embed

        # Encoder
        self.encoder_key = nn.Linear(n_embed, n_embed, bias=False)
        self.encoder_value = nn.Linear(n_embed, n_embed, bias=False)

        # Decoder
        self.decoder_query = nn.Linear(n_embed, n_embed, bias=False)
        # self.decoder_key = nn.Linear(n_embed, n_embed, bias=False)
        # self.decoder_value = nn.Linear(n_embed, n_embed, bias=False)

    def forward(self, x_enc: torch.Tensor, x_dec: torch.Tensor):
        B_enc, T_enc, C_enc = x_enc.shape
        # B_dec, T_dec, C_dec = x_dec.shape

        dec_q = self.decoder_query(x_dec)
        enc_k = self.encoder_key(x_enc)
        enc_v = self.encoder_value(x_enc)

        weights = dec_q @ enc_k.transpose(-2, -1) * (C_enc**0.5)
        weights = weights.softmax(dim=-1)

        out = weights @ enc_v
        return out + x_dec


class FeedForward(nn.Module):
    def __init__(self, n_embed: int = NUM_EMBED, inner_scaling: int = 1):
        super().__init__()
        self.n_embed = n_embed
        self.inner_scaling = inner_scaling
        self.net = nn.Sequential(
            nn.Linear(self.n_embed, self.n_embed * self.inner_scaling),
            nn.ReLU(),
            # nn.Linear(self.n_embed * self.inner_scaling, self.n_embed),
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, dec_vocab_size: int = GO_VOCAB_SIZE):
        super().__init__()
        self.dec_vocab_size = dec_vocab_size

        self.encoder = AAEncoder()
        self.decoder = GODecoder()

        self.cross_attention_head = EncoderDecoderHead()
        self.feed_fwd = FeedForward()
        self.out_head = nn.Linear(
            self.feed_fwd.n_embed * self.feed_fwd.inner_scaling,
            self.dec_vocab_size,
        )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        # enc_input: torch.Tensor,
        # dec_input: torch.Tensor,
        # train: bool = False
    ):

        enc_input = data["seq"]
        dec_input = data["go"]

        # Ensure data has 3 dimensions
        enc_input = enc_input.view(-1, enc_input.size(-1))
        dec_input = dec_input.view(-1, dec_input.size(-1))

        # If we're training, we keep the decoder input as targets.
        # Otherwise, we take the last token, which is <EOS>
        # if not train:
        #     dec_input = dec_input[..., 0]

        # Get embeddings for encoder and decoder
        enc_embed = self.encoder(enc_input)
        dec_embed = self.decoder(dec_input[:, :-1])

        # Cross attention (includes residual)
        x = self.cross_attention_head(enc_embed, dec_embed)

        # Final dense layer
        x = self.feed_fwd(x)

        # Get the logits
        logits = self.out_head(x)

        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = dec_input[:, 1:].view(B * T)
        loss = F.cross_entropy(logits, targets)

        return logits, loss
