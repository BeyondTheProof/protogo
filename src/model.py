"""
This module implements a transformer to translate from protein (amino acid sequence) into GO terms.
It is built on the PyTorch / Lightning architecture.
Written by: Artur Jaroszewicz (@beyondtheproof)
"""

from typing import Dict, Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import lightning as L


AA_MAX_LEN: int = 36000
AA_VOCAB_SIZE: int = 24 + 2  # 24 amino acids and <SOS>, <EOS>

GO_MAX_LEN: int = 256  # T
NUM_EMBED: int = 16  # C
GO_VOCAB_SIZE: int = 18789 + 2  # 18789 GO terms and <SOS>, <EOS>

HEAD_SIZE: int = NUM_EMBED
NUM_HEADS: int = 4
NUM_ENCODER_LAYERS: int = 1
NUM_DECODER_LAYERS: int = 1
INNER_SCALING: int = 4
DROPOUT: float = 0.25


class EncoderHead(L.LightningModule):
    """
    Takes a tensor of size (Batch, Time, Channel), where |Channel| == batch_size and linearly transforms
    to (B, T, head_size), where |head_size| == batch_size, once each for `query`, `key`, and `value`

    Then does matrix multiplication between `query` and `key`, then takes softmax over last dimension,
    and multiplies it by the `value`
    """

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

        self.weight_dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Dot product the queries and the keys, scale by the number of channels
        # q.shape: (B, T, C)
        # k.transpose.shape: (B, C, T)
        # weights.shape: (B, T, T)
        # divide by sqrt(C) to normalize
        weights = q @ k.transpose(-2, -1) * (C**-0.5)
        weights = self.weight_dropout(weights)
        weights = weights.softmax(dim=-1)
        # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        out = weights @ v
        return out


class EncoderMultiHead(L.LightningModule):
    def __init__(
        self,
        num_embed: int,
        head_size: int,
        num_heads: int,
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                EncoderHead(num_embed=num_embed, head_size=head_size)
                for _ in range(num_heads)
            ]
        )

        self.proj = nn.Linear(num_embed, num_embed)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)

        return out


class AAEncoder(L.LightningModule):
    """
    This module takes in a numeric encoding of the tokens, and passes through the (stack of)
    encoder layers.

    It:
     1. Embeds it in a higher-dimensional space (num_embed)
     2. Adds positional encoding + residual
     3. Adds self-attention + residual
    """

    def register_positional_embedding(self):
        """
        Registers a buffer of positional encodings based on alternating sines and cosines of
        decreasing frequency. This is smart, since it's easy to find combinations of similar
        distances, and sine and cosine are out of phase
        """
        # Makes a column vector of shape (max_len, 1). These are the positions of tokens
        position = torch.arange(self.max_len).unsqueeze(1)

        # This is the 'div' term because of the -log(10_000)
        # We do range(0, 2, ..., num_embed-2) because each of these values goes to a specific
        # channel, and we have alternating sines and cosines, so we would double to get num_embed.
        # PE_pos = sin(pos / 10_000^(2i / d_model)
        div_term = torch.exp(
            # ---------- 2i ------------
            torch.arange(0, self.num_embed, 2)
            # ----- 10_000^ -----   -- / d_model --
            * (-math.log(10000.0) / self.num_embed)
        )

        # Positional embedding has the same dimension as token embedding
        # There is a `1` here in the second dimension
        pe = torch.zeros(self.max_len, 1, self.num_embed)

        # position * div_term is a column times a row, yielding a (pos, num_embed // 2)
        # Even positions get sin, odd positions get cos
        # pe.shape: (position, 1, num_embed)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def __init__(
        self,
        vocab_size: int = AA_VOCAB_SIZE,
        num_embed: int = NUM_EMBED,
        num_heads: int = NUM_HEADS,
        max_len: int = AA_MAX_LEN,
        num_layers: int = NUM_ENCODER_LAYERS,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.num_embed = num_embed
        self.num_heads = num_heads
        self.head_size = num_embed // self.num_heads
        self.max_len = max_len
        self.num_layers = num_layers

        self.embedding_table = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.num_embed
        )
        self.register_positional_embedding()
        self.ln = nn.LayerNorm(self.num_embed)
        self.self_attention = nn.Sequential(
            *[
                EncoderMultiHead(
                    num_embed=self.num_embed,
                    head_size=self.head_size,
                    num_heads=self.num_heads,
                )
                for _ in range(self.num_layers)
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
        # add positional embedding (up to the time-length of x)
        x = x + self.pe[: x.size(0)]
        x = x + self.self_attention(self.ln(x))  # add self-attention

        return x


class DecoderHead(L.LightningModule):
    """
    Similar to the EncoderHead, with the only difference being that it masks all positions
    after a given one with torch.tril, filling with -inf logits
    """

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

        self.weight_dropout = nn.Dropout(DROPOUT)

        # These are not weights, but a buffer, so it doesn't update with loss.backward()
        # We take a 1s square matrix of size max_len, then just the lower triangle (with diagonal),
        # setting the rest to 0s
        self.register_buffer("tril", torch.tril(torch.ones(max_len, max_len)))

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Dot product the queries and the keys, scale by the number of channels
        # q.shape: (B, T, C)
        # k.transpose.shape: (B, C, T)
        # weights.shape: (B, T, T)
        weights = q @ k.transpose(-2, -1) * (C**-0.5)

        # Wherever tril (up to size T) is 0 (upper triangle), set the corresponding weights to -inf
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = self.weight_dropout(weights)
        weights = weights.softmax(dim=-1)

        # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        out = weights @ v
        return out


class DecoderMultiHead(L.LightningModule):
    def __init__(
        self,
        num_embed: int,
        head_size: int,
        num_heads: int,
        max_len: int = GO_MAX_LEN,
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                DecoderHead(num_embed=num_embed, head_size=head_size, max_len=max_len)
                for _ in range(num_heads)
            ]
        )

        self.proj = nn.Linear(num_embed, num_embed)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)

        return out


class GODecoder(L.LightningModule):
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
        num_heads: int = NUM_HEADS,
        # head_size: int = HEAD_SIZE,
        max_len: int = GO_MAX_LEN,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.num_embed = num_embed
        self.num_heads = num_heads
        # self.head_size = head_size
        self.head_size = num_embed // self.num_heads
        self.max_len = max_len
        self.num_layers = NUM_DECODER_LAYERS

        self.embedding_table = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.num_embed
        )
        self.register_positional_embedding()
        self.self_attention = nn.Sequential(
            *[
                DecoderMultiHead(
                    num_embed=self.num_embed,
                    head_size=self.head_size,
                    max_len=self.max_len,
                    num_heads=self.num_heads,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.ln = nn.LayerNorm(self.num_embed)

    def forward(self, x: torch.Tensor):
        """
        Returns a positional, self-attention embedding of input tokens which are amino acid residues
        """
        # Takes an input of shape (batch, time [max_len], channel [num_embed])

        # Truncate the input if needed, taking the last `max_len` positions
        x = x[-self.max_len :]

        x = self.embedding_table(x)  # embed
        x = x + self.pe[: x.size(0)]  # add positional embedding
        x = x + self.self_attention(self.ln(x))  # add self-attention

        return x


class EncoderDecoderHead(L.LightningModule):
    """
    Encoder-Decoder attention head
    Takes in input from both encoder and decoder and performs the following:
     1. Calculates a query for the decoder
     2. Calculates a key and value for the encoder
     3. Matrix multiplies (series of dot products) between the decoder query and encoder keys
     4. Return encoder values weighted by softmax of 3.
    """

    def __init__(self, head_size: int, num_embed: int = NUM_EMBED):
        super().__init__()
        self.num_embed = num_embed
        self.head_size = head_size

        # Encoder
        self.encoder_key = nn.Linear(self.num_embed, self.head_size, bias=False)
        self.encoder_value = nn.Linear(self.num_embed, self.head_size, bias=False)

        # Decoder
        self.decoder_query = nn.Linear(self.num_embed, self.head_size, bias=False)

    def forward(self, x_enc: torch.Tensor, x_dec: torch.Tensor):
        B_enc, T_enc, C_enc = x_enc.shape

        dec_q = self.decoder_query(x_dec)
        enc_k = self.encoder_key(x_enc)
        enc_v = self.encoder_value(x_enc)

        weights = dec_q @ enc_k.transpose(-2, -1) * (C_enc**0.5)
        weights = weights.softmax(dim=-1)

        out = weights @ enc_v

        return out


class EncoderDecoderMultiHead(L.LightningModule):
    def __init__(self, num_heads: int, num_embed: int = NUM_EMBED):
        super().__init__()
        self.num_embed = num_embed
        self.num_heads = num_heads
        self.head_size = self.num_embed // self.num_heads

        self.heads = nn.ModuleList(
            [EncoderDecoderHead(self.head_size) for _ in range(self.num_heads)]
        )
        self.proj = nn.Linear(self.head_size * self.num_heads, self.num_embed)

    def forward(self, x_enc, x_dec):
        x = torch.cat([h(x_enc, x_dec) for h in self.heads], dim=-1)
        x = self.proj(x)
        return x


class FeedForward(L.LightningModule):
    """
    This is the final layer in a transformer. It takes the output of the encoder-decoder head
    and passes it through a dense, fully-connected layer
    """

    def __init__(self, num_embed: int = NUM_EMBED, inner_scaling: int = 1):
        super().__init__()
        self.num_embed = num_embed
        self.inner_scaling = inner_scaling
        self.net = nn.Sequential(
            nn.Linear(self.num_embed, self.num_embed * self.inner_scaling),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(self.num_embed * self.inner_scaling, self.num_embed),
        )

    def forward(self, x):
        return self.net(x)


class Transformer(L.LightningModule):
    """
    Takes in data from both input and output language, and:
     1. Gets embedding for the input (encoder) and output (decoder)
     2. Adds positional embedding
     3. Adds full self-attention for encoder, masked self-attention for decoder (autoregressive framework)
     4. Adds encoder-decoder attention (cross-attention)
     5. Passes through fully connected layer
     6. Passes through a layer to output vocabulary size to get token logits
    """

    def __init__(
        self,
        dec_vocab_size: int = GO_VOCAB_SIZE,
        num_heads: int = NUM_HEADS,
        ffwd_inner_scaling: int = INNER_SCALING,
    ):
        super().__init__()
        self.dec_vocab_size = dec_vocab_size
        self.ffwd_inner_scaling = ffwd_inner_scaling

        self.encoder = AAEncoder(num_heads=num_heads)
        self.decoder = GODecoder(num_heads=num_heads)
        self.cross_attention_head = EncoderDecoderMultiHead(num_heads=num_heads)
        self.post_x_attn_ln = nn.LayerNorm(self.cross_attention_head.num_embed)
        self.feed_fwd = FeedForward(inner_scaling=self.ffwd_inner_scaling)
        self.post_ffwd_ln = nn.LayerNorm(self.feed_fwd.num_embed)
        self.out_head = nn.Linear(self.feed_fwd.num_embed, self.dec_vocab_size)
        self.out_sos = None
        self.out_eos = None

    def forward(self, data: Dict[str, torch.Tensor]):
        enc_input = data["seq"]
        dec_input = data["go"]

        # Ensure data has 2 dimensions: (B, T)
        enc_input = enc_input.view(-1, enc_input.size(-1))
        dec_input = dec_input.view(-1, dec_input.size(-1))

        # Get embeddings for encoder and decoder
        enc_embed = self.encoder(enc_input)
        # For the decoder, we are always going to predict the next value,
        # so we don't take the last position (there's no next value)
        dec_embed = self.decoder(dec_input)

        # Cross attention (includes residual)
        x_attn = dec_embed + self.cross_attention_head(enc_embed, dec_embed)

        # Final dense layer
        x_attn = self.feed_fwd(self.post_x_attn_ln(x_attn))

        # Get the logits
        logits = self.out_head(self.post_ffwd_ln(x_attn))

        return logits

    def get_loss(self, data):
        if (self.out_sos is None) or (self.out_eos is None):
            self.check_update_sos_eos(data["go"])

        enc_input = data["seq"]
        dec_input = data["go"][:, :-1]
        dec_target = data["go"][:, 1:]

        logits = self({"seq": enc_input, "go": dec_input})

        # We reshape the data so that batch and time are treated as B*T separate observations
        # We may want to reshape only to get the loss, but not when we're generating an output sequence
        B, T, C = logits.shape
        logits = logits.view(B * T, C)

        # Targets are always 1 token delayed from the input to the decoder
        # We reshape instead of view because we don't have a contiguous tensor here (dec_input[:, 1:])
        targets = dec_target.reshape(B * T)
        loss = F.cross_entropy(logits, targets.type(torch.long))

        return loss

    def configure_optimizers(self):
        return Adam(self.parameters())

    def check_update_sos_eos(self, dec_input: torch.tensor):
        if dec_input.dim() == 1:
            out_sos = dec_input[0].detach()
            out_eos = dec_input[-1].detach()
        else:
            assert dec_input.dim() == 2, dec_input.dim()
            out_sos = dec_input[0, 0].detach()
            out_eos = dec_input[0, -1].detach()

        if self.out_sos is None:
            self.out_sos = out_sos
        elif not self.out_sos == out_sos:
            raise ValueError(f"Previous found {self.out_sos=}, now getting {out_sos=}")

        if self.out_eos is None:
            self.out_eos = out_eos
        elif not self.out_eos == out_eos:
            raise ValueError(f"Previous found {self.out_eos=}, now getting {out_eos=}")

    def training_step(self, data):
        loss = self.get_loss(data)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, data):
        loss = self.get_loss(data)
        self.log("val_loss", loss)
        return loss

    def generate(
        self,
        data: Dict[str, torch.Tensor],
        from_sos: bool = True,
        max_out_len: int = 1000,
    ):
        enc_input = data["seq"]
        if enc_input.dim() == 1:
            enc_input = enc_input.unsqueeze(0)
        if from_sos:
            if self.out_sos is None:
                raise ValueError("self.out_sos is None")
            dec_input = self.out_sos.repeat(enc_input.size(0), 1)
        else:
            dec_input = data["go"]

        if self.out_eos is None:
            raise ValueError("self.out_eos is None")

        end_token = self.out_eos.tolist()
        with torch.no_grad():
            while dec_input.size(-1) < max_out_len:
                logits, _ = self({"seq": enc_input, "go": dec_input})
                next_token = torch.argmax(logits, -1)
                dec_input = torch.cat(
                    [dec_input, next_token[..., -1].unsqueeze(1)], dim=-1
                )

                # Check if we're terminating the generation
                last_tokens = set(dec_input[..., -1].detach().tolist())
                if (len(last_tokens) == 1) and (last_tokens.pop() == end_token):
                    break

        return dec_input
