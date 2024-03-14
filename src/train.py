"""
This module implements trains a transformer to translate from protein (amino acid sequence) into GO terms.
It is built on the PyTorch / Lightning architecture.
Written by: Artur Jaroszewicz (@beyondtheproof)
"""

import argparse
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

import lightning as L

from .data import ProteinDataset, ProteinDataLoader, BATCH_SIZE
from .model import Transformer


def train_model_torch(model: nn.Module, dl_train, dl_val, args):
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    val_losses = []

    for epoch in range(args.num_epochs):
        for step, train_batch in tqdm(
            enumerate(dl_train),
            desc=f"Training {epoch=}",
            disable=False,
            total=dl_train.num_batches,
        ):
            if step % args.eval_interval == 0:
                # We want to make sure we're not updating the model here
                with torch.no_grad():
                    _val_losses = []
                    for val_batch in dl_val:
                        logits, loss = model(val_batch)
                        _val_losses.append(loss.item())
                    val_losses.append(np.mean(_val_losses))

            logits, loss = model(train_batch)
            # Reset the gradients so we're not accumulating
            optimizer.zero_grad(set_to_none=True)
            # Take the derivative
            loss.backward()
            # Take a step
            optimizer.step()


def train_model_lit(model: L.LightningModule, dl_train, dl_val, args):
    trainer = L.Trainer(max_epochs=args.max_epochs, accelerator=args.device)
    trainer.fit(model, train_dataloaders=dl_train, val_dataloaders=dl_val)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filepath",
        type=str,
        default="/Users/arturj/work/deep_learning/projects/data/uniprotkb_taxonomy_id_2759_AND_model_or_2024_02_29.tsv.gz",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    ds = ProteinDataset.from_csv(args.filepath)
    ds_train, ds_val = ds.split_into_train_and_val()
    dl_train = ProteinDataLoader(
        ds_train, num_workers=args.num_workers, batch_size=args.batch_size
    )
    dl_val = ProteinDataLoader(
        ds_val, num_workers=args.num_workers, batch_size=args.batch_size
    )

    transformer = Transformer()

    train_model_lit(transformer, dl_train, dl_val, args)
