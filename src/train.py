"""
This module implements trains a transformer to translate from protein (amino acid sequence) into GO terms.
It is built on the PyTorch / Lightning architecture.
Written by: Artur Jaroszewicz (@beyondtheproof)
"""

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.optim import Adam

# import lightning as L
from .data import EOS


def train_model(model: nn.Module, data):
    # transformer =
    pass
