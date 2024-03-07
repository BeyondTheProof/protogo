"""
This module generates (and potentially perturbs) protein-GO term data.
Raw data was downloaded from UniProt
https://www.uniprot.org/uniprotkb?query=%28taxonomy_id%3A2759%29&facets=model_organism%3A9606

Written by: Artur Jaroszewicz (@beyondtheproof)
"""

from typing import List

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import tensorboard

import lightning as L

COLUMNS = ["seq", "go"]


class ProteinDataset(Dataset):
    @staticmethod
    def load_data(filepath: str, dropna: bool = True) -> pd.DataFrame:
        # Entry   Entry Name   Gene Names   Sequence   Gene Ontology (GO)
        data = (
            pd.read_csv(filepath, sep="\t")
            .rename(columns={"Sequence": "seq", "Gene Ontology (GO)": "go"})
            .set_index("Entry")
        )
        if dropna:
            # We may want to drop NAs because we want to predict if GO terms are unknown (?)ı
            data = data.dropna()
        return data

    # @staticmethod
    def split_go_terms(self, _input: str):
        return [term.lstrip(" ") for term in _input.split(";")]

    def get_vocabularies(self):
        vocabularies = {}
        for col in COLUMNS:
            if col == "seq":
                all_chars = "".join([val for val in self.raw_data[col]])
                vocabulary = sorted(list(set(all_chars)))
            elif col == "go":
                all_go_terms = np.concatenate(
                    [*[val.split(";") for val in self.raw_data[col]]]
                )
                vocabulary = sorted(list(set(all_go_terms)))
                vocabulary = [v.lstrip(" ") for v in vocabulary]
            else:
                raise NotImplementedError(f"{col} not implemented")

            vocabularies[col] = vocabulary
        return vocabularies

    def encode(self, input_str: List[str], col: str):
        return [self.encoding[col][s] for s in input_str]

    def decode(self, input_encoding: List[int], col: str):
        return "".join([self.decoding[col][i] for i in input_encoding])

    def __init__(self, filepath: str):
        super().__init__()
        self.raw_data = self.load_data(filepath)
        self.vocabs = self.get_vocabularies()
        self.encoding = {
            col: {word: idx for idx, word in enumerate(self.vocabs[col])}
            for col in COLUMNS
        }
        self.decoding = {
            col: {idx: word for idx, word in enumerate(self.vocabs[col])}
            for col in COLUMNS
        }

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, item):
        data_line = self.raw_data.iloc[item]
        X, y = data_line[COLUMNS]
        return self.encode(X, COLUMNS[0]), self.encode(
            self.split_go_terms(y), COLUMNS[1]
        )


class ProteinDataLoader(DataLoader):
    def __init__(self, dataset: ProteinDataset):
        super().__init__(dataset)

    def get_batch(self):
        pass
