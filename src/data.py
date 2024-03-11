"""
This module generates (and potentially perturbs) protein-GO term data.
Raw data was downloaded from UniProt
https://www.uniprot.org/uniprotkb?query=%28taxonomy_id%3A2759%29&facets=model_organism%3A9606

Written by: Artur Jaroszewicz (@beyondtheproof)
"""

from typing import List

import pandas as pd
import numpy as np
import math
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Sampler, BatchSampler, RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence

import lightning as L

COLUMNS = ["seq", "go"]
SOS = "<SOS>"
EOS = "<EOS>"
BATCH_SIZE = 16


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
            # We may want to drop NAs because we want to predict if GO terms are unknown (?)
            data = data.dropna()

        # We sort by the sequence length so we can get more useful batches
        data["seq_len"] = data["seq"].apply(lambda s: len(s))
        data.sort_values("seq_len", inplace=True)
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
                vocabulary = set(all_go_terms)
                vocabulary = [v.strip() for v in vocabulary]
                vocabulary = sorted(list(set(vocabulary)))
            else:
                raise NotImplementedError(f"{col} not implemented")

            vocabulary.append(SOS)
            vocabulary.append(EOS)
            vocabularies[col] = vocabulary
        return vocabularies

    def encode(self, input_str: List[str], col: str):
        return torch.tensor(
            [self.encoding[col][s] for s in input_str], dtype=torch.float32
        )

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

    def __getitem__(self, item: int):
        data_line = self.raw_data.iloc[item]
        X, y = data_line[COLUMNS]

        # We finish the AA sequence with EOS to say we're done
        X_enc = self.encode(list(X) + [EOS], COLUMNS[0])

        # We start the GO terms with SOS to say we're starting to generate from nothing
        # We also end with EOS to say we're done generating terms
        y_enc = self.encode([SOS] + self.split_go_terms(y) + [EOS], COLUMNS[1])

        return {"seq": X_enc, "go": y_enc}

    def collate_batch(self, batch):
        # Assuming each item is a tuple (data, label)
        X_input = [item[COLUMNS[0]] for item in batch]
        y_input = [item[COLUMNS[1]] for item in batch]

        # Pad the data
        X_padded = pad_sequence(
            X_input, batch_first=True, padding_value=self.encoding["seq"][EOS]
        )
        y_padded = pad_sequence(
            y_input, batch_first=True, padding_value=self.encoding["go"][EOS]
        )

        return {COLUMNS[0]: X_padded, COLUMNS[1]: y_padded}


class ProteinBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset: ProteinDataset,
        sampler_class: Sampler = RandomSampler,
        batch_size: int = BATCH_SIZE,
        drop_last: bool = False,
    ):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_batches = math.ceil(len(dataset) / batch_size)
        self.sampler = sampler_class(range(self.num_batches))
        self.dataset = dataset

        # Create the batches of idxs
        batches = []
        for batch_num in range(self.num_batches):
            start_idx = batch_num * self.batch_size
            end_idx = min((batch_num + 1) * self.batch_size, len(self.dataset))
            batches.append(list(range(start_idx, end_idx)))

        if self.drop_last and len(batches[-1]) != self.batch_size:
            batches = batches[:-1]

        self.batches = batches

    def __iter__(self):
        for idx in self.sampler:
            yield self.batches[idx]

    def __len__(self):
        return len(self.batches)


class ProteinDataLoader(DataLoader):
    def __init__(
        self,
        dataset: ProteinDataset,
        batch_size: int = BATCH_SIZE,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        batch_sampler = ProteinBatchSampler(
            dataset,
            sampler_class=RandomSampler if shuffle else SequentialSampler,
            batch_size=batch_size,
            drop_last=drop_last,
        )
        super().__init__(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=dataset.collate_batch,
        )
