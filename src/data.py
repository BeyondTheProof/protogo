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
from torch.utils.data import (
    Dataset,
    DataLoader,
    Sampler,
    BatchSampler,
    RandomSampler,
    SequentialSampler,
)
from torch.nn.utils.rnn import pad_sequence

# import lightning as L

COLUMNS = ["seq", "go"]
SOS = "<SOS>"
EOS = "<EOS>"
BATCH_SIZE = 16


class ProteinDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        vocabs=None,
        encoding=None,
        decoding=None,
        max_seq_size: int = 5000,
    ):
        super().__init__()

        # We sort by the sequence length so we can get more useful batches
        data["seq_len"] = data["seq"].apply(lambda s: len(s))
        data.query(f"seq_len <= {max_seq_size}", inplace=True)
        data.sort_values("seq_len", inplace=True)
        self.raw_data = data
        self.max_seq_size = max_seq_size

        self.vocabs = self.get_vocabularies() if vocabs is None else vocabs
        if encoding is None:
            self.encoding = {
                col: {word: idx for idx, word in enumerate(self.vocabs[col])}
                for col in COLUMNS
            }
        else:
            self.encoding = encoding
        if decoding is None:
            self.decoding = {
                col: {idx: word for idx, word in enumerate(self.vocabs[col])}
                for col in COLUMNS
            }
        else:
            self.decoding = decoding

    @classmethod
    def from_csv(cls, filepath: str, dropna: bool = True):
        """
        Reads in a tsv into a pandas dataframe, renaming some columns to be easier to handle.
        Also calculates the length of each AA sequence to make it easier to batch later
        :param filepath:
        :param dropna: drops rows that don't have any GO terms (this can be used as a prediction set)
        :return:
        """
        # Entry   Entry Name   Gene Names   Sequence   Gene Ontology (GO)
        data = (
            pd.read_csv(filepath, sep="\t")
            .rename(columns={"Sequence": "seq", "Gene Ontology (GO)": "go"})
            .set_index("Entry")
        )
        if dropna:
            # We may want to drop NAs because we want to predict if GO terms are unknown (?)
            data = data.dropna()

        return cls(data)

    def split_go_terms(self, input_go_str: str):
        """
        Takes in a string of ';' separated GO terms and returns a list
        """
        return [term.strip(" ") for term in input_go_str.split(";")]

    def get_vocabularies(self):
        """
        Goes through the input and output columns in the pandas dataframe and builds the vocabulary for each.
        - For 'seq', it's a list of all the possible AA residues
        - For 'go', a list of all the possible GO terms
        Also, adds the SOS and EOS tokens
        :return: a dict of {col: [v1, v2, v3, ...]}
        """
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

    def encode(self, input_strs: List[str], col: str):
        """
        Converts from a vocabulary space to numerical representation (an integer)
        :param input_strs: a list of strings. E.g., for col='seq', ['A', 'H', 'N', 'V']
        :param col: the specific encoder we're using ('seq' or 'go')
        :return:
        """
        return torch.tensor(
            [self.encoding[col][s] for s in input_strs], dtype=torch.int
        )

    def decode(self, input_encoding: List[int], col: str):
        """
        The inverse of encode -- takes a list of integers and returns a string
        :param input_encoding:
        :param col:
        :return:
        """
        out = [self.decoding[col][i] for i in input_encoding]
        if col == "seq":
            return "".join(out)
        elif col == "go":
            return "; ".join(out)
        else:
            raise NotImplementedError(col)

    def split_into_train_and_val(self, frac_val: float = 0.05):
        """
        Splits off a part of the dataset to a validation dataset, re-initializing both
        """
        self.raw_data["order"] = np.random.random(len(self))
        self.raw_data.sort_values("order", inplace=True)
        self.raw_data["is_train"] = self.raw_data.order < 1 - frac_val

        # split into train and val
        train_data = self.raw_data.query("is_train").reset_index()
        val_data = self.raw_data.query("~is_train").reset_index()

        kwargs = {
            "vocabs": self.vocabs,
            "encoding": self.encoding,
            "decoding": self.decoding,
            "max_seq_size": self.max_seq_size,
        }
        ds_train = ProteinDataset(train_data, **kwargs)
        ds_val = ProteinDataset(val_data, **kwargs)

        return ds_train, ds_val

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, item: int):
        """
        Takes a single integer position in the length of the data and returns the integer-encoded data
        :param item:
        :return: a dict of {col: tensor}
        """
        try:
            item = int(item)
        except:
            raise ValueError(f"{item=} should be an int")

        data_line = self.raw_data.iloc[item]
        X, y = data_line[COLUMNS]

        # We finish the AA sequence with EOS to say we're done
        X_enc = self.encode(list(X) + [EOS], COLUMNS[0])

        # We start the GO terms with SOS to say we're starting to generate from nothing
        # We also end with EOS to say we're done generating terms
        y_enc = self.encode([SOS] + self.split_go_terms(y) + [EOS], COLUMNS[1])

        return {"seq": X_enc, "go": y_enc}

    def collate_batch(self, batch):
        """
        This is used with a batch sampler to combine samples into a batch.
        First, it extracts the requested key, then pads with EOSs at the end to make everything the same length
        :param batch:
        :return:
        """
        collated = {}
        for col in COLUMNS:
            _input = [item[col] for item in batch]
            _padded = pad_sequence(
                _input, batch_first=True, padding_value=self.encoding[col][EOS]
            )
            collated[col] = _padded

        return collated


class ProteinBatchSampler(BatchSampler):
    """
    This is a sampler that returns a batch of consecutive idxs (AA seq data is sorted by length).
    It randomly samples from the length of the AA seq data, so it's shuffled, but padded nicely
    """

    def __init__(
        self,
        dataset: ProteinDataset,
        sampler_class: Sampler = RandomSampler,
        batch_size: int = BATCH_SIZE,
        drop_last: bool = False,
    ):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.dataset_len = len(dataset)
        self.num_batches = math.ceil(self.dataset_len / self.batch_size)
        self.sampler = sampler_class(range(self.num_batches))

        # Create the batches of idxs
        batches = []
        for batch_num in range(self.num_batches):
            start_idx = batch_num * self.batch_size
            end_idx = min((batch_num + 1) * self.batch_size, self.dataset_len)
            batches.append(list(range(start_idx, end_idx)))

        if self.drop_last and len(batches[-1]) != self.batch_size:
            batches = batches[:-1]

        self.batches = batches
        self.num_batches = len(self.batches)

    def __iter__(self):
        """
        returns a batch of idxs
        """
        for idx in self.sampler:
            yield self.batches[idx]

    def __len__(self):
        return self.num_batches


class ProteinDataLoader(DataLoader):
    def __init__(
        self,
        dataset: ProteinDataset,
        batch_size: int = BATCH_SIZE,
        shuffle: bool = True,
        drop_last: bool = False,
        num_workers: int = 4,
        persistent_workers: bool = True,
    ):
        """
        A torch data loader that returns a batch of data
        :param dataset:
        :param batch_size:
        :param shuffle: if False, returns all data in order of AA seq length
        :param drop_last: drops last batch if it's not of size batch_size
        """
        # The sampler returns a batch of idxs, nothing else
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_sampler = ProteinBatchSampler(
            dataset,
            sampler_class=RandomSampler if shuffle else SequentialSampler,
            batch_size=batch_size,
            drop_last=drop_last,
        )
        self.num_batches = self.batch_sampler.num_batches
        # This takes in the batch sampler and collate function to combine the datapoints
        super().__init__(
            dataset,
            batch_sampler=self.batch_sampler,
            collate_fn=dataset.collate_batch,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )
