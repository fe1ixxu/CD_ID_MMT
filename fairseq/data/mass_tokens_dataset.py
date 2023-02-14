# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache

import numpy as np
import torch
from fairseq.data import Dictionary, data_utils

from . import BaseWrapperDataset, LRUCacheDataset


class MassTokensDataset(BaseWrapperDataset):
    """
    A wrapper Dataset for masked language modeling.

    Input items are masked according to the specified masking probability.

    Args:
        dataset: Dataset to wrap.
        sizes: Sentence lengths
        vocab: Dictionary with the vocabulary and special tokens.
        pad_idx: Id of pad token in vocab
        mask_idx: Id of mask token in vocab
        return_masked_tokens: controls whether to return the non-masked tokens
            (the default) or to return a tensor with the original masked token
            IDs (and *pad_idx* elsewhere). The latter is useful as targets for
            masked LM training.
        seed: Seed for random number generator for reproducibility.
        mask_prob: probability of replacing a token with *mask_idx*.
        leave_unmasked_prob: probability that a masked token is unmasked.
        random_token_prob: probability of replacing a masked token with a
            random token from the vocabulary.
        freq_weighted_replacement: sample random replacement words based on
            word frequencies in the vocab.
        mask_whole_words: only mask whole words. This should be a byte mask
            over vocab indices, indicating whether it is the beginning of a
            word. We will extend any mask to encompass the whole word.
        bpe: BPE to use for whole-word masking.
        mask_multiple_length : repeat each mask index multiple times. Default
            value is 1.
        mask_stdev : standard deviation of masks distribution in case of
            multiple masking. Default value is 0.
    """

    @classmethod
    def apply_mask(cls, dataset: torch.utils.data.Dataset, *args, **kwargs):
        """Return the source and target datasets for masked LM training."""
        dataset = LRUCacheDataset(dataset)
        return (
            LRUCacheDataset(cls(dataset, *args, **kwargs, return_masked_tokens=False)),
            LRUCacheDataset(cls(dataset, *args, **kwargs, return_masked_tokens=True)),
        )

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        vocab: Dictionary,
        pad_idx: int,
        mask_idx: int,
        return_masked_tokens: bool = False,
        seed: int = 1,
        mask_prob: float = 0.15,
        leave_unmasked_prob: float = 0.1,
        random_token_prob: float = 0.1,
        freq_weighted_replacement: bool = False,
        mask_whole_words: torch.Tensor = None,
        mask_multiple_length: int = 1,
        mask_stdev: float = 0.0,
    ):
        assert 0.0 < mask_prob < 1.0
        assert 0.0 <= random_token_prob <= 1.0
        assert 0.0 <= leave_unmasked_prob <= 1.0
        assert random_token_prob + leave_unmasked_prob <= 1.0
        assert mask_multiple_length >= 1
        assert mask_stdev >= 0.0

        self.dataset = dataset
        self.vocab = vocab
        self.pad_idx = pad_idx
        self.mask_idx = mask_idx
        self.return_masked_tokens = return_masked_tokens
        self.seed = seed
        self.mask_prob = mask_prob
        self.leave_unmasked_prob = leave_unmasked_prob
        self.random_token_prob = random_token_prob
        self.mask_whole_words = mask_whole_words
        self.mask_multiple_length = mask_multiple_length
        self.mask_stdev = mask_stdev

        if random_token_prob > 0.0:
            if freq_weighted_replacement:
                weights = np.array(self.vocab.count)
            else:
                weights = np.ones(len(self.vocab))
            weights[: self.vocab.nspecial] = 0
            self.weights = weights / weights.sum()

        self.epoch = 0

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return True  # only the noise changes, not item sizes

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, index: int):
        return self.__getitem_cached__(self.seed, self.epoch, index)

    @lru_cache(maxsize=8)
    def __getitem_cached__(self, seed: int, epoch: int, index: int):
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            item = self.dataset[index]
            sz = len(item)

            assert (
                self.mask_idx not in item
            ), "Dataset contains mask_idx (={}), this is not expected!".format(
                self.mask_idx,
            )

            # decide elements to mask
            mask = np.full(sz, False)
            mask_idc = self.mask_interval(sz)
            try:
                mask[mask_idc] = True
            except:  # something wrong
                print(
                    "Assigning mask indexes {} to mask {} failed!".format(
                        mask_idc, mask
                    )
                )
                raise


            if self.return_masked_tokens:
                # exit early if we're just returning the masked tokens
                # (i.e., the targets for masked LM training)
                new_item = np.full(len(mask), self.pad_idx)
                new_item[mask] = item[torch.from_numpy(mask.astype(np.uint8)) == 1]
                return torch.from_numpy(new_item)

            new_item = np.copy(item)
            new_item[mask] = self.mask_idx
            return torch.from_numpy(new_item)


    def mask_interval(self, l):
        mask_length = round(l * self.mask_prob)
        mask_length = max(1, mask_length)
        mask_start  = np.random.randint(0, l - mask_length)
        return np.array(range(mask_start, mask_start + mask_length))
