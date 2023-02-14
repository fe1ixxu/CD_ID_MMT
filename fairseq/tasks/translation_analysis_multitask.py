# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import itertools
import logging
import os
import time
import math

import numpy as np
import torch
import torch.nn.functional as F
import random

from fairseq import utils
from fairseq.data import (
    FairseqDataset,
    BaseWrapperDataset,
    LanguagePairDataset,
    ConcatDataset,
    ListDataset,
    Dictionary,
    IdDataset,
    MaskTokensDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    PadDataset,
    PrependTokenDataset,
    RawLabelDataset,
    ResamplingDataset,
    SortDataset,
    TokenBlockDataset,
    data_utils,
    encoders,
    iterators,
    AppendTokenDataset,
    DenoisingDataset,

)

from copy import copy
from fairseq.data.multilingual.multilingual_data_manager import (
    MultilingualDatasetManager,
)
from fairseq.data.multilingual.multilingual_utils import LangTokStyle, get_lang_tok
from fairseq.data.multilingual.sampling_method import SamplingMethod
from fairseq.data.encoders.utils import get_whole_word_mask
from fairseq.optim.amp_optimizer import AMPOptimizer
from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.utils import FileContentsAction
from fairseq import metrics
from .denoising import DenoisingTask
from .translation_multi_simple_epoch import TranslationMultiSimpleEpochTask
from collections import defaultdict


###
def get_time_gap(s, e):
    return (
        datetime.datetime.fromtimestamp(e) - datetime.datetime.fromtimestamp(s)
    ).__str__()

def X_loss(logits, pad_mask):
    pad_mask = pad_mask.view(-1)
    non_pad_mask = ~pad_mask
    dict_size = logits[0].size(-1)

    m = sum(logits) / len(logits)
    m = m.float().view(-1, dict_size)[non_pad_mask]

    kl_all = 0
    for l in logits:
        l = l.float().view(-1, dict_size)[non_pad_mask]
        d = (l-m) * (torch.log(l) - torch.log(m))
        kl_all += d.sum()
    return kl_all / len(logits)
###


logger = logging.getLogger(__name__)

class SampleDataset(BaseWrapperDataset):
    def __init__(self, dataset, sample, seed):
        assert sample <= len(dataset), "Sample value must be smaller than the original size!"
        self.dataset = dataset
        self.sample = sample
        self.indices = None
        if self.sample > 0:
            random.seed(seed)
            self.indices = random.sample(list(range(len(dataset))), sample)
    def __getitem__(self, index):
        if self.sample > 0:
            return self.dataset[self.indices[index]]
        else:
            return self.dataset[index]
    def __len__(self):
        if self.sample > 0:
            return self.sample
        else:
            return len(self.dataset)
    @property
    def sizes(self):
        if self.indices:
            return self.dataset.sizes[...,self.indices]
        else:
            return self.dataset.sizes
        
    def size(self, index):
        if self.indices:
            return self.dataset.size(index)[...,self.indices]
        else:
            return self.dataset.size

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        return np.arange(self.sample, dtype=np.int64)
    
        
class SimpleSSLModelDataset(BaseWrapperDataset):
    """
    A simple MLM dataset used for being compatible with 
    the input of the denoisng and translation tasks in 
    multitask learning.
    """

    def __init__(self, dataset, task_name, append_lgtoken):
        self.dataset = dataset
        self.task_name = task_name
        self.append_lgtoken = append_lgtoken
        

    def __getitem__(self, index):
        if "net_input.src_tokens" in self.dataset[index]:
            if self.append_lgtoken:
                self.dataset[index]["net_input.src_tokens"][-1] = self.dataset[index]["tgt_lang"]
            ## This is for the MLM task
            return {
                "id": index,
                "source": self.dataset[index]["net_input.src_tokens"],
                "target": self.dataset[index]["target"],
                "task_name": self.task_name,
                "lang_id": self.dataset[index]["tgt_lang"],
                "decoder_target": self.dataset[index]["decoder_target"],
            }
        else:
            ## This is for the MMT and DAE task
            if isinstance(self.dataset[index], tuple):
                if self.append_lgtoken:
                    self.dataset[index][1]["source"][-1] = self.dataset[index][1]["tgt_lang"]
                return (
                    self.dataset[index][0],
                    {
                        "id": index,
                        "source": self.dataset[index][1]["source"],
                        "target": self.dataset[index][1]["target"],
                        "task_name": self.task_name,
                        "lang_id": self.dataset[index][1]["tgt_lang"],
                    }  
                )         
            else:
                if self.append_lgtoken:
                    self.dataset[index]["source"][-1] = self.dataset[index]["tgt_lang"]
                return {
                    "id": index,
                    "source": self.dataset[index]["source"],
                    "target": self.dataset[index]["target"],
                    "task_name": self.task_name,
                    "lang_id": self.dataset[index]["tgt_lang"],
                }

@register_task("translation_analysis_multitask3")
class TranslationSslAnalysisMultitaskTask3(TranslationMultiSimpleEpochTask):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt:off
        ## general arguments
        parser.add_argument('--ssl-data', default=None,
                            help='monolingual data location')
        parser.add_argument('--ssl-tasks', default="dae", 
                            help='SSL tasks involved')
        parser.add_argument('--nossl-langs', default='',
                            help='languages that will not included in SSL training')
        parser.add_argument('--ssl-max-sample', default=-1, type=int,
                            help='max samples from ssl training data')


        ## arguments for translation
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='inference source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='inference target language')
        parser.add_argument('--lang-pairs', default=None, metavar='PAIRS',
                            help='comma-separated list of language pairs (in training order): en-de,en-fr,de-fr',
                            action=FileContentsAction)
        parser.add_argument('--keep-inference-langtok', action='store_true',
                            help='keep language tokens in inference output (e.g. for analysis or debugging)')
        parser.add_argument('--one-dataset-per-batch', action='store_true',
                            help='limit each minibatch to one sub-dataset (typically lang direction)')
        SamplingMethod.add_arguments(parser)
        MultilingualDatasetManager.add_args(parser)

        ## arguments for mlm
        parser.add_argument(
            "--sample-break-mode",
            default="complete",
            choices=["none", "complete", "complete_doc", "eos"],
            help='If omitted or "none", fills each sample with tokens-per-sample '
            'tokens. If set to "complete", splits samples only at the end '
            "of sentence, but may include multiple sentences per sample. "
            '"complete_doc" is similar but respects doc boundaries. '
            'If set to "eos", includes only one sentence per sample.',
        )
        parser.add_argument(
            "--mask-prob",
            default=0.15,
            type=float,
            help="probability of replacing a token with mask",
        )
        parser.add_argument(
            "--leave-unmasked-prob",
            default=0.1,
            type=float,
            help="probability that a masked token is unmasked",
        )
        parser.add_argument(
            "--random-token-prob",
            default=0.1,
            type=float,
            help="probability of replacing a token with a random token",
        )
        parser.add_argument(
            "--freq-weighted-replacement",
            action="store_true",
            help="sample random replacement words based on word frequencies",
        )
        parser.add_argument(
            "--mask-whole-words",
            default=False,
            action="store_true",
            help="mask whole words; you may also want to set --bpe",
        )
        parser.add_argument(
            "--multilang-sampling-alpha",
            type=float,
            default=1.0,
            help="smoothing alpha for sample rations across multiple datasets",
        )

        ## arguments for denoising
        parser.add_argument("--add-lang-token", default=False, action="store_true")

        parser.add_argument(
            "--no-whole-word-mask-langs",
            type=str,
            default="",
            metavar="N",
            help="languages without spacing between words dont support whole word masking",
        )

        ## arguments for inner/intra-distillation
        parser.add_argument(
            "--id-alpha",
            type=float,
            default=1.0,
            help="weights for consistency loss",
        )

        parser.add_argument(
            "--beta",
            type=float,
            default=1.0,
            help="weights for ssl loss",
        )

        parser.add_argument("--not-shared-proj", action='store_true',
                            help='Encoder and decoder will use different projection layer if set true')
        parser.add_argument("--not-token-block", action='store_true',
                            help='Disable tokenblock and make ssl training and mt training data more mixed')
        parser.add_argument("--enable-id-mmt", action='store_true',
                            help='Enable id MMT Training')
        parser.add_argument("--lg", default=None,
                            help='Enable id MMT Training')
        parser.add_argument("--ssl", default="",
                            help='Enable id MMT Training')
        parser.add_argument("--append-lgtoken", action='store_true',
                            help='Enable Language token appened for the source side')

    def __init__(self, args, langs, dicts, training):
        ## Multilingual Translation 
        super().__init__(args, langs, dicts, training)
    
        ## MLM and DAE
        self.seed = args.seed        
        for lang, d in self.dicts.items():
            self.mask_idx = d.add_symbol("<mask>")
            self.dictionary = d

        self.args = args
        self.all_score = [0, 0]
        self.all_score = defaultdict(lambda : [0, 0])
    
    @classmethod
    def setup_task(cls, args, **kwargs):
        ## multilingual translation setup
        if getattr(args, "train_subset", None) is not None:
            if not all(s.startswith("train") for s in args.train_subset.split(",")):
                raise ValueError('Train splits should be named like "train*".')

        langs, dicts, training = MultilingualDatasetManager.prepare(
            cls.load_dictionary, args, **kwargs
        )

        ## MLM and DAE task setup
        paths, dicts = cls.setup_ssl_task(args, dicts, **kwargs)

        return cls(args, langs, dicts, training)


    @classmethod
    def setup_ssl_task(cls, args, dicts, **kwargs):
        paths = args.ssl_data.split(":")
        assert len(paths) > 0

        data_path = paths[0]
        if args.langs is None:
            languages = sorted(
                [
                    name
                    for name in os.listdir(data_path)
                    if os.path.isdir(os.path.join(data_path, name))
                ]
            )
        else:
            languages = args.langs

        args.ssl_tasks = args.ssl_tasks.split(",")
        args.nossl_langs = args.nossl_langs.split(",")

        if args.add_lang_token:
            for lang, d in dicts.items():
                d.add_symbol("__{}__".format(lang))
                logger.info("{} dictionary: {} types".format(lang, len(d)))


        if not hasattr(args, "shuffle_instance"):
            args.shuffle_instance = False
        return paths, dicts

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        assert not ("mlm" in self.args.ssl_tasks and "id_mlm" in self.args.ssl_tasks), "Only one of mlm and id_mlm is allowed!"
        assert not ("mlm" in self.args.ssl_tasks and "id_mlm2" in self.args.ssl_tasks), "Only one of mlm and id_mlm2 is allowed!"
        assert not ("dae" in self.args.ssl_tasks and "id_dae" in self.args.ssl_tasks), "Only one of dae and id_dae is allowed!"
        self.load_translation_dataset(split=split, epoch=epoch, combine=combine, **kwargs)
        ## Ignore the valid data for the SSL task
        # if "valid" not in split:
        #     if "mlm" in self.args.ssl_tasks:
        #         self.load_mlm_dataset(split=split, epoch=epoch, combine=combine, task_name="mlm", **kwargs)
        #     if "id_mlm" in self.args.ssl_tasks:
        #         self.load_mlm_dataset(split=split, epoch=epoch, combine=combine, task_name="id_mlm", **kwargs)
        #     if "id_mlm2" in self.args.ssl_tasks:
        #         self.load_mlm_dataset(split=split, epoch=epoch, combine=combine, task_name="id_mlm2", **kwargs)
        #     if "dae" in self.args.ssl_tasks:
        #         self.load_dae_dataset(split=split, epoch=epoch, combine=combine, task_name="dae", **kwargs)
        #     if "id_dae" in self.args.ssl_tasks:
        #         self.load_dae_dataset(split=split, epoch=epoch, combine=combine, task_name="id_dae", **kwargs)

        # with data_utils.numpy_seed(self.args.seed + epoch):
        #     shuffle = np.random.permutation(len(self.datasets[split]))
        #     sort_order = [shuffle]
        #     for i in range(1, self.datasets[split].sizes.ndim+1):
        #         sort_order.append(self.datasets[split].sizes[:,-i])
            
        #     self.datasets[split] = SortDataset(
        #         self.datasets[split],
        #         sort_order=sort_order,
        #     )

    def load_translation_dataset(self, split, epoch=1, combine=False, **kwargs):
        ## Multilingual Translation
        if split in self.datasets:
            dataset = self.datasets[split]
            if self.has_sharded_data(split):
                if self.args.virtual_epoch_size is not None:
                    if dataset.load_next_shard:
                        shard_epoch = dataset.shard_epoch
                    else:
                        # no need to load next shard so skip loading
                        # also this avoid always loading from beginning of the data
                        return
                else:
                    shard_epoch = epoch
        else:
            # estimate the shard epoch from virtual data size and virtual epoch size
            shard_epoch = self.data_manager.estimate_global_pass_epoch(epoch)
        logger.info(f"loading data for {split} epoch={epoch}/{shard_epoch}")
        logger.info(f"mem usage: {data_utils.get_mem_usage()}")
        if split in self.datasets:
            del self.datasets[split]
            logger.info("old dataset deleted manually")
            logger.info(f"mem usage: {data_utils.get_mem_usage()}")
        split_datasets = self.data_manager.load_dataset(
            split,
            self.training,
            epoch=epoch,
            combine=combine,
            shard_epoch=shard_epoch,
            **kwargs,
        )

        for split, dataset in split_datasets.items():
            self.datasets[split] = SimpleSSLModelDataset(dataset, task_name="mmt", append_lgtoken=self.args.append_lgtoken)

    def load_mlm_dataset(self, split, epoch=1, combine=False, task_name="mlm", **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.ssl_data.split(":")
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)

        if self.langs is None:
            languages = sorted(
                [
                    name
                    for name in os.listdir(data_path)
                    if os.path.isdir(os.path.join(data_path, name))
                ]
            )
        else:
            languages = copy(self.langs)
            for name in languages:
                p = os.path.join(data_path, name)
                assert os.path.exists(p), "data not found: {}".format(p)

        for lang in self.args.nossl_langs:
            if len(lang) > 0 and lang in languages:
                languages.remove(lang)
        assert len(languages) > 0, "At least one languages for SSL training!"

        logger.info(f"Training on {len(languages)} languages: {languages}, languages in {self.args.nossl_langs} has been removed for ssl training.")
        logger.info(
            "Language to id mapping: ", {lang: id for id, lang in enumerate(languages)}
        )

        mask_whole_words = self._get_whole_word_mask()
        lang_datasets = []
        for lang_id, language in enumerate(languages):
            if language in self.args.nossl_langs:
                continue

            split_path = os.path.join(data_path, language, split)

            dataset = data_utils.load_indexed_dataset(
                split_path,
                self.source_dictionary,
                self.args.dataset_impl,
                combine=combine,
            )
            if dataset is None:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, split_path)
                )

            # sampling dataset with `sample` instances, disabled when args.ssl_max_sample < 0
            dataset = SampleDataset(dataset, self.args.ssl_max_sample, self.args.seed)

            eos_token = (
                self.source_dictionary.index("__{}__".format(language))
                if self.args.add_lang_token
                else self.source_dictionary.bos()
            )
            if not self.args.not_token_block:
                dataset = TokenBlockDataset(
                    dataset,
                    dataset.sizes,
                    self.args.tokens_per_sample - 1,  # one less for <s>
                    pad=self.source_dictionary.pad(),
                    eos=eos_token,
                    break_mode=self.args.sample_break_mode,
                )
                logger.info("loaded {} blocks from: {}".format(len(dataset), split_path))
            

            # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
            # bos_token = (
            #     self.source_dictionary.index("__{}__".format(language))
            #     if self.args.add_lang_token
            #     else self.source_dictionary.bos()
            # )

            # dataset = PrependTokenDataset(dataset, bos_token)
            src_dataset, tgt_dataset = MaskTokensDataset.apply_mask(
                dataset,
                self.source_dictionary,
                pad_idx=self.source_dictionary.pad(),
                mask_idx=self.mask_idx,
                seed=self.args.seed,
                mask_prob=self.args.mask_prob,
                leave_unmasked_prob=self.args.leave_unmasked_prob,
                random_token_prob=self.args.random_token_prob,
                freq_weighted_replacement=self.args.freq_weighted_replacement,
                mask_whole_words=mask_whole_words,
            )

            lang_dataset = NestedDictionaryDataset(
                { 
                    "net_input": {
                        "src_tokens": PadDataset(
                            src_dataset,
                            pad_idx=self.source_dictionary.pad(),
                            left_pad=False,
                        ),
                        "src_lengths": NumelDataset(src_dataset, reduce=False),
                    },
                    "target": PadDataset(
                        tgt_dataset,
                        pad_idx=self.source_dictionary.pad(),
                        left_pad=False,
                    ),
                    "decoder_target":PadDataset(
                        dataset,
                        pad_idx=self.source_dictionary.pad(),
                        left_pad=False,
                    ),
                    "nsentences": NumSamplesDataset(),
                    "ntokens": NumelDataset(src_dataset, reduce=True),
                    "tgt_lang": RawLabelDataset([self.source_dictionary.index("__{}__".format(language))]* src_dataset.sizes.shape[0]),
                },
                sizes=[src_dataset.sizes],
            )
            
            lang_datasets.append(lang_dataset)
        dataset_lengths = np.array(
            [len(d) for d in lang_datasets],
            dtype=float,
        )

        logger.info(
            "loaded total {} blocks for all languages".format(
                dataset_lengths.sum(),
            )
        )
        if split == self.args.train_subset:
            # For train subset, additionally up or down sample languages.
            sample_probs = self._get_sample_prob(dataset_lengths)
            logger.info(
                "Sample probability by language: ",
                {
                    lang: "{0:.4f}".format(sample_probs[id])
                    for id, lang in enumerate(languages)
                },
            )
            size_ratio = (sample_probs * dataset_lengths.sum()) / dataset_lengths
            logger.info(
                "Up/Down Sampling ratio by language: ",
                {
                    lang: "{0:.2f}".format(size_ratio[id])
                    for id, lang in enumerate(languages)
                },
            )

            resampled_lang_datasets = [
                ResamplingDataset(
                    lang_datasets[i],
                    size_ratio=size_ratio[i],
                    seed=self.args.seed,
                    epoch=epoch,
                    replace=size_ratio[i] >= 1.0,
                )
                for i, d in enumerate(lang_datasets)
            ]
            dataset = ConcatDataset(resampled_lang_datasets)
        else:
            dataset = ConcatDataset(lang_datasets)
            lang_splits = [split]
            for lang_id, lang_dataset in enumerate(lang_datasets):
                split_name = split + "_" + languages[lang_id]
                lang_splits.append(split_name)
                self.datasets[split_name] = lang_dataset


            # [TODO]: This is hacky for now to print validation ppl for each
            # language individually. Maybe need task API changes to allow it
            # in more generic ways.
            if split in self.args.valid_subset:
                self.args.valid_subset = self.args.valid_subset.replace(
                    split, ",".join(lang_splits)
                )

        with data_utils.numpy_seed(self.args.seed + epoch):
            shuffle = np.random.permutation(len(dataset))
        
        if split in self.datasets:
            self.datasets[split] = ConcatDataset(
                [self.datasets[split],
                SimpleSSLModelDataset(
                    SortDataset(
                        dataset,
                        sort_order=[
                            shuffle,
                            dataset.sizes,
                        ],
                    ),
                    task_name=task_name,
                )]
            )
        else:
            self.datasets[split] = SimpleSSLModelDataset(
                SortDataset(
                    dataset,
                    sort_order=[
                        shuffle,
                        dataset.sizes,
                    ],
                ),
                task_name=task_name,
                append_lgtoken=self.args.append_lgtoken,
            )

    def load_dae_dataset(self, split, epoch=1, combine=False, task_name="dae", **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.ssl_data.split(":")
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)

        if self.langs is None:
            languages = sorted(
                [
                    name
                    for name in os.listdir(data_path)
                    if os.path.isdir(os.path.join(data_path, name))
                ]
            )
        else:
            languages = copy(self.langs)
            for name in languages:
                p = os.path.join(data_path, name)
                assert os.path.exists(p), "data not found: {}".format(p)
        for lang in self.args.nossl_langs:
            ## Filter empty strings
            if len(lang) > 0 and lang in languages:
                languages.remove(lang)
        assert len(languages) > 0, "At least one languages for SSL training!"

        logger.info(f"Training on {len(languages)} languages: {languages}, languages in {self.args.nossl_langs} has been removed for ssl training.")
        logger.info(
            "Language to id mapping: ", {lang: id for id, lang in enumerate(languages)}
        )

        mask_whole_words = self._get_whole_word_mask()
        language_without_segmentations = self.args.no_whole_word_mask_langs.split(",")
        lang_datasets = []
        for language in languages:
            split_path = os.path.join(data_path, language, split)

            dataset = data_utils.load_indexed_dataset(
                split_path,
                self.source_dictionary,
                self.args.dataset_impl,
                combine=combine,
            )
            if dataset is None:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, split_path)
                )
            # sampling dataset with `sample` instances, disabled when args.ssl_max_sample < 0
            dataset = SampleDataset(dataset, self.args.ssl_max_sample, self.args.seed)

            # bos_token = (
            #     self.source_dictionary.index("__{}__".format(language))
            #     if self.args.add_lang_token
            #     else self.source_dictionary.eos()
            # )
            eos_token = (
                self.source_dictionary.index("__{}__".format(language))
                if self.args.add_lang_token
                else self.source_dictionary.bos()
            )
            if not self.args.not_token_block:
                dataset = TokenBlockDataset(
                    dataset,
                    dataset.sizes,
                    self.args.tokens_per_sample - 2,  # one less for <s>
                    pad=self.source_dictionary.pad(),
                    eos=self.dictionary.eos(),
                    break_mode=self.args.sample_break_mode,
                )
                logger.info("loaded {} blocks from: {}".format(len(dataset), split_path))

            # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
            # dataset = PrependTokenDataset(dataset, bos_token)
            # dataset = AppendTokenDataset(dataset, self.dictionary.eos())

            lang_mask_whole_words = (
                mask_whole_words
                if language not in language_without_segmentations
                else None
            )
            lang_dataset = DenoisingDataset(
                dataset,
                dataset.sizes,
                self.dictionary,
                self.mask_idx,
                lang_mask_whole_words,
                shuffle=self.args.shuffle_instance,
                seed=self.seed,
                args=self.args,
                eos=None,
                lang_id=self.source_dictionary.index("__{}__".format(language)),
            )
            lang_datasets.append(lang_dataset)

        dataset_lengths = np.array(
            [len(d) for d in lang_datasets],
            dtype=float,
        )
        logger.info(
            "loaded total {} blocks for all languages".format(
                int(dataset_lengths.sum()),
            )
        )
        if split == self.args.train_subset:
            # For train subset, additionally up or down sample languages.
            sample_probs = self._get_sample_prob(dataset_lengths)
            logger.info(
                "Sample probability by language: {}".format(
                    {
                        lang: "{0:.4f}".format(sample_probs[id])
                        for id, lang in enumerate(languages)
                    }
                )
            )
            size_ratio = (sample_probs * dataset_lengths.sum()) / dataset_lengths
            logger.info(
                "Up/Down Sampling ratio by language: {}".format(
                    {
                        lang: "{0:.2f}".format(size_ratio[id])
                        for id, lang in enumerate(languages)
                    }
                )
            )

            resampled_lang_datasets = [
                ResamplingDataset(
                    lang_datasets[i],
                    size_ratio=size_ratio[i],
                    seed=self.args.seed,
                    epoch=epoch,
                    replace=size_ratio[i] >= 1.0,
                )
                for i, d in enumerate(lang_datasets)
            ]
            dataset = ConcatDataset(
                resampled_lang_datasets,
            )
        else:
            dataset = ConcatDataset(lang_datasets)
            lang_splits = [split]
            for lang_id, lang_dataset in enumerate(lang_datasets):
                split_name = split + "_" + languages[lang_id]
                lang_splits.append(split_name)
                self.datasets[split_name] = lang_dataset

            if split in self.args.valid_subset:
                self.args.valid_subset = self.args.valid_subset.replace(
                    split, ",".join(lang_splits)
                )

        with data_utils.numpy_seed(self.args.seed + epoch):
            shuffle = np.random.permutation(len(dataset))

        if split in self.datasets:
            self.datasets[split] = ConcatDataset(
                [self.datasets[split],
                SimpleSSLModelDataset(
                    SortDataset(
                        dataset,
                        sort_order=[
                            shuffle,
                            dataset.sizes,
                        ],
                    ),
                    task_name=task_name,
                )]
            )
        else:
            self.datasets[split] = SimpleSSLModelDataset(
                    SortDataset(
                        dataset,
                        sort_order=[
                            shuffle,
                            dataset.sizes,
                        ],
                    ),
                    task_name=task_name,
                    append_lgtoken=self.args.append_lgtoken,
                )
    def _get_most_left_pad_ind(self, tensor):
        assert tensor.ndim == 2
        inds = (tensor==self.source_dictionary.pad()).nonzero()
        if len(inds) > 0:
            return min(inds, key=lambda x:x[1])[1]
        else:
            return tensor.shape[1]

    def _get_task_loss(self, sample, net_output, encoder_out, net_output2, encoder_out2, model, criterion, task_name, device):
        indices = torch.tensor([i for i, task in enumerate(sample["task_name"]) if task == task_name]).to(device)
        if len(indices) == 0:
            # dummy 0 loss
            return [torch.tensor([0]).to(device)]* 5 + [indices]

        id_mlm_consis_loss = torch.tensor([0]).to(device)
        id_dae_consis_loss = torch.tensor([0]).to(device)
        id_mmt_consis_loss = torch.tensor([0]).to(device)

        if task_name == "mlm":
            min_ind = self._get_most_left_pad_ind(sample["net_input"]["src_tokens"][indices, :])
            if self.args.not_shared_proj:
                logits = model.encoder.output_layer(encoder_out["encoder_out"][0].transpose(0,1)[indices, :min_ind, :])
            else:
                logits = model.decoder.output_layer(encoder_out["encoder_out"][0].transpose(0,1)[indices, :min_ind, :])
            ## we concatenate dae and mlm target toghther, so the length of source could be different from the target, we remove the extra padding tokens
            loss, nll_loss = criterion.compute_loss(
                model,
                (logits, None),
                {
                    "nsentences": sample["nsentences"],
                    "ntokens": sample["ntokens"],
                    "target": sample["target"][indices, :min_ind],
                },
                reduce=True,
            )  
        elif task_name == "id_mlm":
            min_ind = self._get_most_left_pad_ind(sample["net_input"]["src_tokens"][indices, :])
            if self.args.not_shared_proj:
                logits_encoder = model(**sample["net_input"], step="encoder_proj_layer", layer_input=encoder_out["encoder_out"][0].transpose(0,1)[indices, :min_ind, :])
                # logits_encoder = model.encoder.output_layer(encoder_out["encoder_out"][0].transpose(0,1)[indices, :min_ind, :])
            else:
                logits_encoder = model(**sample["net_input"], step="decoder_proj_layer", layer_input=encoder_out["encoder_out"][0].transpose(0,1)[indices, :min_ind, :])
                # logits_encoder = model.decoder.output_layer(encoder_out["encoder_out"][0].transpose(0,1)[indices, :min_ind, :])
            
            loss_encoder, nll_loss_encoder = criterion.compute_loss(
                model,
                (logits_encoder, None),
                {
                    "nsentences": sample["nsentences"],
                    "ntokens": sample["ntokens"],
                    "target": sample["target"][indices, :min_ind],
                },
                reduce=True,
            )
            
            logits_decoder = net_output[0][indices, :min_ind, :]
            loss_decoder, nll_loss_decoder = criterion.compute_loss(
                model,
                (logits_decoder, None),
                {
                    "nsentences": sample["nsentences"],
                    "ntokens": sample["ntokens"],
                    "target": sample["target"][indices, :min_ind],
                },
                reduce=True,
            )
            if self.args.id_alpha > 0:
                logits_encoder, logits_decoder = logits_encoder.float(), logits_decoder.float()
                logits_encoder, logits_decoder = F.softmax(logits_encoder, dim=-1), F.softmax(logits_decoder, dim=-1)
                pad_mask = sample["target"][indices, :min_ind].eq(criterion.padding_idx)
                id_mlm_consis_loss = X_loss([logits_encoder, logits_decoder], pad_mask)
                loss = loss_encoder + loss_decoder + self.args.id_alpha * id_mlm_consis_loss
            else:
                loss = loss_encoder + loss_decoder
            nll_loss = nll_loss_encoder + nll_loss_decoder

        elif task_name == "id_mlm2":
            #### Pass 1 encoder
            min_ind = self._get_most_left_pad_ind(sample["net_input"]["src_tokens"][indices, :])
            if self.args.not_shared_proj:
                logits_encoder = model(**sample["net_input"], step="encoder_proj_layer", layer_input=encoder_out["encoder_out"][0].transpose(0,1)[indices, :min_ind, :])
                # logits_encoder = model.encoder.output_layer(encoder_out["encoder_out"][0].transpose(0,1)[indices, :min_ind, :])
            else:
                logits_encoder = model(**sample["net_input"], step="decoder_proj_layer", layer_input=encoder_out["encoder_out"][0].transpose(0,1)[indices, :min_ind, :])
                # logits_encoder = model.decoder.output_layer(encoder_out["encoder_out"][0].transpose(0,1)[indices, :min_ind, :])
            
            loss_encoder, nll_loss_encoder = criterion.compute_loss(
                model,
                (logits_encoder, None),
                {
                    "nsentences": sample["nsentences"],
                    "ntokens": sample["ntokens"],
                    "target": sample["target"][indices, :min_ind],
                },
                reduce=True,
            )

            ### Pass 2 encoder
            if self.args.not_shared_proj:
                logits_encoder2 = model(**sample["net_input"], step="encoder_proj_layer", layer_input=encoder_out2["encoder_out"][0].transpose(0,1)[indices, :min_ind, :])
            else:
                logits_encoder2 = model(**sample["net_input"], step="decoder_proj_layer", layer_input=encoder_out2["encoder_out"][0].transpose(0,1)[indices, :min_ind, :])

            loss_encoder2, nll_loss_encoder2 = criterion.compute_loss(
                model,
                (logits_encoder2, None),
                {
                    "nsentences": sample["nsentences"],
                    "ntokens": sample["ntokens"],
                    "target": sample["target"][indices, :min_ind],
                },
                reduce=True,
            )

            if self.args.id_alpha > 0:
                logits_encoder, logits_encoder2 = logits_encoder.float(), logits_encoder2.float()
                logits_encoder, logits_encoder2 = F.softmax(logits_encoder, dim=-1), F.softmax(logits_encoder2, dim=-1)
                pad_mask = sample["target"][indices, :min_ind].eq(criterion.padding_idx)
                id_mlm_consis_loss_encoder = X_loss([logits_encoder, logits_encoder2], pad_mask)

            del logits_encoder, logits_encoder2

            ### Pass 1 decoder
            logits_decoder = net_output[0][indices, :min_ind, :]
            loss_decoder, nll_loss_decoder = criterion.compute_loss(
                model,
                (logits_decoder, None),
                {
                    "nsentences": sample["nsentences"],
                    "ntokens": sample["ntokens"],
                    "target": sample["target"][indices, :min_ind],
                },
                reduce=True,
            )
            
            ## Pass 2 decoder 
            logits_decoder2 = net_output2[0][indices, :min_ind, :]
            loss_decoder2, nll_loss_decoder2 = criterion.compute_loss(
                model,
                (logits_decoder2, None),
                {
                    "nsentences": sample["nsentences"],
                    "ntokens": sample["ntokens"],
                    "target": sample["target"][indices, :min_ind],
                },
                reduce=True,
            )
            loss_encoder = 0.5 * (loss_encoder + loss_encoder2)
            loss_decoder = 0.5 * (loss_decoder + loss_decoder2)
            nll_loss_encoder = 0.5 * (nll_loss_encoder + nll_loss_encoder2)
            nll_loss_decoder = 0.5 * (nll_loss_decoder + nll_loss_decoder2)
            if self.args.id_alpha > 0:
                logits_decoder, logits_decoder2 = logits_decoder.float(), logits_decoder2.float()
                logits_decoder, logits_decoder2 = F.softmax(logits_decoder, dim=-1), F.softmax(logits_decoder2, dim=-1)
                pad_mask = sample["target"][indices, :min_ind].eq(criterion.padding_idx)
                id_mlm_consis_loss_decoder = X_loss([logits_decoder, logits_decoder2], pad_mask)
                
                id_mlm_consis_loss = 0.5 *(id_mlm_consis_loss_encoder + id_mlm_consis_loss_decoder)
                loss = loss_encoder + loss_decoder + self.args.id_alpha * id_mlm_consis_loss
            else:
                loss = loss_encoder + loss_decoder
            nll_loss = 0.5 * (nll_loss_encoder + nll_loss_decoder)

        elif task_name == "id_dae":
            min_ind_src = self._get_most_left_pad_ind(sample["net_input"]["src_tokens"][indices, :])
            min_ind_tgt = self._get_most_left_pad_ind(sample["target"][indices, :])
            min_ind = min(min_ind_src, min_ind_tgt)
            if self.args.not_shared_proj:
                logits_encoder = model(**sample["net_input"], step="encoder_proj_layer", layer_input=encoder_out["encoder_out"][0].transpose(0,1)[indices, :min_ind, :])
                # logits_encoder = model.encoder.output_layer(encoder_out["encoder_out"][0].transpose(0,1)[indices, :min_ind, :])
            else:
                logits_encoder = model(**sample["net_input"], step="decoder_proj_layer", layer_input=encoder_out["encoder_out"][0].transpose(0,1)[indices, :min_ind, :])
                # logits_encoder = model.decoder.output_layer(encoder_out["encoder_out"][0].transpose(0,1)[indices, :min_ind, :])
            loss_encoder, nll_loss_encoder = criterion.compute_loss(
                model,
                (logits_encoder, None),
                {
                    "nsentences": sample["nsentences"],
                    "ntokens": sample["ntokens"],
                    "target": sample["target"][indices, :min_ind],
                },
                reduce=True,
            )

            logits_decoder = net_output[0][indices, :, :]
            loss_decoder, nll_loss_decoder = criterion.compute_loss(
                model,
                (logits_decoder, None),
                {
                    "nsentences": sample["nsentences"],
                    "ntokens": sample["ntokens"],
                    "target": sample["target"][indices, :],
                },
                reduce=True,
            )
            if self.args.id_alpha > 0:
                logits_encoder, logits_decoder = logits_encoder.float(), logits_decoder[:, :min_ind, :].float()
                logits_encoder, logits_decoder = F.softmax(logits_encoder, dim=-1), F.softmax(logits_decoder, dim=-1)
                pad_mask = sample["target"][indices, :min_ind].eq(criterion.padding_idx)
                id_dae_consis_loss = X_loss([logits_encoder, logits_decoder], pad_mask)
                loss = loss_encoder + loss_decoder + self.args.id_alpha * id_dae_consis_loss
            else:
                loss = loss_encoder + loss_decoder

            nll_loss = nll_loss_encoder + nll_loss_decoder

        elif task_name == "mmt":
            logits = net_output[0][indices, :, :]
            loss, nll_loss = criterion.compute_loss(
                model,
                (logits, None),
                {
                    "nsentences": sample["nsentences"],
                    "ntokens": sample["ntokens"],
                    "target": sample["target"][indices, :],
                },
                reduce=True,
            )   
            if self.args.enable_id_mmt and self.args.id_alpha > 0:
                # second pass
                logits2 = net_output2[0][indices, :, :]
                loss2, nll_loss2 = criterion.compute_loss(
                    model,
                    (logits2, None),
                    {
                        "nsentences": sample["nsentences"],
                        "ntokens": sample["ntokens"],
                        "target": sample["target"][indices, :],
                    },
                    reduce=True,
                )   

                logits, logits2 = logits.float(), logits2.float()
                logits, logits2 = F.softmax(logits, dim=-1), F.softmax(logits2, dim=-1)
                pad_mask = sample["target"][indices, :].eq(criterion.padding_idx)
                id_mmt_consis_loss = X_loss([logits, logits2], pad_mask)
                loss = 0.5 * (loss + loss2) + self.args.id_alpha * id_mmt_consis_loss
                nll_loss = 0.5 * (nll_loss + nll_loss2)
        else:
            logits = net_output[0][indices, :, :]
            loss, nll_loss = criterion.compute_loss(
                model,
                (logits, None),
                {
                    "nsentences": sample["nsentences"],
                    "ntokens": sample["ntokens"],
                    "target": sample["target"][indices, :],
                },
                reduce=True,
            )   
        
        return loss, nll_loss, id_mlm_consis_loss, id_dae_consis_loss, id_mmt_consis_loss, indices


    def _get_loss(self, sample, model, criterion):
        assert hasattr(
            criterion, "compute_loss"
        ), "translation_thor task requires the criterion to implement the compute_loss() method"
        device = sample["net_input"]["src_tokens"].device
        batch_size = sample["net_input"]["src_tokens"].shape[0]

        # Wrap all forward into forward function to accommodate fully_sharded
        encoder_out = model(**sample["net_input"], step="encoder")
        net_output = model(**sample["net_input"], step="decoder", encoder_out=encoder_out)

        if "id_mlm2" in sample["task_name"] or (self.args.enable_id_mmt and "mmt" in sample["task_name"]):
            # second pass
            encoder_out2 = model(**sample["net_input"], step="encoder")
            net_output2 = model(**sample["net_input"], step="decoder", encoder_out=encoder_out)
        else:
            encoder_out2 = None
            net_output2 = None

        mlm_loss, mlm_nll_loss, _, _, _, mlm_ind = self._get_task_loss(sample, net_output, encoder_out, net_output2, encoder_out2, model, criterion, "mlm", device)
        dae_loss, dae_nll_loss, _, _, _, dae_ind = self._get_task_loss(sample, net_output, encoder_out, net_output2, encoder_out2, model, criterion, "dae", device)
        id_mlm_loss, id_mlm_nll_loss, id_mlm_consis_loss, _, _, id_mlm_ind = self._get_task_loss(sample, net_output, encoder_out, net_output2, encoder_out2, model, criterion, "id_mlm", device)
        id_mlm_loss, id_mlm_nll_loss, id_mlm_consis_loss, _, _, id_mlm_ind = self._get_task_loss(sample, net_output, encoder_out, net_output2, encoder_out2, model, criterion, "id_mlm2", device)
        id_dae_loss, id_dae_nll_loss, _, id_dae_consis_loss, _, id_dae_ind = self._get_task_loss(sample, net_output, encoder_out, net_output2, encoder_out2, model, criterion, "id_dae", device)
        mmt_loss, mmt_nll_loss, _, _, id_mmt_consis_loss, mmt_ind = self._get_task_loss(sample, net_output, encoder_out, net_output2, encoder_out2, model, criterion, "mmt", device)

        loss = mmt_loss + self.args.beta *( dae_loss + mlm_loss + id_mlm_loss + id_dae_loss )
        nll_loss = mmt_nll_loss + self.args.beta * (dae_nll_loss + mlm_nll_loss + id_mlm_nll_loss + id_dae_nll_loss)
        sample_size = (
            sample["target"].size(0) if criterion.sentence_avg else sample["ntokens"]
        )
        mmt_sample_size = (
            sample["target"][mmt_ind,:].size(0) if criterion.sentence_avg else
            max(int(sample["ntokens"] * (len(mmt_ind)/ batch_size)), 1)
        )
        mlm_sample_size = (
            sample["target"][mlm_ind,:].size(0) if criterion.sentence_avg else
            max(int(sample["ntokens"] * (len(mlm_ind)/ batch_size)), 1)
        )
        dae_sample_size = (
            sample["target"][dae_ind,:].size(0) if criterion.sentence_avg else
            max(int(sample["ntokens"] * (len(dae_ind)/ batch_size)), 1)
        )

        id_mlm_sample_size = (
            sample["target"][id_mlm_ind,:].size(0) if criterion.sentence_avg else
            max(int(sample["ntokens"] * (len(id_mlm_ind)/ batch_size)), 1)
        )

        id_dae_sample_size = (
            sample["target"][id_dae_ind,:].size(0) if criterion.sentence_avg else
            max(int(sample["ntokens"] * (len(id_dae_ind)/ batch_size)), 1)
        )

        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "mmt_loss": mmt_loss.data,
            "mlm_loss": mlm_loss.data,
            "dae_loss": dae_loss.data,
            "id_mlm_loss": id_mlm_loss.data,
            "id_mlm_consis_loss": id_mlm_consis_loss.data,
            "id_dae_loss": id_dae_loss.data,
            "id_dae_consis_loss": id_dae_consis_loss.data,
            "id_mmt_consis_loss": id_mmt_consis_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "mmt_sample_size": mmt_sample_size,
            "mlm_sample_size": mlm_sample_size,
            "dae_sample_size": dae_sample_size,
            "id_mlm_sample_size": id_mlm_sample_size,
            "id_dae_sample_size": id_dae_sample_size,
        }
        return loss, sample_size, logging_output

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        
        # if "id_mlm" in sample["task_name"]:
        #     print("TASK file print")
        #     print(sample["task_name"])
        # print(sample["net_input"]["src_tokens"], sample["net_input"]["src_tokens"].shape)
        # print(sample["target"], sample["target"].shape)
        # print(sample["net_input"]["prev_output_tokens"], sample["net_input"]["prev_output_tokens"].shape)
        # exit(0)
        
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                loss, sample_size, logging_output = self._get_loss(sample, model, criterion)

        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)

        def freeze_params(layer):
            for param in layer.parameters():
                param.grad = None

        if getattr(self.cfg, "freeze_up_to_layer", None):
            for module in model.modules():
                if not isinstance(module, torch.nn.ModuleList):
                    continue
                for layer in module[: getattr(self.cfg, "freeze_up_to_layer", None)]:
                    freeze_params(layer)

        # STEP = 1
        # all_score = []
        # for name, params in model.named_parameters():
        #     grad = params.grad.clone().detach().view(-1)
        #     params = params.clone().detach().view(-1)
        #     score = torch.abs(grad*params)
        #     score = score.to("cpu")
        #     all_score.append(score)

        # all_score = torch.cat(all_score, dim=-1).type(torch.float32)
        # self.all_score[0] = (self.all_score[0] * self.all_score[1] + all_score) / (self.all_score[1] + 1)
        # self.all_score[1] += 1
        # for name, params in model.named_parameters():
        #     params.grad = None
        # if self.all_score[1] == 500:
        #     torch.save(self.all_score[0], f"/checkpoint/haoranxu/SSL/analysis/m8_32k_500/{self.args.lg}_id{self.args.ssl}")
        #     exit(0)

        STEP = 1
        flag = True
        # for task in sample["task_name"]:
        #     if task != "mmt":
        #         flag = False
        if flag:
            for name, params in model.named_parameters():
                grad = params.grad.clone().detach()
                params = params.clone().detach()
                score = torch.abs(grad*params)
                score = score.to("cpu").type(torch.float32)
                self.all_score[name][0] = (self.all_score[name][0] * self.all_score[name][1] + score) / (self.all_score[name][1] + 1)
                self.all_score[name][1] += 1

        for name, params in model.named_parameters():
            params.grad = None
        if self.all_score[name][1] == 500:
            torch.save(dict(self.all_score), f"/checkpoint/haoranxu/SSL/analysis/m8_32k_500_eng_xx/dict_except_{self.args.lg}_id")
            exit(0)

        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = self._get_loss(sample, model, criterion)
        return loss, sample_size, logging_output

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        with torch.no_grad():
            _, tgt_langtok_spec = self.args.langtoks["main"]
            if not self.args.lang_tok_replacing_bos_eos:
                if prefix_tokens is None and tgt_langtok_spec:
                    tgt_lang_tok = self.data_manager.get_decoder_langtok(
                        self.args.target_lang, tgt_langtok_spec
                    )
                    src_tokens = sample["net_input"]["src_tokens"]
                    bsz = src_tokens.size(0)
                    prefix_tokens = (
                        torch.LongTensor([[tgt_lang_tok]]).expand(bsz, 1).to(src_tokens)
                    )

                return generator.generate(
                    models,
                    sample,
                    prefix_tokens=prefix_tokens,
                    constraints=constraints,
                    bos_token=self.target_dictionary.index(f"__{self.args.target_lang}__")
                )
            else:
                return generator.generate(
                    models,
                    sample,
                    prefix_tokens=prefix_tokens,
                    bos_token=self.data_manager.get_decoder_langtok(
                        self.args.target_lang, tgt_langtok_spec
                    )
                    if tgt_langtok_spec
                    else self.target_dictionary.index(f"__{self.args.target_lang}__"),
                )

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        # follows the reduce_metrics() function in label_smoothed_cross_entropy.py
        mmt_loss_sum = sum(log.get("mmt_loss", 0) for log in logging_outputs)
        dae_loss_sum = sum(log.get("dae_loss", 0) for log in logging_outputs)
        mlm_loss_sum = sum(log.get("mlm_loss", 0) for log in logging_outputs)
        id_mlm_loss_sum = sum(log.get("id_mlm_loss", 0) for log in logging_outputs)
        id_dae_loss_sum = sum(log.get("id_dae_loss", 0) for log in logging_outputs)
        id_mlm_consis_loss_sum = sum(log.get("id_mlm_consis_loss", 0) for log in logging_outputs)
        id_dae_consis_loss_sum = sum(log.get("id_dae_consis_loss", 0) for log in logging_outputs)
        id_mmt_consis_loss_sum = sum(log.get("id_mmt_consis_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        mmt_sample_size = sum(log.get("mmt_sample_size", 0) for log in logging_outputs)
        mlm_sample_size = sum(log.get("mlm_sample_size", 0) for log in logging_outputs)
        dae_sample_size = sum(log.get("dae_sample_size", 0) for log in logging_outputs)
        id_mlm_sample_size = sum(log.get("id_mlm_sample_size", 0) for log in logging_outputs)
        id_dae_sample_size = sum(log.get("id_dae_sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "mmt_loss", mmt_loss_sum / mmt_sample_size / math.log(2), mmt_sample_size, round=3
        )
        metrics.log_scalar(
            "dae_loss", dae_loss_sum / dae_sample_size / math.log(2), dae_sample_size, round=3
        )
        metrics.log_scalar(
            "mlm_loss", mlm_loss_sum / mlm_sample_size / math.log(2), mlm_sample_size, round=3
        )
        metrics.log_scalar(
            "id_mlm_loss", id_mlm_loss_sum / id_mlm_sample_size / math.log(2), id_mlm_sample_size, round=3
        )
        metrics.log_scalar(
            "id_mlm_consis_loss", id_mlm_consis_loss_sum / id_mlm_sample_size / math.log(2), id_mlm_sample_size, round=3
        )
        metrics.log_scalar(
            "id_dae_loss", id_dae_loss_sum / id_dae_sample_size / math.log(2), id_dae_sample_size, round=3
        )
        metrics.log_scalar(
            "id_dae_consis_loss", id_dae_consis_loss_sum / id_dae_sample_size / math.log(2), id_dae_sample_size, round=3
        )
        metrics.log_scalar(
            "id_mmt_consis_loss", id_mmt_consis_loss_sum / mmt_sample_size / math.log(2), mmt_sample_size, round=3
        )



    def _get_whole_word_mask(self):
        # create masked input and targets
        if self.args.mask_whole_words:
            bpe = encoders.build_bpe(self.args)
            if bpe is not None:

                def is_beginning_of_word(i):
                    if i < self.source_dictionary.nspecial:
                        # special elements are always considered beginnings
                        return True
                    if i >= len(self.source_dictionary) - len(self.dicts) - 1:
                        # larger than indices of lang tokens and masks
                        return True
                    tok = self.source_dictionary[i]
                    if tok.startswith("madeupword"):
                        return True
                    try:
                        return bpe.is_beginning_of_word(tok)
                    except ValueError:
                        return True

                mask_whole_words = torch.ByteTensor(
                    list(map(is_beginning_of_word, range(len(self.source_dictionary))))
                )
        else:
            mask_whole_words = None
        return mask_whole_words

    def _get_sample_prob(self, dataset_lens):
        """
        Get smoothed sampling porbability by languages. This helps low resource
        languages by upsampling them.
        """
        prob = dataset_lens / dataset_lens.sum()
        smoothed_prob = prob**self.args.multilang_sampling_alpha
        smoothed_prob = smoothed_prob / smoothed_prob.sum()
        return smoothed_prob

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        if constraints is not None:
            raise NotImplementedError(
                "Constrained decoding with the multilingual_translation task is not supported"
            )

        src_data = ListDataset(src_tokens, src_lengths)
        src_lang = self.source_dictionary.index("__{}__".format(self.args.source_lang))
        tgt_lang = self.source_dictionary.index("__{}__".format(self.args.target_lang))
        dataset = LanguagePairDataset(src_data, src_lengths, self.source_dictionary, src_lang=src_lang, tgt_lang=tgt_lang)

        dataset = SimpleSSLModelDataset(
            dataset,
            task_name="mmt",
            append_lgtoken=self.args.append_lgtoken,
            )
        return dataset
    
