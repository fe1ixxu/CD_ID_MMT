import os
from copy import deepcopy
from glob import glob

import yaml
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--bilingual', type=str, required=True, help='input bilingual file')
parser.add_argument('--multilingual', type=str, required=True, help='input multilingual file')
parser.add_argument('--monolingual', type=str, required=True, help='input monolingual file')
parser.add_argument('--vocab_size', type=int, required=True, help='vocabulary size')
parser.add_argument('--langs', type=str, required=True, help='language lists')




args = parser.parse_args()

MULTILINGUAL_DIR = (
    # "/large_experiments/nllb/mmt/multilingual_bin/multilingual_public+mined.ssl.v1.256k"
    # "/checkpoint/haoranxu/toy/multilingual"
    # "/checkpoint/haoranxu/SSL/data/bilingual/eng_wol.8k.bible"
    args.multilingual
)
BILINGUAL_DIR = (
    # "/large_experiments/nllb/mmt/multilingual_bin/bilingual_public+mined.ssl.v1.8k"
    # "/checkpoint/haoranxu/toy/bilingual"
    # "/checkpoint/haoranxu/SSL/data/bilingual/eng_wol.8k.bible"
    args.bilingual
)
# LANGS = "amh cjk eng ewe fon fuv hau ibo kam kik kin kmb kon lin lug luo nso nya orm run som ssw swh tir tsn tso twi umb wol xho zul".split()
# LANGS = "wol eng".split()
LANGS = args.langs.split(",")
VOCAB_SIZE = args.vocab_size
# mono_root = "/large_experiments/nllb/mmt/phrase_mining/sentences_lid210"
mono_root = args.monolingual

# Iterate throug all directions. Figure out which ones are valid.
valid_directions = set()
for src in LANGS:
    for tgt in LANGS:
        if os.path.exists(f"{MULTILINGUAL_DIR}/retrieved_data/train.{src}-{tgt}.{src}"):
            valid_directions.add(f"{src}-{tgt}")

with open(f"{MULTILINGUAL_DIR}/config.yaml", "r") as fin:
    multilingual_config = yaml.safe_load(fin)

# Produce the main configs for the bilingual models
for direction in valid_directions:
    print(f"Writing main config for {direction}")
    config = deepcopy(multilingual_config)
    config["train_corpora"] = {
        key: val for key, val in config["train_corpora"].items() if key == direction
    }
    config["valid_corpora"] = {
        key: val for key, val in config["valid_corpora"].items() if key == direction
    }
    config["test_corpora"] = {
        key: val for key, val in config["test_corpora"].items() if key == direction
    }
    config["source_vocab_config"]["vocab_build_params"]["vocab_size"] = VOCAB_SIZE
    config["target_vocab_config"]["vocab_build_params"]["vocab_size"] = VOCAB_SIZE
    config["executor_config"]["slurm_partition"] = "devaccel"
    os.makedirs(f"{BILINGUAL_DIR}/{direction}", exist_ok=True)
    # with open(f"{BILINGUAL_DIR}/{direction}/config.yaml", "w") as fout:
    #    yaml.dump(config, fout)

# Produces the monolingual configs for the bilingual models
base_mono_config = yaml.safe_load(
    """executor_config:
  log_folder: mono_executor_logs
  cluster: slurm
  slurm_partition: learnaccel
preprocessing_config:
  sample_size: null
  max_tokens: null
  moses_config:
    script_directory: examples/nllb/modeling/preprocessing/moses
    lowercase: false
    normalize_punctuation: true
    remove_non_printing_chars: false
    deescape_special_chars: false
  preprocess_source: true
  preprocess_target: false
  tag_secondary_data: false
shard_size: 100000
binarize_workers: 60
random_seed: 0"""
)

for direction in valid_directions:
    src, tgt = direction.split("-")
    src_path = list(glob(f"{mono_root}/{src}/*"))[0]
    tgt_path = list(glob(f"{mono_root}/{tgt}/*"))[0]
    print(f"Writing main config for {direction}")
    config = deepcopy(base_mono_config)
    config["corpora"] = {
        src: {
            "values": {
                "sentences_lid210": {
                    "local_paths": {
                        "path": src_path,
                        "num_lines": None,
                        "compression": None,
                    }
                }
            }
        },
        tgt: {
            "values": {
                "sentences_lid210": {
                    "local_paths": {
                        "path": tgt_path,
                        "num_lines": None,
                        "compression": None,
                    }
                }
            }
        },
    }
    config["built_vocab"] = {
        "model_file": f"{MULTILINGUAL_DIR}/vocab_bin/sentencepiece.source.{VOCAB_SIZE}.model",
        "vocab_file": f"{MULTILINGUAL_DIR}/vocab_bin/sentencepiece.source.{VOCAB_SIZE}.vocab",
        "dict_file": None,
    }
    with open(f"{BILINGUAL_DIR}/{direction}/mono_config.yaml", "w") as fout:
       yaml.dump(config, fout)

# Produces the monolingual config for the multilingual model
config = deepcopy(base_mono_config)
config["corpora"] = {}
config["built_vocab"] = {
    "model_file": f"{MULTILINGUAL_DIR}/vocab_bin/sentencepiece.source.{VOCAB_SIZE}.model",
    "vocab_file": f"{MULTILINGUAL_DIR}/vocab_bin/sentencepiece.source.{VOCAB_SIZE}.vocab",
    "dict_file": None,
}
for lang in LANGS:
    path = list(glob(f"{mono_root}/{lang}/*"))[0]
    config["corpora"][lang] = {
        "values": {
            "sentences_lid210": {
                "local_paths": {"path": path, "num_lines": None, "compression": None}
            }
        }
    }
with open(f"{MULTILINGUAL_DIR}/mono_config.yaml", "w") as fout:
    yaml.dump(config, fout)