#!/bin/bash

KENLM=/private/home/jeanm/src/kenlm/build/bin/lmplz
MOSES_DIR=/private/home/jeanm/src/mosesdecoder
clean_corpus=${MOSES_DIR}/scripts/training/clean-corpus-n.perl
replace_punc=${MOSES_DIR}/scripts/tokenizer/replace-unicode-punctuation.perl
norm_punc=${MOSES_DIR}/scripts/tokenizer/normalize-punctuation.perl
tokenizer=${MOSES_DIR}/scripts/tokenizer/tokenizer.perl
escape=${MOSES_DIR}/scripts/tokenizer/escape-special-chars.perl
train_truecaser=${MOSES_DIR}/scripts/recaser/train-truecaser.perl
truecase=${MOSES_DIR}/scripts/recaser/truecase.perl
spm_encode=/private/home/jeanm/src/sentencepiece/build/src/spm_encode


# SRC_MONO_DIR=/large_experiments/nllb/mmt/phrase_mining/sentences_lid210
# SRC_MULTI_DATA_DIR=/checkpoint/haoranxu/SSL/data/bilingual/eng_wol.8k.bible
# #/large_experiments/nllb/mmt/multilingual_bin/multilingual_public+mined.ssl.v1.256k
# SRC_BI_DATA_DIR=/checkpoint/haoranxu/SSL/data/bilingual/eng_wol.8k.bible
# #/large_experiments/nllb/mmt/multilingual_bin/bilingual_public+mined.ssl.v1.8k

# OUT_DIR=/checkpoint/haoranxu/SSL/data/moses_public+mined.ssl.v1/
# LANGS=${1:-wol,eng}
# #${1:-amh,cjk,eng,ewe,fon,fuv,hau,ibo,kam,kik,kin,kmb,kon,lin,lug,luo,nso,nya,orm,run,som,ssw,swh,tir,tsn,tso,twi,umb,wol,xho,zul}
# PARTITION=learnfair

SRC_MULTI_DATA_DIR=${1}
SRC_BI_DATA_DIR=${2}
OUT_DIR=${3}
LANGS=${4}
PARTITION=${5}
VOCAB_SIZE=${6}


mkdir -p ${OUT_DIR}/corpora
mkdir -p ${OUT_DIR}/scripts
mkdir -p ${OUT_DIR}/models/smt
mkdir -p ${OUT_DIR}/models/lms
mkdir -p ${OUT_DIR}/models/truecasers


for src in ${LANGS//,/ }; do
  for tgt in ${LANGS//,/ }; do
    spm_model=${SRC_BI_DATA_DIR}/vocab_bin/sentencepiece.source.${VOCAB_SIZE}.model

    # Skip invalid directions
    if [[ ${src} != 'eng' ]]; then
      continue
    fi
    if [[ ${src} == 'eng' && ${tgt} == 'eng' ]]; then
      continue
    fi

    ####################
    # MONOLINGUAL: BPE #
    ####################

    if [[ -f ${OUT_DIR}/models/lms/spm.${src}-${tgt}.${src} && -f ${OUT_DIR}/models/lms/spm.${src}-${tgt}.${tgt} && -f ${OUT_DIR}/corpora/spm.mono.${src}-${tgt}.${src} && -f ${OUT_DIR}/corpora/spm.mono.${src}-${tgt}.${tgt} ]]; then
      echo "Skipped: Training ${src} and ${tgt} BPE LMs (${src}-${tgt} dict)"
    else
      echo "Training ${src} and ${tgt} BPE LMs (${src}-${tgt} dict)"
      script=${OUT_DIR}/scripts/models.lms.spm.${src}-${tgt}.sh
      cat << EOF > ${script}
#!/bin/bash
if [ ! -f ${OUT_DIR}/corpora/spm.mono.${src}-${tgt}.${src} ]; then
  echo "Applying BPE to ${src} mono data (${src}-${tgt} dict)"
  ${spm_encode} --model=${spm_model} --output_format=piece \
    < ${OUT_DIR}/corpora/truecase.mono.${src} \
    > ${OUT_DIR}/corpora/spm.mono.${src}-${tgt}.${src}
fi
if [ ! -f ${OUT_DIR}/corpora/spm.mono.${src}-${tgt}.${tgt} ]; then
  echo "Applying BPE to ${tgt} mono data (${src}-${tgt} dict)"
  ${spm_encode} --model=${spm_model} --output_format=piece \
    < ${OUT_DIR}/corpora/truecase.mono.${tgt} \
    > ${OUT_DIR}/corpora/spm.mono.${src}-${tgt}.${tgt}
fi

if [ ! -f ${OUT_DIR}/models/lms/spm.${src}-${tgt}.${src} ]; then
  echo "Training ${src} BPE LMs (${src}-${tgt} dict)"
  ${KENLM} --prune 0 0 3 -o 5 --discount_fallback \
    < ${OUT_DIR}/corpora/spm.mono.${src}-${tgt}.${src} \
    > ${OUT_DIR}/models/lms/spm.${src}-${tgt}.${src}
fi
if [ ! -f ${OUT_DIR}/models/lms/spm.${src}-${tgt}.${tgt} ]; then
  echo "Training ${tgt} BPE LMs (${src}-${tgt} dict)"
  ${KENLM} --prune 0 0 3 -o 5 --discount_fallback \
    < ${OUT_DIR}/corpora/spm.mono.${src}-${tgt}.${tgt} \
    > ${OUT_DIR}/models/lms/spm.${src}-${tgt}.${tgt}
fi
EOF
      sbatch -J models.lms.spm.${src}-${tgt} --partition=${PARTITION} --nodes=1 --ntasks-per-node=1 \
        --gpus-per-node=0 --cpus-per-task=8 --mem=150G --time=1200 ${script}
    fi

    ##########################
    # BITEXT: PREPARE SPLITS #
    ##########################

    for split in train test valid; do
      raw_corpus_pref=${SRC_MULTI_DATA_DIR}/retrieved_data/${split}.${src}-${tgt}

      if [[ -f ${OUT_DIR}/corpora/spm.${split}.${src}-${tgt}.${src} && -f ${OUT_DIR}/corpora/spm.${split}.${src}-${tgt}.${tgt} ]]; then
        echo "Skipped: Processing ${split}.${src}-${tgt}"
        continue
      fi
      echo "Processing ${split}.${src}-${tgt}"
      script=${OUT_DIR}/scripts/corpora.${split}.${src}-${tgt}.sh
      cat << EOF > ${script}
#!/bin/bash

# Clean data if necessary
if [ "${split}" == "train" ]; then
  echo "Cleaning ${src}-${tgt}.${split}"
  ${clean_corpus} ${raw_corpus_pref} ${src} ${tgt} ${OUT_DIR}/corpora/clean.${split}.${src}-${tgt} 1 50
else  # only the training set needs to be cleaned!
  echo "No need to clean ${src}-${tgt}.${split}"
  cp ${raw_corpus_pref}.${src} ${OUT_DIR}/corpora/clean.${split}.${src}-${tgt}.${src}
  cp ${raw_corpus_pref}.${tgt} ${OUT_DIR}/corpora/clean.${split}.${src}-${tgt}.${tgt}
fi

# Normalise data
if [ ! -f ${OUT_DIR}/corpora/norm.${split}.${src}-${tgt}.${src} ]; then
  echo "Normalising ${split}.${src}-${tgt}.${src}"
  cat ${OUT_DIR}/corpora/clean.${split}.${src}-${tgt}.${src} > ${OUT_DIR}/corpora/norm.${split}.${src}-${tgt}.${src}
fi
if [ ! -f ${OUT_DIR}/corpora/norm.${split}.${src}-${tgt}.${tgt} ]; then
  echo "Normalising ${split}.${src}-${tgt}.${tgt}"
  cat ${OUT_DIR}/corpora/clean.${split}.${src}-${tgt}.${tgt} > ${OUT_DIR}/corpora/norm.${split}.${src}-${tgt}.${tgt}
fi

# Truecase data
if [ ! -f ${OUT_DIR}/corpora/truecase.${split}.${src}-${tgt}.${src} ]; then
  echo "Truecasing ${split}.${src}-${tgt}.${src}"
  $truecase --model ${OUT_DIR}/models/truecasers/${src} \
    < ${OUT_DIR}/corpora/norm.${split}.${src}-${tgt}.${src} \
    > ${OUT_DIR}/corpora/truecase.${split}.${src}-${tgt}.${src}
fi
if [ ! -f ${OUT_DIR}/corpora/truecase.${split}.${src}-${tgt}.${tgt} ]; then
  echo "Truecasing ${split}.${src}-${tgt}.${tgt}"
  $truecase --model ${OUT_DIR}/models/truecasers/${tgt} \
    < ${OUT_DIR}/corpora/norm.${split}.${src}-${tgt}.${tgt} \
    > ${OUT_DIR}/corpora/truecase.${split}.${src}-${tgt}.${tgt}
fi

# BPE data
if [ ! -f ${OUT_DIR}/corpora/spm.${split}.${src}-${tgt}.${src} ]; then
  echo "Applying BPE to ${split}.${src}-${tgt}.${src}"
  ${spm_encode} --model=${spm_model} --output_format=piece \
    < ${OUT_DIR}/corpora/truecase.${split}.${src}-${tgt}.${src} \
    > ${OUT_DIR}/corpora/spm.${split}.${src}-${tgt}.${src}
fi
if [ ! -f ${OUT_DIR}/corpora/spm.${split}.${src}-${tgt}.${tgt} ]; then
  echo "Applying BPE to ${split}.${src}-${tgt}.${src}"
  ${spm_encode} --model=${spm_model} --output_format=piece \
    < ${OUT_DIR}/corpora/truecase.${split}.${src}-${tgt}.${tgt} \
    > ${OUT_DIR}/corpora/spm.${split}.${src}-${tgt}.${tgt}
fi
EOF
      sbatch -J corpora.${split}.${src}-${tgt} --partition=${PARTITION} --nodes=1 --ntasks-per-node=1 \
        --gpus-per-node=0 --cpus-per-task=8 --mem=50G --time=1200 ${script}
    done
  done
done
