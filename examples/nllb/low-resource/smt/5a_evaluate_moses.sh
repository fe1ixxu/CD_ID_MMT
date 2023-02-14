#!/bin/bash

# OUT_DIR=/checkpoint/haoranxu/SSL/data/moses_public+mined.ssl.v1
# /checkpoint/jeanm/nllb/moses_public+mined.ssl.v1
# RAW_CORPORA_DIR=/checkpoint/haoranxu/SSL/data/bilingual/eng_wol.8k.bible
# /large_experiments/nllb/mmt/multilingual_bin/bilingual_public+mined.ssl.v1.8k
OUT_DIR=${1}
RAW_CORPORA_DIR=${2}
LANGS=${3}
VOCAB_SIZE=${4}

EVAL_PARTITION=test
FLORES_PARTITION=devtest
FLORES=/large_experiments/nllb/mmt/flores101/${FLORES_PARTITION}
FLORES_BETA=/large_experiments/nllb/mmt/flores101_beta/${FLORES_PARTITION}

MOSES_DIR=/private/home/jeanm/src/mosesdecoder
moses=${MOSES_DIR}/bin/moses
recase=${MOSES_DIR}/scripts/recaser/detruecase.perl
unescape=${MOSES_DIR}/scripts/tokenizer/deescape-special-chars.perl
spm_decode=/private/home/jeanm/src/sentencepiece/build/src/spm_decode

SACREBLEU=/private/home/jeanm/src/sacrebleu/sacrebleu/sacrebleu.py

# LANGS=${1:-wol,eng}
# LANGS=${1:-amh,cjk,ewe,fon,fuv,hau,ibo,kam,kik,kin,kmb,kon,lin,lug,luo,nso,nya,orm,run,som,ssw,swh,tir,tsn,tso,twi,umb,wol,xho,zul}

for lang in ${LANGS//,/ }; do
    if [ -f ${RAW_CORPORA_DIR}/vocab_bin/sentencepiece.source.${VOCAB_SIZE}.model ]; then
        spm_model=${RAW_CORPORA_DIR}/vocab_bin/sentencepiece.source.${VOCAB_SIZE}.model
    elif [ -f ${RAW_CORPORA_DIR}/vocab_bin/sentencepiece.source.${VOCAB_SIZE}.model ]; then
        spm_model=${RAW_CORPORA_DIR}/vocab_bin/sentencepiece.source.${VOCAB_SIZE}.model
    fi
    if [ -f ${OUT_DIR}/corpora/spm.${EVAL_PARTITION}.eng-${lang}.eng ]; then
        data_pref=${OUT_DIR}/corpora/spm.${EVAL_PARTITION}.eng-${lang}
    elif [ -f ${OUT_DIR}/corpora/spm.${EVAL_PARTITION}.${lang}-eng.eng ]; then
        data_pref=${OUT_DIR}/corpora/spm.${EVAL_PARTITION}.${lang}-eng
    fi
  

  model_dir=${OUT_DIR}/models/smt/eng-${lang}
  # check if it's in flores101 or beta
  flores_path=${FLORES}/${lang}.${FLORES_PARTITION}
  if [ ! -f ${flores_path} ]; then
    flores_path=${FLORES_BETA}/${lang}.${FLORES_PARTITION}
  fi

  ######

#   if [ -f ${model_dir}/${EVAL_PARTITION}.bleu ]; then
#     echo Already evaluated: eng-${lang}.
  if [ -f ${model_dir}/model/moses.ini ]; then
    echo Translating eng data into ${lang}...
    $moses -threads 10 -f ${model_dir}/model/moses.ini < ${data_pref}.eng \
      | ${spm_decode} --model ${spm_model} \
      | ${recase} \
      > ${model_dir}/${EVAL_PARTITION}.${lang}
    lang=${lang} SACREBLEU_FORMAT=text python ${SACREBLEU} -tok spm ${flores_path} \
      < ${model_dir}/${EVAL_PARTITION}.${lang} \
      > ${model_dir}/${EVAL_PARTITION}.bleu
  else
    echo Path not found: ${model_dir}/tuning/mert/moses.ini
  fi

  ######

  model_dir=${OUT_DIR}/models/smt/${lang}-eng
#   if [ -f ${model_dir}/${EVAL_PARTITION}.bleu ]; then
#     echo Already evaluated: ${lang}-eng.
  if [ -f ${model_dir}/model/moses.ini ]; then
    echo Translating ${lang} data into eng...
    $moses -threads 10 -f ${model_dir}/model/moses.ini < ${data_pref}.${lang} \
      | ${spm_decode} --model ${spm_model} \
      | ${recase} \
      > ${model_dir}/${EVAL_PARTITION}.eng
    lang=${lang} SACREBLEU_FORMAT=text python ${SACREBLEU} -tok spm ${FLORES}/eng.${FLORES_PARTITION} \
      < ${model_dir}/${EVAL_PARTITION}.eng \
      > ${model_dir}/${EVAL_PARTITION}.bleu
  else
    echo Path not found: ${model_dir}/tuning/mert/moses.ini
  fi

done

for lang in ${LANGS//,/ }; do
    if [ ${lang} != eng ]; then
        model_dir=${OUT_DIR}/models/smt/eng-${lang}
        echo "eng-${lang} is:"
        cat ${model_dir}/${EVAL_PARTITION}.bleu

        model_dir=${OUT_DIR}/models/smt/${lang}-eng
        echo "${lang}-eng is:"
        cat ${model_dir}/${EVAL_PARTITION}.bleu
    fi
done