#!/bin/bash
STEP=${1}
BILINGUAL_DIR=${2:-/checkpoint/haoranxu/SSL/data/bilingual/eng_wol.1k}
MULTILINGUAL_DIR=${3:-/checkpoint/haoranxu/SSL/data/bilingual/eng_wol.1k}
MONOLINGUAL_DIR=${4:-/checkpoint/haoranxu/SSL/data/monolingual/for_low_resource}
LANGS=${5:-wol,eng}
VOCAB_SIZE=${6:-1000}
OUT_DIR=${7:-/checkpoint/haoranxu/SSL/data/mose_data/mose_data.1k}
PARTITION=${8:-devlab}

if [ ${STEP} == "0" ]; then
    python ./examples/nllb/low-resource/smt/0_make_bitexts_config.py \
        --bilingual ${BILINGUAL_DIR} \
        --multilingual ${MULTILINGUAL_DIR} \
        --monolingual ${MONOLINGUAL_DIR} \
        --vocab_size ${VOCAB_SIZE} \
        --langs ${LANGS}

    ## Copy vocab_bin to the bilingual vocab bin
    for src in ${LANGS//,/ }; do
        for tgt in ${LANGS//,/ }; do
            if [ -d ${BILINGUAL_DIR}/${src}-${tgt} ]; then
                cp -r ${MULTILINGUAL_DIR}/vocab_bin ${BILINGUAL_DIR}/${src}-${tgt}/
            fi
        done
    done
elif [ ${STEP} == "1" ]; then
    bash ./examples/nllb/low-resource/smt/1_prepare_mono_data.sh \
        ${MONOLINGUAL_DIR} \
        ${OUT_DIR} \
        ${LANGS} \
        ${PARTITION}
elif [ ${STEP} == "2" ]; then
    bash ./examples/nllb/low-resource/smt/2_prepare_smt_data_and_lms.sh \
        ${MULTILINGUAL_DIR} \
        ${BILINGUAL_DIR} \
        ${OUT_DIR} \
        ${LANGS} \
        ${PARTITION} \
        ${VOCAB_SIZE}
elif [ ${STEP} == "3" ]; then
    bash ./examples/nllb/low-resource/smt/3a_train_moses.sh \
        ${OUT_DIR} \
        ${LANGS} \
        ${PARTITION}
elif [ ${STEP} == "4" ]; then
    bash ./examples/nllb/low-resource/smt/4_tune.sh \
        ${OUT_DIR} \
        ${LANGS} \
        ${PARTITION}
elif [ ${STEP} == "5" ]; then
    bash ./examples/nllb/low-resource/smt/5a_evaluate_moses.sh \
        ${OUT_DIR} \
        ${BILINGUAL_DIR} \
        ${LANGS} \
        ${VOCAB_SIZE}
fi
