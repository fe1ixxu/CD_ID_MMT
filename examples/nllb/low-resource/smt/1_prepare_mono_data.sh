#!/bin/bash

KENLM=/private/home/jeanm/src/kenlm/build/bin/lmplz
MOSES_DIR=/private/home/jeanm/src/mosesdecoder
clean_corpus=${MOSES_DIR}/scripts/training/clean-corpus-n.perl
replace_punc=${MOSES_DIR}/scripts/tokenizer/replace-unicode-punctuation.perl
norm_punc=${MOSES_DIR}/scripts/tokenizer/normalize-punctuation.perl
tokenizer=${MOSES_DIR}/scripts/tokenizer/tokenizer.perl
train_truecaser=${MOSES_DIR}/scripts/recaser/train-truecaser.perl
truecase=${MOSES_DIR}/scripts/recaser/truecase.perl
spm_encode=/private/home/jeanm/src/sentencepiece/build/src/spm_encode


# SRC_MONO_DIR=/large_experiments/nllb/mmt/phrase_mining/sentences_lid210
# SRC_MULTI_DATA_DIR=/checkpoint/haoranxu/SSL/data/bilingual/eng_wol.8k.bible #/large_experiments/nllb/mmt/multilingual_bin/multilingual_public+mined.ssl.v1.256k
# SRC_BI_DATA_DIR=/checkpoint/haoranxu/SSL/data/bilingual/eng_wol.8k.bible #/large_experiments/nllb/mmt/multilingual_bin/bilingual_public+mined.ssl.v1.8k
# OUT_DIR=/checkpoint/haoranxu/SSL/data/moses_public+mined.ssl.v1/
# LANGS=${1:-wol,eng}
# #${1:-amh,cjk,eng,ewe,fon,fuv,hau,ibo,kam,kik,kin,kmb,kon,lin,lug,luo,nso,nya,orm,run,som,ssw,swh,tir,tsn,tso,twi,umb,wol,xho,zul}
# PARTITION=learnfair

SRC_MONO_DIR=${1}
OUT_DIR=${2}
LANGS=${3}
PARTITION=${4}


mkdir -p ${OUT_DIR}/corpora
mkdir -p ${OUT_DIR}/scripts
mkdir -p ${OUT_DIR}/models/smt
mkdir -p ${OUT_DIR}/models/lms
mkdir -p ${OUT_DIR}/models/truecasers


#####################################
# MONOLINGUAL: NORMALISE & TRUECASE #
#####################################
for src in ${LANGS//,/ }; do
  if [ -f ${OUT_DIR}/corpora/truecase.mono.${src} ]; then
    echo "Skipped: Preparing ${src} monolingual corpus"
    continue
  fi
  echo Preparing ${src} monolingual corpus
  script=${OUT_DIR}/scripts/corpora.mono.${src}.sh
cat << EOF > ${script}
#!/bin/bash

echo Preparing ${src} monolingual corpus
cat ${SRC_MONO_DIR}/${src}*.txt | shuf -n 30000000 | ${replace_punc} \
  | ${norm_punc} -l en > ${OUT_DIR}/corpora/norm.mono.${src}

echo Training ${src} truecaser
${train_truecaser} --model ${OUT_DIR}/models/truecasers/${src} \
  --corpus ${OUT_DIR}/corpora/norm.mono.${src}

echo Truecasing ${src} monolingual data
${truecase} --model ${OUT_DIR}/models/truecasers/${src} \
  < ${OUT_DIR}/corpora/norm.mono.${src} \
  > ${OUT_DIR}/corpora/truecase.mono.${src}
EOF
  sbatch -J corpora.mono.${src} --partition=${PARTITION} --nodes=1 --ntasks-per-node=1 \
    --gpus-per-node=0 --cpus-per-task=8 --mem=50G --time=1200 ${script}
done
