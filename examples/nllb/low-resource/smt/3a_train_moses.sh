#!/bin/bash

MOSES_DIR=/private/home/jeanm/src/mosesdecoder
# OUT_DIR=/checkpoint/haoranxu/toy/moses_public+mined.ssl.v1
# OUT_DIR=/checkpoint/haoranxu/SSL/data/moses_public+mined.ssl.v1
#/checkpoint/jeanm/nllb/moses_public+mined.ssl.v1

train_model=${MOSES_DIR}/scripts/training/train-model.perl

# LANGS=${1:-wol,eng}
# ${1:-eng,wol}
#${1:-amh,cjk,eng,ewe,fon,fuv,hau,ibo,kam,kik,kin,kmb,kon,lin,lug,luo,nso,nya,orm,run,som,ssw,swh,tir,tsn,tso,twi,umb,wol,xho,zul}
# PARTITION=learnfair

OUT_DIR=${1}
LANGS=${2}
PARTITION=${3}


mkdir -p ${OUT_DIR}/logs

for src in ${LANGS//,/ }; do
  for tgt in ${LANGS//,/ }; do
  lm_model_pref=${OUT_DIR}/models/lms/spm.${src}-${tgt}
  if [ ! -f ${lm_model_pref}.${src} ] || [ ! -f ${lm_model_pref}.${tgt} ] || [ ! -f ${OUT_DIR}/corpora/spm.train.${src}-${tgt}.${src} ] || [ ! -f ${OUT_DIR}/corpora/spm.train.${src}-${tgt}.${tgt} ]; then
    echo "Skipped ${src}-${tgt}: No such direction"
    continue
  fi 

  if [ ! -d ${OUT_DIR}/models/smt/${src}-${tgt} ]; then
    echo "Training ${src}-${tgt}..."
    train_dir=${OUT_DIR}/models/smt/${src}-${tgt}
    mkdir -p $train_dir
    rm -rf $train_dir/*
    script=${train_dir}/train.sh
    cat << EOF > ${script}
#!/bin/bash
${train_model} -root-dir ${train_dir} \
  -external-bin-dir /private/home/jeanm/src/mgiza/mgizapp/bin \
  --mgiza --parallel --max-phrase-length 5 \
  --corpus ${OUT_DIR}/corpora/spm.train.${src}-${tgt} \
  -lm 0:5:${lm_model_pref}.${tgt} \
  --f ${src} --e ${tgt} \
  1> ${OUT_DIR}/logs/train.smt.${src}-${tgt}.out \
  2> >(tee ${OUT_DIR}/logs/train.smt.${src}-${tgt}.err)
EOF
    sbatch -J models.smt.spm.${src}-${tgt} --partition=${PARTITION} --nodes=1 --ntasks-per-node=1 \
      --gpus-per-node=0 --cpus-per-task=8 --mem=150G --time=2400 ${script}
  fi
################
  if [ ! -d ${OUT_DIR}/models/smt/${tgt}-${src} ]; then
    echo "Training ${tgt}-${src}..."
    train_dir=${OUT_DIR}/models/smt/${tgt}-${src}
    mkdir -p $train_dir
    rm -rf $train_dir/*
    script=${train_dir}/train.sh
    cat << EOF > ${script}
#!/bin/bash
$train_model -root-dir ${train_dir} \
  -external-bin-dir /private/home/jeanm/src/mgiza/mgizapp/bin \
  --mgiza --parallel --max-phrase-length 5 \
  --corpus ${OUT_DIR}/corpora/spm.train.${src}-${tgt} \
  -lm 0:5:${lm_model_pref}.${src} \
  --e ${src} --f ${tgt} \
  1> ${OUT_DIR}/logs/train.smt.${tgt}-${src}.out \
  2> >(tee ${OUT_DIR}/logs/train.smt.${tgt}-${src}.err)
EOF
    sbatch -J models.smt.spm.${tgt}-${src} --partition=${PARTITION} --nodes=1 --ntasks-per-node=1 \
      --gpus-per-node=0 --cpus-per-task=8 --mem=150G --time=2400 ${script}
  fi
done
done
