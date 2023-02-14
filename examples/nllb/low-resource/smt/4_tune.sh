#!/bin/bash

MOSES_DIR=/private/home/jeanm/src/mosesdecoder
# OUT_DIR=/checkpoint/haoranxu/SSL/data/moses_public+mined.ssl.v1/
#/checkpoint/jeanm/nllb/moses_public+mined.ssl.v1

tune_model=${MOSES_DIR}/scripts/training/mert-moses.pl
moses=${MOSES_DIR}/bin/moses

# LANGS=${1:-wol,eng}
#${1:-amh,cjk,eng,ewe,fon,fuv,hau,ibo,kam,kik,kin,kmb,kon,lin,lug,luo,nso,nya,orm,run,som,ssw,swh,tir,tsn,tso,twi,umb,wol,xho,zul}
# PARTITION=learnfair

OUT_DIR=${1}
LANGS=${2}
PARTITION=${3}

for src in ${LANGS//,/ }; do
  for tgt in ${LANGS//,/ }; do
  train_dir=${OUT_DIR}/models/smt/${src}-${tgt}
  tune_dir=${train_dir}/tuning
  ini_path=${train_dir}/model/moses.ini
  if [ ! -f ${ini_path} ]; then
    echo "Skipping ${src}-${tgt} as the base model is missing..."
    continue
  fi
  if [ ! -f ${tune_dir}/mert/moses.ini ]; then
    mkdir -p ${tune_dir}
    rm -rf ${tune_dir}/*
    script=${train_dir}/tune.sh
  if [ -f ${OUT_DIR}/corpora/spm.valid.${src}-${tgt}.${tgt} ]; then
    echo "Tuning ${src}-${tgt}..."
    cat << EOF > ${script}
#!/bin/bash
${tune_model} \
  --decoder-flags="-threads all" \
  --mertargs="--sctype BLEU --scconfig weights:0.6+0.4 --threads all" \
  --working-dir ${tune_dir}/mert \
  --input ${OUT_DIR}/corpora/spm.valid.${src}-${tgt}.${tgt} \
  --refs ${OUT_DIR}/corpora/spm.valid.${src}-${tgt}.${src} \
  --decoder ${moses} \
  --config ${ini_path} \
  > ${tune_dir}/mert.output
EOF
  sbatch -J tune.smt.spm.${src}-${tgt} --partition=${PARTITION} --nodes=1 --ntasks-per-node=1 \
    --gpus-per-node=0 --cpus-per-task=8 --mem=150G --time=2400 ${script}
  fi

  if [ -f ${OUT_DIR}/corpora/spm.valid.${tgt}-${src}.${tgt} ]; then
    echo "Tuning ${src}-${tgt}..."
    cat << EOF > ${script}
#!/bin/bash
${tune_model} \
  --decoder-flags="-threads all" \
  --mertargs="--sctype BLEU --scconfig weights:0.6+0.4 --threads all" \
  --working-dir ${tune_dir}/mert \
  --input ${OUT_DIR}/corpora/spm.valid.${tgt}-${src}.${tgt} \
  --refs ${OUT_DIR}/corpora/spm.valid.${tgt}-${src}.${src} \
  --decoder ${moses} \
  --config ${ini_path} \
  > ${tune_dir}/mert.output
EOF
  sbatch -J tune.smt.spm.${src}-${tgt} --partition=${PARTITION} --nodes=1 --ntasks-per-node=1 \
    --gpus-per-node=0 --cpus-per-task=8 --mem=150G --time=2400 ${script}
  fi
fi


  #######

  # train_dir=${OUT_DIR}/models/smt/${tgt}-${src}
#   tune_dir=${train_dir}/tuning
#   ini_path=${train_dir}/model/moses.ini
#   if [ ! -f ${ini_path} ]; then
#     echo "Skipping ${tgt}-${src} as the base model is missing..."
#     continue
#   fi
#   if [ ! -f ${tune_dir}/mert/moses.ini ]; then
#     echo "Tuning ${src}-${tgt}..."
#     mkdir -p ${tune_dir}
#     rm -rf ${tune_dir}/*
#     script=${train_dir}/tune.sh
#     cat << EOF > ${script}
# #!/bin/bash
# ${tune_model} \
#   --decoder-flags="-threads all" \
#   --mertargs="--sctype BLEU --scconfig weights:0.6+0.4 --threads all" \
#   --working-dir ${tune_dir}/mert \
#   --input ${OUT_DIR}/corpora/spm.valid.eng-${lang}.eng \
#   --refs ${OUT_DIR}/corpora/spm.valid.eng-${lang}.${lang} \
#   --decoder ${moses} \
#   --config ${ini_path} \
#   > ${tune_dir}/mert.output
# EOF
#   sbatch -J tune.smt.spm.eng-${lang} --partition=${PARTITION} --nodes=1 --ntasks-per-node=1 \
#     --gpus-per-node=0 --cpus-per-task=8 --mem=150G --time=2400 ${script}
#   fi
done
done
