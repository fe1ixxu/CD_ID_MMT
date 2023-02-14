#!/bin/bash

#SBATCH --job-name=m8_xx_eng
#SBATCH --output=./logs/sample-%j.out
#SBATCH --error=./logs/sample-%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=4320
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:8
#SBATCH --mem 128G
#SBATCH --constraint=volta32gb
#SBATCH --signal=B:USR1@180
#SBATCH --mail-user haoranxu@fb.com
#SBATCH --mail-type end
#SBATCH --partition=nllb,learnaccel

source ~/.bashrc
conda activate fairseq-model

trap_handler () {
    echo "Caught signal: " $1
    # SIGTERM must be bypassed
    if [ "$1" = "TERM" ]; then
        echo "bypass sigterm"
    else
        # Submit a new job to the queue
        echo "Requeuing " $SLURM_JOB_ID
        scontrol requeue $SLURM_JOB_ID
    fi
}


# Install signal handler
trap 'trap_handler USR1' USR1
trap 'trap_handler TERM' TERM

LANGS='fuv,kmb,lug,nya,swh,umb,wol,zul,eng'
LANG_PAIRS='fuv-eng,kmb-eng,lug-eng,nya-eng,swh-eng,umb-eng,wol-eng,zul-eng'
SRCS='fuv,kmb,lug,nya,swh,umb,wol,zul'
tgt=eng

DATA_DIR=/checkpoint/haoranxu/SSL/data/multilingual/m8_32k/
DATA_BIN=${DATA_DIR}/data_bin/shard000/

SAVE_PATH=${1}
TASKS=${2:-id_cd} # only change this one, e.g., id_mlm, mlm, dae, id_dae, cd, id_cd
TYPE=all
ID_MMT=${3:-'id_mmt'}
ALPHA=${4:-5}
MASK_RATIO=${5:-0.30}
TEMP=${6:-1}
MAX_UPDATES=${7:-100000}
APPEND=${8:-'append'}
SHARED=${9:-share}
BLOCK=${10:-block}
SUBSET=${11:--1} # number of monolingual data to use, -1 for all
REMOVE=${12:-"noremove"} # remove language from ssl

export WANDB_NAME=${SLURM_JOB_ID}_id_xx_eng_${TASKS}_${TYPE}_alpha${ALPHA}_mask${MASK_RATIO}_${ID_MMT}_temp_${TEMP}_${SHARED}_${BLOCK}_sub${SUBSET}_${REMOVE}

if [ ${SHARED} == "share" ]; then
    SHARED=''
    ARCH=transformer
    FREQ=2
    MAX_TOKENS=4096
else
    SHARED='--not-shared-proj'
    ARCH=transformer_multitask
    FREQ=2
    MAX_TOKENS=4096
fi

if [ ${BLOCK} == "block" ]; then
    BLOCK=''
else
    BLOCK='--not-token-block'
fi

if [ ${ID_MMT} == 'id_mmt' ]; then
    ID_MMT='--enable-id-mmt'
else
    ID_MMT=''
fi

if [ ${APPEND} == "append" ]; then
    APPEND='--append-lgtoken'
else
    APPEND=''
fi

mkdir -p ${SAVE_PATH}
RANDOM_PORT=155$(( $RANDOM % 50 + 1 ))
## Train  ## --mask is for dae, and --mask-prob is for mlm
 srun --job-name e_${ALPHA}_${MASK_RATIO} --output ${SAVE_PATH}/train.%j --error ${SAVE_PATH}/train.stderr.%j --mail-user haoranxu@fb.com --mail-type end \
 --nodes=4 --ntasks-per-node=1 --time=4320 --cpus-per-task=10 --gpus-per-node 8 --constraint volta32gb \
 --open-mode append --unbuffered --cpu-bind=map_ldom:0,0,0,0,1,1,1,1 \
 python train.py --distributed-world-size 32 --distributed-port ${RANDOM_PORT} ${DATA_BIN} --arch ${ARCH}  --task translation_ssl_multitask \
 --lang-pairs ${LANG_PAIRS} --langs ${LANGS} --sampling-method temperature --sampling-temperature ${TEMP} --ssl-resample 0.50 \
 --ssl-tasks ${TASKS} --nossl-langs ${REMOVE} --ssl-max-sample ${SUBSET} --beta 0.5 --id-alpha ${ALPHA} ${SHARED} ${BLOCK} ${ID_MMT} ${APPEND}\
 --ssl-data /checkpoint/haoranxu/SSL/data/monolingual/m8_32k/${TYPE} --sentencepiece-model $DATA_DIR/vocab_bin/sentencepiece.source.32000.model \
 --sample-break-mode "complete" --mask ${MASK_RATIO} --mask-random 0.1 --insert 0.0 --permute 0.1667 --poisson-lambda 3.0 --replace-length 1 --rotate 0 \
 --mask-whole-words --multilang-sampling-alpha 0.7 --mask-prob ${MASK_RATIO} --leave-unmasked-prob 0.1 --random-token-prob 0.1 --bpe sentencepiece \
 --encoder-layers 6 --decoder-layers 6 --encoder-ffn-embed-dim 4096 --decoder-ffn-embed-dim 4096 \
 --encoder-embed-dim 1024 --decoder-embed-dim 1024 --encoder-attention-heads 16 --decoder-attention-heads 16 --attention-dropout 0.1 --relu-dropout 0.0 \
 --decoder-normalize-before --encoder-normalize-before --share-all-embeddings --max-source-positions 512 --max-target-positions 512 \
 --max-update ${MAX_UPDATES} --update-freq ${FREQ}  --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' --lr-scheduler inverse_sqrt \
 --warmup-init-lr 1e-07 --warmup-updates 8000 --lr 0.001 --stop-min-lr 1e-09 --clip-norm 0.0 --dropout 0.3 --weight-decay 0.0 --criterion label_smoothed_cross_entropy \
 --label-smoothing 0.1 --best-checkpoint-metric loss --max-tokens ${MAX_TOKENS}  --validate-interval-updates 500 --save-interval-updates 500 --save-interval 2 \
 --keep-interval-updates 1  --validate-interval 500  --seed 42 --log-format simple --log-interval 100 \
 --fp16 --optimizer adam --min-params-to-wrap 100000000  --use-local-shard-size \
 --save-dir ${SAVE_PATH}  --skip-invalid-size-inputs-valid-test --memory-efficient-fp16  --wandb-project M8_32K --ddp-backend fully_sharded


# # Evaluate
mkdir -p ${SAVE_PATH}/results
for src in ${SRCS//,/ }; do
    echo predict $src to $tgt
    FSRC=${DATA_DIR}/retrieved_data/test.${tgt}-${src}.${src}
    FTGT=${DATA_DIR}/retrieved_data/test.${tgt}-${src}.${tgt}
    FOUT=${SAVE_PATH}/results/predict.${tgt}-${src}.${tgt}


    cat ${FSRC} | \
    python interactive.py ${DATA_BIN} --path $SAVE_PATH/checkpoint_best-shard0.pt \
        --langs ${LANGS} \
        --lang-pairs ${LANG_PAIRS} \
        --task translation_ssl_multitask ${APPEND} \
        --bpe "sentencepiece" \
        --ssl-data /checkpoint/haoranxu/SSL/data/monolingual/m8_32k/${TYPE} \
        --sentencepiece-model ${DATA_DIR}/vocab_bin/sentencepiece.source.32000.model \
        --source-lang ${src} --target-lang ${tgt} \
        --buffer-size 1024 --batch-size 100 \
        --no-progress-bar |\
    grep -P "^D-" | cut -f 3- > $FOUT > ${FOUT}

    # SACREBLEU=/private/home/jeanm/src/sacrebleu/sacrebleu/sacrebleu.py
    SACREBLEU_FORMAT=text sacrebleu -tok spm --width 2 $FTGT \
        < $FOUT \
        > $FOUT.bleu

    sacrebleu -m chrf --chrf-word-order 2 -tok spm --width 2 ${FTGT} < ${FOUT} > ${FOUT}.chrf 
done

# ## Print
for src in ${SRCS//,/ }; do
    FOUT=${SAVE_PATH}/results/predict.${tgt}-${src}.${tgt}
    echo -------------------
    echo ${src}-${tgt}
    cat $FOUT.bleu | cut -d ' ' -f 3
    cat $FOUT.chrf | grep score
done

python /private/home/haoranxu/nllb/inference/get_mean_score.py \
    --input ${SAVE_PATH}/results/ \
    --engtgt 1
