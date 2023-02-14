LANGS=fas,zho_Hans,rus,kor,ara_Arab
OUT_MULTI=/checkpoint/haoranxu/SSL/data/multilingual/m6_64k
OUT_MONO=/checkpoint/haoranxu/SSL/data/monolingual/m6_64k
MULTILINGUAL_DIR=/large_experiments/nllb/mmt/multilingual_bin/flores200.en_xx_en.v4.1.256k/
MONOLINGUAL_DIR=/large_experiments/nllb/mmt/multilingual_bin/flores200.monolingual.v2.256k/retrieved_data
src=eng

source activate fairseq-moe

# mkdir -p ${OUT_MULTI}/retrieved_data
# for lang in ${LANGS//,/ }; do
#     for split in train valid test; do
#         if [ -f ${MULTILINGUAL_DIR}/retrieved_data/${split}.eng-${lang}.eng ]; then
#             pair=eng-${lang}
#         else
#             pair=${lang}-eng
#         fi
#         cp ${MULTILINGUAL_DIR}/retrieved_data/${split}.${pair}.eng ${OUT_MULTI}/retrieved_data/${split}.eng-${lang}.eng
#         cp ${MULTILINGUAL_DIR}/retrieved_data/${split}.${pair}.${lang} ${OUT_MULTI}/retrieved_data/${split}.eng-${lang}.${lang}
#     done
# done

# mkdir -p ${OUT_MULTI}/vocab_bin
# python /private/home/haoranxu/fairseq-py/scripts/spm_train.py \
#     --input=$(echo $(ls ${OUT_MULTI}/retrieved_data/*) | sed 's/ /,/g') \
#     --model_prefix=${OUT_MULTI}/vocab_bin/sentencepiece.source.64000 --vocab_size=64000 --character_coverage=0.9999999995 \
#     --input_sentence_size=1000000

# mkdir -p ${OUT_MULTI}/tok
# cut -f 1 ${OUT_MULTI}/vocab_bin/sentencepiece.source.64000.vocab | tail -n +4 | sed "s/$/ 100/g" > ${OUT_MULTI}/dict.txt
 
for lang in ${LANGS//,/ }; do
    pair=${src}-${lang}
    for split in 'train' 'valid' 'test'; do
        python scripts/spm_encode.py \
        --model ${OUT_MULTI}/vocab_bin/sentencepiece.source.64000.model \
        --input ${OUT_MULTI}/retrieved_data/${split}.${pair}.${src} \
        --outputs ${OUT_MULTI}/tok/${split}.${pair}.${src}

        python scripts/spm_encode.py \
        --model ${OUT_MULTI}/vocab_bin/sentencepiece.source.64000.model \
        --input ${OUT_MULTI}/retrieved_data/${split}.${pair}.${lang} \
        --outputs ${OUT_MULTI}/tok/${split}.${pair}.${lang}
    done

    fairseq-preprocess --task "translation" --source-lang ${src} --target-lang ${lang} \
    --trainpref ${OUT_MULTI}/tok/train.${pair} --validpref ${OUT_MULTI}/tok/valid.${pair}  --testpref ${OUT_MULTI}/tok/test.${pair}  \
    --destdir  ${OUT_MULTI}/data_bin/shard000 --dataset-impl 'mmap' --padding-factor 1 --workers 32 \
    --srcdict ${OUT_MULTI}/dict.txt --tgtdict ${OUT_MULTI}/dict.txt

done

## All monolingual 64K
# LANGS=${LANGS},eng
# mkdir -p ${OUT_MONO}/all
# mkdir -p ${OUT_MONO}/tok
# mkdir -p ${OUT_MONO}/retrieved_data
# mkdir -p ${OUT_MONO}/all_tok

# cut -f 1 ${OUT_MULTI}/vocab_bin/sentencepiece.source.32000.vocab | tail -n +4 | sed "s/$/ 100/g" > ${OUT_MONO}/dict.txt
# for lang in ${LANGS//,/ }; do
#     ## train
#     cat ${MONOLINGUAL_DIR}/train.*.${lang} | shuf -n 15000000 > ${OUT_MONO}/retrieved_data/train.${lang}
#     python scripts/spm_encode.py \
#     --model ${OUT_MULTI}/vocab_bin/sentencepiece.source.32000.model \
#     --input ${OUT_MONO}/retrieved_data/train.${lang} \
#     --outputs ${OUT_MONO}/tok/train.${lang}
#     cat ${OUT_MULTI}/tok/train.*.${lang} ${OUT_MONO}/tok/train.${lang} | shuf > ${OUT_MONO}/all_tok/train.${lang}
#     cat ${OUT_MULTI}/tok/valid.*.${lang}  > ${OUT_MONO}/all_tok/valid.${lang}

#     fairseq-preprocess --only-source --srcdict ${OUT_MONO}/dict.txt \
#     --trainpref ${OUT_MONO}/all_tok/train.${lang} --validpref ${OUT_MONO}/all_tok/valid.${lang} \
#     --destdir ${OUT_MONO}/all/${lang}  --workers 60

# done


## Move all to mixed-all

# mkdir -p ${OUT_MONO}/mixed-all/shard000
# cp ${OUT_MONO}/dict.txt ${OUT_MONO}/mixed-all/shard000
# LANGS=${LANGS},eng
# for lang in ${LANGS//,/ }; do
#     for split in 'train' 'valid'; do
#         cp ${OUT_MONO}/all/${lang}/${split}.bin ${OUT_MONO}/mixed-all/shard000/${split}.${lang}-${lang}.${lang}.bin
#         cp ${OUT_MONO}/all/${lang}/${split}.idx ${OUT_MONO}/mixed-all/shard000/${split}.${lang}-${lang}.${lang}.idx
#     done
# done