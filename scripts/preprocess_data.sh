LANGS=fuv,kmb,lug,nya,swh,umb,wol,zul
OUT_MULTI=/checkpoint/haoranxu/SSL/data/multilingual/m8_32k
OUT_MONO=/checkpoint/haoranxu/SSL/data/monolingual/m8_32k_raw
MULTILINGUAL_DIR=/large_experiments/nllb/mmt/multilingual_bin/flores200.en_xx_en.v4.1.256k/
MONOLINGUAL_DIR=/large_experiments/nllb/mmt/multilingual_bin/flores200.monolingual.v2.256k/retrieved_data
src=eng

source activate fairseq-mbart

# Train sentencepiece
# mkdir -p ${OUT_MULTI}/vocab_bin
# python /private/home/haoranxu/fairseq-py/scripts/spm_train.py \
#     --input=$(echo $(ls ${OUT_MULTI}/retrieved_data/*) | sed 's/ /,/g') \
#     --model_prefix=${OUT_MULTI}/vocab_bin/sentencepiece.source.32000 --vocab_size=32000 --character_coverage=0.9999999995 \
#     --input_sentence_size=2000000


## Tokenize by 256000 vocab
# mkdir -p ${OUT_MULTI}/tok
# cut -f 1 ${MULTILINGUAL_DIR}/vocab_bin/sentencepiece.source.256000.vocab | tail -n +4 | sed "s/$/ 100/g" > ${OUT_MULTI}/dict.txt
 
# for lang in ${LANGS//,/ }; do
#     if [ -f ${MULTILINGUAL_DIR}/retrieved_data/train.${src}-${lang}.${src} ]; then
#         pair=${src}-${lang}
#     else
#         pair=${lang}-${src}
#     fi
#     for split in 'train' 'valid' 'test'; do
        # python scripts/spm_encode.py \
        # --model ${MULTILINGUAL_DIR}/vocab_bin/sentencepiece.source.256000.model \
        # --input ${MULTILINGUAL_DIR}/retrieved_data/${split}.${pair}.${src} \
        # --outputs ${OUT_MULTI}/tok/${split}.${pair}.${src}

#         python scripts/spm_encode.py \
#         --model ${MULTILINGUAL_DIR}/vocab_bin/sentencepiece.source.256000.model \
#         --input ${MULTILINGUAL_DIR}/retrieved_data/${split}.${pair}.${lang} \
#         --outputs ${OUT_MULTI}/tok/${split}.${pair}.${lang}
#     done

#     fairseq-preprocess --task "translation" --source-lang ${src} --target-lang ${lang} \
#     --trainpref ${OUT_MULTI}/tok/train.${pair} --validpref ${OUT_MULTI}/tok/valid.${pair}  --testpref ${OUT_MULTI}/tok/test.${pair}  \
#     --destdir  ${OUT_MULTI}/data_bin/shard000 --dataset-impl 'mmap' --padding-factor 1 --workers 32 \
#     --srcdict ${OUT_MULTI}/dict.txt --tgtdict ${OUT_MULTI}/dict.txt


# done

## Tokenize by 32000 vocab
# mkdir -p ${OUT_MULTI}/tok
# cut -f 1 ${OUT_MULTI}/vocab_bin/sentencepiece.source.32000.vocab | tail -n +4 | sed "s/$/ 100/g" > ${OUT_MULTI}/dict.txt
 
# for lang in ${LANGS//,/ }; do
#     if [ -f ${MULTILINGUAL_DIR}/retrieved_data/train.${src}-${lang}.${src} ]; then
#         pair=${src}-${lang}
#     else
#         pair=${lang}-${src}
#     fi
#     for split in 'train' 'valid' 'test'; do
#         python scripts/spm_encode.py \
#         --model ${OUT_MULTI}/vocab_bin/sentencepiece.source.32000.model \
#         --input ${MULTILINGUAL_DIR}/retrieved_data/${split}.${pair}.${src} \
#         --outputs ${OUT_MULTI}/tok/${split}.${pair}.${src}

#         python scripts/spm_encode.py \
#         --model ${OUT_MULTI}/vocab_bin/sentencepiece.source.32000.model \
#         --input ${MULTILINGUAL_DIR}/retrieved_data/${split}.${pair}.${lang} \
#         --outputs ${OUT_MULTI}/tok/${split}.${pair}.${lang}
#     done

#     fairseq-preprocess --task "translation" --source-lang ${src} --target-lang ${lang} \
#     --trainpref ${OUT_MULTI}/tok/train.${pair} --validpref ${OUT_MULTI}/tok/valid.${pair}  --testpref ${OUT_MULTI}/tok/test.${pair}  \
#     --destdir  ${OUT_MULTI}/data_bin/shard000 --dataset-impl 'mmap' --padding-factor 1 --workers 32 \
#     --srcdict ${OUT_MULTI}/dict.txt --tgtdict ${OUT_MULTI}/dict.txt


# done


## bitext monolingual
# LANGS=${LANGS},eng
# mkdir -p ${OUT_MONO}/bitext

# cp ${OUT_MULTI}/dict.txt ${OUT_MONO}/bitext
# for lang in ${LANGS//,/ }; do
#     mkdir -p ${OUT_MONO}/bitext/tmp
#     for split in 'train' 'valid'; do
#         cat ${OUT_MULTI}/tok/${split}.*.${lang} > ${OUT_MONO}/bitext/tmp/${split}.${lang}
#     done

#     fairseq-preprocess --only-source --srcdict ${OUT_MONO}/bitext/dict.txt \
#     --trainpref ${OUT_MONO}/bitext/tmp/train.${lang} --validpref ${OUT_MONO}/bitext/tmp/valid.${lang} \
#     --destdir ${OUT_MONO}/bitext/${lang}  --workers 60

#     rm -rf ${OUT_MONO}/bitext/tmp

# done



## Move bitext to mixed 
# LANGS=${LANGS},eng
# mkdir -p ${OUT_MONO}/mixed-bitext/shard000/
# for lang in ${LANGS//,/ }; do
#     for split in 'train' 'valid'; do
#         cp ${OUT_MONO}/bitext/${lang}/${split}.bin ${OUT_MONO}/mixed-bitext/shard000/${split}.${lang}-${lang}.${lang}.bin
#         cp ${OUT_MONO}/bitext/${lang}/${split}.idx ${OUT_MONO}/mixed-bitext/shard000/${split}.${lang}-${lang}.${lang}.idx
#     done
# done

# Move retrived data:
# mkdir -p ${OUT_MULTI}/retrieved_data/
# for lang in ${LANGS//,/ }; do
#     cp ${MULTILINGUAL_DIR}/retrieved_data/train.eng-${lang}.eng ${OUT_MULTI}/retrieved_data
#     cp ${MULTILINGUAL_DIR}/retrieved_data/train.eng-${lang}.${lang} ${OUT_MULTI}/retrieved_data
#     cp ${MULTILINGUAL_DIR}/retrieved_data/valid.eng-${lang}.eng ${OUT_MULTI}/retrieved_data
#     cp ${MULTILINGUAL_DIR}/retrieved_data/valid.eng-${lang}.${lang} ${OUT_MULTI}/retrieved_data
#     cp ${MULTILINGUAL_DIR}/retrieved_data/test.eng-${lang}.eng ${OUT_MULTI}/retrieved_data
#     cp ${MULTILINGUAL_DIR}/retrieved_data/test.eng-${lang}.${lang} ${OUT_MULTI}/retrieved_data
# done


## All monolingual
# LANGS=${LANGS},eng
# mkdir -p ${OUT_MONO}/all

# cut -f 1 ${MULTILINGUAL_DIR}/vocab_bin/sentencepiece.source.256000.vocab | tail -n +4 | sed "s/$/ 100/g" > ${OUT_MONO}/all/dict.txt
# for lang in ${LANGS//,/ }; do
#     mkdir -p ${OUT_MONO}/all/tmp

#     ## train
#     cat ${MONOLINGUAL_DIR}/train.*.${lang} | shuf -n 3000000 > ${OUT_MONO}/all/tmp/train.mono.${lang}
#     python scripts/spm_encode.py \
#     --model ${MULTILINGUAL_DIR}/vocab_bin/sentencepiece.source.256000.model \
#     --input ${OUT_MONO}/all/tmp/train.mono.${lang}\
#     --outputs ${OUT_MONO}/all/tmp/train.mono.tok.${lang}
#     cat ${OUT_MULTI}/tok/train.*.${lang} ${OUT_MONO}/all/tmp/train.mono.tok.${lang} | shuf > ${OUT_MONO}/all/tmp/train.${lang}
#     cat ${OUT_MULTI}/tok/valid.*.${lang}  > ${OUT_MONO}/all/tmp/valid.${lang}

#     fairseq-preprocess --only-source --srcdict ${OUT_MONO}/all/dict.txt \
#     --trainpref ${OUT_MONO}/all/tmp/train.${lang} --validpref ${OUT_MONO}/all/tmp/valid.${lang} \
#     --destdir ${OUT_MONO}/all/${lang}  --workers 60

#     rm -rf ${OUT_MONO}/all/tmp

# done

## All monolingual 32K
# LANGS=${LANGS},eng
# mkdir -p ${OUT_MONO}/all

# cut -f 1 ${OUT_MULTI}/vocab_bin/sentencepiece.source.32000.vocab | tail -n +4 | sed "s/$/ 100/g" > ${OUT_MONO}/all/dict.txt
# for lang in ${LANGS//,/ }; do
#     mkdir -p ${OUT_MONO}/all/tmp

#     ## train
#     cat ${MONOLINGUAL_DIR}/train.*.${lang} | shuf -n 3000000 > ${OUT_MONO}/all/tmp/train.mono.${lang}
#     python scripts/spm_encode.py \
#     --model ${OUT_MULTI}/vocab_bin/sentencepiece.source.32000.model \
#     --input ${OUT_MONO}/all/tmp/train.mono.${lang}\
#     --outputs ${OUT_MONO}/all/tmp/train.mono.tok.${lang}
#     cat ${OUT_MULTI}/tok/train.*.${lang} ${OUT_MONO}/all/tmp/train.mono.tok.${lang} | shuf > ${OUT_MONO}/all/tmp/train.${lang}
#     cat ${OUT_MULTI}/tok/valid.*.${lang}  > ${OUT_MONO}/all/tmp/valid.${lang}

#     fairseq-preprocess --only-source --srcdict ${OUT_MONO}/all/dict.txt \
#     --trainpref ${OUT_MONO}/all/tmp/train.${lang} --validpref ${OUT_MONO}/all/tmp/valid.${lang} \
#     --destdir ${OUT_MONO}/all/${lang}  --workers 60

#     rm -rf ${OUT_MONO}/all/tmp

# done



## Move all to mixed-all

# mkdir -p ${OUT_MONO}/mixed-all/shard000
# cp ${OUT_MONO}/all/dict.txt ${OUT_MONO}/mixed-all/shard000
# LANGS=${LANGS},eng
# for lang in ${LANGS//,/ }; do
#     for split in 'train' 'valid'; do
#         cp ${OUT_MONO}/all/${lang}/${split}.bin ${OUT_MONO}/mixed-all/shard000/${split}.${lang}-${lang}.${lang}.bin
#         cp ${OUT_MONO}/all/${lang}/${split}.idx ${OUT_MONO}/mixed-all/shard000/${split}.${lang}-${lang}.${lang}.idx
#     done
# done



### Phrase replacement
# LANGS=fuv-eng,kmb-eng,lug-eng,nya-eng,swh-eng,umb-eng,wol-eng,zul-eng,eng-fuv,eng-kmb,eng-lug,eng-nya,eng-swh,eng-umb,eng-wol,eng-zul
# for lg in ${LANGS//,/ }; do
#     python examples/nllb/modeling/prepare_data/phrase_table_clean.py --input /checkpoint/haoranxu/SSL/data/phrase_tables/m8_32k/${lg} --output /checkpoint/haoranxu/SSL/data/phrase_tables/m8_32k/${lg}.tb
# done

# LANGS='fuv,kmb,lug,nya,umb,wol'

# cp  /checkpoint/haoranxu/SSL/data/multilingual/m8_32k/dict.txt /checkpoint/haoranxu/SSL/data/multilingual/m8_32k_pr/
# cp  -r /checkpoint/haoranxu/SSL/data/multilingual/m8_32k/vocab_bin /checkpoint/haoranxu/SSL/data/multilingual/m8_32k_pr/
# cp /checkpoint/haoranxu/SSL/data/multilingual/m8_32k_pr/tok/valid* /checkpoint/haoranxu/SSL/data/multilingual/m8_32k_pr/tok/
# cp /checkpoint/haoranxu/SSL/data/multilingual/m8_32k_pr/tok/test* /checkpoint/haoranxu/SSL/data/multilingual/m8_32k_pr/tok/
# for tgt in ${LANGS//,/ }; do
# python examples/nllb/modeling/prepare_data/phrase_replacement.py --input /checkpoint/haoranxu/SSL/data/monolingual/m8_32k_raw/tok/${tgt}.txt --output /checkpoint/haoranxu/SSL/data/multilingual/m8_32k_pr/tok/tmp.eng-${tgt}.eng --table_file /checkpoint/haoranxu/SSL/data/phrase_tables/m8_32k/${tgt}-eng.tb
# # cp /checkpoint/haoranxu/SSL/data/monolingual/m8_32k_raw/tok/${tgt}.txt /checkpoint/haoranxu/SSL/data/multilingual/m8_32k_pr/tok/tmp.eng-${tgt}.${tgt}
# cat /checkpoint/haoranxu/SSL/data/multilingual/m8_32k/tok/train.eng-${tgt}.eng /checkpoint/haoranxu/SSL/data/multilingual/m8_32k_pr/tok/tmp.eng-${tgt}.eng > /checkpoint/haoranxu/SSL/data/multilingual/m8_32k_pr/tok/train.eng-${tgt}.eng
# cat /checkpoint/haoranxu/SSL/data/multilingual/m8_32k/tok/train.eng-${tgt}.${tgt}  /checkpoint/haoranxu/SSL/data/monolingual/m8_32k_raw/tok/${tgt}.txt >  /checkpoint/haoranxu/SSL/data/multilingual/m8_32k_pr/tok/train.eng-${tgt}.${tgt}

# python examples/nllb/modeling/prepare_data/phrase_replacement.py --input /checkpoint/haoranxu/SSL/data/monolingual/m8_32k_raw/tok/eng.txt --output /checkpoint/haoranxu/SSL/data/multilingual/m8_32k_pr/tok/tmp.${tgt}-eng.${tgt} --table_file /checkpoint/haoranxu/SSL/data/phrase_tables/m8_32k/eng-${tgt}.tb
# # cp /checkpoint/haoranxu/SSL/data/monolingual/m8_32k_raw/tok/eng.txt /checkpoint/haoranxu/SSL/data/multilingual/m8_32k_pr/tok/tmp.${tgt}-eng.eng
# cat /checkpoint/haoranxu/SSL/data/multilingual/m8_32k/tok/train.eng-${tgt}.${tgt} /checkpoint/haoranxu/SSL/data/multilingual/m8_32k_pr/tok/tmp.${tgt}-eng.${tgt} > /checkpoint/haoranxu/SSL/data/multilingual/m8_32k_pr/tok/train.${tgt}-eng.${tgt}
# cat /checkpoint/haoranxu/SSL/data/multilingual/m8_32k/tok/train.eng-${tgt}.eng /checkpoint/haoranxu/SSL/data/monolingual/m8_32k_raw/tok/eng.txt > /checkpoint/haoranxu/SSL/data/multilingual/m8_32k_pr/tok/train.${tgt}-eng.eng
# rm /checkpoint/haoranxu/SSL/data/multilingual/m8_32k_pr/tok/tmp*


# python preprocess.py --task translation_ssl_multitask3 --source-lang eng --target-lang ${tgt} \
# --trainpref /checkpoint/haoranxu/SSL/data/multilingual/m8_32k_pr/tok/train.eng-${tgt} \
# --validpref /checkpoint/haoranxu/SSL/data/multilingual/m8_32k_pr/tok/valid.eng-${tgt} \
# --testpref /checkpoint/haoranxu/SSL/data/multilingual/m8_32k_pr/tok/test.eng-${tgt}  \
# --destdir   /checkpoint/haoranxu/SSL/data/multilingual/m8_32k_pr/eng-xx/ --dataset-impl 'mmap' --padding-factor 1 --workers 32 \
# --srcdict  /checkpoint/haoranxu/SSL/data/multilingual/m8_32k_pr/dict.txt --tgtdict  /checkpoint/haoranxu/SSL/data/multilingual/m8_32k_pr/dict.txt

# python preprocess.py --task translation_ssl_multitask3 --source-lang ${tgt} --target-lang eng \
# --trainpref /checkpoint/haoranxu/SSL/data/multilingual/m8_32k_pr/tok/train.${tgt}-eng \
# --validpref /checkpoint/haoranxu/SSL/data/multilingual/m8_32k_pr/tok/valid.eng-${tgt} \
# --testpref /checkpoint/haoranxu/SSL/data/multilingual/m8_32k_pr/tok/test.eng-${tgt}  \
# --destdir   /checkpoint/haoranxu/SSL/data/multilingual/m8_32k_pr/xx-eng/ --dataset-impl 'mmap' --padding-factor 1 --workers 32 \
# --srcdict  /checkpoint/haoranxu/SSL/data/multilingual/m8_32k_pr/dict.txt --tgtdict  /checkpoint/haoranxu/SSL/data/multilingual/m8_32k_pr/dict.txt

# done