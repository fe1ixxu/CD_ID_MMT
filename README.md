This is the repo for the intern project [Language-Aware Multilingual Machine Translation with Self-Supervised Learning](https://fb.workplace.com/groups/831302610278251/permalink/8170600219681750)

## Building Virtual Environment:
```
conda create -n fairseq-model python=3.8
source activate fairseq-model
pip install -e ./
pip install -r requirements.txt
```

## Training

Train a model with CD+ID on M8 xxx->eng:
```
bash runs/run_m8_xx_eng.sh ${SAVE_DIR}
```

`${SAVE_DIR}` is the place where you save the model. Dataset is located in the `${DATA_DIR}` in the file. The default method is CD+ID. If you want to run other SSL method such as DAE without ID, you can run:

```
bash runs/run_m8_xx_eng.sh ${SAVE_DIR} dae no_id_mmt
```

The SSL methods only supports for the following options:
* `dae`: naive DAE
* `id_dae`: DAE with intra-distillation
* `mlm`: naive MLM
* `id_mlm`: MLM with intra-distillation
* `cd` : Concurrent Denoising
* `id_cd`: CD with intra-distillation

To default settings will enable intra-distillation on MMT, to disable it, just append `no_id_mmt` after the task option (as the example shown above)

Similar commands also holds for other directions and M15 dataset.
