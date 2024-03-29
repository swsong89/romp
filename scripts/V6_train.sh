TRAIN_CONFIGS='configs/v6_train.yml'

GPUS=$(cat $TRAIN_CONFIGS | shyaml get-value ARGS.gpu)
DATASET=$(cat $TRAIN_CONFIGS | shyaml get-value ARGS.dataset)
TAB=$(cat $TRAIN_CONFIGS | shyaml get-value ARGS.tab)

# CUDA_VISIBLE_DEVICES=${GPUS} python -u -m romp.train --configs_yml=${TRAIN_CONFIGS}
CUDA_VISIBLE_DEVICES='1' python -u -m romp.train --configs_yml=${TRAIN_CONFIGS}
CUDA_VISIBLE_DEVICES='2' python -u  romp/train.py --configs_yml='configs/v6_train.yml'
#CUDA_VISIBLE_DEVICES=${GPUS} nohup python -u -m romp.train --configs_yml=${TRAIN_CONFIGS} > 'log/'${TAB}'_'${DATASET}'_g'${GPUS}.log 2>&1 &