TRAIN_CONFIGS='configs/v1_hrnet_3dpw_ft.yml'
#cat $TRAIN_CONFIGS
GPUS=$(cat $TRAIN_CONFIGS | shyaml get-value ARGS.gpu)
#cat $GPUS
DATASET=$(cat $TRAIN_CONFIGS | shyaml get-value ARGS.dataset)
#cat $DATASET
TAB=$(cat $TRAIN_CONFIGS | shyaml get-value ARGS.tab)
#cat $TAB

CUDA_VISIBLE_DEVICES=${GPUS} python -u -m romp.train --configs_yml=${TRAIN_CONFIGS}
#CUDA_VISIBLE_DEVICES=${GPUS} nohup python -u -m romp.train --configs_yml=${TRAIN_CONFIGS} > 'log/'${TAB}'_'${DATASET}'_g'${GPUS}.log 2>&1 &