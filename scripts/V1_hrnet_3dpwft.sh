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


CUDA_VISIBLE_DEVICES='0' python -u romp/train.py --configs_yml='configs/v1_hrnet_3dpw_ft.yml'
CUDA_VISIBLE_DEVICES='0' python -u romp/train.py --configs_yml='configs/check_v1_hrnet_3dpw_ft.yml'

# CUDA_VISIBLE_DEVICES='1'
python -u -m romp.train --configs_yml=${TRAIN_CONFIGS}
# CUDA_VISIBLE_DEVICES=${GPUS} nohup python -u -m romp.train --configs_yml=${TRAIN_CONFIGS} > 'log/'${TAB}'_'${DATASET}'_g'${GPUS}.log 2>&1 &

#     os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

CUDA_VISIBLE_DEVICES='1' python -u  romp/train.py >> /data2/2020/ssw/romp/log/hrnet_cm64_V1_hrnet_pw3d_ft/hrnet_cm64_V1_hrnet_pw3d_ft.log 2>&1
