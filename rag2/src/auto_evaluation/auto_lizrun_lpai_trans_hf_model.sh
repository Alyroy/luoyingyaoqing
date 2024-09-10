#!/bin/bash

TRAIN_MODEL_DIR=$1
MODEL_CKPT_DIR=$2
HF_MODEL_CKPT_DIR=$3 
QUEUE_NAME=$4
CUR_DIR=$5

lizrun lpai start -c "bash /lpai/volumes/ssai-nlu-bd/nlu/app/gongwuxuan/_init_server_.sh; bash ${CUR_DIR}/auto_trans_hf_model.sh ${TRAIN_MODEL_DIR} ${MODEL_CKPT_DIR} ${HF_MODEL_CKPT_DIR ${CUR_DIR}}" \
-j convert-hf-model \
-i reg-ai.chehejia.com/ssai/lizr/cu118/py310/pytorch:2.1.0-multinode-flashattn-2.3.2-fsdp  \
-p ${QUEUE_NAME} \
-n 1