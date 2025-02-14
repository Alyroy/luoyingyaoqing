#!/bin/bash

# 全局参数设置
CURRENT_DIR=/mnt/pfs-guan-ssai/nlu/renhuimin/lisft_qwen/lisft/examples/Qwen_72b_FSDP/Qwen72b_convet_model/
cd $CURRENT_DIR

MODEL_CKPT_DIR=$1
JOB_NAME=$2
QUEUE_NAME=$3

bash lizrun_run_convert_qwen72b.sh ${MODEL_CKPT_DIR} ${JOB_NAME} ${QUEUE_NAME}