#!/bin/bash

CUR_DIR=$1
JOB_NAME=$2
QUEUE_NAME=$3
M_CNT=$4
EVAL_MODEL=$5
EVAL_TIMESTAMP=$6
INPUT_DIR=$7
OUTPUT_DIR=$8
EVAL_COL=$9

echo "JOB_NAME:${JOB_NAME}"
lizrun lpai start -c "bash /lpai/volumes/ssai-nlu-bd/nlu/app/gongwuxuan/_init_server_.sh; bash ${CUR_DIR}/auto_livis_moe_run_inference_api_mp.sh ${EVAL_MODEL} ${EVAL_TIMESTAMP} ${CUR_DIR} ${INPUT_DIR} ${OUTPUT_DIR} ${EVAL_COL}" \
    -j ${JOB_NAME} \
    -i reg-ai.chehejia.com/ssai/lizr/cu118/py310/pytorch:2.1.0-multinode-flashattn-2.3.2-fsdp  \
    -p ${QUEUE_NAME}  \
    -n ${M_CNT} \
    -w pytorch