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
temperature=${10}
top_p=${11}
repetition=${12}

echo "JOB_NAME:${JOB_NAME}"
lizrun start -c "bash ${CUR_DIR}/auto_livis_moe_run_inference_api_mp.sh ${EVAL_MODEL} ${EVAL_TIMESTAMP} ${CUR_DIR} ${INPUT_DIR} ${OUTPUT_DIR} ${EVAL_COL} ${temperature} ${top_p} ${repetition}" \
    -j ${JOB_NAME} \
    -i reg-ai.chehejia.com/ssai/lizr/cu121/py310/pytorch:2.4.0-multinode-flashattn2.6.3-vllm0.6.2-liktoken1.0.6  \
    -p ${QUEUE_NAME}  \
    -n ${M_CNT} \
    -w pytorch