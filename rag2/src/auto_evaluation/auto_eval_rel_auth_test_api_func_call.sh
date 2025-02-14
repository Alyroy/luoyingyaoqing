#!/bin/bash

CURRENT_DIR=$1
INPUT_LOG_COL=$2 #模型日志输入
INPUT_PREDICT_COL=$3 #模型预测结果
EVAL_INPUT_DIR=$4
EVAL_OUTPUT_DIR_REL=$5 # 相关性输出文件夹
EVAL_OUTPUT_DIR_AUTH=$6 # 真实性输出文件夹
IP=$7

cd ${CURRENT_DIR}
bash auto_eval_log_relevance_qwen_api_func_call.sh $CURRENT_DIR $INPUT_LOG_COL $INPUT_PREDICT_COL $EVAL_INPUT_DIR $EVAL_OUTPUT_DIR_REL $IP &
bash auto_eval_log_authenticity_qwen_api_func_call.sh $CURRENT_DIR $INPUT_LOG_COL $INPUT_PREDICT_COL $EVAL_INPUT_DIR $EVAL_OUTPUT_DIR_AUTH $IP
