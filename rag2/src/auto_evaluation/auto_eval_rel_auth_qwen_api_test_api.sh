#!/bin/bash

CURRENT_DIR=$1
INPUT_LOG_COL=$2 #模型日志输入
EVAL_INPUT_DIR=$3
EVAL_OUTPUT_DIR_REL=$4 # 相关性输出文件夹
EVAL_OUTPUT_DIR_AUTH=$5 # 真实性输出文件夹
IP=$6

bash auto_eval_log_relevance_qwen_api_test_api.sh $CURRENT_DIR $INPUT_LOG_COL $EVAL_INPUT_DIR $EVAL_OUTPUT_DIR_REL $IP &
bash auto_eval_log_authenticity_qwen_api_test_api.sh $CURRENT_DIR $INPUT_LOG_COL $EVAL_INPUT_DIR $EVAL_OUTPUT_DIR_AUTH $IP