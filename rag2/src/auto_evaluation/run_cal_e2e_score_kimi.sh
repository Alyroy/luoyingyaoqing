#!/bin/bash

CURRENT_DIR=/mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/auto_evaluation/
EVAL_FOLDER=/mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/data/test_data/rag_exp_test/eval_results/1209_implementation_fc/scb-fp8-1209-72b-ep2/
JOB_NAME=scb-fp8-1209-72b-ep2
FILE_NAME='fp8.csv'

cd $CURRENT_DIR

python calculate_all_kimi_metric.py --eval_folder ${EVAL_FOLDER} --job_name ${JOB_NAME} --file ${FILE_NAME}