#!/bin/bash

CURRENT_DIR=$1
EVAL_FOLDER=$2
JOB_NAME=$3
FILE_NAME=$4
auth_metric=$5
rel_metric=$6

if [ -z "$FILE_NAME" ]; then
  FILE_NAME='select-new-1231-lvfc_form-2k_final_input.csv1113.out.csv'
fi

if [ -z "$auth_metric" ]; then
  auth_metric='authenticity_test_api_eval'
fi

if [ -z "$rel_metric" ]; then
  rel_metric='relevance_test_api_eval'
fi

cd $CURRENT_DIR

python calculate_all_kimi_metric.py --eval_folder ${EVAL_FOLDER} --job_name ${JOB_NAME} --file ${FILE_NAME} --auth_metric ${auth_metric} --rel_metric ${rel_metric}