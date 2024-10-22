#!/bin/bash

CURRENT_DIR=$1
EVAL_FOLDER=$2
JOB_NAME=$3

cd $CURRENT_DIR

python calculate_all_e2e_metric.py --eval_folder ${EVAL_FOLDER} --job_name ${JOB_NAME}