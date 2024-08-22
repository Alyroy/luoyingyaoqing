#!/bin/bash
source ~/.bashrc
conda activate rhm_env

# CURRENT_DIR=$(cd $(dirname $0); pwd)
CURRENT_DIR=/mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/weekly_run/
cd $CURRENT_DIR

input_folder="/mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/data/train_data/sft/jsonl/latest/renhuimin_assistant_sft/"
output_folder="/mnt/pfs-guan-ssai/nlu/renhuimin/data/trained_querys/"

python get_train_query.py --input_folder $input_folder --output_folder $output_folder

