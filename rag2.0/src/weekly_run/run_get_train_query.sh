#!/bin/bash

input_folder="/workspace/renhuimin/pro_rag/data/train_data/sft/jsonl/v20240808_new"
output_folder="/mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/rag2.0/data/distillation_data/used_querys/2024-08-09/"

python get_train_query.py --input_folder $input_folder --output_folder $output_folder

