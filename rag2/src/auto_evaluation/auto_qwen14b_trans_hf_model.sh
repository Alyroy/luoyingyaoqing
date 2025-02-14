#!/bin/bash

# 全局参数设置
CURRENT_DIR=/mnt/pfs-guan-ssai/nlu/renhuimin/lisft_qwen/lisft/examples/Qwen_14b_FSDP
cd $CURRENT_DIR

MODEL_CKPT_DIR=$1
HF_MODEL_CKPT_DIR=$2

echo "begin covert model ${MODEL_CKPT_DIR}"
python convert_fsdp_model_to_hf_model.py --input_model_path ${MODEL_CKPT_DIR} --output_model_path ${HF_MODEL_CKPT_DIR}

