#!/bin/bash
CUR_DIR=$1
MODEL_CKPT_DIR=$2
HF_MODEL_CKPT_DIR=$3 


cd $CUR_DIR

echo "begin covert model ${MODEL_CKPT_DIR}"
python convert_fsdp_model_to_hf_model.py --input_model_path ${MODEL_CKPT_DIR} --output_model_path ${HF_MODEL_CKPT_DIR}

base_mode_path=/mnt/pfs-guan-ssai/nlu/lizr/wangheqing/base_model/MindGPT-2.0-32K
token_config_path=/mnt/pfs-guan-ssai/nlu/lizr/wangheqing/lisft/auto-eval/evalset/professionsft/config
cp ${base_mode_path}/config.json ${HF_MODEL_CKPT_DIR}/
cp ${token_config_path}/tokenizer_config.json ${HF_MODEL_CKPT_DIR}/
cp ${token_config_path}/tokenizer.json  ${HF_MODEL_CKPT_DIR}/