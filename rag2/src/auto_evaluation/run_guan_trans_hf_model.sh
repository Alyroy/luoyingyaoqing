#!/bin/bash
CUR_DIR=/mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/auto_evaluation/
MODEL_CKPT_DIR=/mnt/pfs-gv8sxa/nlu/team/renhuimin/checkpoints/moe_cft_train20241129_child1w_sft10w_n4b4lr1e5/checkpoint-2520
HF_MODEL_CKPT_DIR=${MODEL_CKPT_DIR}/hf_model

cd $CUR_DIR

echo "begin covert model ${MODEL_CKPT_DIR}"
python convert_fsdp_model_to_hf_model.py --input_model_path ${MODEL_CKPT_DIR} --output_model_path ${HF_MODEL_CKPT_DIR}

base_mode_path=/mnt/pfs-guan-ssai/nlu/lizr/wangheqing/base_model/MindGPT-2.0-32K
token_config_path=/mnt/pfs-guan-ssai/nlu/lizr/wangheqing/lisft/auto-eval/evalset/professionsft/config
cp ${base_mode_path}/config.json ${HF_MODEL_CKPT_DIR}/
cp ${token_config_path}/tokenizer_config.json ${HF_MODEL_CKPT_DIR}/
cp ${token_config_path}/tokenizer.json  ${HF_MODEL_CKPT_DIR}/