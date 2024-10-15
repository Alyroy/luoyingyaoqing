#!/bin/bash
pip install git+https://github.com/huggingface/accelerate
pip install https://test-space-internal-cache.s3.bj.bcebos.com/cache/ssai-training/litiktoken/litiktoken-0.0.1-py3-none-any.whl
pip install aiohttp
pip install pyltp
# pip install vllm==0.5.3.post1
pip install vllm==0.6.2
pip install blobfile

# 评估候选模型
EVAL_MODEL=$1
# 当前时间时间戳,即推理结果的保存目录名
EVAL_TIMESTAMP=$2
# quick_eval路径
CUR_DIR=$3
# 测试数据集名称
INPUT_DIR=$4
OUTPUT_DIR=$5
EVAL_COL=$6
cd $CUR_DIR

tiktoken_model_path=/mnt/pfs-guan-ssai/nlu/lvjianwei/models/MindGPT-2.0-32K/tokenizer.model

python livis_moe_inference_assistant_mp.py --input_file ${INPUT_DIR} --output_path ${OUTPUT_DIR} --model ${EVAL_MODEL} --tiktoken_path ${tiktoken_model_path} --time_stamp ${EVAL_TIMESTAMP} --batch_size 5 --turn_mode moe --eval_col ${EVAL_COL}

echo "完成所有文件的推理" > ${OUTPUT_DIR}/${EVAL_TIMESTAMP}/.done
