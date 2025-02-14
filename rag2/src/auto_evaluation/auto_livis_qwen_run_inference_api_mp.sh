#!/bin/bash

mkdir -p ~/.pip && \
echo "[global]" > ~/.pip/pip.conf && \
echo "trusted-host = http://mirrors.aliyun.com" >> ~/.pip/pip.conf && \
echo "index-url = https://mirrors.aliyun.com/pypi/simple/" >> ~/.pip/pip.conf

pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip config set install.trusted-host http://mirrors.aliyun.com

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
temperature=$7
top_p=$8
repetition=$9
cd $CUR_DIR

tiktoken_model_path=/mnt/pfs-guan-ssai/nlu/lvjianwei/models/MindGPT-2.0-32K/tokenizer.model

python vllm_qwen2.5_inference.py --input_file ${INPUT_DIR} --output_path ${OUTPUT_DIR} --model ${EVAL_MODEL} --tiktoken_path ${tiktoken_model_path} --time_stamp ${EVAL_TIMESTAMP} --batch_size 2 --turn_mode qwen --eval_col ${EVAL_COL} --temperature ${temperature} --top_p ${top_p} --repetition ${repetition}

# 检查 OUTPUT_DIR 下是否有 .csv 文件
csv_file_count=$(find "${OUTPUT_DIR}/${EVAL_TIMESTAMP}" -type f -name "*.csv" | wc -l)

# 如果存在 .csv 文件，生成 .done 文件
if [ $csv_file_count -gt 0 ]; then
    echo "完成所有文件的推理" > "${OUTPUT_DIR}/${EVAL_TIMESTAMP}/.done"
else
    echo "未找到 .csv 文件, 未生成 .done 文件"
fi