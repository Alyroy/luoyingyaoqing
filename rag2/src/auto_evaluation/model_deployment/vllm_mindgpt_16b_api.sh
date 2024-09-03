#!/bin/bash
# 获取当前脚本路径，进入当前脚本目录
CURRENT_DIR=$(cd $(dirname $0); pwd)
cd $CURRENT_DIR

MODEL=/mnt/pfs-guan-ssai/nlu/data/16B-generator-sft-model/16b_sft_generator_mindgpt_20240607_140w_v632k_model_1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
HOST=${HOSTNAME//-/.}  # b区服务器的主机结点
PORT=9000
echo "mindgpt 16b api服务启动地址 http://${HOST}:${PORT}"
# 并行部署
for ((i=0;i<8;i++)); do
    # export CUDA_VISIBLE_DEVICES=$i
    python -m vllm.entrypoints.openai.api_server \
            --host $HOST \
            --port $PORT$i  \
            --model $MODEL \
            --served-model-name mindgpt \
            --dtype bfloat16 \
            --max-model-len 10000 \
            --disable-log-stats \
            --tokenizer $MODEL \
            --disable-log-stats \
            --chat-template template_mindgpt_task.jinja \
            --tensor-parallel-size 4
    break
done

wait