#!/bin/bash
CURRENT_DIR=$(cd $(dirname $0); pwd)
# cd $CURRENT_DIR
cd /mnt/pfs-guan-ssai/nlu/data/tianxy/vllm_inference_chunk/src/


# CUDA INFO
nvcc_version_output=$(nvcc --version)
[[ $nvcc_version_output =~ ([0-9]+\.[0-9]+) ]]
cuda_version=${BASH_REMATCH[1]}
echo "Detected CUDA version: $cuda_version"
expected_version="12.1"

# 检查CUDA版本是否满足需要
if [ $cuda_version != $expected_version ]
then
    echo "CUDA version mismatch. Expected: $expected_version, Detected: $cuda_version"
    exit
fi

pip install pyarrow pandas tiktoken


SERVER_NAME=$(hostname -i)  # b区服务器的主机结点
SERVER_PORT=8000
echo "qwen2 72b api服务启动地址 http://${SERVER_NAME}:${SERVER_PORT}"
MODEL=/mnt/pfs-guan-ssai/nlu/lizr/models/Qwen2-72B-Instruct
export CUDA_VISIBLE_DEVICES=0,1,2,3 # 决定使用哪张卡
python -m vllm.entrypoints.openai.api_server \
        --host $SERVER_NAME \
        --port $SERVER_PORT \
        --model $MODEL \
        --served-model-name qwen2_72b \
        --dtype bfloat16 \
        --disable-log-stats \
        --enforce-eager \
        --gpu-memory-utilization 0.9 \
        --tensor-parallel-size 4 &

        
MODEL=/mnt/pfs-guan-ssai/nlu/data/16B-generator-sft-model/16b_sft_generator_mindgpt_20240607_140w_v632k_model_1
export CUDA_VISIBLE_DEVICES=4,5,6,7
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
