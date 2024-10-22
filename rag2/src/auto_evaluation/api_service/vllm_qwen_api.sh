#!/bin/bash
CURRENT_DIR=$(cd $(dirname $0); pwd)
cd $CURRENT_DIR

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



# MODEL_PATH=/mnt/pfs-guan-ssai/nlu/lizr/models/Qwen2-72B-Instruct
MODEL_PATH=/mnt/pfs-guan-ssai/nlu/lizr/models/Qwen2.5-72B-Instruct

SERVER_NAME=${HOSTNAME//-/.}  # b区服务器的主机结点
SERVER_PORT=8012
echo "api服务启动地址 http://${SERVER_NAME}:${SERVER_PORT}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name qwen \
    --port ${SERVER_PORT} \
    --host ${SERVER_NAME} \
    --trust-remote-code \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 8

# MODEL=/mnt/pfs-guan-ssai/nlu/lizr/models/Qwen2-72B-Instruct
# a=-1
# for ((i=0;i<8;i+=4)); do
#     a=$(($a+1))
#     export CUDA_VISIBLE_DEVICES=$i,$(($i+1)),$(($i+2)),$(($i+3))
#     echo $CUDA_VISIBLE_DEVICES
#     # exit
#     python -m vllm.entrypoints.openai.api_server \
#         --host $SERVER_NAME \
#         --port 800$a  \
#         --model $MODEL \
#         --served-model-name qwen \
#         --dtype bfloat16 \
#         --gpu-memory-utilization 0.9 \
#         --tensor-parallel-size 4 &
# done

wait
# MODEL=/mnt/pfs-guan-ssai/nlu/data/tianxy/MODELS/Qwen1.5-14B-Chat
# for ((i=0;i<8;i++)); do
#     export CUDA_VISIBLE_DEVICES=$i
#     echo $CUDA_VISIBLE_DEVICES
#     # exit
#     python -m vllm.entrypoints.openai.api_server \
#         --host $SERVER_NAME \
#         --port 800$i  \
#         --model $MODEL \
#         --served-model-name qwen \
#         --dtype bfloat16 \
#         --gpu-memory-utilization 0.9 \
#         --tensor-parallel-size 1 &
# done