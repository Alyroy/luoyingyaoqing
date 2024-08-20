#!/bin/bash
cd /lpai/volumes/ssai-nlu-bd/nlu/app/gongwuxuan/tools/sshpass/
tar -xvf sp.tar.gz
cd ./sshpass-1.10
./configure
make install
rm -r ../sshpass-1.10

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

SERVER_NAME=$(hostname -i)  # b区服务器的主机结点
SERVER_PORT=8000
BROAD_CAST_FILE=/lpai/volumes/ssai-nlu-bd/nlu/app/gongwuxuan/pubilc/Qwen2_72B_running_url.log
TGT_SERVER_PATH=/mnt/pfs-guan-ssai/nlu/gongwuxuan/public
MODEL=/lpai/volumes/ssai-nlu-bd/lizr/models/Qwen2-72B-Instruct

echo "qwen2 72b api服务启动地址 http://${SERVER_NAME}:${SERVER_PORT}"
echo "http://${SERVER_NAME}:${SERVER_PORT}" | tee ${BROAD_CAST_FILE}
sshpass -p gwx199987 scp -r -P 32010 ${BROAD_CAST_FILE} root@172.24.136.34:${TGT_SERVER_PATH}

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # 决定使用哪张卡
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
wait