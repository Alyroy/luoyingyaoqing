#!/bin/bash
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip config set global.trusted-host mirrors.aliyun.com
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_IB_GID_INDEX=3
export ONEFLOW_COMM_NET_IB_GID_INDEX=3  # for pipeline parallel only
export ONEFLOW_COMM_NET_IB_HCA=mlx5_1:1 # ,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1 # for pipeline parallel only

echo "-------------------------------------------"
pip3 list | grep torch

WORK_DIR=/mnt/pfs-guan-ssai/nlu/jiajuntong/code/rag_tool/rag2/src/tool_kg_search/1b_model_deployment/  # 工作目录需要修改

cd $WORK_DIR
pip3 install -r $WORK_DIR/requirements.txt -i https://mirrors.aliyun.com/pypi/simple

set -xv
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

SERVER_NAME=$(hostname -i)  # b区服务器的主机结点
BROAD_CAST_FILE=/mnt/pfs-guan-ssai/nlu/jiajuntong/public/api_url.log
# 检查并创建目录
mkdir -p "$(dirname "$BROAD_CAST_FILE")"
echo "http://${SERVER_NAME}:16073/ligpt_with_api/search" | tee ${BROAD_CAST_FILE}

python ligpt_api_app_new.py

sleep 10000000000