#!/bin/bash

QUEUE_NAME=$1
# 设置每次的时间戳
stamp=$(date +%Y-%m-%d-%H-%M-%S)

lizrun lpai start -c "bash -c /lpai/volumes/ssai-nlu-bd/nlu/app/renhuimin/rag_tool/src/tool_llm_response/model_deployment/lpai_vllm_qwen2_72b_api.sh" \
   -i reg-ai.chehejia.com/ssai/lizr/cu121/py310/pytorch:2.1.2-vllm0.3.1 \
   -p ${QUEUE_NAME} \
   -n 1 \
   -j eval-qwen2-72b-${stamp}