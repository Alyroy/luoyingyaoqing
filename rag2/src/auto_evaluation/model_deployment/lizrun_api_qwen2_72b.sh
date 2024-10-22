# 设置每次的时间戳
stamp=$(date +%Y-%m-%d-%H-%M-%S)

lizrunv2 start -c "bash -c /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/auto_evaluation/model_deployment/vllm_qwen2_72b_api.sh" \
   -j eval1-${stamp} \
   -i reg-ai.chehejia.com/ssai/lizr/cu121/py310/pytorch:2.1.2-vllm0.3.1 \
   -p base \
   -n 1


lizrunv2 start -c "bash -c /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/auto_evaluation/model_deployment/vllm_qwen2_72b_api.sh" \
   -j eval2-${stamp} \
   -i reg-ai.chehejia.com/ssai/lizr/cu121/py310/pytorch:2.1.2-vllm0.3.1 \
   -p base \
   -n 1


lizrunv3 start -c "bash -c /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/auto_evaluation/model_deployment/vllm_qwen2_72b_api.sh" \
   -j eval2-${stamp} \
   -i reg-ai.chehejia.com/ssai/lizr/cu121/py310/pytorch:2.1.2-vllm0.3.1 \
   -p base \
   -n 1
