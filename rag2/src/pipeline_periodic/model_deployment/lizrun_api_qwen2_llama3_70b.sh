# 设置每次的时间戳
stamp=$(date +%Y-%m-%d-%H-%M-%S)

lizrunv2 start -c "bash -c /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/weekly_run/model_deployment/vllm_qwen2_72b_llama3_70b_api.sh" \
   -j eval-${stamp} \
   -i reg-ai.chehejia.com/ssai/lizr/cu121/py310/pytorch:2.1.2-vllm0.3.1 \
   -p security \
   -n 1


# lizrun start -c "bash,-c,  /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/rag2.0/src/auto_evaluation/model_deployment/vllm_qwen2_72b_llama3_70b_api.sh" \
#    -j eval-${stamp} \
#    -i reg-ai.chehejia.com/ssai/lizr/cu121/py310/pytorch:2.1.2-vllm0.3.1 \
#    -p nlu_nlu \
#    -t all \
#    -n 1
