# 设置每次的时间戳
stamp=$(date +%Y-%m-%d-%H-%M-%S)

lizrunv2 lpai start -c "bash -c /lpai/volumes/ssai-nlu-bd/nlu/app/gongwuxuan/code/rag_tool/rag2.0/src/tool_llm_response/model_deployment/lpai_vllm_llama3_70b_api.sh" \
   -j eval-llama3-70b-${stamp} \
   -i reg-ai.chehejia.com/ssai/lizr/cu121/py310/pytorch:2.1.2-vllm0.3.1 \
   -p app-bd \
   -n 1


# lizrun start -c "bash,-c,  /mnt/pfs-guan-ssai/nlu/data/renhuimin/eval_tools/vllm_inference_chunk/src/部署模型/vllm_qwen2_72b_api.sh" \
#    -j qwen2-72b-${stamp} \
#    -i reg-ai.chehejia.com/ssai/lizr/cu121/py310/pytorch:2.1.2-vllm0.3.1 \
#    -p app \
#    -t all \
#    -n 1
