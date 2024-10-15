# 设置每次的时间戳
stamp=$(date +%Y-%m-%d-%H-%M-%S)

# lizrun start -c "bash,-c,  /mnt/pfs-guan-ssai/nlu/data/tianxy/DeepSeek-Coder/inference/run_inference.sh" \
#    -j eval-${stamp} \
#    -i reg-ai.chehejia.com/ssai/lizr/cu118/py310/pytorch:2.1.0-multinode-flashattn-2.3.2 \
#    -p sft \
#    -t all \
#    -n 1

   
lizrun start -c "bash,-c,  /mnt/pfs-guan-ssai/nlu/data/renhuimin/eval_tools/vllm_inference_chunk/src/部署模型/vllm_mindgpt_16b_api.sh" \
   -j mindgpt-${stamp} \
   -i reg-ai.chehejia.com/ssai/lizr/cu121/py310/pytorch:2.1.2-vllm0.3.1 \
   -p sft \
   -t all \
   -n 1