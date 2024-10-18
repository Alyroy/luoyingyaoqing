# 设置每次的时间戳
stamp=$(date +%Y-%m-%d-%H-%M-%S)

# lizrun start -c "bash,-c,  /mnt/pfs-guan-ssai/nlu/data/tianxy/DeepSeek-Coder/inference/run_inference.sh" \
#    -j eval-${stamp} \
#    -i reg-ai.chehejia.com/ssai/lizr/cu118/py310/pytorch:2.1.0-multinode-flashattn-2.3.2 \
#    -p sft \
#    -t all \
#    -n 1

   
# lizrun start -c "bash,-c,  /mnt/pfs-guan-ssai/nlu/data/tianxy/vllm_inference_chunk/src/vllm_qwen110b_api.sh" \
#    -j qwen-api-${stamp} \
#    -i reg-ai.chehejia.com/ssai/lizr/cu121/py310/pytorch:2.1.2-vllm0.3.1 \
#    -p sftn \
#    -t all \
#    -n 1

# root_path=`pwd`
job_name=$1
queue_name=$2
root_path=$3
echo "job_name:${job_name}"
lizrun start -c "${root_path}/api_service/vllm_qwen_api.sh" \
   -j ${job_name} \
   -i reg-ai.chehejia.com/ssai/lizr/cu121/py310/pytorch:2.1.2-vllm0.3.1 \
   -p ${queue_name} \
   -n 1
