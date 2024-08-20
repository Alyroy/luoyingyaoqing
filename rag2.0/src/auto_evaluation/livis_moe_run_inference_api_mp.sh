pip install git+https://github.com/huggingface/accelerate
pip install https://test-space-internal-cache.s3.bj.bcebos.com/cache/ssai-training/litiktoken/litiktoken-0.0.1-py3-none-any.whl
pip install aiohttp
pip install pyltp
pip install vllm==0.5.3.post1
pip install blobfile

CURRENT_DIR=$(cd $(dirname $0); pwd)
cd $CURRENT_DIR

# EVAL_MODEL="/mnt/pfs-guan-ssai/nlu/lizr/wangheqing/lisft/model/16b_generator_mindgpt_20240725_155w_v7moe_32k_liptm_model_2/checkpoint-9333/hf_model"
# EVAL_MODEL="/mnt/pfs-guan-ssai/nlu/luhengtong/li-safe-rlhf/output/sft-mind-gpt-v7moe-0725_dpo_dpo-0727_2560_n12b3e2_0728-seed42/ckpt-2910/"
# EVAL_MODEL="/mnt/pfs-guan-ssai/nlu/lizr/wangheqing/lisft/model/16b_generator_mindgpt_20240801_160w_v7moe_32k_liptm_model_1/checkpoint-4902/hf_model"
EVAL_MODEL="/mnt/pfs-guan-ssai/nlu/luhengtong/li-safe-rlhf/output/sft-mind-gpt-v7moe-0801_dpo_dpo-v7-0802_2560_n16b3e2_0804-seed42/ckpt-2166"
tiktoken_model_path="none"
eval_column="resp中间结果"
# tiktoken_model_path=/mnt/pfs-guan-ssai/nlu/lvjianwei/models/MindGPT-2.0-32K/tokenizer.model

# if [ ! -f ${EVAL_MODEL}/tokenizer_config.json ]; then
#     if [[ "${tiktoken_model_path}" != "none" ]]; then
#         cp config/tokenizer_config.json ${EVAL_MODEL}
#     fi
# fi

# if [ ! -f ${EVAL_MODEL}/tokenizer.json ]; then
#     if [[ "${tiktoken_model_path}" != "none" ]]; then
#         cp config/tokenizer.json ${EVAL_MODEL}
#     fi
# fi


t_stamp="moe_dpo_0802"

input_dir="/mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/rag2.0/data/test_data/v20240804/input"
output_dir="/mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/rag2.0/data/test_data/v20240804/"

# readarray -t file_array < <(find "$input_dir" -type f -not -path "*/.ipynb_checkpoints/*" -exec basename {} \;)
declare -a file_array=("手机APP_泛化集_人工_2024-07-30.csv") 

for fs in "${file_array[@]}"
do
    python livis_moe_inference_assistant_mp.py --input_file ${input_dir}/${fs} --output_path ${output_dir} --model ${EVAL_MODEL} --tiktoken_path ${tiktoken_model_path} --time_stamp ${t_stamp} --batch_size 2 --turn_mode moe --eval_col ${eval_column}
done
