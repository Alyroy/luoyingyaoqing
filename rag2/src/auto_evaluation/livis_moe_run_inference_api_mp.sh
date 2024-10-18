pip install git+https://github.com/huggingface/accelerate
pip install https://test-space-internal-cache.s3.bj.bcebos.com/cache/ssai-training/litiktoken/litiktoken-0.0.1-py3-none-any.whl
pip install aiohttp
pip install pyltp
# pip install vllm==0.5.3.post1
pip install vllm==0.6.2
pip install blobfile


CURRENT_DIR=$(cd $(dirname $0); pwd)
cd $CURRENT_DIR


tiktoken_model_path=/mnt/pfs-guan-ssai/nlu/lvjianwei/models/MindGPT-2.0-32K/tokenizer.model

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


EVAL_MODEL="/lpai/volumes/ssai-nlu-bd/lizr/wangheqing/lisft/model/16b_generator_mindgpt_20240814_165w_v7moe_32k_liptm_model_1/checkpoint-4935/hf_model"
t_stamp="moe_sft0814_baseline"

input_dir="/mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/data/test_data/app_self_test/v20240827/input_data/"
output_dir="/mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/data/test_data/app_self_test/v20240827/"


python livis_moe_inference_assistant_mp.py --input_file ${input_dir} --output_path ${output_dir} --model ${EVAL_MODEL} --tiktoken_path ${tiktoken_model_path} --time_stamp ${t_stamp} --batch_size 5 --turn_mode moe --eval_col model_13b_input

