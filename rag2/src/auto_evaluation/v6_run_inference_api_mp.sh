pip install accelerate==0.20.3
pip install aiohttp
pip install pyltp
CURRENT_DIR=$(cd $(dirname $0); pwd)
cd $CURRENT_DIR

model="/mnt/pfs-guan-ssai/nlu/renhuimin/checkpoints/16b_generator_mindgpt_rag_cft_20240729_1k_v6_0628_32k_1/checkpoint-168"
t_stamp="v6_0729_cft"

input_dir="/mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/rag2.0/data/app_self_test/input_data/"
output_dir="/mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/rag2.0/data/app_self_test/"

declare -a file_array=("手机APP_泛化集_人工_0723_2024-07-23T02_36_04.680-结果集_修正.csv") 
# readarray -t file_array < <(find "$input_dir" -type f -not -path "*/.ipynb_checkpoints/*" -exec basename {} \;)

for fs in "${file_array[@]}"
do
    python inference_api_assistant_mp.py --input_file ${input_dir}/${fs} --output_path ${output_dir} --model ${model} --time_stamp ${t_stamp} --batch_size 2 --turn_mode 13b --eval_col resp中间结果
done