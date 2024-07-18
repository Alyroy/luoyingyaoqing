pip install accelerate==0.20.3
pip install aiohttp
pip install pyltp
CURRENT_DIR=$(cd $(dirname $0); pwd)
cd $CURRENT_DIR

# model="/mnt/pfs-guan-ssai/nlu/data/lisunzhu/checkpoints/mindgpt_20240124_166w_v4base_seq6144_mode_1/checkpoint-6600/13b_generator_mindgpt_20240124_166w_v4base_seq6144_mode_1"
# model="/mnt/pfs-guan-ssai/nlu/data/renhuimin/checkpoints/mindgpt_20240131_167w_v4base_seq6144_model_2/checkpoint-7800/13b_generator_mindgpt_20240131_167w_v4base_seq6144_model_2"
# model="/mnt/pfs-guan-ssai/nlu/data/lisunzhu/checkpoints/mindgpt_20240228_150w_reason_v2_seq6144_mode_1/checkpoint-8000/13b_generator_mindgpt_20240228_150w_reason_v2_seq6144_mode_1"
# model="/mnt/pfs-guan-ssai/nlu/data/16B-generator-sft-model/16b_sft_generator_mindgpt_20240320_170w_v6base50p_model_1"
# model="/mnt/pfs-guan-ssai/nlu/data/13B-generator-sft-model/13b_generator_mindgpt_20240327_170w_v4base_seq6144_mode_1"
# t_stamp="obs_rel_16B_v6_0320"
# model="/mnt/pfs-guan-ssai/nlu/data/16B-generator-sft-model/16b_sft_generator_mindgpt_20240410_170w_v6base_b176c4q5_model_1"
# model="/mnt/pfs-guan-ssai/nlu/data/wangheqing/v6_16b_scripts_new/lisft/model/16b_generator_mindgpt_20240424_140w_v6base_model_1/checkpoint-4500"
# model="/mnt/pfs-guan-ssai/nlu/data/renhuimin/checkpoints/16b_mindgpt_20240504_11w_v6base_model_carcomparision_1/checkpoint-2722"
# model="/mnt/pfs-guan-ssai/nlu/data/renhuimin/checkpoints/16b_mindgpt_20240502_140w_v6base_model_ragskill_all_1/checkpoint-9000"
# model="/mnt/pfs-guan-ssai/nlu/data/16B-generator-sft-model/16b_sft_generator_mindgpt_20240424_140w_v6base_model_1"
# model="/mnt/pfs-guan-ssai/nlu/data/wangheqing/v6_16b_scripts_new/lisft/model/16b_generator_mindgpt_20240511_140w_v632k_model_1/checkpoint-21873"
# model="/mnt/pfs-guan-ssai/nlu/data/wangheqing/v6_16b_scripts_new/lisft/model/16b_generator_mindgpt_20240524_140w_v632k_model_1/checkpoint-26208"
# model="/mnt/pfs-guan-ssai/nlu/data/16B-generator-sft-model/16b_sft_generator_mindgpt_20240607_140w_v632k_model_1"
# model="/mnt/pfs-guan-ssai/nlu/data/renhuimin/checkpoints/16b_generator_mindgpt_20240614_7k_v6_32k_model_2/checkpoint-702"
# model="/mnt/pfs-guan-ssai/nlu/data/renhuimin/checkpoints/16b_generator_mindgpt_20240614_7k_v6_32k_model_2/checkpoint-1212"
# model="/mnt/pfs-guan-ssai/nlu/data/renhuimin/checkpoints/16b_generator_rag_new_20240626_64k_v6_32k_model_1/checkpoint-3024"
# model="/mnt/pfs-guan-ssai/nlu/data/renhuimin/checkpoints/16b_generator_rag_new_20240627_66k_v6_32k_model_1/checkpoint-3132"
# model="/mnt/pfs-guan-ssai/nlu/data/renhuimin/checkpoints/16b_generator_mindgpt_20240625_33k_v6_32k_model_1/checkpoint-6336"
# model="/mnt/pfs-guan-ssai/nlu/data/renhuimin/checkpoints/16b_generator_rag_full_nofollow_20240625_11w_v6_32k_model_1/checkpoint-5628"
# model="/mnt/pfs-guan-ssai/nlu/data/renhuimin/checkpoints/16b_generator_rag_cft_multi_20240705_11w_v6_32k_model_1/checkpoint-10416"

model="/mnt/pfs-guan-ssai/nlu/data/16B-generator-sft-model/16b_sft_generator_mindgpt_20240607_140w_v632k_model_1"
t_stamp="v6_0607"

input_dir="/mnt/pfs-guan-ssai/nlu/data/renhuimin/pro_rag/data/eval_data/v20240717/input_data/"
output_dir="/mnt/pfs-guan-ssai/nlu/data/renhuimin/pro_rag/data/eval_data/v20240717/"

declare -a file_array=("测试集_奥运赛事_200_json.csv" "测试集_奥运赛事_200_text.csv") 
# readarray -t file_array < <(find "$input_dir" -type f -not -path "*/.ipynb_checkpoints/*" -exec basename {} \;)

for fs in "${file_array[@]}"
do
    python inference_api_assistant_mp.py --input_file ${input_dir}/${fs} --output_path ${output_dir} --model ${model} --time_stamp ${t_stamp} --batch_size 2 --turn_mode 13b
done


model="/mnt/pfs-guan-ssai/nlu/data/renhuimin/checkpoints/16b_generator_rag_base_structure_20240716_13w_v6_32k_model_1/checkpoint-10000"
t_stamp="v6_0716_base_new_rag"

input_dir="/mnt/pfs-guan-ssai/nlu/data/renhuimin/pro_rag/data/eval_data/v20240717/input_data/"
output_dir="/mnt/pfs-guan-ssai/nlu/data/renhuimin/pro_rag/data/eval_data/v20240717/"

declare -a file_array=("测试集_奥运赛事_200_json.csv" "测试集_奥运赛事_200_text.csv") 
# readarray -t file_array < <(find "$input_dir" -type f -not -path "*/.ipynb_checkpoints/*" -exec basename {} \;)

for fs in "${file_array[@]}"
do
    python inference_api_assistant_mp.py --input_file ${input_dir}/${fs} --output_path ${output_dir} --model ${model} --time_stamp ${t_stamp} --batch_size 2 --turn_mode 13b
done



model="/mnt/pfs-guan-ssai/nlu/data/renhuimin/checkpoints/16b_generator_rag_cft_structure_20240716_13w_v6_32k_model_2/checkpoint-6192"
t_stamp="v6_0716_cft_new_rag"

input_dir="/mnt/pfs-guan-ssai/nlu/data/renhuimin/pro_rag/data/eval_data/v20240717/input_data/"
output_dir="/mnt/pfs-guan-ssai/nlu/data/renhuimin/pro_rag/data/eval_data/v20240717/"

# declare -a file_array=("测试集_奥运赛事_200_json.csv" "测试集_奥运赛事_200_text.csv") 
readarray -t file_array < <(find "$input_dir" -type f -not -path "*/.ipynb_checkpoints/*" -exec basename {} \;)

for fs in "${file_array[@]}"
do
    python inference_api_assistant_mp.py --input_file ${input_dir}/${fs} --output_path ${output_dir} --model ${model} --time_stamp ${t_stamp} --batch_size 2 --turn_mode 13b
done