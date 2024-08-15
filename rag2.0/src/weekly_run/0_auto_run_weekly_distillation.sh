#!/bin/bash

# 参数
start_date="2024-08-08"
end_date="2024-08-12"
model_list=("qwen2_72b" "llama3_70b")
url_list=("http://172.24.136.30:8000/v1" " http://172.24.136.30:8001/v1")

api_url="http://172.24.139.95:16073/ligpt_with_api/search"

output_folder="/workspace/renhuimin/pro_rag/data/distillation_data/v20240814_applog/"
dpo_outpt="dpo_by0812.csv"
sft_outpt="sft_by0812"

# 启动第一个和第二个脚本并在后台运行
./1_auto_weekly_distillation_log.sh $start_date $end_date "$(IFS=,; echo "${model_list[*]}")" "$(IFS=,; echo "${url_list[*]}")" > log.run_weekly_distillation_log 2>&1 &
pid1=$!

./2_auto_weekly_distillation_extand.sh $start_date $end_date "$(IFS=,; echo "${model_list[*]}")" "$(IFS=,; echo "${url_list[*]}")" "$api_url" > log.run_weekly_distillation_extand 2>&1 &
pid2=$!

wait $pid1
wait $pid2

# 运行第三个脚本
./3_auto_get_dpo_sft_data.sh $start_date $end_date $output_folder $dpo_outpt $sft_outpt > log.get_sft_dpo 2>&1 &