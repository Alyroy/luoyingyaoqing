#!/bin/bash

# 函数：生成日期列表
generate_date_list() {
    local start_date=$1
    local end_date=$2
    local dates=()

    # 使用GNU date生成日期列表
    current_date="$start_date"
    while [[ "$current_date" < "$end_date" ]] || [[ "$current_date" == "$end_date" ]]; do
        dates+=("$current_date")
        current_date=$(date -I -d "$current_date + 1 day")
    done

    echo "${dates[@]}"
}

# 指定开始和结束日期，定义模型列表和 URL 列表
start_date=$1
end_date=$2
model_list=(${3//,/ })
url_list=(${4//,/ })
api_url=$5

# 获取生成的日期列表
dates=($(generate_date_list $start_date $end_date))

# 循环调用Python文件
for date in "${dates[@]}"; do
    echo "step1: 蒸馏原始日志数据, start at $(date)"
    cd /workspace/renhuimin/pro_rag/src/weekly_run/
    python weekly_distillation_extand_sft.py --date $date --api_url $api_url

    echo "step 2: 自动筛选高质量回复数据, start at $(date)"
    cd /workspace/renhuimin/pro_rag/src/auto_filter/
    folder='/mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/rag2.0/data/distillation_data/log_data/'${date}'/extension/'
    initial_folder=${folder}"gpt4_data/"
    readarray -t initial_inputs < <(find "$initial_folder" -type f -name "*.csv" -not -path "*/.ipynb_checkpoints/*" | sort)
    style_filter_output_base_dir=${folder}"style_filter_output"
    correct_filter_output_base_dir=${folder}"correct_filter_output"
    # 将模型列表和 URL 列表转换为字符串以便传递
    model_list_str=$(IFS=","; echo "${model_list[*]}")
    url_list_str=$(IFS=","; echo "${url_list[*]}")
    
    # 对每个初始输入文件进行处理
    for initial_input in "${initial_inputs[@]}"; do
        # 提取文件名（包括扩展名和不包括扩展名的部分）
        filename_with_ext="${initial_input##*/}"
        speech_filename="${filename_with_ext%.*}"
    
        # 创建输出目录
        mkdir -p "$style_filter_output_base_dir"
        mkdir -p "$correct_filter_output_base_dir"
    
        # Step 1: 筛选话术
        echo "step2.1: start at $(date)"
        echo "Running auto_filter_style to filter the response data for $initial_input..."
        bash ./auto_filter_style.sh "$initial_input" "$style_filter_output_base_dir" "$model_list_str" "$url_list_str"
        if [ $? -ne 0 ]; then
            echo "Error in running run_auto_filter_style.sh for $initial_input"
            exit 1
        fi
        echo "Style filtered data generated at $style_filter_output_base_dir"
    
        # Step 2: 话术正确后，筛选回复正确
        echo "step2.2: start at $(date)"
        style_filtered_file="${style_filter_output_base_dir}/${speech_filename}_speech_style_filter_保留.csv"
        echo "Running auto_filter_correct to filter the response data for $style_filtered_file..."
        bash ./auto_filter_correct.sh "$style_filtered_file" "$correct_filter_output_base_dir" "$model_list_str" "$url_list_str"
        if [ $? -ne 0 ]; then
            echo "Error in running run_auto_filter_correct.sh for $style_filtered_file"
            exit 1
        fi
        echo "Filtered data generated at $correct_filter_output_base_dir."
    done
    
    echo "all finished at $(date)"
    echo "All steps completed successfully for all input files."
done