#!/bin/bash
CURRENT_DIR=$(cd $(dirname $0); pwd)
cd $CURRENT_DIR

# initial_folder="/workspace/renhuimin/rag_llm/data/v20240612/initial_input/"
# # 定义初始输入文件列表
# initial_inputs=(
#     ${initial_folder}"20240527-体育查询-送标1.csv"
#     ${initial_folder}"20240527-体育查询-送标2.csv"
#     # 可以在这里添加更多的文件路径
# )

folder="/workspace/renhuimin/pro_rag/data/distillation_data/v20240808/"
# 定义初始文件夹
initial_folder=${folder}"gpt4_data/"
# 查找初始文件夹中的所有数据文件，并将其添加到初始输入文件列表中
readarray -t initial_inputs < <(find "$initial_folder" -type f -name "*.csv" -not -path "*/.ipynb_checkpoints/*" | sort)

style_filter_output_base_dir=${folder}"style_filter_output"
correct_filter_output_base_dir=${folder}"correct_filter_output"

# 定义模型列表和 URL 列表
model_list=("gpt4o")
url_list=("https://rhm-gpt4.fc.chj.cloud/gpt4o")

# 将模型列表和 URL 列表转换为字符串以便传递
model_list_str=$(IFS=","; echo "${model_list[*]}")
url_list_str=$(IFS=","; echo "${url_list[*]}")

echo "all start at $(date)"
# 对每个初始输入文件进行处理
for initial_input in "${initial_inputs[@]}"; do
    # 提取文件名（包括扩展名和不包括扩展名的部分）
    filename_with_ext="${initial_input##*/}"
    speech_filename="${filename_with_ext%.*}"

    # 创建输出目录
    mkdir -p "$style_filter_output_base_dir"
    mkdir -p "$correct_filter_output_base_dir"

    # Step 1: 筛选话术
    echo "step1: start at $(date)"
    echo "Running auto_filter_style to filter the response data for $initial_input..."
    bash ./auto_filter_style.sh "$initial_input" "$style_filter_output_base_dir" "$model_list_str" "$url_list_str"
    if [ $? -ne 0 ]; then
        echo "Error in running run_auto_filter_style.sh for $initial_input"
        exit 1
    fi
    echo "Style filtered data generated at $style_filter_output_base_dir"

    # Step 2: 话术正确后，筛选回复正确
    echo "step2: start at $(date)"
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