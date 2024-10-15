#!/bin/bash
source ~/.bashrc
conda activate rhm_env

# CURRENT_DIR=$(cd $(dirname \$0); pwd)
CURRENT_DIR=/mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/
cd $CURRENT_DIR
source "common/periodical_job_template.sh"
JOB_ROOT_DIR=${CURRENT_DIR}
source "common/common_scripts.sh"

JOB_NAME="filter_log_single_rag_car"

# current_date=$(date "+%Y-%m-%d")
# date=$(date -d "${current_date} -2 day" "+%Y-%m-%d")
# date="2024-08-14"
# dates=("2024-08-28" "2024-08-29" "2024-08-30" "2024-08-31" "2024-09-01" "2024-09-02")
start_date="2024-10-08"
end_date="2024-10-11"

# 生成日期列表
current_date="$start_date"
while [[ "$current_date" < "$end_date" || "$current_date" == "$end_date" ]]; do
    dates+=("$current_date")
    current_date=$(date -I -d "$current_date + 1 day")
done

get_model_url() {
    local model_name=$1
    local log_file

    case $model_name in
        "qwen2_72b")
            log_file="/mnt/pfs-guan-ssai/nlu/gongwuxuan/public/Qwen2_72B_running_url.log"
            ;;
        "qwen25_72b")
            log_file="/mnt/pfs-guan-ssai/nlu/gongwuxuan/public/Qwen25_72B_running_url.log"
            ;;
        *)
            echo "Unknown model name: $model_name"
            exit 1
            ;;
    esac

    if [ -f "${log_file}" ]; then
        cat "${log_file}"
    else
        echo "Log file ${log_file} does not exist. Exiting."
        exit 1
    fi
}

CheckDependencyFunction() {
    local raw_extension_type=$1
    local done_file="/mnt/pfs-guan-ssai/nlu/renhuimin/data/log_distillation_car/${date}/${raw_extension_type}/single_True_rag_True/gpt4_data/.done"
    if [ ! -f $done_file ]; then
        PrintLog "$done_file is not ready."
        exit 1
    fi

    PrintLog "$done_file is ready."
}

ExecuteJobFunction() {
    local raw_extension_type=$1
    cd /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/auto_filter/
    folder="/mnt/pfs-guan-ssai/nlu/renhuimin/data/log_distillation_car/${date}/${raw_extension_type}/single_True_rag_True/"
    initial_folder="${folder}gpt4_data/"
    readarray -t initial_inputs < <(find "$initial_folder" -type f -name "*.csv" -not -path "*/.ipynb_checkpoints/*" | sort)
    style_filter_output_base_dir="${folder}style_filter_output"
    correct_filter_output_base_dir="${folder}correct_filter_output"
    # 自动获取url
    model_list=("qwen2_72b" "qwen25_72b")
    url_list=()
    
    for model in "${model_list[@]}"; do
        url_list+=("$(get_model_url ${model})/v1")
    done
    # 将模型列表和 URL 列表转换为字符串以便传递
    model_list_str=$(IFS=","; echo "${model_list[*]}")
    url_list_str=$(IFS=","; echo "${url_list[*]}")
    
    # 对每个初始输入处理
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
}

CheckDoneFunction() {
    local raw_extension_type=$1
    folder="/mnt/pfs-guan-ssai/nlu/renhuimin/data/log_distillation_car/${date}/${raw_extension_type}/single_True_rag_True/"
    initial_folder="${folder}gpt4_data/"

    # 使用 readarray 和 find 获取文件列表
    readarray -t initial_inputs < <(find "$initial_folder" -type f -name "*.csv" -not -path "*/.ipynb_checkpoints/*" | sort)

    style_filter_output_base_dir="${folder}style_filter_output"
    correct_filter_output_base_dir="${folder}correct_filter_output"
    
    # 初始设置 all_files_present 变量
    all_files_present=true

    # 循环检查文件是否存在
    for initial_input in "${initial_inputs[@]}"; do
        filename_with_ext="${initial_input##*/}"
        speech_filename="${filename_with_ext%.*}"
        correct_filtered_file="${correct_filter_output_base_dir}/${speech_filename}_speech_style_filter_保留_correct_ans_filter_保留.csv"
        echo $correct_filtered_file

        if [ ! -f "${correct_filtered_file}" ]; then
            all_files_present=false
            break
        fi
    done

    # 创建 good.done 文件 
    if [ "$all_files_present" = true ]; then
        touch "${style_filter_output_base_dir}/.done"
        echo ".done file created"
    else
        echo "All sytle filter required files are not present."
    fi
    # 创建 good.done 文件
    if [ "$all_files_present" = true ]; then
        touch "${correct_filter_output_base_dir}/.done"
        echo ".done file created"
    else
        echo "All correct filter required files are not present."
    fi
}

# for date in "${dates[@]}"; do
#     echo "[INFO] Processing raw date: $date"
#     CheckDependencyFunction "raw" $date
#     if [ $? -eq 0 ]; then
#         ExecuteJobFunction "raw" $date
#         CheckDoneFunction "raw" $date
#     else
#         echo "[ERROR] Dependency check failed for date: $date"
#     fi
# done

for date in "${dates[@]}"; do
    echo "[INFO] Processing extension date: $date"
    CheckDependencyFunction "extension" $date
    if [ $? -eq 0 ]; then
        ExecuteJobFunction "extension" $date
        CheckDoneFunction "extension" $date
    else
        echo "[ERROR] Dependency check failed for date: $date"
    fi
done
