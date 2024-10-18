#!/bin/bash
source ~/.bashrc
conda activate rhm_env

CURRENT_DIR=/mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/
cd $CURRENT_DIR
source "common/periodical_job_template.sh"
JOB_ROOT_DIR=${CURRENT_DIR}
source "common/common_scripts.sh"

JOB_NAME="get_atomic_capacity_single_rag"

current_date=$(date "+%Y-%m-%d")
date=$(date -d "${current_date} -3 day" "+%Y-%m-%d") # 滞后模型打标一天
# date='2024-08-13'

CheckDependencyFunction() {
    local car_app_feishu_type=$1
    local raw_extension_type=$2
    local atomic_type=$3
    local done_file="/mnt/pfs-guan-ssai/nlu/renhuimin/data/${car_app_feishu_type}/${date}/${raw_extension_type}/single_True_rag_True/${atomic_type}/.done"
    if [ ! -f $done_file ]; then
        PrintLog "$done_file is not ready."
        return 1  # 将 exit 改为 return
    fi

    PrintLog "$done_file is ready."
    return 0
}

ExecuteJobFunction() {
    local car_app_feishu_type=$1
    local raw_extension_type=$2
    local atomic_type=$3
    infolder="/mnt/pfs-guan-ssai/nlu/renhuimin/data/${car_app_feishu_type}/${date}/${raw_extension_type}/single_True_rag_True/${atomic_type}/"
    outfolder="/mnt/pfs-guan-ssai/nlu/renhuimin/data/${car_app_feishu_type}/${date}/${raw_extension_type}/single_True_rag_True/${atomic_type}_atomic/"
    mkdir -p $outfolder
    # readarray -t initial_inputs < <(find "$infolder" -type f -name "filter.csv" -not -path "*/.*/*" -not -name ".*" | sort)

    readarray -t initial_inputs < <(find "$infolder" -type f -name "*filter.csv" ! -path "*/.*/*" | sort)
        
    # 打印开始时间
    begin_time=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[INFO] begin_time: ${begin_time}" 
    echo "Processing: $initial_inputs"
    
    for initial_input in "${initial_inputs[@]}"; do
        INPUT_PATH="${initial_input}"
        # 提取最里层文件名
        base_filename=$(basename "$INPUT_PATH")
        OUTPUT_PATH="$outfolder/${base_filename}"  # 在这里指定输出路径
        ANS_COL="parser_gpt4"
        EVAL_COL="qwen2_72b_eval_response"
        prompt_folder="/mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/conf/synonyms_prompts/"
        SOURCE="rag原子能力-回复话术判断"

        # 调用Python脚本
        python get_correct_filter_atomic_capacity.py \
        --input_path "$INPUT_PATH" \
        --output_path "$OUTPUT_PATH" \
        --ans_col "$ANS_COL" \
        --eval_col "$EVAL_COL" \
        --prompt_folder "$prompt_folder" \
        --source "$SOURCE" \
    
    done

    # 打印结束时间
    end_time=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[INFO] end_time: $end_time "

    duration=$(( $(date +%s -d "${end_time}") - $(date +%s -d "${begin_time}") ))
    echo "[INFO] duration: $(( duration / 60 )) min "
    return 0
}

# 参数组合
car_app_feishu_types=("log_distillation" "log_distillation_car") # 
raw_extension_types=("raw" "extension") # 
atomic_types=("correct_filter_output") #

for car_app_feishu_type in "${car_app_feishu_types[@]}"; do
    for raw_extension_type in "${raw_extension_types[@]}"; do
        for atomic_type in "${atomic_types[@]}"; do
            # 检查依赖并执行任务，出错继续
            CheckDependencyFunction "$car_app_feishu_type" "$raw_extension_type" "$atomic_type" # || continue
            ExecuteJobFunction "$car_app_feishu_type" "$raw_extension_type" "$atomic_type"  # || continue
        done
    done
done