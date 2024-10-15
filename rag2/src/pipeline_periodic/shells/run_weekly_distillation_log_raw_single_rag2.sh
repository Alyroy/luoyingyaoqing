#!/bin/bash
source ~/.bashrc
conda activate rhm_env

# CURRENT_DIR=$(cd $(dirname $0); pwd)
CURRENT_DIR=/mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/
cd $CURRENT_DIR
source "common/periodical_job_template.sh"
JOB_ROOT_DIR=${CURRENT_DIR}
source "common/common_scripts.sh"

JOB_NAME="distillate_raw_log_single_rag"

# 日期列表
start_date="2024-10-08"
end_date="2024-10-11"

# 生成日期列表
current_date="$start_date"
while [[ "$current_date" < "$end_date" || "$current_date" == "$end_date" ]]; do
    dates+=("$current_date")
    current_date=$(date -I -d "$current_date + 1 day")
done

CheckDependencyFunction() {
    local date=$1
    local done_file="/mnt/pfs-guan-ssai/nlu/wangxiaoyuan/online-data/${date}/prod/${date}_rule_labeled.csv.gpt_labeled.csv"
    if [ ! -f $done_file ]; then
        PrintLog "$done_file is not ready."
        return 1
    fi

    PrintLog "$done_file is ready."
    return 0
}

ExecuteJobFunction() {
    local date=$1
    model_url="https://rhm-gpt4.fc.chj.cloud/gpt4o/conversation"
    input_file="/mnt/pfs-guan-ssai/nlu/wangxiaoyuan/online-data/${date}/prod/${date}_rule_labeled.csv.gpt_labeled.csv"
    base_output_path="/mnt/pfs-guan-ssai/nlu/renhuimin/data/log_distillation/"
    prompt_path="/mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/conf/generation_prompts/generation_单轮RAG日志蒸馏.txt"

    # 打印开始时间
    begin_time=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[INFO] begin_time: ${begin_time}" 

    # 对日志数据回复糟糕的部分进行蒸馏
    python get_raw_log_distillation.py --is_single_flag --is_rag_flag --model_name 'gpt4o' --model_url ${model_url} --base_output_path ${base_output_path} --prompt_path ${prompt_path} --input_file ${input_file}

    # 打印结束时间
    end_time=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[INFO] end_time: $end_time"

    duration=$(($(date +%s -d "${end_time}")-$(date +%s -d "${begin_time}")));
    echo "[INFO] duration: `expr $duration / 60` min "
}

for date in "${dates[@]}"; do
    echo "[INFO] Processing date: $date"
    CheckDependencyFunction $date
    if [ $? -eq 0 ]; then
        ExecuteJobFunction $date
    else
        echo "[ERROR] Dependency check failed for date: $date"
        continue # 跳过当前日期，继续处理下一个
    fi
done