#!/bin/bash
source ~/.bashrc
conda activate rhm_env

# CURRENT_DIR=$(cd $(dirname $0); pwd)
CURRENT_DIR=/mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/weekly_run/
cd $CURRENT_DIR
source "common/periodical_job_template.sh"
JOB_ROOT_DIR=${CURRENT_DIR}
source "common/common_scripts.sh"


JOB_NAME="app_log_distillation_report"

INFOLDER="/mnt/pfs-guan-ssai/nlu/renhuimin/data/log_distillation/"
SINGLE_RAG_TYPE="single_True_rag_True"

# current_date=$(date "+%Y-%m-%d")
# TARGET_DATE=$(date -d "${current_date} -2 day" "+%Y-%m-%d")
# TARGET_DATE="2024-08-14"
dates=("2024-08-13" "2024-08-14" "2024-08-15" "2024-08-16" "2024-08-17" "2024-08-18" "2024-08-19" "2024-08-20")

CheckDependencyFunction() {
    local log_type=$1
    local single_rag_type=$2
    local TARGET_DATE=$3
    local done_file="${INFOLDER}/${TARGET_DATE}/${log_type}/${single_rag_type}/correct_filter_output/cd .done"
    if [ ! -f $done_file ]; then
        PrintLog "$done_file is not ready."
        exit 1
    fi

    PrintLog "$done_file is ready."
}

# 定义运行报告的函数
ExecuteJobFunction() {
    local log_type=$1
    local single_rag_type=$2
    local TARGET_DATE=$3
    python report.py --target_date $TARGET_DATE --log_type $log_type --infolder $INFOLDER --single_rag_type $single_rag_type
}


for date in "${dates[@]}"; do
    echo "[INFO] Processing date: $date"
    CheckDependencyFunction "raw" $SINGLE_RAG_TYPE $date
    if [ $? -eq 0 ]; then
        ExecuteJobFunction "raw" $SINGLE_RAG_TYPE $date
    else
        echo "[ERROR] Dependency check failed for date: $date"
    fi

    CheckDependencyFunction "extension" $SINGLE_RAG_TYPE $date
    if [ $? -eq 0 ]; then
        ExecuteJobFunction "extension" $SINGLE_RAG_TYPE $date
    else
        echo "[ERROR] Dependency check failed for date: $date"
    fi
done