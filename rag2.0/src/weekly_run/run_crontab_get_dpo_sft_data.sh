#!/bin/bash
source ~/.bashrc
conda activate rhm_env

CURRENT_DIR="/mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/weekly_run/"
cd "$CURRENT_DIR"

# Calculate last week's Monday and last week's Sunday.
start_date=$(date -d "last monday - 1 week" +%Y-%m-%d)
end_date=$(date -d "last sunday" +%Y-%m-%d)
# start_date="2024-08-13"
# end_date="2024-08-18"

in_folder="/mnt/pfs-guan-ssai/nlu/renhuimin/data/log_distillation/"
out_folder_base="/mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/data/routine_label_data/${start_date}_${end_date}/"
source="livis日志回流"

run_python_script() {
    local raw_extension_type=$1
    local single_rag_type=$2
    local dpo_outpt=$3
    local sft_outpt=$4

    python get_dpo_csv_sft_jsonl.py --start_date "$start_date" --end_date "$end_date" --in_folder "$in_folder" --out_folder "$out_folder_base" --dpo_outpt "$dpo_outpt" --sft_outpt "$sft_outpt" --source $source --raw_extension_type $raw_extension_type --single_rag_type $single_rag_type
}

# Run for /raw/single_True_rag_True/
run_python_script "raw" "single_True_rag_True" "raw_dpo_${start_date}_${end_date}_送标" "raw_sft_${start_date}_${end_date}"

# Run for /extension/single_True_rag_True/
run_python_script "extension" "single_True_rag_True" "extension_dpo_${start_date}_${end_date}_送标" "extension_sft_${start_date}_${end_date}"