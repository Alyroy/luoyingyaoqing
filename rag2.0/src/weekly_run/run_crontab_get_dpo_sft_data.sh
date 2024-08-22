#!/bin/bash
source ~/.bashrc
conda activate rhm_env

CURRENT_DIR="/mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/weekly_run/"
cd "$CURRENT_DIR"

# Calculate last week's Monday and last week's Sunday.
start_date=$(date -d "last monday - 1 week" +%Y-%m-%d)
end_date=$(date -d "last sunday" +%Y-%m-%d)

in_folder="/mnt/pfs-guan-ssai/nlu/renhuimin/data/log_distillation/"
out_folder_base="/mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/data/routine_label_data/${start_date}_${end_date}/"
source="livis日志回流"

run_python_script() {
    local cate=$1
    local dpo_outpt=$2
    local sft_outpt=$3

    python get_dpo_csv_sft_jsonl.py --start_date "$start_date" --end_date "$end_date" --in_folder "$in_folder" --out_folder "$out_folder_base" --cate "$cate" --dpo_outpt "$dpo_outpt" --sft_outpt "$sft_outpt" --source $source
}

# Run for /raw/single_True_rag_True/
run_python_script "/raw/single_True_rag_True/" "raw_dpo_${start_date}_${end_date}_送标.csv" "raw_sft_${start_date}_${end_date}"

# Run for /extension/single_True_rag_True/
run_python_script "/extension/single_True_rag_True/" "extension_dpo_${start_date}_${end_date}_送标.csv" "extension_sft_${start_date}_${end_date}"