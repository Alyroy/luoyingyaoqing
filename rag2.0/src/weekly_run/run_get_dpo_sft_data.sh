#!/bin/bash

start_date="2024-08-06"
end_date=="2024-08-06"
out_folder="/workspace/renhuimin/pro_rag/data/distillation_data/v20240813_applog/"
dpo_outpt=""
sft_outpt=""
    
python get_dpo_csv_sft_jsonl.py --start_date $start_date --end_date $end_date --out_folder $out_folder --dpo_outpt $dpo_outpt --sft_outpt $sft_outpt