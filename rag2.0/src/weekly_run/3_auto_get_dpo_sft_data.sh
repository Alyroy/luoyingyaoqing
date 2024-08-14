#!/bin/bash

start_date=$1
end_date=$2
out_folder=$3
dpo_outpt=$4
sft_outpt=$5
    
python get_dpo_csv_sft_jsonl.py --start_date $start_date --end_date $end_date --out_folder $out_folder --dpo_outpt $dpo_outpt --sft_outpt $sft_outpt