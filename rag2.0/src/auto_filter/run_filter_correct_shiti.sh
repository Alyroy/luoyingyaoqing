#!/bin/bash

# 模型名称与url，需一一对应，以空格间隔
# 模型可以为 ['qwen2_72b', 'qwen1.5_72b', 'qwen1.5_110b', 'autoj', 'deepseek', 'gpt4', 'mindgpt','wenxin']
model_list=("gpt4o" "qwen")
url_list=("https://rhm-gpt4.fc.chj.cloud/gpt4o" "http://172.24.136.6:7000/v1")

# 每个chunk处理的线程数
thread_num=20
# chunk数
chunk_num=5
# 模型的temperature值
temperature=0.1

# 评估列表，即query obs ans的自定义列名
eval_column_list=("user-query" "observation" "parser_gpt4")
save_column=多模型筛选回复结果

# 指标和prompt地址，二者需要同时修改
metric=correct_ans_filter
prompt_path=/workspace/renhuimin/pro_rag/conf/filter_prompts/filter_correct.txt

# 输入路径，可以是目录也可以是文件
input_dir=/workspace/renhuimin/pro_rag/data/distillation_data/v20240724/gpt4数据蒸馏/实体泛化query
# 输出路径
output_dir=/workspace/renhuimin/pro_rag/data/distillation_data/v20240724/correct_filter_output_实体query泛化/

python multi_filter_assistant.py  \
    --model_list "${model_list[@]}" \
    --url_list "${url_list[@]}" \
    --eval_column_list "${eval_column_list[@]}" \
    --save_column $save_column \
    --metric $metric \
    --input_dir $input_dir \
    --output_dir $output_dir \
    --prompt_path $prompt_path \
    --thread_num $thread_num \
    --chunk_num $chunk_num \
    --temperature $temperature \
    --concat_prompt_flag