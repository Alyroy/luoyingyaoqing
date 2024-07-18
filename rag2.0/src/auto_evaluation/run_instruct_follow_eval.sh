#!/bin/bash

# 模型名称与url，需一一对应，以空格间隔
# 模型可以为[qwen deepseek autoj gpt4]
model_list=('gpt4') #("qwen2_72b")
url_list=('https://rhm-gpt4.fc.chj.cloud/gpt4_turbo') # 'http://172.24.136.106:8000/v1'

# 评估列表，即query obs ans的自定义列名
eval_column_list=("user-query" "observation" "predict_output")
save_column=指令遵循打分

# 指标和prompt地址，二者需要同时修改
# [authenticity, relevance] 
metric=instruct_follow
prompt_path=/mnt/pfs-guan-ssai/nlu/data/renhuimin/eval_tools/infer_eval/auto_eval/prompts/eval-instruct-follow-prompts.txt

# 输入路径，可以是目录也可以是文件
input_dir=/mnt/pfs-guan-ssai/nlu/data/renhuimin/eval_tools/0_data/0628_模型收益测试/output_instruct_follow/
# 输出路径
output_dir=/mnt/pfs-guan-ssai/nlu/data/renhuimin/eval_tools/0_data/0628_模型收益测试/指令遵循打分/

# 每个chunk处理的线程数
thread_num=20
# chunk数
chunk_num=4
# 模型的temperature值
temperature=0.1

python evaluation.py  \
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
    --temperature $temperature