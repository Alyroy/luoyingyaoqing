#!/bin/bash
CURRENT_DIR=$(cd $(dirname $0); pwd)
cd $CURRENT_DIR

# 参数化输入和输出路径
input_dir=$1
output_dir=$2
model_list_str=$3
url_list_str=$4

# 将字符串转换为数组
IFS=',' read -r -a model_list <<< "$model_list_str"
IFS=',' read -r -a url_list <<< "$url_list_str"

# 每个chunk处理的线程数
thread_num=20
# chunk数
chunk_num=10
# 模型的temperature值
temperature=0.5

# 评估列表，即query obs ans的自定义列名
eval_column_list=("user-query" "observation" "assistant")
save_column=多模型筛选回复结果

# 指标和prompt地址，二者需要同时修改
metric=correct_ans_filter
prompt_path=/mnt/pfs-guan-ssai/nlu/data/renhuimin/pro_rag/conf/filter_prompts/filter_correct.txt

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
    --concat_prompt_flag #默认进入代码后拼接prompt，若提前拼好，此处可设置为 --no_concat_prompt_flag; 提前拼好后默认取eval_column_list[0]为eval_col，如果是qwen打分，会把eval_col改成llm_prompts
