#!/bin/bash
CURRENT_DIR=$1
INPUT_LOG_COL=$2
EVAL_INPUT_DIR=$3
EVAL_OUTPUT_DIR_REL=$4 # 相关性输出文件夹
gpt4O_url=$5
eval_column_list_str=$6 # 自定义query obs output字段名

cd $CURRENT_DIR

echo ${gpt4O_url}

model_list=("qwen")
url_list=("${gpt4O_url}")

# 评估列表，即query obs ans的自定义列名 or 日志列、输出列、输出列
if [ -n "$eval_column_list_str" ]; then
  IFS=',' read -r -a eval_column_list <<< "$eval_column_list_str"
else
  eval_column_list=("${INPUT_LOG_COL}" "query" "predict_output") #log_col, query_col, output_col
fi
save_column=丰富性打分

# 指标和prompt地址，二者需要同时修改
# [authenticity, relevance] 
metric=richness_test_api_eval
# prompt_path=/mnt/pfs-guan-ssai/nlu/nlu/renhuimin/rag_tool/rag2.0/src/auto_evaluation/prompts/relevance-prompts-resp.txt
prompt_path=/mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/auto_evaluation/prompts/richness-prompts.txt

# prompt拼接方式
# user_obs_ans_concat(输入user-query, observation, assistant列后拼接prompt), model_13b_log(输入13b output 后处理拼接prompt), with_prompt(已拼接好prompt)
eval_mode=model_13b_log
# 每个chunk处理的线程数
thread_num=8
# chunk数
chunk_num=5
# 模型的temperature值
temperature=0
input_text_type='function_call' # 'default' 或 'function_call'，相关性无所谓，只看query和predict_output


python evaluation.py  \
    --model_list "${model_list[@]}" \
    --url_list "${url_list[@]}" \
    --eval_column_list "${eval_column_list[@]}" \
    --save_column $save_column \
    --metric $metric \
    --input_dir $EVAL_INPUT_DIR \
    --output_dir $EVAL_OUTPUT_DIR_REL \
    --prompt_path $prompt_path \
    --thread_num $thread_num \
    --chunk_num $chunk_num \
    --temperature $temperature \
    --eval_mode $eval_mode \
    --input_text_type $input_text_type