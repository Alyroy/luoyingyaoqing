#!/bin/bash
CURRENT_DIR=$1
INPUT_LOG_COL=$2
INPUT_PREDICT_COL=$3
EVAL_INPUT_DIR=$4
EVAL_OUTPUT_DIR_REL=$5 # 相关性输出文件夹
IP=$6

cd $CURRENT_DIR

QWEN_URL=http://${IP}:8012/v1
echo ${QWEN_URL}

model_list=("qwen")
url_list=("${QWEN_URL}")

# 评估列表，即query obs ans的自定义列名 or 日志列、输出列、输出列
eval_column_list=("${INPUT_LOG_COL}" "query" "${INPUT_PREDICT_COL}") #log_col, query_col, output_col
save_column=相关性打分

# 指标和prompt地址，二者需要同时修改
# [authenticity, relevance] 
metric=relevance
# prompt_path=/mnt/pfs-guan-ssai/nlu/nlu/renhuimin/rag_tool/rag2.0/src/auto_evaluation/prompts/relevance-prompts-resp.txt
prompt_path=/mnt/pfs-guan-ssai/nlu/chihuixuan/rag_tool/rag2/src/auto_evaluation/prompts/chx/common-relevance.txt

# prompt拼接方式
# user_obs_ans_concat(输入user-query, observation, assistant列后拼接prompt), model_13b_log(输入13b output 后处理拼接prompt), with_prompt(已拼接好prompt)
eval_mode=model_13b_log
# 每个chunk处理的线程数
thread_num=20
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
    --max_tokens 4000 \
    --input_text_type $input_text_type