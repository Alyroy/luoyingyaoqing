#!/bin/bash
CURRENT_DIR=$(cd $(dirname $0); pwd)
cd $CURRENT_DIR
cd ../

# 模型名称与url，需一一对应，以空格间隔
# 模型可以为[qwen deepseek autoj gpt4]
# model_list=('qwen2_72b') #("qwen2_72b")
# url_list=('http://172.24.136.106:8000/v1') #("http://172.24.136.236:8000/v1")
model_list=('gpt4o') #("qwen2_72b")
# url_list=('https://rhm-gpt4.fc.chj.cloud/gpt4o') #("http://172.24.136.236:8000/v1")
url_list=('https://gongwuxuan-llm-test.fc.chj.cloud/gpt4o/conversation')

# 评估列表，即query obs ans的自定义列名 or 日志列、输出列、输出列
eval_column_list=("resp中间结果" "predict_output" "predict_output") #为适配格式，需要输入3个列，可重复，最后一个为output列
save_column=相关性打分

# 指标和prompt地址，二者需要同时修改
# [authenticity, relevance] 
metric=relevance
# prompt_path=/mnt/pfs-guan-ssai/nlu/nlu/renhuimin/rag_tool/rag2.0/src/auto_evaluation/prompts/relevance-prompts-resp.txt
prompt_path=/mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/rag2.0/src/auto_evaluation/prompts/common-relevance.txt

# prompt拼接方式
# user_obs_ans_concat(输入user-query, observation, assistant列后拼接prompt), model_13b_log(输入13b output 后处理拼接prompt), with_prompt(已拼接好prompt)
eval_mode=model_13b_log
# 每个chunk处理的线程数
thread_num=5
# chunk数
chunk_num=2
# 模型的temperature值
temperature=0.1

# 输入路径，可以是目录也可以是文件
input_dir=/mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/rag2.0/data/test_data/scb_0804_selected/output/
# 输出路径
output_dir=/mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/rag2.0/data/test_data/scb_0804_selected/相关性打分_gpt4_0804_dpo/

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
    --temperature $temperature \
    --eval_mode $eval_mode