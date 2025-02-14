#!/bin/bash
# 3. 启动qwen api
CURRENT_DIR=/mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/auto_evaluation/
cd $CURRENT_DIR

stamp=$(date +%Y-%m-%d-%H-%M-%S)
QUEUE_NAME=wq-sft
QWEN_JOBNAME1=qwenrel-single
echo sh api_service/lizrun_api_qwen_auto.sh ${QWEN_JOBNAME1} ${QUEUE_NAME} ${CURRENT_DIR} | tee -a $LOG_FILE
sh api_service/lizrun_api_qwen_auto.sh ${QWEN_JOBNAME1} ${QUEUE_NAME} ${CURRENT_DIR} | tee -a $LOG_FILE
sleep 120

# 获取qwen api的IP
cnt=1
USER_NAME=renhuimin
IP1=`lizrun pool get -p ${QUEUE_NAME} -d | grep Running | grep ${QWEN_JOBNAME1}-${USER_NAME} | awk -F " " '{print($2)}'`
until [ ${IP1} ]; do
    echo "存在IP1的qwen-api任务未就位！！！" | tee -a $LOG_FILE
    IP1=`lizrun pool get -p ${QUEUE_NAME} -d | grep Running | grep ${QWEN_JOBNAME1}-${USER_NAME} | awk -F " " '{print($2)}'`
    sleep 600
    cnt=$(expr $cnt + 1)
done
echo "IP1的qwen-api任务已就位！！！"`date` | tee -a $LOG_FILE
IP1=`echo ${IP1} | awk '{gsub("-", ".", $0); print $0}'`
echo "IP1:"${IP1} | tee -a $LOG_FILE

        
INPUT_LOG_COL='model_13b_input'
EVAL_INPUT_DIRS=('/mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/data/test_data/rag_exp_test/eval_results/1111_implementation_fc/lf-14b-1111-code8k-ep2/手机APP_泛化集_2024-08-23T21_34_46.379.csv_fc.csv.lf-14b-1111-code8k-ep2.csv','/mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/data/test_data/rag_exp_test/eval_results/1111_implementation_fc/lf-14b-1111-code8k-ep3/APP同分布开发集16b评估-0821.csv_fc.csv.lf-14b-1111-code8k-ep3.csv')
EVAL_OUTPUT_DIR_RELS=('/mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/data/test_data/rag_exp_test/eval_results/1111_implementation_fc/lf-14b-1111-code8k-ep2/真实性打分/','/mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/data/test_data/rag_exp_test/eval_results/1111_implementation_fc/lf-14b-1111-code8k-ep3/真实性打分/')

QWEN_URL=http://${IP1}:8012/v1
echo ${QWEN_URL}
model_list=("qwen")
url_list=("${QWEN_URL}")
# 评估列表，即query obs ans的自定义列名 or 日志列、输出列、输出列
eval_column_list=("${INPUT_LOG_COL}" "query" "predict_output") #log_col, query_col, output_col
save_column=相关性打分

# 指标和prompt地址，二者需要同时修改
# [authenticity, relevance] 
metric=relevance_test_api_eval
# prompt_path=/mnt/pfs-guan-ssai/nlu/nlu/renhuimin/rag_tool/rag2.0/src/auto_evaluation/prompts/relevance-prompts-resp.txt
prompt_path=/mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/auto_evaluation/prompts/common-relevance.txt

# prompt拼接方式
# user_obs_ans_concat(输入user-query, observation, assistant列后拼接prompt), model_13b_log(输入13b output 后处理拼接prompt), with_prompt(已拼接好prompt)
eval_mode=model_13b_log
# 每个chunk处理的线程数
thread_num=10
# chunk数
chunk_num=5
# 模型的temperature值
temperature=0
input_text_type='function_call' # 'default' 或 'function_call'，相关性无所谓，只看query和predict_output

for i in "${!MODEL_CKPT_DIR_LIST[@]}"; do
    EVAL_INPUT_DIR=${EVAL_INPUT_DIRS[$i]}
    EVAL_OUTPUT_DIR_REL=${EVAL_OUTPUT_DIR_RELS[$i]}
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
done
lizrun stop ${QWEN_JOBNAME1}-${USER_NAME}