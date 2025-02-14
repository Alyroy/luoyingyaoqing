#!/bin/bash
################################################
# 说明
# 1. 推理
# 2. 评估 丰富性、重复性、真实性、相关性
# 3. 统计指标
################################################
# 全局参数设置
CURRENT_DIR=/mnt/pfs-guan-ssai/nlu/jiajuntong/code/rag_tool/rag2/src/auto_evaluation
cd $CURRENT_DIR

# 配置模型参数
declare -a MODEL_CKPT_DIR_LIST=(
    "/mnt/pfs-gv8sxa/nlu/team/renhuimin/checkpoints/mindgpt-pro-v2.5d_rag-72b_20250113_1_v2.5.7"
)

declare -a QUEUE_NAME_LIST=("security")
declare -a JOB_NAME_LIST=("infer-swa")
USER_NAME=jiajuntong

# 输入与输出目录设定
INPUT_DIR=${CURRENT_DIR}'/eval_data/safety_functioncall' # query扩展形式  livis_query_fc_1114/select-new-1231-lvfc_form-2k_final_input.csv1113.out.csv 旧测试集
OUTPUT_DIR=${CURRENT_DIR}'/eval_result'
EVAL_COL=instruction # 无需替换
INPUT_PREDICT_COL=predict_output # 无需替换
# 指标统计
file=safety_functioncall_0116
auth_metric=authenticity # 无需替换
rel_metric=relevance # 无需替换
# 推理参数
HF_MODEL_CKPT_DIR_SUFFIX=''
temperature=0.7
top_p=0.95
repetition=1

# 评估丰富度、重复度时需要设置
gpt4O_url_list=("xxx") # 你自己的url
eval_richness_name='richness_eval_v1' # richness_eval_1 旧版丰富度, richness_eval 新版丰富度（指标会提升10PP）
eval_repeat_name='repeat_eval' # repeat_eval 重复评估 
# 存储所有后台任务的PID
pids=()

# 遍历所有模型进行推理和评估
for i in "${!MODEL_CKPT_DIR_LIST[@]}"; do
    MODEL_CKPT_DIR=${MODEL_CKPT_DIR_LIST[$i]}
    QUEUE_NAME=${QUEUE_NAME_LIST[$i]}
    JOB_NAME=${JOB_NAME_LIST[$i]}
    gpt4O_url=${gpt4O_url_list[$i]}
    
    HF_MODEL_CKPT_DIR=${MODEL_CKPT_DIR}${HF_MODEL_CKPT_DIR_SUFFIX}
    
    # 创建日志文件
    LOG_FILE=${OUTPUT_DIR}/${JOB_NAME}/"log.${JOB_NAME}"

    (
        # # 0. 判断模型是否训练完成
        # cnt=1
        # start_time=$(date +%s)
        # echo "${MODEL_CKPT_DIR}/config.json"
        # until [ -f "${MODEL_CKPT_DIR}/config.json" ]; do
        #     echo "等待 ${MODEL_CKPT_DIR} 模型训练完毕...已等待 ${cnt} 小时" | tee -a $LOG_FILE
        #     sleep 3600
        #     cnt=$(expr $cnt + 1)
        #     current_time=$(date +%s)
        #     elapsed_time=$(expr $current_time - $start_time)
        #     if [ $elapsed_time -gt 86400 ]; then
        #         echo "等待超过24小时，停止任务." | tee -a $LOG_FILE
        #         exit 1
        #     fi
        # done

        # 1. 启动推理脚本
        echo "Starting inference for job: ${JOB_NAME}" | tee -a $LOG_FILE
        echo sh auto_lizrun_guan_livis_qwen_swa.sh ${CURRENT_DIR} ${JOB_NAME} ${QUEUE_NAME} 1 ${HF_MODEL_CKPT_DIR} ${JOB_NAME} ${INPUT_DIR} ${OUTPUT_DIR} ${EVAL_COL} ${temperature} ${top_p} ${repetition} | tee -a $LOG_FILE
        sh auto_lizrun_guan_livis_qwen_swa.sh ${CURRENT_DIR} ${JOB_NAME} ${QUEUE_NAME} 1 ${HF_MODEL_CKPT_DIR} ${JOB_NAME} ${INPUT_DIR} ${OUTPUT_DIR} ${EVAL_COL} ${temperature} ${top_p} ${repetition} | tee -a $LOG_FILE

        # 2. 判断推理是否完成
        # cnt=1
        # start_time=$(date +%s)
        # until [ -f "${OUTPUT_DIR}/${JOB_NAME}/.done" ]; do
        #     echo "等待${JOB_NAME}推理任务结束...${cnt}*10分钟" | tee -a $LOG_FILE
        #     sleep 600
        #     cnt=$(expr $cnt + 1)
        #     current_time=$(date +%s)
        #     elapsed_time=$(expr $current_time - $start_time)
        #     if [ $elapsed_time -gt 86400 ]; then
        #         echo "等待超过24小时，停止任务." | tee -a $LOG_FILE
        #         exit 1
        #     fi
        # done

        # 并行执行的步骤
        # {
            # # 丰富度任务评估 
            # INPUT_LOG_COL=${EVAL_COL}
            # EVAL_INPUT_DIR=${OUTPUT_DIR}/${JOB_NAME}/
            # EVAL_OUTPUT_DIR_RICH=${OUTPUT_DIR}/${JOB_NAME}/丰富性打分/
            # echo bash auto_eval_log_richness_test_api.sh ${CURRENT_DIR} ${INPUT_LOG_COL} ${EVAL_INPUT_DIR} ${EVAL_OUTPUT_DIR_RICH} ${gpt4O_url} ${eval_richness_name} | tee -a $LOG_FILE
            # bash auto_eval_log_richness_test_api.sh ${CURRENT_DIR} ${INPUT_LOG_COL} ${EVAL_INPUT_DIR} ${EVAL_OUTPUT_DIR_RICH} ${gpt4O_url} ${eval_richness_name} | tee -a $LOG_FILE &
        
            # # 重复度任务评估 
            # INPUT_LOG_COL=${EVAL_COL}
            # EVAL_INPUT_DIR=${OUTPUT_DIR}/${JOB_NAME}/
            # EVAL_OUTPUT_DIR_REPEAT=${OUTPUT_DIR}/${JOB_NAME}/重复性打分/
            # echo bash auto_eval_log_richness_test_api.sh ${CURRENT_DIR} ${INPUT_LOG_COL} ${EVAL_INPUT_DIR} ${EVAL_OUTPUT_DIR_REPEAT} ${gpt4O_url} ${eval_repeat_name} | tee -a $LOG_FILE
            # bash auto_eval_log_richness_test_api.sh ${CURRENT_DIR} ${INPUT_LOG_COL} ${EVAL_INPUT_DIR} ${EVAL_OUTPUT_DIR_REPEAT} ${gpt4O_url} ${eval_repeat_name} | tee -a $LOG_FILE &
            
            # Wait for both background processes to finish before exiting the block
        #     wait
        # } &

        # 3. 启动qwen api
        # (
        #     stamp=$(date +%Y%m%d-%H%M%S)
        #     QWEN_JOBNAME1=qwenapi-${JOB_NAME}-${stamp}
        #     echo sh api_service/lizrun_api_qwen_auto.sh ${QWEN_JOBNAME1} ${QUEUE_NAME} ${CURRENT_DIR} | tee -a $LOG_FILE
        #     sh api_service/lizrun_api_qwen_auto.sh ${QWEN_JOBNAME1} ${QUEUE_NAME} ${CURRENT_DIR} | tee -a $LOG_FILE
        #     sleep 120
            
        #     # 获取qwen api的IP
        #     cnt=1
        #     start_time=$(date +%s)
        #     IP1=`lizrun pool get -p ${QUEUE_NAME} -d | grep Running | grep ${QWEN_JOBNAME1}-${USER_NAME} | awk -F " " '{print($2)}'`
        #     until [ ${IP1} ]; do
        #         echo "存在IP1的qwen-api任务未就位！！！" | tee -a $LOG_FILE
        #         IP1=`lizrun pool get -p ${QUEUE_NAME} -d | grep Running | grep ${QWEN_JOBNAME1}-${USER_NAME} | awk -F " " '{print($2)}'`
        #         sleep 120
        #         cnt=$(expr $cnt + 1)
        #         current_time=$(date +%s)
        #         elapsed_time=$(expr $current_time - $start_time)
        #         if [ $elapsed_time -gt 86400 ]; then
        #             echo "等待超过24小时，停止任务." | tee -a $LOG_FILE
        #             exit 1
        #         fi
        #     done
        #     echo "IP1的qwen-api任务已就位！！！"`date` | tee -a $LOG_FILE
        #     IP1=`echo ${IP1} | awk '{gsub("-", ".", $0); print $0}'`
        #     echo "IP1:"${IP1} | tee -a $LOG_FILE
            
        #     # 4. 评估模型
        #     echo "开始评估已启动" | tee -a $LOG_FILE
        #     INPUT_LOG_COL=${EVAL_COL}
        #     EVAL_INPUT_DIR=${OUTPUT_DIR}/${JOB_NAME}/
        #     EVAL_OUTPUT_DIR_REL=${OUTPUT_DIR}/${JOB_NAME}/相关性打分/
        #     EVAL_OUTPUT_DIR_AUTH=${OUTPUT_DIR}/${JOB_NAME}/真实性打分/

        #     # echo sh auto_eval_rel_auth_test_api.sh ${CURRENT_DIR} ${INPUT_LOG_COL} ${EVAL_INPUT_DIR} ${EVAL_OUTPUT_DIR_REL} ${EVAL_OUTPUT_DIR_AUTH} ${IP1} | tee -a $LOG_FILE
        #     # sh auto_eval_rel_auth_test_api.sh ${CURRENT_DIR} ${INPUT_LOG_COL} ${EVAL_INPUT_DIR} ${EVAL_OUTPUT_DIR_REL} ${EVAL_OUTPUT_DIR_AUTH} ${IP1} | tee -a $LOG_FILE
        #     echo bash auto_eval_rel_auth_test_api_func_call.sh ${CURRENT_DIR} ${INPUT_LOG_COL} ${INPUT_PREDICT_COL} ${EVAL_INPUT_DIR} ${EVAL_OUTPUT_DIR_REL} ${EVAL_OUTPUT_DIR_AUTH} ${IP1}
        #     bash auto_eval_rel_auth_test_api_func_call.sh ${CURRENT_DIR} ${INPUT_LOG_COL} ${INPUT_PREDICT_COL} ${EVAL_INPUT_DIR} ${EVAL_OUTPUT_DIR_REL} ${EVAL_OUTPUT_DIR_AUTH} ${IP1}
            
        #     # 停止qwen api任务
        #     lizrun stop ${QWEN_JOBNAME1}-${USER_NAME}
        # ) &

        # 等待并行任务完成
        #wait

        # 5. 统计所有分数
        # echo "计算所有分数" | tee -a $LOG_FILE
        # EVAL_INPUT_DIR=${OUTPUT_DIR}/${JOB_NAME}/
        # echo sh auto_cal_e2e_score_kimi.sh ${CURRENT_DIR} ${EVAL_INPUT_DIR} ${JOB_NAME} ${file} ${auth_metric} ${rel_metric} | tee -a $LOG_FILE
        # sh auto_cal_e2e_score_kimi.sh ${CURRENT_DIR} ${EVAL_INPUT_DIR} ${JOB_NAME} ${file} ${auth_metric} ${rel_metric} | tee -a $LOG_FILE

    ) &  # 在后台执行

    # 保存PID
    pids+=($!)

    # 提交任务之间等待1分钟
    sleep 60
done

# 等待所有后台进程完成
for pid in "${pids[@]}"; do
    wait $pid
done

echo "所有模型推理和评估任务已完成。"
