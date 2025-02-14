#!/bin/bash
#nohup ./hf-72B-begin.sh >hf-m13000-s1debug.eval 2>&1 &
# 全局参数设置
CURRENT_DIR=$(cd $(dirname $0); pwd) # 必须在mnt/下
cd $CURRENT_DIR 

# 配置模型参数
declare -a MODEL_CKPT_DIR_LIST=(
    "/mnt/pfs-mc0p4k/nlu/team/dingyifeng/nlp/qwen_cpt/sft/output/MindGPT2.5D-baseon13500steps/checkpoint-7497/hf_model"
)

declare -a QUEUE_NAME_LIST=("base-130")
declare -a JOB_NAME_LIST=("test-git")
USER_NAME=xxx

# 输入与输出目录设定
INPUT_DIR=${CURRENT_DIR}'/eval_data/livis_query_fc_1213/' # query扩展形式
OUTPUT_DIR=${CURRENT_DIR}'/eval_results/'
EVAL_COL=livis_model_input
HF_MODEL_CKPT_DIR_SUFFIX=''
temperature=0.7 #0.7 # 0.9
top_p=0.8 #0.8 # 0.95
repetition=1.05 #1.05 # 1

# 评估丰富度时需要设置
gpt4O_url=xxx

# 存储所有后台任务的PID
pids=()

# 遍历所有模型进行推理和评估
for i in "${!MODEL_CKPT_DIR_LIST[@]}"; do
    MODEL_CKPT_DIR=${MODEL_CKPT_DIR_LIST[$i]}
    QUEUE_NAME=${QUEUE_NAME_LIST[$i]}
    JOB_NAME=${JOB_NAME_LIST[$i]}
    
    HF_MODEL_CKPT_DIR=${MODEL_CKPT_DIR}${HF_MODEL_CKPT_DIR_SUFFIX}
    
    # 创建日志文件
    LOG_FILE="log.${JOB_NAME}"

    (
        # 0. 判断模型是否训练完成
        # cnt=1
        # echo "${MODEL_CKPT_DIR}/config.json"
        # until [ -f "${MODEL_CKPT_DIR}/config.json" ]; do
        #     echo "等待 ${MODEL_CKPT_DIR} 模型训练完毕...已等待 ${cnt} 小时" | tee -a $LOG_FILE
        #     sleep 3600
        #     cnt=$(expr $cnt + 1)
        # done


        # 1. 启动推理脚本
        echo "Starting inference for job: ${JOB_NAME}" | tee -a $LOG_FILE
        echo sh hf-72B-middle.sh ${CURRENT_DIR} ${JOB_NAME} ${QUEUE_NAME} 1 ${HF_MODEL_CKPT_DIR} ${JOB_NAME} ${INPUT_DIR} ${OUTPUT_DIR} ${EVAL_COL} ${temperature} ${top_p} ${repetition} | tee -a $LOG_FILE
        sh hf-72B-middle.sh ${CURRENT_DIR} ${JOB_NAME} ${QUEUE_NAME} 1 ${HF_MODEL_CKPT_DIR} ${JOB_NAME} ${INPUT_DIR} ${OUTPUT_DIR} ${EVAL_COL} ${temperature} ${top_p} ${repetition} | tee -a $LOG_FILE

        # 2. 判断推理是否完成
        cnt=1
        until [ -f "${OUTPUT_DIR}/${JOB_NAME}/.done" ]; do
            echo "等待${JOB_NAME}推理任务结束...${cnt}*10分钟" | tee -a $LOG_FILE
            sleep 600
            cnt=$(expr $cnt + 1)
        done

        # 3. 启动qwen api
        stamp=$(date +%Y-%m-%d-%H-%M-%S)
        QWEN_JOBNAME1=qwenapi-${JOB_NAME}
        echo sh api_service/lizrun_api_qwen_auto.sh ${QWEN_JOBNAME1} ${QUEUE_NAME} ${CURRENT_DIR} | tee -a $LOG_FILE
        sh api_service/lizrun_api_qwen_auto.sh ${QWEN_JOBNAME1} ${QUEUE_NAME} ${CURRENT_DIR} | tee -a $LOG_FILE
        sleep 300
        
        # 获取qwen api的IP
        cnt=1
        IP1=`lizrun pool get -p ${QUEUE_NAME} -d | grep Running | grep ${QWEN_JOBNAME1}-${USER_NAME} | awk -F " " '{print($2)}'`
        until [ ${IP1} ]; do
            echo "存在IP1的qwen-api任务未就位！！！" | tee -a $LOG_FILE
            IP1=`lizrun pool get -p ${QUEUE_NAME} -d | grep Running | grep ${QWEN_JOBNAME1}-${USER_NAME} | awk -F " " '{print($2)}'`
            sleep 120
            cnt=$(expr $cnt + 1)
        done
        echo "IP1的qwen-api任务已就位！！！"`date` | tee -a $LOG_FILE
        IP1=`echo ${IP1} | awk '{gsub("-", ".", $0); print $0}'`
        echo "IP1:"${IP1} | tee -a $LOG_FILE
        
        # 4. 评估模型
        echo "开始评估已启动" | tee -a $LOG_FILE
        INPUT_LOG_COL=${EVAL_COL}
        EVAL_INPUT_DIR=${OUTPUT_DIR}/${JOB_NAME}/
        EVAL_OUTPUT_DIR_REL=${OUTPUT_DIR}/${JOB_NAME}/相关性打分/
        EVAL_OUTPUT_DIR_AUTH=${OUTPUT_DIR}/${JOB_NAME}/真实性打分/

        echo sh auto_eval_rel_auth_test_api.sh ${CURRENT_DIR} ${INPUT_LOG_COL} ${EVAL_INPUT_DIR} ${EVAL_OUTPUT_DIR_REL} ${EVAL_OUTPUT_DIR_AUTH} ${IP1} | tee -a $LOG_FILE
        sh auto_eval_rel_auth_test_api.sh ${CURRENT_DIR} ${INPUT_LOG_COL} ${EVAL_INPUT_DIR} ${EVAL_OUTPUT_DIR_REL} ${EVAL_OUTPUT_DIR_AUTH} ${IP1} | tee -a $LOG_FILE

        # 丰富度任务评估 
        # INPUT_LOG_COL=${EVAL_COL}
        # EVAL_INPUT_DIR=${OUTPUT_DIR}/${JOB_NAME}/
        # EVAL_OUTPUT_DIR_RICH=${OUTPUT_DIR}/${JOB_NAME}/丰富性打分/
        # echo bash auto_eval_log_richness_test_api.sh ${CURRENT_DIR} ${INPUT_LOG_COL} ${EVAL_INPUT_DIR} ${EVAL_OUTPUT_DIR_RICH} ${gpt4O_url} | tee -a $LOG_FILE
        # bash auto_eval_log_richness_test_api.sh ${CURRENT_DIR} ${INPUT_LOG_COL} ${EVAL_INPUT_DIR} ${EVAL_OUTPUT_DIR_RICH} ${gpt4O_url} | tee -a $LOG_FILE

        # 停止qwen api任务
        lizrun stop ${QWEN_JOBNAME1}-${USER_NAME}

        # 5. 统计所有分数
        echo "计算所有分数" | tee -a $LOG_FILE
        EVAL_INPUT_DIR=${OUTPUT_DIR}/${JOB_NAME}/
        echo sh auto_cal_e2e_score_kimi.sh ${CURRENT_DIR} ${EVAL_INPUT_DIR} ${JOB_NAME} | tee -a $LOG_FILE
        sh auto_cal_e2e_score_kimi.sh ${CURRENT_DIR} ${EVAL_INPUT_DIR} ${JOB_NAME} | tee -a $LOG_FILE
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