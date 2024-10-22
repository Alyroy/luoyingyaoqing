#!/bin/bash

# 全局参数设置
CURRENT_DIR=/mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/auto_evaluation/
cd $CURRENT_DIR

# 配置模型参数
declare -a MODEL_CKPT_DIR_LIST=(
    "/mnt/pfs-guan-ssai/nlu/lizr/renhuimin/checkpoints/mindgpt21_14b_stage1_train20240903_norag_144w_model1"
    "/mnt/pfs-guan-ssai/nlu/lizr/renhuimin/checkpoints/mindgpt21_14b_stage1_train20241017_norag_216w_model1"
    "/mnt/pfs-guan-ssai/nlu/lizr/renhuimin/checkpoints/mindgpt21_14b_stage1_20240903_stage2_train20241017_rag_30w_model1"
    "/mnt/pfs-guan-ssai/nlu/lizr/renhuimin/checkpoints/mindgpt21_14b_stage1_20241017_stage2_train20241017_rag_30w_model1"
    
    # 可以在这里添加更多的模型路径
)

declare -a QUEUE_NAME_LIST=("base" "base" "base" "base")  # 根据需要添加更多队列名
declare -a JOB_NAME_LIST=("qwen0903stage1" "qwen1017stage1" "qwen0903stage2" "qwen1017stage2")  # 根据需要添加更多任务名

# 训练模型转格式脚本路径
HF_MODEL_CKPT_DIR_SUFFIX=""

# 输入与输出目录设定
INPUT_DIR='/mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/data/test_data/rag_exp_test/input_data/'
OUTPUT_DIR="/mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/data/test_data/rag_exp_test/eval_results/1025_implementation/"
EVAL_COL=model_13b_input  # 评估列名，16b模型input，同日志格式

# 存储所有后台任务的PID
pids=()

# 遍历所有模型进行推理和评估
for i in "${!MODEL_CKPT_DIR_LIST[@]}"; do
    MODEL_CKPT_DIR=${MODEL_CKPT_DIR_LIST[$i]}
    QUEUE_NAME=${QUEUE_NAME_LIST[$i]}
    JOB_NAME=${JOB_NAME_LIST[$i]}
    
    HF_MODEL_CKPT_DIR=${MODEL_CKPT_DIR}${HF_MODEL_CKPT_DIR_SUFFIX}
    (
        # 0. 判断模型是否训练完成
        cnt=1
        echo "${MODEL_CKPT_DIR}/config.json"
        until [ -f "${MODEL_CKPT_DIR}/config.json" ]; do
            echo "等待 ${MODEL_CKPT_DIR} 模型训练完毕...已等待 ${cnt} 小时"
            sleep 3600
            cnt=$(expr $cnt + 1)
        done

        # 1. 启动推理脚本
        echo "Starting inference for job: ${JOB_NAME}"
        echo sh auto_lizrun_guan_livis_qwen.sh ${CURRENT_DIR} ${JOB_NAME} ${QUEUE_NAME} 1 ${HF_MODEL_CKPT_DIR} ${JOB_NAME} ${INPUT_DIR} ${OUTPUT_DIR} ${EVAL_COL}
        sh auto_lizrun_guan_livis_qwen.sh ${CURRENT_DIR} ${JOB_NAME} ${QUEUE_NAME} 1 ${HF_MODEL_CKPT_DIR} ${JOB_NAME} ${INPUT_DIR} ${OUTPUT_DIR} ${EVAL_COL}

        # 2. 判断推理是否完成
        cnt=1
        until [ -f "${OUTPUT_DIR}/${JOB_NAME}/.done" ]; do
            echo "等待${JOB_NAME}推理任务结束...${cnt}*10分钟"
            sleep 600
            cnt=$(expr $cnt + 1)
        done

        # 3. 启动qwen api
        stamp=$(date +%Y-%m-%d-%H-%M-%S)
        QWEN_JOBNAME1=qwen-api1-${JOB_NAME}
        echo sh api_service/lizrun_api_qwen_auto.sh ${QWEN_JOBNAME1} ${QUEUE_NAME} ${CURRENT_DIR}
        sh api_service/lizrun_api_qwen_auto.sh ${QWEN_JOBNAME1} ${QUEUE_NAME} ${CURRENT_DIR}
        sleep 120
        
        # 获取qwen api的IP
        cnt=1
        USER_NAME=renhuimin
        IP1=`lizrun pool get -p ${QUEUE_NAME} -d | grep Running | grep ${QWEN_JOBNAME1}-${USER_NAME} | awk -F " " '{print($2)}'`
        until [ ${IP1} ]; do
            echo "存在IP1的qwen-api任务未就位！！！"
            IP1=`lizrun pool get -p ${QUEUE_NAME} -d | grep Running | grep ${QWEN_JOBNAME1}-${USER_NAME} | awk -F " " '{print($2)}'`
            sleep 300
            cnt=$(expr $cnt + 1)
        done
        echo "IP1的qwen-api任务已就位！！！"`date`
        IP1=`echo ${IP1} | awk '{gsub("-", ".", $0); print $0}'`
        echo "IP1:"${IP1}
        
        # 4. 评估模型
        echo "开始评估已启动"
        INPUT_LOG_COL=model_13b_input
        EVAL_INPUT_DIR=${OUTPUT_DIR}/${JOB_NAME}/  # 模型推理结果文件夹
        EVAL_OUTPUT_DIR_REL=${OUTPUT_DIR}/${JOB_NAME}/相关性打分/
        EVAL_OUTPUT_DIR_AUTH=${OUTPUT_DIR}/${JOB_NAME}/真实性打分/
        echo sh auto_eval_rel_auth_test_api.sh ${CURRENT_DIR} ${INPUT_LOG_COL} ${EVAL_INPUT_DIR} ${EVAL_OUTPUT_DIR_REL} ${EVAL_OUTPUT_DIR_AUTH} ${IP1}
        sh auto_eval_rel_auth_test_api.sh ${CURRENT_DIR} ${INPUT_LOG_COL} ${EVAL_INPUT_DIR} ${EVAL_OUTPUT_DIR_REL} ${EVAL_OUTPUT_DIR_AUTH} ${IP1}

        # 停止qwen api任务
        lizrun stop ${QWEN_JOBNAME1}-${USER_NAME}

        # 5. 统计所有分数
        echo "计算所有分数"
        EVAL_INPUT_DIR=${OUTPUT_DIR}/${JOB_NAME}/  # 模型推理结果文件夹
        echo sh auto_cal_e2e_score.sh ${CURRENT_DIR} ${EVAL_INPUT_DIR} ${JOB_NAME}
        sh auto_cal_e2e_score.sh ${CURRENT_DIR} ${EVAL_INPUT_DIR} ${JOB_NAME}
    ) &  # 在后台执行

    # 保存PID
    pids+=($!)
done

# 等待所有后台进程完成
for pid in "${pids[@]}"; do
    wait $pid
done

echo "所有模型推理和评估任务已完成。"