#!/bin/bash

stamp=$(date +%Y-%m-%d-%H-%M-%S)
# 全局参数设置
CURRENT_DIR=$(cd $(dirname $0); pwd) # 必须在mnt/下
cd $CURRENT_DIR 

# 配置模型参数
declare -a MODEL_CKPT_DIR_LIST=(
    "/mnt/pfs-gv8sxa/nlu/team/gongwuxuan/model/llama_factory/lf_s1-v1107_s2-v1112_dpo-safe-and-rich-v1223-4k_25d-72b_1225-1_1e6-ep4/checkpoint-3484"
)

declare -a QUEUE_NAME_LIST=("wq-app")
declare -a JOB_NAME_LIST=("test-online-infer")
declare -a MODEL_NAME_LIST=("local-mindgt2.5.4dpo")

# 输入与输出目录设定
# INPUT_DIR=${CURRENT_DIR}'/eval_data/livis_query_fc_1114/'
INPUT_DIR="/mnt/pfs-guan-ssai/nlu/gongwuxuan/code/rag_tool/rag2/src/auto_evaluation/eval_data/livis_query_fc_1114/select-new-1231-lvfc_form-2k_final_input.csv1113.out.csv"
OUTPUT_DIR=${CURRENT_DIR}'/eval_results/'
EVAL_COL=livis_model_input
SERVER_PORT=8001

# 支持推理参数设置
temperature=0.9 #0.7 # 0.9
top_p=0.95 #0.8 # 0.95
repetition=1 #1.05 # 1

# 存储所有后台任务的PID
pids=()
USER=gongwuxuan

# 遍历所有模型进行推理和评估
for i in "${!MODEL_CKPT_DIR_LIST[@]}"; do
    MODEL_CKPT_DIR=${MODEL_CKPT_DIR_LIST[$i]}
    QUEUE_NAME=${QUEUE_NAME_LIST[$i]}
    JOB_NAME=${JOB_NAME_LIST[$i]}
    MODEL_NAME=${MODEL_NAME_LIST[$i]}

    HF_MODEL_CKPT_DIR=${MODEL_CKPT_DIR}${HF_MODEL_CKPT_DIR_SUFFIX}
    
    # 创建日志文件
    LOG_FILE=${OUTPUT_DIR}/${JOB_NAME}/"log.${JOB_NAME}"

    (
        # 0.启动vllm线上推理api
        INFER_JOB_NMAE=infer-${JOB_NAME}
        echo bash /mnt/pfs-guan-ssai/nlu/gongwuxuan/tools/model_deployment/vllm_custom_api.sh ${MODEL_CKPT_DIR} ${MODEL_NAME} ${SERVER_PORT}

        lizrunv2 start -c "bash /mnt/pfs-guan-ssai/nlu/gongwuxuan/tools/model_deployment/vllm_custom_api.sh ${MODEL_CKPT_DIR} ${MODEL_NAME} ${SERVER_PORT}" \
        -j ${INFER_JOB_NMAE} \
        -i reg-ai.chehejia.com/ssai/lizr/cu124/py310/pytorch:pytorch-2.5.1-multinode-flashattn-2.6.3-vllm0.6.5-liktoken1.0.6 \
        -p ${QUEUE_NAME} \
        -n 1

        # 1. 获取线上部署模型 api
        cnt=1
        IP1=`lizrun pool get -p ${QUEUE_NAME} -d | grep Running | grep ${INFER_JOB_NMAE}-${USER_NAME} | awk -F " " '{print($2)}'`
        until [ ${IP1} ]; do
            echo "存在IP1的qwen-api任务未就位！！！" | tee -a $LOG_FILE
            IP1=`lizrun pool get -p ${QUEUE_NAME} -d | grep Running | grep ${INFER_JOB_NMAE}-${USER_NAME} | awk -F " " '{print($2)}'`
            sleep 30
            cnt=$(expr $cnt + 1)
        done
        echo "IP1的qwen-api任务已就位！！！"`date` | tee -a $LOG_FILE
        IP1=`echo ${IP1} | awk '{gsub("-", ".", $0); print $0}'`
        echo "IP1:"${IP1} | tee -a $LOG_FILE

        # 1. 启动推理脚本
        echo "Starting inference for job: ${JOB_NAME}" | tee -a $LOG_FILE
        python vllm_online_inference.py \
            --input_file ${INPUT_DIR} \
            --output_path ${OUTPUT_DIR} \
            --model_name ${MODEL_NAME} \
            --url http://${IP1}:${SERVER_PORT}/v1 \
            --batch_num 100 --time_stamp ${JOB_NAME} \
            --turn_mode vllm_online --eval_col ${EVAL_COL} \
            --temperature ${temperature} --top_p ${top_p} --repetition ${repetition}
        
        echo "完成所有文件的推理" > ${OUTPUT_DIR}/${JOB_NAME}/.done

    ) &  # 在后台执行

    # 保存PID
    pids+=($!)

    # 提交任务之间等待10s
    sleep 10
done

# 等待所有后台进程完成
for pid in "${pids[@]}"; do
    wait $pid
done

echo "所有模型推理任务已提交。"