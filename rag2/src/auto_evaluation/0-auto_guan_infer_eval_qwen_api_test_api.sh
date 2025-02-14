#!/bin/bash
# 全局参数设置
# CURRENT_DIR=$(cd $(dirname $0); pwd)
CURRENT_DIR=/mnt/pfs-guan-ssai/nlu/chihuixuan/rag_tool/rag2/src/auto_evaluation/
cd $CURRENT_DIR
# MODEL_CKPT_DIR="/lpai/volumes/ssai-nlu-bd/lizr/wangheqing/lisft/model/16b_generator_mindgpt_20240903_172w_v7moe_32k_liptm_model_1_new/checkpoint-5121/"
# MODEL_CKPT_DIR="/lpai/volumes/ssai-nlu-bd/lizr/wangheqing/lisft/model/16b_generator_mindgpt_20240827_165w_v7moe_32k_liptm_model_1/checkpoint-4986"
# MODEL_CKPT_DIR="/lpai/volumes/ssai-nlu-bd/lizr/renhuimin/mindgpt/v7moe_sft_0814_cft_0903_new/checkpoint-1380"
MODEL_CKPT_DIR="/mnt/pfs-guan-ssai/nlu/renhuimin/checkpoints/16b_sft_mindgpt_20240911_norag_140w_v7moe_32k_model_1/checkpoint-4299"
HF_MODEL_CKPT_DIR=${MODEL_CKPT_DIR}/hf_model

# 1. 启动转换模型脚本
# 模型训练转格式脚本路径
TRAIN_MODEL_DIR=/mnt/pfs-guan-ssai/nlu/renhuimin/lisft/

# echo sh auto_trans_hf_model.sh $TRAIN_MODEL_DIR $MODEL_CKPT_DIR $HF_MODEL_CKPT_DIR $CURRENT_DIR
# sh auto_trans_hf_model.sh $TRAIN_MODEL_DIR $MODEL_CKPT_DIR $HF_MODEL_CKPT_DIR $CURRENT_DIR

# 2. 启动推理脚本
QUEUE_NAME="base" # GPU机群队列名
M_CNT=1 # GPU机数
JOB_NAME="testapi" # 推理任务名称
EVAL_MODEL=${HF_MODEL_CKPT_DIR} # 评估模型路径
EVAL_TIMESTAMP=$JOB_NAME #-`date +%Y%m%d` # 推理结果路径
INPUT_DIR='/mnt/pfs-guan-ssai/nlu/chihuixuan/data/rag/yuqing/input_data_0927_1/'
# OUTPUT_DIR="/mnt/pfs-guan-ssai/nlu/chihuixuan/data/rag/yuqing/"
OUTPUT_DIR='/mnt/pfs-guan-ssai/nlu/chihuixuan/data/rag/livis/testapi1021/'
# INPUT_DIR="/mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/data/test_data/app_self_test/v20240903/input_data/test_时效性摘要.csv"
# OUTPUT_DIR="/mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/data/test_data/app_self_test/v20240903/"
EVAL_COL=model_13b_input # 评估列名，16b模型input，同日志格式
# echo sh auto_lizrun_guan_livis_moe.sh ${CURRENT_DIR} ${JOB_NAME} ${QUEUE_NAME} ${M_CNT} ${EVAL_MODEL} ${EVAL_TIMESTAMP} ${INPUT_DIR} ${OUTPUT_DIR} ${EVAL_COL}
# sh auto_lizrun_guan_livis_moe.sh ${CURRENT_DIR} ${JOB_NAME} ${QUEUE_NAME} ${M_CNT} ${EVAL_MODEL} ${EVAL_TIMESTAMP} ${INPUT_DIR} ${OUTPUT_DIR} ${EVAL_COL}

# 3. 启动qwen api
# 3.1 判断推理是否完成
# 循环，直到文件存在
# cnt=1
# echo "${OUTPUT_DIR}/${EVAL_TIMESTAMP}/.done"
# until [ -f "${OUTPUT_DIR}/${EVAL_TIMESTAMP}/.done" ]; do
#   echo "等待${EVAL_TIMESTAMP}推理任务结束...${cnt}*5分钟"
#   sleep 300
#   cnt=$(expr $cnt + 1)
# done

stamp=$(date +%Y-%m-%d-%H-%M-%S)
QWEN_JOBNAME1=qwen-api1-${JOB_NAME}-${stamp}
echo sh api_service/lizrun_api_qwen_auto.sh ${QWEN_JOBNAME1} ${QUEUE_NAME} ${CURRENT_DIR}
sh api_service/lizrun_api_qwen_auto.sh ${QWEN_JOBNAME1} ${QUEUE_NAME} ${CURRENT_DIR}
sleep 120
cnt=1
USER_NAME=chihuixuan
IP1=`lizrun pool get -p ${QUEUE_NAME} -d |grep Running | grep ${QWEN_JOBNAME1}-${USER_NAME} |awk -F " " '{print($2)}'`
until [ ${IP1} ]; do
    echo "存在IP1的qwen-api任务未就位！！！"
    IP1=`lizrun pool get -p ${QUEUE_NAME} -d |grep Running | grep ${QWEN_JOBNAME1}-${USER_NAME} |awk -F " " '{print($2)}'`
    echo "IP1:"${IP1}
    sleep 300
    cnt=$(expr $cnt + 1)
done
echo "IP1的qwen-api任务已就位！！！"`date`
IP1=`echo ${IP1} | awk '{gsub("-", ".", $0); print $0}'`
echo "IP1:"${IP1}

echo "开始评估已启动"
# 4. 获取三机的ip，并启动run_eval_auto.sh脚本
INPUT_LOG_COL=model_13b_input
EVAL_INPUT_DIR=${OUTPUT_DIR} # /${EVAL_TIMESTAMP}/ # 模型推理结果文件夹
EVAL_OUTPUT_DIR_REL=${OUTPUT_DIR}/相关性打分/ # /${JOB_NAME}/相关性打分/
EVAL_OUTPUT_DIR_AUTH=${OUTPUT_DIR}/真实性打分/ # /${JOB_NAME}/真实性打分/
echo sh auto_eval_rel_auth_test_api.sh ${CURRENT_DIR} ${INPUT_LOG_COL} ${EVAL_INPUT_DIR} ${EVAL_OUTPUT_DIR_REL} ${EVAL_OUTPUT_DIR_AUTH} ${IP1}
sh auto_eval_rel_auth_test_api.sh ${CURRENT_DIR} ${INPUT_LOG_COL} ${EVAL_INPUT_DIR} ${EVAL_OUTPUT_DIR_REL} ${EVAL_OUTPUT_DIR_AUTH} ${IP1}

lizrun stop ${QWEN_JOBNAME1}-${USER_NAME}