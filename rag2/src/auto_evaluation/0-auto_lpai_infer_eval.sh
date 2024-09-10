#!/bin/bash
# 全局参数设置
# CURRENT_DIR=$(cd $(dirname $0); pwd)
CURRENT_DIR=/mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/auto_evaluation/
cd $CURRENT_DIR
# MODEL_CKPT_DIR="/lpai/volumes/ssai-nlu-bd/lizr/wangheqing/lisft/model/16b_generator_mindgpt_20240903_172w_v7moe_32k_liptm_model_1_new/checkpoint-5121/"
MODEL_CKPT_DIR="/lpai/volumes/ssai-nlu-bd/lizr/wangheqing/lisft/model/16b_generator_mindgpt_20240827_165w_v7moe_32k_liptm_model_1/checkpoint-4986"
HF_MODEL_CKPT_DIR=${MODEL_CKPT_DIR}/hf_model

# 1. 启动转换模型脚本
# 模型训练转格式脚本路径
TRAIN_MODEL_DIR=/mnt/pfs-guan-ssai/nlu/renhuimin/lisft/
QUEUE_NAME="app-bd" # GPU机群队列名

echo sh auto_lizrun_lpai_trans_hf_model.sh $TRAIN_MODEL_DIR $MODEL_CKPT_DIR $HF_MODEL_CKPT_DIR $QUEUE_NAME
sh auto_lizrun_lpai_trans_hf_model.sh $TRAIN_MODEL_DIR $MODEL_CKPT_DIR $HF_MODEL_CKPT_DIR $QUEUE_NAME

# 2. 启动推理脚本
M_CNT=1 # GPU机数
JOB_NAME="sft0827" # 推理任务名称
EVAL_MODEL=${HF_MODEL_CKPT_DIR} # 评估模型路径
EVAL_TIMESTAMP=$JOB_NAME #-`date +%Y%m%d` # 推理结果路径
INPUT_DIR="/mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/data/test_data/app_self_test/v20240903/input_data/"
OUTPUT_DIR="/mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/data/test_data/app_self_test/v20240903/"
EVAL_COL=model_13b_input # 评估列名，16b模型input，同日志格式
echo sh auto_lizrun_lpai_livis_moe.sh ${CURRENT_DIR} ${JOB_NAME} ${QUEUE_NAME} ${M_CNT} ${EVAL_MODEL} ${EVAL_TIMESTAMP} ${INPUT_DIR} ${OUTPUT_DIR} ${EVAL_COL}
sh auto_lizrun_lpai_livis_moe.sh ${CURRENT_DIR} ${JOB_NAME} ${QUEUE_NAME} ${M_CNT} ${EVAL_MODEL} ${EVAL_TIMESTAMP} ${INPUT_DIR} ${OUTPUT_DIR} ${EVAL_COL}

# 3. 启动qwen api
# 3.1 判断推理是否完成
# 循环，直到文件存在
cnt=1
echo "${OUTPUT_DIR}/${EVAL_TIMESTAMP}/.done"
until [ -f "${OUTPUT_DIR}/${EVAL_TIMESTAMP}/.done" ]; do
  echo "等待${EVAL_TIMESTAMP}推理任务结束...${cnt}*5分钟"
  sleep 300
  cnt=$(expr $cnt + 1)
done

echo "开始评估已启动"
# 4. 获取三机的ip，并启动run_eval_auto.sh脚本
INPUT_LOG_COL=model_13b_input
EVAL_INPUT_DIR=${OUTPUT_DIR}/${EVAL_TIMESTAMP}/ # 模型推理结果文件夹
EVAL_OUTPUT_DIR_REL=${OUTPUT_DIR}/${JOB_NAME}_相关性打分/
EVAL_OUTPUT_DIR_AUTH=${OUTPUT_DIR}/${JOB_NAME}_真实性打分/
echo sh auto_eval_rel_auth.sh ${CURRENT_DIR} ${INPUT_LOG_COL} ${EVAL_INPUT_DIR} ${EVAL_OUTPUT_DIR_REL} ${EVAL_OUTPUT_DIR_AUTH}
sh auto_eval_rel_auth.sh ${CURRENT_DIR} ${INPUT_LOG_COL} ${EVAL_INPUT_DIR} ${EVAL_OUTPUT_DIR_REL} ${EVAL_OUTPUT_DIR_AUTH}