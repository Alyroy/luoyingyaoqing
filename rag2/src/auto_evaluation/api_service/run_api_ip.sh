
CURRENT_DIR=/mnt/pfs-guan-ssai/nlu/chihuixuan/rag_tool/rag2/src/auto_evaluation/
cd ${CURRENT_DIR}
QUEUE_NAME="wq-app" # GPU机群队列名
M_CNT=1 # GPU机数
JOB_NAME="ragmoe0927yuqing1" # 推理任务名称

stamp=$(date +%Y-%m-%d-%H-%M-%S)
QWEN_JOBNAME1=qwen-api1-${JOB_NAME}
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
