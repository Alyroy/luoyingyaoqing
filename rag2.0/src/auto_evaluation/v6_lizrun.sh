## 修改为你的shell的绝对路径 run_inference_gsb / run_inference

lizrun start -c "bash,-c, /mnt/pfs-guan-ssai/nlu/data/renhuimin/pro_rag/src/auto_evaluation/v6_run_inference_api_mp.sh" \
   -j test-structure-0716-v2${stamp} \
   -i reg-ai.chehejia.com/ssai/lizr/cu118/py310/pytorch:2.1.0-multinode-flashattn-2.3.2-fsdp \
   -p base3\
   -n 1
   # -t nvidia-geforce-rtx-4090 \
 