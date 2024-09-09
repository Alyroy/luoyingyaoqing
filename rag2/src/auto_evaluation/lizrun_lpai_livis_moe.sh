# 修改为你的shell的绝对路径 run_inference_gsb / run_inference
# lizrun start -c "bash,-c, /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/rag2.0/src/auto_evaluation/livis_moe_run_inference_api_mp.sh" \
#    -j test-moe0802-dpo-strategry-v4${stamp} \
#    -i reg-ai.chehejia.com/ssai/lizr/cu118/py310/pytorch:2.1.0-multinode-flashattn-2.3.2-fsdp \
#    -p app \
#    -n 1

# lizrunv2 start -c "bash -c /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/auto_evaluation/livis_moe_run_inference_api_mp.sh" \
#    -j test-0816-tourism-memory-v2${stamp} \
#    -i reg-ai.chehejia.com/ssai/lizr/cu118/py310/pytorch:2.3.0-multinode-flashattn2.5.6-vllm0.4.2-liktoken1.0.6-patch \
#    -p sft \
#    -n 1
#    # -t nvidia-geforce-rtx-4090 



lizrun lpai start -c "bash /lpai/volumes/ssai-nlu-bd/nlu/app/gongwuxuan/_init_server_.sh; bash /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/auto_evaluation/livis_moe_run_inference_api_mp1.sh" \
    -j eval-0903-sft \
    -i reg-ai.chehejia.com/ssai/lizr/cu118/py310/pytorch:2.1.0-multinode-flashattn-2.3.2-fsdp  \
    -p base-bd  \
    -n 1 \
    -w pytorch


lizrun lpai start -c "bash /lpai/volumes/ssai-nlu-bd/nlu/app/gongwuxuan/_init_server_.sh; bash /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/auto_evaluation/livis_moe_run_inference_api_mp2.sh" \
    -j eval-0903-sft \
    -i reg-ai.chehejia.com/ssai/lizr/cu118/py310/pytorch:2.1.0-multinode-flashattn-2.3.2-fsdp  \
    -p base-bd  \
    -n 1 \
    -w pytorch