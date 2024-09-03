stamp=$(date +%Y-%m-%d-%H-%M-%S)

# lizrunv2 start -c "/mnt/pfs-guan-ssai/nlu/xinhongsheng/stanford_alpaca_dynamics_2048_dev/02_train_dist_fsdp.sh" \
#     -j v6-1b-api-server-model-0819 \
#     -i reg-ai.chehejia.com/ssai/lizr/cu118/py310/pytorch:2.0.0-multinode-flashattn \
#     -n 1 \
#     -p sft \
    # -t nvidia-geforce-rtx-4090 




lizrunv2 start -c "/mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/tool_kg_search/1b_model_deployment/run_api_server.sh" \
    -j v6-1b-api-server-${stamp} \
    -i reg-ai.chehejia.com/ssai/lizr/cu118/py310/pytorch:2.0.0-multinode-flashattn \
    -p app-4090 \
    -n 1 \
    # -t nvidia-geforce-rtx-4090 
