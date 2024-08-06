# lizrun start -c "bash,-c, /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/rag2.0/src/auto_evaluation/v6_run_inference_api_mp.sh" \
#    -j test-v6-0729cft-app-self-v3${stamp} \
#    -i reg-ai.chehejia.com/ssai/lizr/cu118/py310/pytorch:2.1.0-multinode-flashattn-2.3.2-fsdp \
#    -p app\
#    -n 1


lizrunv2 start -c "bash -c /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/rag2.0/src/auto_evaluation/v6_run_inference_api_mp.sh" \
   -j test-v6-0729cft-app-self-v3${stamp} \
   -i reg-ai.chehejia.com/ssai/lizr/cu118/py310/pytorch:2.1.0-multinode-flashattn-2.3.2-fsdp \
   -p base3\
   -n 1