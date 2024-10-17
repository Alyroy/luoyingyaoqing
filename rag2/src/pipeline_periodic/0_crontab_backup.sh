# 每天8点 统计 T-3 自动化蒸馏筛选数据数量
0 8 * * * bash /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/shells/run_crontab_report.sh >> /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/crontab_logs/report.log 2>&1

# 每周五早8点 自动转化上周一至周日 sft和dpo训练数据，dpo数据送人工标注，sft数据自动加入训练数据
0 8 * * 5 bash /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/shells/run_crontab_get_dpo_sft_data.sh >> /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/crontab_logs/weekly_get_dpo_sft_data.log 2>&1

# 每周一凌晨2点全量同步训练数据至向量化路径
0 2 * * 1 bash /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/shells/run_crontab_get_train_query.sh >> /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/crontab_logs/weekly_get_train_query.log 2>&1

##########################################################################################
# 蒸馏livis线上日志单轮
##########################################################################################

# 每天晚上10点 蒸馏 T-2 线上日志原始query
0 22 * * * bash /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/shells/run_crontab_distillation_log_raw_single_rag.sh >> /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/crontab_logs/distillation_log_raw_single_rag.log 2>&1

# 每天晚上10点 蒸馏 T-2 线上日志泛化query
0 22 * * * bash /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/shells/run_crontab_distillation_log_extension_single_rag.sh >> /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/crontab_logs/distillation_log_extension_single_rag.log 2>&1

# 每天早晨1点 自动化筛选 T-3 蒸馏数据 
0 1 * * * bash /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/shells/run_crontab_filter_log_single_rag.sh >> /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/crontab_logs/filter_log_single_rag.log 2>&1

# 每天早晨10点 自动化筛选 T-3 蒸馏数据 
0 10 * * * bash /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/shells/run_crontab_get_atomic_capacity_single_rag.sh >> /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/crontab_logs/get_atomic_capacity_single_rag.log 2>&1

##########################################################################################
# 蒸馏car线上日志单轮
##########################################################################################

# 每天晚上8点 蒸馏 T-2 线上日志原始query(car)
0 20 * * * bash /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/shells/run_crontab_car_distillation_log_raw_single_rag.sh >> /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/crontab_logs/car_distillation_log_raw_single_rag.log 2>&1

# 每天晚上8点 蒸馏 T-2 线上日志泛化query(car)
0 20 * * * bash /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/shells/run_crontab_car_distillation_log_extension_single_rag.sh >> /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/crontab_logs/car_distillation_log_extension_single_rag.log 2>&1

# 每天早晨2点 自动化筛选 T-3 蒸馏数据 
0 2 * * * bash /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/shells/run_crontab_car_filter_log_single_rag.sh >> /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/crontab_logs/car_filter_log_single_rag.log 2>&1

##########################################################################################
# 蒸馏livis线上日志多轮
##########################################################################################

# 每天晚上9点 蒸馏 T-2 线上日志原始query(livis multi)
0 21 * * * bash /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/shells/run_crontab_distillation_log_raw_multi_rag.sh >> /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/crontab_logs/livis_distillation_log_raw_multi_rag.log 2>&1

# 每天晚上8点 蒸馏 T-2 线上日志泛化query(livis multi)
0 21 * * * bash /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/shells/run_crontab_distillation_log_extension_multi_rag.sh >> /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/crontab_logs/livis_distillation_log_extension_multi_rag.log 2>&1

# 每天早晨2点 自动化筛选 T-3 蒸馏数据 
0 3 * * * bash /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/shells/run_crontab_filter_log_multi_rag.sh >> /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/crontab_logs/livis_filter_log_multi_rag.log 2>&1


##########################################################################################
# 蒸馏car线上日志多轮
##########################################################################################

# 每天晚上11点 蒸馏 T-2 线上日志原始query(car multi)
0 23 * * * bash /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/shells/run_crontab_car_distillation_log_raw_multi_rag.sh >> /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/crontab_logs/car_distillation_log_raw_multi_rag.log 2>&1

# 每天晚上11点 蒸馏 T-2 线上日志泛化query(car multi)
0 23 * * * bash /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/shells/run_crontab_car_distillation_log_extension_multi_rag.sh >> /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/crontab_logs/car_distillation_log_extension_multi_rag.log 2>&1

# 每天早晨4点 自动化筛选 T-3 蒸馏数据 
0 4 * * * bash /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/shells/run_crontab_car_filter_log_multi_rag.sh >> /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/crontab_logs/car_filter_log_multi_rag.log 2>&1


# 0 2 * * * bash /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/shells/run_crontab_car_filter_log_single_rag.sh >> /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/crontab_logs/car_filter_log_single_rag.log 2>&1
# 0 20 * * * bash /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/shells/run_crontab_car_distillation_log_raw_single_rag.sh >> /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/crontab_logs/car_distillation_log_raw_single_rag.log 2>&1
# 0 20 * * * bash /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/shells/run_crontab_car_distillation_log_extension_single_rag.sh >> /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/crontab_logs/car_distillation_log_extension_single_rag.log 2>&1

# 0 8 * * 5 bash /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/shells/run_crontab_get_dpo_sft_data.sh >> /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/crontab_logs/weekly_get_dpo_sft_data.log 2>&1
# 0 2 * * 1 bash /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/shells/run_crontab_get_train_query.sh >> /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/crontab_logs/weekly_get_train_query.log 2>&1

# 0 8 * * * bash /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/shells/run_crontab_report.sh >> /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/crontab_logs/report.log 2>&1
# 0 22 * * * bash /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/shells/run_crontab_distillation_log_raw_single_rag.sh >> /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/crontab_logs/distillation_log_raw_single_rag.log 2>&1
# 0 22 * * * bash /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/shells/run_crontab_distillation_log_extension_single_rag.sh >> /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/crontab_logs/distillation_log_extension_single_rag.log 2>&1
# 0 1 * * * bash /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/shells/run_crontab_filter_log_single_rag.sh >> /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/pipeline_periodic/crontab_logs/filter_log_single_rag.log 2>&1

# 0 3 * * 0 bash /mnt/pfs-guan-ssai/nlu/renhuimin/backup.sh >> /mnt/pfs-guan-ssai/nlu/renhuimin/log.backup 2>&1 &