# 每天晚上10点 蒸馏 T-2 线上日志原始query
0 22 * * * bash /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/weekly_run/run_crontab_distillation_log_raw_single_rag.sh >> /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/weekly_run/crontab_logs/distillation_log_raw_single_rag.log 2>&1

# 每天晚上10点 蒸馏 T-2 线上日志泛化query
0 22 * * * bash /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/weekly_run/run_crontab_distillation_log_extension_single_rag.sh >> /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/weekly_run/crontab_logs/distillation_log_extension_single_rag.log 2>&1

# 每天早晨6点 自动化筛选 T-3 蒸馏数据 
0 6 * * * bash /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/weekly_run/run_crontab_filter_log_single_rag.sh >> /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/weekly_run/crontab_logs/filter_log_single_rag.log 2>&1

# 每天下午18点 统计 T-3 自动化蒸馏筛选数据数量
0 18 * * * bash /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/weekly_run/run_crontab_report.sh >> /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/weekly_run/crontab_logs/report.log 2>&1

# 每周五早8点 自动转化上周一至周日 sft和dpo训练数据，dpo数据送人工标注，sft数据自动加入训练数据
0 8 * * 5 bash /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/weekly_run/run_crontab_get_dpo_sft_data.sh >> /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/weekly_run/crontab_logs/weekly_get_dpo_sft_data.log 2>&1

# 每周一凌晨2点全量同步训练数据至向量化路径
0 2 * * 1 bash /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/weekly_run/run_crontab_get_train_query.sh >> /mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/src/weekly_run/crontab_logs/weekly_get_train_query.log 2>&1