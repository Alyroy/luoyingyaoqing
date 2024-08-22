# 定时蒸馏及泛化回复差的线上日志

## 定时任务一览
每天晚上10点 蒸馏 T-2 线上日志原始query：
`0 22 * * * bash run_crontab_distillation_log_raw_single_rag.sh` 

每天晚上10点 蒸馏 T-2 线上日志泛化query：
`0 22 * * * bash run_crontab_distillation_log_extension_single_rag.sh` 

每天早晨6点 自动化筛选 T-3 蒸馏数据 ：
`0 6 * * * bash run_crontab_filter_log_single_rag.sh` 

每天下午18点 统计 T-3 自动化蒸馏筛选数据数量：
`0 18 * * * bash run_crontab_report.sh` 

每周五早8点 自动转化上周一至周日 sft和dpo训练数据，dpo数据送人工标注，sft数据自动加入训练数据：
`0 8 * * 5 bash run_crontab_get_dpo_sft_data.sh` 

每周一凌晨2点全量同步训练数据至向量化路径：
`0 2 * * 1 bash run_crontab_get_train_query.sh` 

注：**run_weekly**系列sh为手动设置时间的运行脚本

## 脚本使用说明
- run_crontab_distillation_log_raw_single_rag.sh
```
CheckDependencyFunction() 检测数据回流是否完成打标,gpt_labeled存在执行下一步
ExecuteJobFunction() 蒸馏相关性、逻辑性中或差的线上日志，需自行设置蒸馏模型url及蒸馏prompt
```
- run_crontab_distillation_log_extension_single_rag.sh
```
get_api_model_url() 获取api url
CheckDependencyFunction() 检测数据回流是否完成打标,gpt_labeled存在执行下一步
ExecuteJobFunction() 对相关性、逻辑性中或差的线上日志进行泛化，并获取最新的api及obs，再进行蒸馏
```
- run_crontab_filter_log_single_rag.sh
```
get_model_url() 获取qwen2及llama3 vllm url
CheckDependencyFunction() 检测蒸馏数据是否完成
ExecuteJobFunction() 首先单过滤回复话术是否有问题，其次双模型对真实性、相关性简易打分，双模型均通过后存为训练数据
CheckDoneFunction() 检测双模型是否完成筛选，完成后生成.done文件
```
- run_crontab_report.sh
```
CheckDependencyFunction() 检测双模型筛选是否完成
ExecuteJobFunction() 分别统计 原始raw 及泛化extension 数量
支持字段为：
- 时间
- 原始query总量
- 原始逻辑性低或中数量
- 原始相关性低或中数量
- 数据蒸馏筛选后总量
- 数据蒸馏筛选后逻辑性低或中数量
- 数据蒸馏筛选后相关性低或中数量
- 蒸馏回复平均长度
- 蒸馏回复长度10分位
- 蒸馏回复长度90分位
```
- run_crontab_get_dpo_sft_data.sh
```
将上一周蒸馏的数据转为dpo及sft格式，
- 双模型通过的，dpo转为人工标注格式，sft转为训练数据格式
- 双模型未通过的，sft转为人工标注格式
```
- run_crontab_get_train_query.sh
```
每周抽取全量的RAG训练数据query存入向量数据转换路径
```