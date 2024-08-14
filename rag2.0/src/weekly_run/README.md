### 启动脚本
```
chmod +x 0_auto_run_weekly_distillation.sh 1_auto_weekly_distillation_log.sh 2_auto_weekly_distillation_extand.sh 3_auto_get_dpo_sft_data.sh
./0_auto_run_weekly_distillation.sh
```

### 使用说明
- 1_auto_weekly_distillation_log.sh 
	- 对线上日志进行蒸馏和双模型自动化筛选

- 2_auto_weekly_distillation_extand.sh
	- 对线上日志进行蒸馏和双模型自动化筛选

- 3_auto_get_dpo_sft_data.sh
    - 自动提前dpo送标数据及sft训练数据
 
- 0_auto_run_weekly_distillation.sh
    - 串联上述3个脚本
    - 注意，首先需要部署qwen2及llama3，此步骤主要用于双模型打分筛选蒸馏后的数据，亦可选择其他大模型进行筛选，按需部署


### 参数说明
```
start_date="2024-08-08" # 筛选开始日期
end_date="2024-08-12" # 筛选结束日期，在代码内部会直接 链接/mnt/pfs-guan-ssai/nlu/wangxiaoyuan/online-data/{date}/
model_list=("qwen2_72b" "llama3_70b") # 自动化筛选模型名称
url_list=("http://172.24.136.30:8000/v1" " http://172.24.136.30:8001/v1") # 自动化筛选模型地址，vllm需部署，智能云可自行替换

api_url="http://172.24.139.95:16073/ligpt_with_api/search" # api url，用于泛化问构造api obs assistant，最新链接可找辛洪生

output_folder="/workspace/renhuimin/pro_rag/data/distillation_data/v20240813_applog/" # 数据存储路径
dpo_outpt="dpo_by0807.csv" # dpo存储名称，多个文件会merge为一个
sft_outpt="sft_by0807" # sft存储名称，csv和jsonl会分别存储，多个文件merge为一个
```
注意，蒸馏gpt4的url，需修改文件内对应的gpt4url
`weekly_distillation_extand_sft.py`
`weekly_distillation_log_sft_dpo.py`