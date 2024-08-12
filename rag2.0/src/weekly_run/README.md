
### 启动脚本
```
chmod +x run_weekly_distillation.sh run_weekly_distillation_extand.sh run_weekly_distillation_log.sh
./run_weekly_distillation.sh
```

### 使用说明
- run_weekly_distillation_log.sh
	- 对线上日志进行蒸馏和筛选
	- 蒸馏回复，使用时，需在weekly_distillation_log_sft_dpo.py 中的ZnyConfig蒸馏步骤修改为自己的gpt4接口

- run_weekly_distillation_extand.sh
	- 对泛化query进行蒸馏和筛选
	- 蒸馏数据的url需在weekly_distillation_log_sft_dpo.py中修改