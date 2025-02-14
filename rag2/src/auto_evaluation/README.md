### 更新记录
【2025.1.10】
1. 更新测试集eval_data/livis_query_fc_0110
2. 更新指标统计计算
3. 提供线下评估方式，优化评估速度1小时1200条 真实性+相关
4. 启动参考0-auto_qwen_infer_eval_qwen_api_swa.sh

【2024.12.30】
1. 支持vllm swa推理
2. 支持本地评估真实性、相关性
3. 更新统计指标
4. 使用参考 0-auto_qwen_infer_eval_qwen_api_swa.sh
5. 支持模型通过vllm api server部署的url进行推理

### MOE LLM 推理
#### 脚本运行
```
./lizrun_livis_moe.sh
```
#### 推理参数说明

| 参数名                             | 类型 | 必要性 | 默认值 | 描述 |
|------------------------------------|------|--------|--------|------|
| --input_file                       |str      |必须        |N/A        |      |
| --output_path                      |str      |必须        |N/A        |      |
| --model                            |str      |必须        |N/A        |      |
| --tiktoken_path                    |str      |可选        |"none"        |"none"表示自动load tokenizer，或者自己添加tokenizer地址      |
| --time_stamp                       |str      |必须        |N/A        |output_path后新建保存文件夹      |
| --batch_size                       |int      |可选        |20        |RAG推理下建议设置5以内      |
| --turn_mode                        |str      |必须       |moe        |moe 或 dense(v6)      |
| --eval_col                         |str      |可选        |xx        |如果提供eval_col，模型自动load该列，并直接进行推理，数据格式同model_13b_input      |
| --dosample_flag/--no_dosample_flag |bool      |可选        |True        |是否开启或关闭dosample，默认开启      |
| --api_flag/--no_api_flag           |bool      |可选        |True        |如果没有设置eval_col，脚本会自动拼接user thought api observation, api_flag表示拼接为RAG格式，no_api_flag表示拼接为非RAG格式      |
| --temperature                      |float      |可选        |0.9        |      |
| --top_k                            |int      |可选        |50        |      |
| --top_p                            |float      |可选        |0.95        |      |
| --repetition                       |float      |可选        |1        |      |


### 评估
#### 脚本运行
- 解析日志真实性、相关性评估
```
cd shell_eval
./livis_run_authenticity_eval.sh # 真实性
./livis_run_relevance_eval.sh # 相关性
```
- 拆分user obs assistant 真实性相关性评估
```
cd shell_eval
./run_authenticity_eval.sh # 真实性
./run_relevance_eval.sh # 相关性
```
#### 参数说明
| 参数名                  | 类型 | 必要性 | 默认值 | 描述 |
|------------------------------------|------|--------|--------|------|
| --model_list           |list      |必须        |('gpt4o' 'wenxin')     |需要评估的模型名称    |
| --url_list             |list      |必须        |N/A      |与模型名称对应的url   |
| --eval_mode             |str      |必须        |user_obs_ans_concat       |model_13b_log表示直接解析日志进行评估；user_obs_ans_concat表示拼接 user obs ans 进行评估；with_prompt表示已拼接好prompt可直接进行评估   |
| --eval_column_list             |list      |必须        |('user_col' 'obs_col' 'ans_col')        |如果上个参数是user_obs_ans_concat，提供对应的3列；如果上个参数是model_13b_log，('log_col' 'ans_col' 'ans_col') ；如果上个参数是with_prompt，('prompt_col' 'prompt_col' 'prompt_col') |
| --save_column             |str      |可选        |eval_output      |评估结果输出列名   |
| --metric             |str      |可选        |authenticity      |评估维度，authenticity or relevance   |
| --input_dir             |str      |可选        |N/A      |输入文件夹or单个文件   |
| --output_dir             |str      |可选        |N/A      |输出文件夹   |
| --prompt_path             |str      |可选        |N/A      |评估prompt绝对路径   |
| --thread_num             |int      |可选        |10      |qwen评估线程数，不超过30   |
| --chunk_num             |int      |可选        |2     |qwen评估chunk数，不超过20   |
| --temperature             |float     |可选        |0.7     |评估模型的温度   |