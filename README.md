# RAG
## 更新记录
【2024.10.15更新】
1. 支持多轮对话调用API，具体使用方法参考rag2/src/tool_kg_search/example.ipynb
2. 优化mindgpt推理脚本，更改镜像提升模型推理速度，支持文件夹推理
3. 修复真实性评估解析脚本bug


【2024.09.03更新】
1. 优化数据格式转化脚本，支持转sft dpo 最终训练数据格式
`DataFormat.gen_sft_unused_data, DataFormat.gen_dpo_unused_data`
[RAG 中间训练格式说明](https://li.feishu.cn/docx/CkxadAXZfoQqOLxexWPcso5enWa)
2. 对齐推理tokenizner

【2024.08.14更新】
1. 增加 weekly_run/ 自动蒸馏[线上日志](https://gitlab.chehejia.com/wangxiaoyuan/online-data-label/-/tree/master/)及生成dpo送标数据和sft训练数据


RAG数据构造流程参考rag2/src/auto_llm_distillation/example.ipynb
## 全局代码结构
```commandline
├── common                     通用代码
├── conf                       参数设置/常用prompt
├── src                        封装工具
│   ├── tool_kg_search         			搜索接口
│   ├── tool_llm_response         		智能云/vllm大模型调用接口
│   ├── tool_rag_generation         	数据格式定义及转换
│   ├── auto_evaluation         	    评估
│   ├── auto_filter         	        自动化筛选蒸馏数据
│   ├── auto_llm_distillation           获取api obs gpt4的蒸馏结果
│   ├── weekly_run           定时蒸馏线上日志
...
```
## 搜索接口tool_kg_search
### 1. 代码结构
```commandline
├── debug.ipynb                		1B API 及搜索调用示例
├── get_1b_output.py                1B API 调用接口
├── kgsearch_llm_obs.py             大模型搜索调用接口
├── search_http_tool.py             搜索通用接口
└── utils.py                  		搜索utils
```
### 2. 运行方式
```commandline
参考debug.ipynb 或 kgsearch_llm_obs.py
```
### 3. 1B API 输入输出数据格式
```commandline
输入：url:str, query:str, cateogry:str
输出：dict. function call 形式，包含thought及API各参数
```
### 4. 搜索 输入输出数据格式
```commandline
输入：search_env:str(环境), query:str, api:dict
输出：list[dict] 搜索返回内容
```
## 智能云/vllm大模型调用接口tool_llm_response

### 1. 代码结构
```commandline
├── debug.ipynb                		各类大模型调用示例
├── call_llm_with_zny.py            智能云大模型调用接口
├── call_llm_with_vllm.py           vllm部署大模型调用接口
└── model_deployment                vllm模型部署脚本
```
### 2. 运行方式
```commandline
参考debug.ipynb
```
### 3. 输入输出数据格式
```commandline
输入：df:pd.DataFrame,包含大模型的input列
输出：df:pd.DataFrame,新增大模型的response列
```
### 4. 智能云脚本输入参数
```commandline
url # 智能云GPT api
model_name # 目前支持['gpt4o','gpt4','wenxin']
temperature # llm输出温度，一半设置为0-1，1表示随机 0表示greedy search
smax_retries # 调用gpt报错后最多重试 max_retries 次
qps # 多线程 qps数量，与各账号设置有关，一般不超过10
max_concurrent # 异步多线程参数，一般10或者20，太大会接口超过qps
asyncio_flag # False=普通多线程，Jupyter或者python均可
query_column_name # llm模型输入列名
response_column_name # llm模型输出列名
...
```
### 5. vllm脚本输入参数
```commandline
model # 模型名称，与部署vllm有关, 例如qwen
url # 模型 api
temperature # 模型temperature值
top_p # 模型top_p值
max_tokens # 最大输出长度
chunk_num # chunk块数，一般不超过10
thread_num # 线程数，一般不超过20
query_column_name # llm模型输入列名
response_column_name # llm模型输出列名
...

使用vllm蒸馏或评估数据时，需要先部署模型，model_deployment内整理常用的开源模型，以qwen为主
```
## 数据格式定义及转换tool_rag_generation

### 1. 代码结构
```commandline
├── debug.ipynb                		数据格式转换示例
├── data_format.py           		定义数据类型及数据格式转换
```
### 2. 运行方式
```commandline
参考debug.ipynb
```
