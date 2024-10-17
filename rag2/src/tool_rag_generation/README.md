### 使用说明
1. 支持 RAG、非RAG，单轮、多轮 csv数据 转jsonl数据（sft中间训练格式, <|xxx|> special token）
2. 支持 RAG、非RAG，单轮、多轮 csv数据 转json数据（模型训练最终格式，instruction)

### 参数说明
1. df中必须字段：id, source, user-query, thought, api, observation, assistant
2. dataformat_obj = DataFormat(api_flag: boolean, multi_flag: boolean) # 数据需拆分RAG、非RAG、单轮、多轮
3. dataformat_obj.gen_sft_data(df) # 生成jsonl
4. dataformat_obj.gen_sft_unused_data(df) # 生成最终训练数据格式 json
5.  dataformat_obj.gen_dpo_unused_data(df) # 生成最终训练数据格式，必须包含chosen，rejected字段

### 数据格式说明
|key|数据结构|描述|
|---|---|---|
|**user-query**|str|**用户query**|
|**assistant**|str|**线上回复**|
|**api**|list[dict]|**API**|
|**thought**|str|**thought**|
|**observation**|list[list[str]]|**observation**|
|**context**|list[dict[str, str]]|**上文信息**|
