import pandas as pd
import requests
import json
#【安全提示】当问题中出现脏话、侮辱词、负面词、敏感词（短语），回答过程尽量不要直接出现脏话、侮辱词汇、负面、敏感词（短语）。回复的最后需要进行劝导，\
    # 以委婉语气劝诫其不要使用或者谨慎使用负面词汇（短语）。 根据【参考知识】和【安全提示】回答问题，回答简练些、不要冗长的回答。不用输出分析过程，\
        # 不要输出参考知识标识和安全提示标识，直接输出回复。

# df = pd.read_json('/mnt/pfs-guan-ssai/nlu/chenjun18/data/202411/kaiche.jsonl', lines=True)
# fw = jsonlines.open('/mnt/pfs-guan-ssai/nlu/chenjun18/rag_tool/rag2/src/auto_llm_distillation/wenxin_w_api_wo_obs_6000.jsonl', 'w')

df = pd.read_csv('/mnt/pfs-guan-ssai/nlu/jiajuntong/code/rag_tool/rag2/src/q_api_obs_lab/obs_0115_filter.csv')
url = 'https://chenjun18-gpt4.fc.chj.cloud/wenxin/zh-conversation'
headers = {
    # 'User-Agent': 'Mozilla/5.0',
    'Content-Type': 'application/json',
    'Accept': 'application/json',  # 指定希望返回 JSON
    # 可以添加其他请求头
}
df['reflect'] = ''
df['answer_w_reflect'] = ''
for i in range(df.shape[0]):
    q = df.iloc[i]['user-query']
    obs = df.iloc[i]['observation']
    data0 = {
    "messages": [
        {"role": "user", "content": f"【参考知识】{obs}\n你是一个经验丰富的政治、军事、法律领域的专家，请你对提供的问题进行意图理解和分析，然后根据【参考知识】生成一些回复策略。\
            回复策略简练些、不要冗长的回答。且回复策略需要能帮助我生成无害、真实、相关的回复。直接输出回复策略,但不要生成回复（示例），不要出现除回复策略以外的任何内容。【问题】是{q}]"}
    ]
    }
    # print(data)
    reflect = requests.post(url, headers=headers, data=json.dumps(data0))
    try:
        reflect=reflect.json()
        if reflect['choices'] and 'content' in reflect['choices'][0]:
            reflect=reflect['choices'][0]['content']
        else:
            reflect=''
        df.loc[i, 'reflect'] = reflect
    except:
        continue
    data1 = {
    "messages": [
        {"role": "user", "content": f"【参考知识】{obs}\n【回复策略】{reflect}\n【安全提示】当问题中出现脏话、侮辱词、负面词、敏感词（短语），\
            回答过程尽量不要直接出现脏话、侮辱词汇、负面、敏感词（短语）。回复的最后需要进行劝导，以委婉语气劝诫其不要使用或者谨慎使用负面词汇（短语）\n \
        你是一个经验丰富的政治、军事、法律领域的专家，请你对提供的问题进行意图理解和分析，然后根据【参考知识】【回复策略】【安全提示】生成回复，\
            不用输出分析过程，不要输出参考知识标识、回复策略标识、安全提示标识，直接输出回复。【问题】是{q}]"}
    ]
    }

    res = requests.post(url, headers=headers, data=json.dumps(data1))
    try:
        res=res.json()
        if res['choices'] and 'content' in res['choices'][0]:
            res=res['choices'][0]['content']
        else:
            res=''
        print('query>>', q,'\n', '参考知识>>', obs, '\n', '回复策略>>', reflect, '\n', '最终回复>>', res)
        df.loc[i, 'answer_w_reflect'] = res
    except:
        continue
    
df.to_csv('/mnt/pfs-guan-ssai/nlu/jiajuntong/code/rag_tool/rag2/src/q_api_obs_lab/wenxin_safety_reflect_w_answer.csv', index=False,encoding='utf-8-sig')  
# df['answer'] = ''

# for i in range(df.shape[0]):
#     q = df.iloc[i]['user-query']
#     obs = df.iloc[i]['observation']
#     reflect = df.iloc[i]['reflect']
#     data = {
#     "messages": [
#         {"role": "user", "content": f"【参考知识】{obs}\n【回复策略】{reflect}\n【安全提示】当问题中出现脏话、侮辱词、负面词、敏感词（短语），\
#             回答过程尽量不要直接出现脏话、侮辱词汇、负面、敏感词（短语）。回复的最后需要进行劝导，以委婉语气劝诫其不要使用或者谨慎使用负面词汇（短语）\n \
#         你是一个经验丰富的政治、军事、法律领域的专家，请你对提供的问题进行意图理解和分析，然后根据【参考知识】【回复策略】【安全提示】生成回复，\
#             不用输出分析过程，不要输出参考知识标识、回复策略标识、安全提示标识，直接输出回复。【问题】是{q}]"}
#     ]
#     }
#     # print(data)
#     headers = {
#     # 'User-Agent': 'Mozilla/5.0',
#     'Content-Type': 'application/json',
#     'Accept': 'application/json',  # 指定希望返回 JSON
#     # 可以添加其他请求头
# }
#     res = requests.post(url, headers=headers, data=json.dumps(data))
#     try:
#         res=res.json()
#         if res['choices'] and 'content' in res['choices'][0]:
#             res=res['choices'][0]['content']
#         else:
#             res=''
#         print('query>>', q,'\n', '参考知识>>', obs, '\n', '回复策略>>', reflect, '\n', '最终回复>>', res)
#         df.loc[i, 'answer_w_reflect'] = res
#     except:
#         continue
# df.to_csv('/mnt/pfs-guan-ssai/nlu/jiajuntong/data/lab3/wenxin_safety300_reflect.csv', index=False,encoding='utf-8-sig')
