import pandas as pd
import requests
import json

df = pd.read_csv('/mnt/pfs-guan-ssai/nlu/chenjun18/rag_tool/rag2/src/auto_llm_distillation/obs_1000_1105.csv')
# df = pd.read_json('/mnt/pfs-guan-ssai/nlu/chenjun18/data/202411/kaiche.jsonl', lines=True)
# fw = jsonlines.open('/mnt/pfs-guan-ssai/nlu/chenjun18/rag_tool/rag2/src/auto_llm_distillation/wenxin_w_api_wo_obs_6000.jsonl', 'w')
df = pd.read_csv('/mnt/pfs-guan-ssai/nlu/chenjun18/data/202412/seqing_query_1209_part2.csv')
df = pd.read_csv('/mnt/pfs-guan-ssai/nlu/chenjun18/data/202412/涉敏_reflect_0103_v1.csv')
df = pd.read_csv('/mnt/pfs-guan-ssai/nlu/jiajuntong/code/rag_tool/rag2/src/auto_llm_distillation/obs_300.csv')
url = 'https://chenjun18-gpt4.fc.chj.cloud/wenxin/zh-conversation'
df['answer_w_reflect'] = ''
for i in range(df.shape[0]):
    q = df.iloc[i]['user-query']
    reflect = df.iloc[i]['answer']
    data = {
    "messages": [
        {"role": "user", "content": f"【回复策略】{reflect}\n根据【回复策略】不用输出分析过程，不要输出回复策略标识，直接输出回复。【问题】是{q}]"}
    ]
    }
    # print(data)
    headers = {
    # 'User-Agent': 'Mozilla/5.0',
    'Content-Type': 'application/json',
    'Accept': 'application/json',  # 指定希望返回 JSON
    # 可以添加其他请求头
}
    res = requests.post(url, headers=headers, data=json.dumps(data))
    try:
        res=res.json()
        if res['choices'] and 'content' in res['choices'][0]:
            res=res['choices'][0]['content']
        else:
            res=''
        print('query>>', q,'\n', '回复策略>>', reflect, '\n', '最终回复>>', res)
        df.loc[i, 'answer_w_reflect'] = res
    except:
        continue
df.to_csv('/mnt/pfs-guan-ssai/nlu/jiajuntong/data/lab3/wenxin_safety300_reflect.csv', index=False,encoding='utf-8-sig')