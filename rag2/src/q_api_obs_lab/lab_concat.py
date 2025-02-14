import pandas as pd
import re

# 读取两个CSV文件
df_instruction = pd.read_csv('/mnt/pfs-guan-ssai/nlu/jiajuntong/code/mindgpt_sft_tools/Eval/data/0115_inf_safety_api_obs/instruction.csv')
df_wenxin = pd.read_csv('wenxin_safety_reflect_w_answer_filter.csv')

# 从query列提取user内容
def extract_user_query(query):
    match = re.search(r'\[unused0\]user\n(.*?)\[unused1\]', query, re.DOTALL)
    return match.group(1) if match else None

# 处理instruction.csv
df_instruction['user-query'] = df_instruction['query'].apply(extract_user_query)
df_instruction['mindgpt_res'] = df_instruction['res']
# 合并数据框
merged_df = pd.merge(
    df_wenxin[['user-query', 'reflect', 'answer_w_reflect']], 
    df_instruction[['user-query', 'mindgpt_res']], 
    on='user-query', 
    how='outer'
)

# 保存结果
merged_df.to_csv('./merged_results.csv', index=False,encoding='utf-8-sig')


# import pandas as pd
# import re
# from fuzzywuzzy import fuzz
# from fuzzywuzzy import process

# # 读取两个CSV文件
# df_instruction = pd.read_csv('/mnt/pfs-guan-ssai/nlu/jiajuntong/code/mindgpt_sft_tools/Eval/data/0115_inf_safety_api_obs/instruction.csv')
# df_wenxin = pd.read_csv('wenxin_safety_reflect_w_answer_filter.csv')

# # 从query列提取user内容
# def extract_user_query(query):
#     match = re.search(r'\[unused0\]user\n(.*?)\[unused1\]', query, re.DOTALL)
#     return match.group(1).strip() if match else None

# # 处理instruction.csv
# df_instruction['user-query'] = df_instruction['query'].apply(extract_user_query)
# df_instruction['mindgpt_res'] = df_instruction['res']
# # 模糊匹配函数
# def fuzzy_match(x, choices, min_score=80):
#     matched = process.extractOne(x, choices, scorer=fuzz.ratio)
#     return matched[0] if matched[1] >= min_score else x

# # 对df_instruction中的user-query进行模糊匹配
# wenxin_queries = df_wenxin['user-query'].tolist()
# df_instruction['user-query'] = df_instruction['user-query'].apply(
#     lambda x: fuzzy_match(x, wenxin_queries) if pd.notna(x) else x
# )

# # 合并数据框
# merged_df = pd.merge(
#     df_wenxin[['user-query', 'reflect', 'answer_w_reflect']], 
#     df_instruction[['user-query', 'mindgpt_res']], 
#     on='user-query', 
#     how='outer'
# )

# # 保存结果
# merged_df.to_csv('./lab2_results.csv', index=False,encoding='utf-8-sig')