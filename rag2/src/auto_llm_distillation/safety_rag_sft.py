import pandas as pd
# pd.set_option('display.max_rows', 100) #设置为80
# pd.set_option('display.max_colwidth', 1500) #设置为80

from tqdm import tqdm
tqdm.pandas() 

import sys
sys.path.append('../')
from tool_llm_response.call_llm_with_zny import CallLLMByZny,ZnyConfig
from tool_kg_search.get_1b_output import get_thought_api
from tool_kg_search.get_api_obs import get_api_df, get_obs_df
from tool_kg_search.search_http_tool import SearchByHttpTool
from tool_kg_search.kgsearch_llm_obs import *
from tool_rag_generation.data_format import DataFormat

# df = pd.read_json('/mnt/pfs-guan-ssai/nlu/chenjun18/data/sft_data/zhwr_3684_v2.json', lines=True)
# df = pd.read_csv('/mnt/pfs-guan-ssai/nlu/chenjun18/data/202410/zhwr_1031_part2.csv')
# df = pd.read_csv('/mnt/pfs-guan-ssai/nlu/chenjun18/data/202410/zhwr_1031_part3.csv')
# df = pd.read_json('/mnt/pfs-guan-ssai/nlu/chenjun18/data/202410/zhwr_1031_part3.json', lines=True)
df = pd.read_csv('/mnt/pfs-guan-ssai/nlu/jiajuntong/code/rag_tool/rag2/src/auto_llm_distillation/query_0115.csv',encoding='utf-8-sig')
df['user-query'] = df['user-query'].astype(str)

print(df.shape)
df.drop_duplicates(inplace=True)
print(df.shape)
# df = pd.read_json('/mnt/pfs-guan-ssai/nlu/chenjun18/data/sft_data/zhwr_80_rag_v1.json', lines=True)


# NLU_API_URL = 'http://172.24.139.47:16073/ligpt_with_api/search'
# NLU_API_URL = 'http://172.24.139.202:16073/ligpt_with_api/search'
NLU_API_URL = 'http://172.24.169.150:16073/ligpt_with_api/search'

# 调用api
df_api = get_api_df(df, col_query='user-query', output_file='./api_0115.csv', 
                    url= NLU_API_URL)

# df_api = pd.read_csv('api_6000.csv')
# print(df_api.shape)
# df_api = df_api[df_api['thought'].str.len() > 6]
# df_api.drop_duplicates(inplace=True)
# print(df_api.shape)

# 调用obs
df_obs = get_obs_df(df_api, length_limit=3000, output_path='./obs_0115.csv') #length_limit 表示obs最大长度，超长pop
print (df_api, df_obs)