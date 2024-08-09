import pandas as pd
import csv
import os
import sys
import re
import ast
import json
import random
from tqdm import tqdm
sys.path.append('../')
from tool_kg_search.get_1b_output import get_thought_api
from tool_kg_search.search_http_tool import SearchByHttpTool
from tool_kg_search.kgsearch_llm_obs import *

tqdm.pandas()

#########
# 获取 thought api
def modify_name_key(data):
    # 遍历列表中的每个字典
    for entry in data:
        # 如果字典里面有 'name' 键
        if 'name' in entry:
            # 获取 'name' 键的值
            value = entry.pop('name')
            # 将键名改为 'apiname'，并赋值
            entry['apiname'] = value
    return data
    
def get_api_df(df,col_query, output_file, url='http://172.24.139.95:16073/ligpt_with_api/search') -> pd.DataFrame:
    """
    根据query和api_name，获取thought和api dict
    修改name to apiname，与obs接口对应
    """
    # 使用 lambda 表达式处理每一行，并检查 api_name 是否为空列表
    df[['Thought', 'API']] = df.progress_apply(
        lambda row: get_thought_api(
            row[col_query], 
            url
        ), 
        axis=1, 
        result_type='expand'
    )

    df['API'] = df['API'].apply(modify_name_key)
    df.to_csv(output_file,index=False)
    df = df.astype(str)
    return df

# def get_api_df(df, col_query, col_apiname, output_file, url='http://172.24.139.95:16073/ligpt_with_api/search'):
#     """
#     根据query和api_name，获取thought和api dict
#     修改name to apiname，与obs接口对应
#     逐行调用 API 并写入 CSV
#     """

#     file_exists = os.path.isfile(output_file) and os.path.getsize(output_file) > 0 
#     # 打开 CSV 文件，并写入表头
#     with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
#         writer = csv.writer(csvfile)
#         # 如果文件不存在或为空，则写入表头
#         if not file_exists:
#             writer.writerow(['user-query', 'apiname', 'Thought', 'API'])  

#         for index, row in df.iterrows():
#             query = row[col_query]            
#             # 调用 API 获取数据
#             thought, api_dict = get_thought_api(query, url)
            
#             # 处理 API 字典
#             api_dict = modify_name_key(api_dict)
            
#             # 写入一行数据到 CSV
#             writer.writerow([query, api_name, thought, api_dict]) 
            
#     return pd.read_csv(output_file)


#########
# 获取obs
def get_raw_obs(query: str,api: list[dict],k: int) -> list[list[dict]]:
    """
    Args:
        query: str 用户query
        api: list of dict, API function call 格式
        k: 随机数，随机返回obs的数量 5-20随机整数
    Returns:
        multi_llm_obs: list[list[dict]] 搜索接口返回的原始内容
    """
    search_env = 'app_dev'     #调用的搜索环境，取值: arch | dev | liping | das | faq ... 
    control_param =  {      # 搜索需要的参数, 默认都不打开
        # "disable-bing-cache":"true",      # 是否不要bing cache
    }
    search_http_tool = SearchByHttpTool(search_env=search_env, 
                                        limit=k,       # -1 则采用搜索内部默认, >0则手动指定 
                                        control_param=control_param)
    multi_llm_obs = get_obs(search_http_tool, query, api)
    return multi_llm_obs


def parser_obs(multi_llm_obs: list[list[dict]], length_limit: int = 20000) -> list[list]:
    """
    解析单个搜索结果
    Args:
        multi_llm_obs:list[list[dict]] 搜索接口返回的原始内容
        length_limit: int 结果的长度限制
    Returns:
        all_ls: list[list] 2D list
    """
    all_ls = []
    for api_obs in multi_llm_obs:
        single_ls = []
        for kg_search in api_obs:
            single_ls.append(kg_search['content'])
        all_ls.append(single_ls)

    # Ensure the length of all_ls does not exceed length_limit
    while len(str(all_ls)) > length_limit:
        all_ls.pop()
    
    # Shuffle single_ls after confirming the length of all_ls
    for single_ls in all_ls:
        random.shuffle(single_ls)

    return all_ls

def get_obs_df(df, length_limit: int = 20000, output_path='processed_data.csv'):
    new_obs_ls = []
    raw_obs_ls = []
    for i in tqdm(range(len(df))):
        query = df.iloc[i]['user-query']
        api = df.iloc[i]['API']
        if isinstance(api, str):
            api = ast.literal_eval(api) 
        k = random.choice([5,10,15,20])
        multi_llm_obs = get_raw_obs(query,api,k)
        new_obs_ls.append(parser_obs(multi_llm_obs, length_limit))
        raw_obs_ls.append(multi_llm_obs)
    
    df.loc[:,'observation'] = new_obs_ls
    # df.loc[:,'raw_observation'] = raw_obs_ls
    df.to_csv(output_path,index=False)
    df = df.astype(str)
    return df


# def get_obs_df(df, length_limit: int = 20000, output_path='processed_data.csv'):
#     """
#     处理数据帧，逐行请求并写入CSV文件，使用csv模块避免数据分割错误。

#     Args:
#         df (pd.DataFrame): 输入数据帧，包含 'user-query' 和 'API' 列。
#         length_limit (int):  观察结果长度限制，默认为 20000。
#         output_path (str): 输出 CSV 文件路径，默认为 'processed_data.csv'。
#     """
#     file_exists = os.path.isfile(output_path) and os.path.getsize(output_path) > 0 
#     with open(output_path, 'a', newline='') as f:
#         writer = csv.writer(f)
#         # 如果文件不存在或为空，则写入表头
#         if not file_exists:
#             writer.writerow(["user-query", "API", "observation", "raw_observation"])  # 写入标题行

#         for i in tqdm(range(len(df))):
#             try:
#                 query = df.iloc[i]['user-query']
#                 api = df.iloc[i]['API']
#                 if isinstance(api, str):
#                     api = ast.literal_eval(api)

#                 k = random.choice([5, 10, 15, 20])
#                 multi_llm_obs = get_raw_obs(query, api, k)  
#                 new_obs = parser_obs(multi_llm_obs, length_limit)

#                 # 使用csv.writer写入数据，避免逗号分割问题
#                 writer.writerow([query, api, new_obs, multi_llm_obs]) 
#             except Exception as e:
#                 print(f"处理第 {i} 行数据时出错: {e}")

#     return pd.read_csv(output_path)