import argparse
import pandas as pd
import os
import sys
import re
import ast
import json
import random
import time
from datetime import datetime,timedelta
from tqdm import tqdm
sys.path.append('../')
from tool_llm_response.call_llm_with_zny import CallLLMByZny,ZnyConfig
from tool_kg_search.get_1b_output import ApiResult,get_thought_api
from tool_kg_search.search_http_tool import SearchByHttpTool
from tool_kg_search.kgsearch_llm_obs import *
sys.path.append('../../')
from common import utils
tqdm.pandas()

def filter_df(df):
    """
    筛选目标筛选
    """
    usecols = ['user-query', 'assistant', 'api', 'thought', 'observation', 'system', 'apiname',
           'assistant_relevance', 'assistant_logic','generalized_question_from_user', 'generalized_question_from_assistant']
    filter_df = df[
    ((df['is_rag'] == 1) | (df['is_rag'] == -1)) & 
    ((df['is_single_turn'] == 1) | (df['is_single_turn'] == -1)) &
    ((df['is_chara'] == 0) | (df['is_chara'] == -1)) &
    ((df['is_math'] == 0) | (df['is_math'] == -1)) &
    ((df['is_child'] == 0) | (df['is_child'] == -1)) &
    ((df['is_simplified'] == 0) | (df['is_simplified'] == -1)) &
    ((df['is_guidance'] == 0) | (df['is_guidance'] == -1)) &
    (df['source'] == 'real')
    ]
    filter_df = filter_df[usecols]
    return filter_df


def get_prompts_df(df: pd.DataFrame, oneshot_prompt:str) -> pd.DataFrame:
    df['prompts'] = df.apply(lambda row: oneshot_prompt + f"""
    ---
    下面是给出的实际问题：
    system:
    {row['system']}
    Observation:
    {row['observation']}
    Question:
    {row['user-query']}
    Answer：
    """, axis=1)
    
    return df

# 定义一个解析函数
def parse_content(text):
    match = re.search(r'\{\{(.+?)\}\}', text, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return None
        
def get_gpt4df(df, outpt, prompt_path):
    oneshot_prompt = utils.read_txt(prompt_path)
    
    prompt_df = get_prompts_df(df, oneshot_prompt)
    gpt4_df = call_zny.get_gpt4api_df(prompt_df)
    gpt4_df.to_csv(outpt,index=False)

    gpt4_df['parser_gpt4'] = gpt4_df['gpt4response'].apply(parse_content)
    gpt4_df.to_csv(outpt,index=False)


def get_distillation_data(date):
    infolder = f'/mnt/pfs-guan-ssai/nlu/wangxiaoyuan/online-data/{date}/'
    outfolder = f'/mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/rag2.0/data/distillation_data/log_data/{date}/raw/'
    utils.create_directory(outfolder)
    files = ['prod/'+f'{date}_rule_labeled.csv.gpt_labeled.csv',
             'test/'+f'{date}_rule_labeled.csv.gpt_labeled.csv']
    dl = []
    for file in files:  
        try:
            df = pd.read_csv(infolder + file)
            df = filter_df(df)
            dl.append(df)
        except Exception as e:
            print('filter data error',e)
    df = pd.concat(dl)
    # 每500条，split files 保证gpt4正常生产
    utils.split_and_save_df(df, chunk_size=500, outfolder=outfolder+'split_data/')

    try:
        call_zny = CallLLMByZny(config)
        prompt_path='/workspace/renhuimin/pro_rag/conf/generation_prompts/generation_单轮日志蒸馏.txt'
        utils.create_directory(outfolder+'gpt4_data/')
        split_files = sorted([f for f in os.listdir(outfolder+'split_data/') if '.ipynb_checkpoints' not in f])
        for file in split_files:
            print(file)
            df_ = pd.read_csv(outfolder+'split_data/'+file)
            get_gpt4df(df_, outfolder+'gpt4_data/'+'gpt4_'+file, prompt_path=prompt_path)
            time.sleep(60)
    except Exception as e:
        print('蒸馏错误',e)


config = ZnyConfig(
    url = 'https://rhm-gpt4.fc.chj.cloud/gpt4o/conversation', # 智能云GPT api
    model_name = 'gpt4o',
    temperature = 0.5, # llm输出温度，zny下的gpt4基本无效，因为是全球节点，还是会有随机性
    max_retries = 5, # 调用gpt报错后最多重试 max_retries 次
    qps = 5, # 多线程 or 异步多线程下，qps，不要超过5
    max_concurrent = 10, # 异步多线程参数，一般10或者20，太大会接口超过qps
    asyncio_flag = False, # True=异步多线程，只能python调用；False=普通多线程，Jupyter或者python均可
    query_column_name = 'prompts', # llm模型输入列名
    response_column_name = 'gpt4response' # llm模型输出列名
)

call_zny = CallLLMByZny(config)


if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='distillate gpt4 for app log.')
    
    # 添加参数
    parser.add_argument('--date', type=str, help='文件日期，格式为 YYYY-MM-DD')
    args = parser.parse_args()
    print(args)

    # step 1 蒸馏数据
    get_distillation_data(args.date)