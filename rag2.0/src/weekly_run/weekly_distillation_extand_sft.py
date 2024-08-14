from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer

import argparse
import pandas as pd
import numpy as np
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
from tool_kg_search.search_http_tool import SearchByHttpTool
from tool_kg_search.kgsearch_llm_obs import *
from tool_kg_search.get_api_obs import get_api_df, get_obs_df
sys.path.append('../../')
from common import utils

tqdm.pandas()

from weekly_distillation_log_sft_dpo import filter_df, get_gpt4df

# 获取泛化query
def get_query_ls(raw_query_ls):
    query_ls = []
    for q in raw_query_ls:
        query_ls.extend(q.split('\n'))
        
    query_ls = [q for q in query_ls if "-" not in q and "**" not in q and q != ""]
    query_ls = list(set(query_ls))
    return query_ls

def get_new_query_ls(df):
    query_ls1_ = df['generalized_question_from_user'].to_list()
    query_ls2_ = df['generalized_question_from_assistant'].to_list()
    
    query_ls1 = get_query_ls(query_ls1_)
    query_ls2 = get_query_ls(query_ls2_)
    query_ls = query_ls1+query_ls2
    
    return query_ls

# 对泛化query聚类筛选
def get_unique_query(query_ls):
    # 1. 文本表示
    BGE_MODEL_PATH = "/mnt/pfs-guan-ssai/nlu/wangxiaoyuan/bge-base-zh"
    model = SentenceTransformer(BGE_MODEL_PATH).cuda()
    embeddings = model.encode(query_ls)
    
    # 2. 计算相似度
    similarity_matrix = cosine_similarity(embeddings)
    
    # 3. 使用聚类算法来识别不相似的query
    threshold = 0.9  # 设定相似度阈值，小于这个值的query相似的
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1-threshold,  # 将距离阈值设为1-相似度阈值
        metric='cosine',
        linkage='average'
    )
    labels = clustering.fit_predict(embeddings)
    
    # 取每个簇第一个元素
    unique_queries = []
    for label in np.unique(labels):
        index = np.where(labels == label)[0][0]
        unique_queries.append(query_ls[index])
        
    return unique_queries

# 执行数据筛选、api、obs、蒸馏
def get_distillation_data(date,api_url):
    # filter data
    infolder = f'/mnt/pfs-guan-ssai/nlu/wangxiaoyuan/online-data/{date}/'
    outfolder = f'/mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/rag2.0/data/distillation_data/log_data/{date}/extension/'
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
            print(e)
    df = pd.concat(dl)
    df.to_csv(outfolder+f'{date}_log_data.csv',index=False)

    # 获取区分大的query
    query_ls = get_new_query_ls(df)
    unique_queries = get_unique_query(query_ls)
    
    # 获取API及obs
    query_df = pd.DataFrame(unique_queries,columns=['user-query'])
    df_api = get_api_df(query_df, col_query='user-query',
                        output_file=outfolder+'api.csv', url=api_url)
    
    df_api = df_api[df_api['api']!='[]']
    df_obs = get_obs_df(df_api, length_limit=20000, output_path=outfolder+f'{date}_obs.csv')
    df_obs = pd.read_csv(outfolder+f'{date}_obs.csv')
    
    # # 每500条，split files 保证gpt4正常生产
    df_obs = df_obs[~df_obs['observation'].isin(['[[]]','[]'])]
    system = "你是一个名字叫做理想同学的AI数字生命体。\n理想同学是一个可靠的智能家庭助手，由理想汽车智能空间部门创造。理想同学使用了理想公司自研MindGPT大语言模型技术。\n理想同学能够理解人类的指令和意图，并且给出合理的、切合问题的、没有歧视、中立的、安全的回复。\n" \
        + "请根据以下文本写一个合适的回复。"
    df_obs['system'] = system
    utils.split_and_save_df(df_obs, chunk_size=500, outfolder=outfolder+'split_data/')
    
    try:
        call_zny = CallLLMByZny(config)
        prompt_path='/workspace/renhuimin/pro_rag/conf/generation_prompts/generation_单轮日志蒸馏.txt'
        utils.create_directory(outfolder+'gpt4_data/')
        split_files = sorted([f for f in os.listdir(outfolder+'split_data/') if '.ipynb_checkpoints' not in f])
        for file in split_files:
            print(file)
            df_ = pd.read_csv(outfolder+'split_data/'+file)
            get_gpt4df(df_, outfolder+f'gpt4_data/{date}_gpt4_'+file, prompt_path=prompt_path)
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
if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='distillate gpt4 for app log.')
    
    # 添加参数
    parser.add_argument('--date', type=str, help='文件日期，格式为 YYYY-MM-DD')
    parser.add_argument('--api_url', type=str, help='API调用url')
    args = parser.parse_args()
    print(args)

    # step 1 蒸馏数据
    get_distillation_data(args.date,args.api_url)