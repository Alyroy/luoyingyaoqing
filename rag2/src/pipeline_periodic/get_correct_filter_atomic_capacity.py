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
import gc
from tqdm import tqdm
from datetime import datetime
sys.path.append('../')
from tool_llm_response.call_llm_with_zny import CallLLMByZny,ZnyConfig
from tool_llm_response.call_llm_with_vllm import CallLLMByVllm,VllmConfig
from tool_kg_search.get_api_obs import get_api_df, get_obs_df
from tool_rag_generation.data_format import DataFormat,gen_multi_sft_data

sys.path.append('../../')
from common import utils,utils_log

tqdm.pandas()


def remove_section(text):
    try:
        start_marker = "## 回答问题："
        end_marker = "## 相关性"
        start_index = text.find(start_marker)
        end_index = text.find(end_marker)
        
        if start_index != -1 and end_index != -1 and end_index > start_index:
            return text[:start_index].strip() + '\n' + text[end_index:].strip()
        return text

    except Exception as e:
        print(f"Error processing remove answer: {e}")
        return text
    
def extract_relevance_section(text):
    try:
        start_marker = "## 相关性"
        end_marker = "## 真实性"
        start_index = text.find(start_marker)
        end_index = text.find(end_marker)
        
        if start_index != -1 and end_index != -1 and end_index > start_index:
            return text[start_index:end_index].strip()
    
    except Exception as e:
        print(f"Error processing filter relevance: {e}")
        return text

def extract_authenticity_section(text):
    try:
        start_marker = "## 真实性"
        end_marker = "综上"
        start_index = text.find(start_marker)
        end_index = text.find(end_marker)
        
        if start_index != -1 and end_index != -1 and end_index > start_index:
            return text[start_index:end_index].strip()
    
    except Exception as e:
        print(f"Error processing filter authenticity: {e}")
        return text


def add_fixed_columns(df, source, task_name, date=None):
    """
    添加固定列到DataFrame并设置创建和更新日期。

    Args：
        df (pd.DataFrame): 要更新的DataFrame。
        source (str): 数据的来源。
        task_name (str): 任务名称。
        date (str, optional): 日期格式为 'YYYY/MM/DD'，默认使用当前日期。

    Returns：
        pd.DataFrame: 更新后的DataFrame，带有附加列。
    """
    df['source'] = source
    df['task_name'] = task_name
    df['produce_source'] = '自动化评估数据'

    if date is None:
        date = datetime.now().strftime('%Y/%m/%d')

    df['create_time'] = date
    df['update_time'] = date
    df['create_user'] = 'renhuimin'
    df['update_user'] = 'renhuimin'
    df['is_reviewed'] = '否'
    df['update_content'] = ''

    return df


def process_dataframe(df, eval_col, ans_col, transform_func, prompt_file, prompt_folder, source, task_name, date):
    """
    处理DataFrame并添加固定列。

    参数:
        df (pd.DataFrame): 要处理的DataFrame。
        eval_col (str): 评估列名。
        ans_col (str): 答案列名。
        transform_func (callable): 用于转换的函数。
        prompt_folder (str): 提示符文件夹路径。
        source (str): 数据来源。
        task_name (str): 任务名称。
        date (str): 日期。

    返回:
        pd.DataFrame: 处理后的DataFrame。
    """
    # 读取提示语
    df_prompt = pd.read_csv(os.path.join(prompt_folder, prompt_file), usecols=['prompts'])
    user_prompt_ls = df_prompt['prompts'].tolist()

    # 转换数据
    df = df.rename(columns={ans_col: 'observation'})   
    df = df[~df['observation'].isna()]
    df['observation'] = df['observation'].apply(lambda x: [[x.replace('[unused8]', '\n')]])
    df['assistant'] = df[eval_col].apply(transform_func)
    df = df.astype(str)

    # 更新'user-query'列
    df['user'] = df['user-query']
    df['user-query'] = df['user'].apply(
        lambda x: f"{random.choice(user_prompt_ls)}\n 大模型回复参考observation, 用户问题是：{x}"
    )

    # 添加固定列
    df = add_fixed_columns(df, source, task_name, date)
    return df


def main(input_path: str, output_path: str, ans_col: str, eval_col: str, prompt_folder: str, source: str, date: str = None):
    """
    处理评估数据并将转换后的数据输出到文件。

    Args:
        input_path (str): 输入数据文件的路径。
        output_path (str): 保存处理后数据的路径前缀。
        ans_col (str): 要评估的列名。
        eval_col (str): 评估列名。
        prompt_folder (str): 提示符CSV文件夹的路径。
        source (str): 数据来源名称。
        date (str, optional): 创建和更新日期，格式为 'YYYY/MM/DD'。默认值为None。

    Retuns:
        None
    """
    df = pd.read_csv(input_path)
    filter_col = '多模型筛选回复结果'
    df = df[df[filter_col] != -1]
    df = df[~df[ans_col].isna()]

    # 选取并重命名需要的列
    target_cols = ['uid', 'user-query', 'thought', 'api', ans_col, eval_col]
    used_cols = [col for col in target_cols if col in df.columns]
    
    # 处理全量数据，task_name设置为准确性评估
    df_full = df.copy()
    df_full = df_full[used_cols]
    df_full = process_dataframe(df_full, eval_col, ans_col, remove_section, 'synonyms_prompts_speech_sytle.csv', prompt_folder, source, '推理-知识推理-回复判断-准确性评估', date)
    df_full.to_csv(f'{output_path}_accuracy.csv', index=False, encoding='utf-8-sig')


    # 随机抽取50%的数据用于相关性评估
    df_relevance = df.sample(frac=0.5, random_state=42).copy()
    df_relevance = df_relevance[used_cols]
    df_relevance = process_dataframe(df_relevance, eval_col, ans_col, extract_relevance_section, 'synonyms_prompts_relevance.csv', prompt_folder, source, '推理-知识推理-回复判断-相关性评估', date)
    df_relevance.to_csv(f'{output_path}_relevance.csv', index=False, encoding='utf-8-sig')

    # 随机抽取50%的数据用于真实性评估
    df_authenticity = df.sample(frac=0.5, random_state=10086).copy()
    df_authenticity = df_authenticity[used_cols]
    df_authenticity = process_dataframe(df_authenticity, eval_col, ans_col, extract_authenticity_section, 'synonyms_prompts_authenticity.csv', prompt_folder, source, '推理-知识推理-回复判断-真实性评估', date)
    df_authenticity.to_csv(f'{output_path}_authenticity.csv', index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='处理原子能力的评估数据。')
    
    parser.add_argument('--input_path', type=str, required=True, help='输入数据文件的文件路径')
    parser.add_argument('--output_path', type=str, required=True, help='处理好数据的文件路径前缀')
    parser.add_argument('--ans_col', type=str, required=True, help='被评估列的列名')
    parser.add_argument('--eval_col', type=str, required=True, help='评估列的列名')
    parser.add_argument('--prompt_folder', type=str, required=True, help='同义改写后的prompt CSV 文件的路径')
    parser.add_argument('--source', type=str, required=True, help='数据的来源名称')
    parser.add_argument('--date', type=str, help='为空表示今天，创建和更新日期，格式为 "YYYY/MM/DD"')

    args = parser.parse_args()
    print(args)

    main(
        input_path=args.input_path,
        output_path=args.output_path,
        ans_col=args.ans_col,
        eval_col=args.eval_col,
        prompt_folder=args.prompt_folder,
        source=args.source,
        date=args.date
    )
    
    print("数据处理完成。")