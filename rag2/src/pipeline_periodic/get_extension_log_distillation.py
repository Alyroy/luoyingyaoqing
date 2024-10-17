import argparse
import sys
import os
import subprocess
import traceback
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import re
import ast
from data_processing import DataFilter
from base_distillation import BaseDistillation
sys.path.append('../')
from tool_kg_search.get_api_obs import get_api_df, get_obs_df
from tool_llm_response.call_llm_with_zny import CallLLMByZny, ZnyConfig
sys.path.append('../../')
from common import utils


class GPT4DistillationExtended(BaseDistillation):
    def __init__(self, config):
        super().__init__(config)

    def split_you_time_command_queries(self, query_ls: list):
        """
        检查query里是否包含时效性、任务型、人设，去掉这类query，以免污染模型
        """
        clean_queries, you_command_time_queries = [], []
        time_word_list = ['今天', '昨天', '明天', '本周', '上周', '今年', '明年', '去年', '最近', '目前', '现在']
        command_pattern = "^(播放|我想听|我想看|我要听|我要看|打开|关闭|找一下|找一家|搜一下|看一下|放个|停止|开始)"

        for query in query_ls:
            bad_query_dict = query.copy()  # Copy the entire dictionary to keep metadata
    
            # 检查时效性
            if any(time_word in query['user-query'] for time_word in time_word_list):
                bad_query_dict['tag'] = '时效性'
            # 检查人设
            elif '你' in query['user-query']:
                bad_query_dict['tag'] = '人设你'
            # 检查任务型
            elif re.search(command_pattern, query['user-query']):
                bad_query_dict['tag'] = '任务型'
            else:
                clean_queries.append(query)
                continue

            you_command_time_queries.append(bad_query_dict)

        return clean_queries, you_command_time_queries

    def get_query_ls_with_metadata(self, df, query_column_name, task_name_column='task_name', uid_column='uid'):
        """
        从 DataFrame 提取 query，并保留 task-name 和 uid 的元数据
        """
        query_ls_with_metadata = []
        uid_exists = uid_column in df.columns  # Check if uid column exists
        task_exists = task_name_column in df.columns
        context_exists = 'context' in df.columns
        for index, row in df.iterrows():
            if pd.isnull(row[query_column_name]):
                continue  # Skip if the query column entry is None or NaN
                
            queries = row[query_column_name].split('\n')
            for query in queries:
                if "-" not in query and "**" not in query and query != "" and "问题：" not in query:
                    query_ls_with_metadata.append({
                        'user-query': query,
                        'task_name': row[task_name_column] if task_exists else "",
                        'uid': row[uid_column] if uid_exists else "",
                        'context': row['context'] if context_exists else "",
                        'datetime': row['datetime'] if context_exists else ""
                    })
        return query_ls_with_metadata

    def get_new_query_ls_with_metadata(self, df):
        query_ls1 = self.get_query_ls_with_metadata(df, 'generalized_question_from_user')
        query_ls2 = self.get_query_ls_with_metadata(df, 'generalized_question_from_assistant')
        query_ls = query_ls1 + query_ls2
        return query_ls

    def update_user_query_with_context(self, row):
        context = row['context']
        query = row['user-query']
        new_query = []
        if isinstance(context, str):
            context = ast.literal_eval(context)
        if len(context)==0:
            new_query.append(query)
        else:
            # 循环提取每个字典中的 user 和 assistant 内容
            for entry in context:
                new_query.append(entry['user'])
                new_query.append(entry['assistant'])
            new_query.append(query)
        return new_query
    
    def get_distillation_data(self, input_file):
        date = utils.extract_date_from_filename(input_file)
        print(f"Processing distillation data for date: {date}")

        is_single_flag = self.config['is_single_flag']
        is_rag_flag = self.config['is_rag_flag']
        outfolder = os.path.join(self.config['base_output_path'], date, f'extension/single_{str(is_single_flag)}_rag_{str(is_rag_flag)}')

        print(f"Input file: {input_file}")
        print(f"Output folder: {outfolder}")
        utils.create_directory(outfolder)

        filter_df = pd.DataFrame()
        print(f"Reading input file: {input_file}")
        try:
            df = pd.read_csv(input_file)
            df = df[~df['taskformer-model-13b-input'].isna()]
            filter_df = DataFilter.filter_bad_df(df, self.config)
            filter_df = DataFilter.get_task_usecols(filter_df)
            filter_df = filter_df.groupby('user-query').first().reset_index()
        except Exception as exc:
            print(f"Exception occurred while processing file: {input_file}")
            traceback.print_exc()
        
        df.to_csv(os.path.join(outfolder, f'{date}_raw_log_data.csv'), index=False, encoding='utf-8-sig')
        print('raw log len:', len(df))

        # 过滤任务型、你的泛化query
        query_ls_with_metadata = self.get_new_query_ls_with_metadata(filter_df)
        clean_queries_with_metadata, you_command_time_queries_dict = self.split_you_time_command_queries(query_ls_with_metadata)
        output_jsonl_file = os.path.join(outfolder, f'{date}_时效人设任务型query.jsonl')
        with open(output_jsonl_file, 'w', encoding='utf-8') as file:
            for item in you_command_time_queries_dict:
                file.write(json.dumps(item, ensure_ascii=False) + '\n')
        print('时效人设任务型query 数量：',len(you_command_time_queries_dict))

        # 获取unique query，每次最多只泛化500条case
        unique_queries_with_metadata = DataFilter.get_unique_query(clean_queries_with_metadata, batch_size=64, BGE_MODEL_PATH=self.config['embedding_model_path'])
        unique_queries_df = pd.DataFrame(unique_queries_with_metadata)
        unique_queries_df.to_csv(os.path.join(outfolder, f'{date}_unique_queries.csv'), index=False, encoding='utf-8-sig')
        unique_queries_df = unique_queries_df.sample(n=min(500,len(unique_queries_df)))
        print('unique queries len:',len(unique_queries_df))

        # 生成api thought obs gpt4
        print('调用API')
        unique_queries_df['new_query'] = unique_queries_df.apply(self.update_user_query_with_context,axis=1)
        df_api = get_api_df(unique_queries_df, col_query='new_query', output_file=os.path.join(outfolder, f'{date}_api.csv'), url=self.config['api_url'])
        df_api = df_api[df_api['api'] != '[]']
        print('有效api数量',len(df_api))
        print('调用obs')
        df_obs = get_obs_df(df_api, length_limit=20000, output_path=os.path.join(outfolder, f'{date}_obs.csv'))
        df_obs = pd.read_csv(os.path.join(outfolder, f'{date}_obs.csv'))

        df_obs = df_obs[~df_obs['observation'].isin(['[[]]', '[]'])]
        # df_obs['datetime'] = ''
        print('有效obs数量',len(df_obs))
        
        system = (
            "你是一个名字叫做理想同学的AI数字生命体。\n"
            "理想同学是一个可靠的智能家庭助手，由理想汽车智能空间部门创造。"
            "理想同学使用了理想公司自研MindGPT大语言模型技术。\n"
            "理想同学能够理解人类的指令和意图，并且给出合理的、切合问题的、没有歧视、"
            "中立的、安全的回复。\n请根据以下文本写一个合适的回复。"
        )
        df_obs['system'] = system
        utils.split_and_save_df(df_obs, chunk_size=500, outfolder=os.path.join(outfolder, 'split_data'))
        prompt_path = self.get_prompt_path()
        self.check_and_generate_gpt4_files(outfolder, prompt_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distillate GPT-4 for app extended log.')
    parser.add_argument('--api_url', type=str, help='API调用url')
    parser.add_argument('--embedding_model_path', type=str, default='/mnt/pfs-guan-ssai/nlu/wangxiaoyuan/bge-base-zh',help='bge路径')
    parser.add_argument('--is_single_flag', dest='is_single_flag', action='store_true', help='是否为单轮数据')
    parser.add_argument('--no_is_single_flag', dest='is_single_flag', action='store_false', help='是否为多轮数据')
    parser.add_argument('--is_rag_flag', dest='is_rag_flag', action='store_true', help='是否为rag数据')
    parser.add_argument('--no_is_rag_flag', dest='is_rag_flag', action='store_false', help='不是rag数据')
    parser.add_argument('--model_url', type=str, help='模型URL')
    parser.add_argument('--model_name', type=str, help='模型名称')
    parser.add_argument('--prompt_path', type=str, help='提示路径')
    parser.add_argument('--input_file', type=str, default='/mnt/pfs-guan-ssai/nlu/wangxiaoyuan/online-data/', help='数据输入路径')
    parser.add_argument('--base_output_path', type=str, default='/mnt/pfs-guan-ssai/nlu/renhuimin/data/log_distillation/', help='基础输出路径')

    parser.set_defaults(is_single_flag=True, is_rag_flag=True)
    args = parser.parse_args()
    print(f"Arguments: {args}")

    config = {
        'is_single_flag': args.is_single_flag,
        'is_rag_flag': args.is_rag_flag,
        'model_name': args.model_name,
        'api_url': args.api_url,
        'embedding_model_path': args.embedding_model_path,
        'prompt_path': args.prompt_path,
        'zny_config': ZnyConfig(
            url=args.model_url,
            model_name=args.model_name,
            temperature=0.5,
            max_retries=5,
            qps=2,
            max_concurrent=10,
            asyncio_flag=False,
            query_column_name='prompts',
            response_column_name='gpt4response'
        ),
        'input_file': args.input_file,
        'base_input_path': '/mnt/pfs-guan-ssai/nlu/wangxiaoyuan/online-data/',
        'base_output_path': args.base_output_path
    }

    distillator = GPT4DistillationExtended(config)
    distillator.get_distillation_data(args.input_file)