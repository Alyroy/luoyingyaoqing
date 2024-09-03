import argparse
import sys
import os
import traceback
import random
import ast
from tqdm import tqdm
tqdm.pandas()
import pandas as pd
from data_processing import DataFilter
from base_distillation import BaseDistillation  # 引用共享模块
sys.path.append('../')
from tool_llm_response.call_llm_with_zny import CallLLMByZny, ZnyConfig
from tool_kg_search.get_api_obs import shuffle_obs
sys.path.append('../')
from common import utils,utils_log

class GPT4Distillation(BaseDistillation):
    def __init__(self, config):
        super().__init__(config)

    def process_obs(self, df):
        result = df['observation'].apply(lambda x: shuffle_obs(x))
        df['observation'] = result.apply(lambda x: x[0])
        df['obs_shuffle_type'] = result.apply(lambda x: x[1])
        return df

    def get_distillation_data(self, input_file):
        date = utils.extract_date_from_filename(input_file)
        print(f"Processing distillation data for date: {date}")
        
        is_single_flag = self.config['is_single_flag']
        is_rag_flag = self.config['is_rag_flag']
        outfolder = os.path.join(self.config['base_output_path'], date, f'raw/single_{str(is_single_flag)}_rag_{str(is_rag_flag)}')

        print(f"Input file: {input_file}")
        print(f"Output folder: {outfolder}")
        utils.create_directory(outfolder)

        prompt_path = self.get_prompt_path()
        filter_df = pd.DataFrame()
        try:
            df = pd.read_csv(input_file)
            df = df[~df['user-query'].isna()]
            # 筛选目标数据
            df = DataFilter.filter_bad_df(df, self.config)
            # 聚类去重
            query_ls = df.to_dict(orient='records')
            unique_queries_with_metadata = DataFilter.get_unique_query(query_ls, batch_size=64, BGE_MODEL_PATH=self.config['embedding_model_path'])            
            filter_df = pd.DataFrame(unique_queries_with_metadata)
            # 提取目标columns
            filter_df = DataFilter.get_task_usecols(filter_df)
            # shuffle obs
            filter_df = self.process_obs(filter_df)
            filter_df = filter_df.groupby('user-query').first().reset_index()
        except Exception as exc:
            print(f"Exception occurred while processing file: {input_file}")
            traceback.print_exc()

        df.to_csv(os.path.join(outfolder, f'{date}_raw_log_data.csv'), index=False, encoding='utf-8-sig')
        utils.split_and_save_df(filter_df, chunk_size=500, outfolder=os.path.join(outfolder, 'split_data'))
        self.check_and_generate_gpt4_files(outfolder, prompt_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='distillate gpt4 for app log.')
    parser.add_argument('--is_single_flag', dest='is_single_flag', action='store_true', help='是否为单轮数据')
    parser.add_argument('--no_is_single_flag', dest='is_single_flag', action='store_false', help='是否为多轮数据')
    parser.add_argument('--is_rag_flag', dest='is_rag_flag', action='store_true', help='是否为rag数据')
    parser.add_argument('--no_is_rag_flag', dest='is_rag_flag', action='store_false', help='不是rag数据')
    parser.add_argument('--model_url', type=str, help='模型URL')
    parser.add_argument('--model_name', type=str, help='模型名称')
    parser.add_argument('--prompt_path', type=str, help='单轮RAG提示路径')
    parser.add_argument('--input_file', type=str, default='/mnt/pfs-guan-ssai/nlu/wangxiaoyuan/online-data/', help='数据输入路径')
    parser.add_argument('--base_output_path', type=str, default='/mnt/pfs-guan-ssai/nlu/renhuimin/data/log_distillation/', help='基础输出路径')
    parser.add_argument('--embedding_model_path', type=str, default='/mnt/pfs-guan-ssai/nlu/wangxiaoyuan/bge-base-zh',help='bge路径')

    parser.set_defaults(is_single_flag=True, is_rag_flag=True)

    args = parser.parse_args()
    print(f"Arguments: {args}")

    config = {
        'is_single_flag': args.is_single_flag,
        'is_rag_flag': args.is_rag_flag,
        'model_name': args.model_name,
        'zny_config': ZnyConfig(
            url=args.model_url,
            model_name=args.model_name,
            temperature=0.5,
            max_retries=5,
            qps=3,
            max_concurrent=10,
            asyncio_flag=False,
            query_column_name='prompts',
            response_column_name='gpt4response'
        ),
        'input_file': args.input_file,
        'base_output_path': args.base_output_path,
        'prompt_path': args.prompt_path,
        'embedding_model_path': args.embedding_model_path
    }

    distillator = GPT4Distillation(config)
    distillator.get_distillation_data(args.input_file)