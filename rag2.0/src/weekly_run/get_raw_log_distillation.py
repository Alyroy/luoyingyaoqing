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
sys.path.append('../')
from common import utils,utils_log

class GPT4Distillation(BaseDistillation):
    def __init__(self, config):
        super().__init__(config)

    def shuffle_obs(self, obs):
        all_ls = obs
        if isinstance(all_ls, str):
            all_ls = ast.literal_eval(all_ls) 
        for single_ls in all_ls:
            random.shuffle(single_ls)
        return all_ls

    def get_distillation_data(self, date):
        print(f"Processing distillation data for date: {date}")

        infolder = os.path.join(self.config['base_input_path'], date)
        is_single_flag = self.config['is_single_flag']
        is_rag_flag = self.config['is_rag_flag']
        outfolder = os.path.join(self.config['base_output_path'], date, f'raw/single_{str(is_single_flag)}_rag_{str(is_rag_flag)}')

        print(f"Input folder: {infolder}")
        print(f"Output folder: {outfolder}")
        utils.create_directory(outfolder)

        prompt_path = self.get_prompt_path()
        files = [f'prod/{date}_rule_labeled.csv.gpt_labeled.csv']
        dl = []

        for file in files:
            file_path = os.path.join(infolder, file)
            print(f"Reading input file: {file_path}")
            try:
                df = pd.read_csv(file_path)
                df = df[~df['user-query'].isna()]
                filter_df = DataFilter.filter_bad_df(df, self.config)
                filter_df = DataFilter.get_task_usecols(filter_df)
                filter_df = filter_df.groupby('user-query').first().reset_index()
                dl.append(filter_df)
            except Exception as exc:
                print(f"Exception occurred while processing file: {file_path}")
                traceback.print_exc()

        if not dl:
            print("Data list is empty, no valid dataframes read.")
            return

        df = pd.concat(dl)
        df['observation'] = df['observation'].apply(self.shuffle_obs)
        df.to_csv(os.path.join(outfolder, f'{date}_log_data.csv'), index=False)
        
        utils.split_and_save_df(df, chunk_size=500, outfolder=os.path.join(outfolder, 'split_data'))

        self.check_and_generate_gpt4_files(outfolder, prompt_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='distillate gpt4 for app log.')
    parser.add_argument('--date', type=str, help='文件日期，格式为 YYYY-MM-DD')
    parser.add_argument('--is_single_flag', dest='is_single_flag', action='store_true', help='是否为单轮数据')
    parser.add_argument('--no_is_single_flag', dest='is_single_flag', action='store_false', help='是否为多轮数据')
    parser.add_argument('--is_rag_flag', dest='is_rag_flag', action='store_true', help='是否为rag数据')
    parser.add_argument('--no_is_rag_flag', dest='is_rag_flag', action='store_false', help='不是rag数据')
    parser.add_argument('--model_url', type=str, help='模型URL')
    parser.add_argument('--model_name', type=str, help='模型名称')
    parser.add_argument('--prompt_path', type=str, help='单轮RAG提示路径')
    parser.add_argument('--base_output_path', type=str, default='/mnt/pfs-guan-ssai/nlu/renhuimin/data/log_distillation/', help='基础输出路径')

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
        'base_input_path': '/mnt/pfs-guan-ssai/nlu/wangxiaoyuan/online-data/',
        'base_output_path': args.base_output_path,
        'prompt_path': args.prompt_path,
    }

    distillator = GPT4Distillation(config)
    distillator.get_distillation_data(args.date)