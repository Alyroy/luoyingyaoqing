import argparse
import pandas as pd
import numpy as np
import os
import gc
import sys
import traceback
import time
import re
from datetime import datetime, timedelta
from tqdm import tqdm
tqdm.pandas()
from data_processing import DataFilter,PromptConstructor

sys.path.append('../')
from tool_llm_response.call_llm_with_zny import CallLLMByZny, ZnyConfig
sys.path.append('../../')
from common import utils


class BaseDistillation:
    def __init__(self, config):
        self.config = config
        self.call_zny = CallLLMByZny(config['zny_config'])

    def get_prompt_path(self):
        return self.config['prompt_path']

    def get_gpt4df(self, df, output_path, prompt_path):
        print(f"Reading oneshot prompt from: {prompt_path}")
        oneshot_prompt = utils.read_txt(prompt_path)
        print("Reading oneshot prompt done.")
        if not oneshot_prompt or oneshot_prompt.strip() == "":
            print("Oneshot prompt is empty, please check the prompt file.")
            return

        print("Building prompts for dataframe entries.")
        df['prompts'] = df.apply(lambda row: PromptConstructor.construct_prompt(row, oneshot_prompt, self.config), axis=1)

        print("Calling GPT-4 API")
        gpt4_df = self.call_zny.get_gpt4api_df(df)
        gpt4_df.to_csv(output_path, index=False, encoding='utf-8-sig')

        print("Parsing GPT-4 response")
        gpt4_df['parser_gpt4'] = gpt4_df['gpt4response'].apply(self.parse_content)
        gpt4_df = gpt4_df[~gpt4_df['parser_gpt4'].isna()]
        gpt4_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        del df, gpt4_df
        gc.collect()
        time.sleep(60)

    @staticmethod
    def parse_content(text):
        match = re.search(r'\{\{(.+?)\}\}', text, re.DOTALL)
        if match:
            return match.group(1)
        else:
            return None

    def check_and_generate_gpt4_files(self, outfolder, prompt_path):
        split_data_folder = os.path.join(outfolder, 'split_data')
        gpt4_data_folder = os.path.join(outfolder, 'gpt4_data')
        utils.create_directory(gpt4_data_folder)
        split_files = sorted([f for f in os.listdir(split_data_folder) if '.ipynb_checkpoints' not in f])
        iterations = 0

        while True:
            missing_files = []
            wrong_data_count = 0
            total_data_count = 0

            for file in split_files:
                gpt4_file_path = os.path.join(gpt4_data_folder, f'gpt4_{file}')
                if not os.path.exists(gpt4_file_path):
                    missing_files.append(file)
                else:
                    df = pd.read_csv(gpt4_file_path)
                    total_data_count += len(df)
                    wrong_data_count += df['gpt4response'].str.contains('<|wrong data|>').sum()

            if not missing_files and (wrong_data_count / total_data_count) <= 0.5:
                print(f"All GPT-4 files are generated and valid. Total files: {len(split_files)}")
                done_file_path = os.path.join(gpt4_data_folder, '.done')
                open(done_file_path, 'w').close()
                break

            for file in missing_files:
                print(f"Regenerating missing GPT-4 file for split data file: {file}")
                df_ = pd.read_csv(os.path.join(split_data_folder, file))
                self.get_gpt4df(
                    df_,
                    os.path.join(gpt4_data_folder, f'gpt4_{file}'),
                    prompt_path
                )

            iterations += 1
            if iterations > 5:  # Add a safeguard against infinite loops
                print("Maximum iterations reached. Please check for potential issues.")
                break
            time.sleep(60)