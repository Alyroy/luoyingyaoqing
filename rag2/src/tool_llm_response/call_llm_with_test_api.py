from statistics import mode
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from typing import Generator
# openai 最新版
from openai import OpenAI
from typing_extensions import Required, NotRequired, TypedDict
from pprint import pprint
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import json


class GenerateConfig(TypedDict):
    model: Required[str]
    temperature: NotRequired[float]
    top_p: NotRequired[float]
    max_tokens: Required[int]
    n: NotRequired[int]
    stop: Required[str]


class TestAPIConfig:
    def __init__(self, model, url, temperature=0.9, top_p=0.9, max_tokens=2048, chunk_num=4, thread_num=10, query_column_name='prompts', model_input_column_name='model_13b_input', predict_column_name='predict_output', response_column_name='response', eval_task='qwen_authenticity_eval', input_text_type='default', test_api='http://172.24.139.92:31696/predict'):
        self.model = model # 模型 name
        self.url = url # 模型 api
        self.temperature = temperature 
        self.top_p = top_p
        self.max_tokens = max_tokens # 最大输出长度
        self.chunk_num = chunk_num
        self.thread_num = thread_num
        self.query_column_name = query_column_name # llm模型输入列名
        self.model_input_column_name = model_input_column_name
        self.predict_column_name = predict_column_name
        self.response_column_name = response_column_name # llm模型输出列名
        self.eval_task = eval_task
        self.input_text_type = input_text_type
        self.test_api = test_api


class CallLLMByTestAPI(object):
    def __init__(self, config: TestAPIConfig):
        self.config = config

        api_key = 'none'
        # self.config.client = OpenAI(base_url=self.config.url, api_key=api_key, max_retries=1)
        self.config.generate_config = { "model": self.config.model,
                            "temperature": self.config.temperature,
                            "top_p": self.config.top_p,
                            "max_tokens": self.config.max_tokens,
                            "stop": ['<|endoftext|>'],
                            }
        
    def complete_response(self, messages, client, stream=False, **kwargs):
        try:
            completion = requests.post(client, json=messages)
            yield json.dumps(completion.json(), indent=4, ensure_ascii=False) # .choices[0].message.content
        except Exception as e:
            print('=' * 25)
            print(e)
            print('=' * 25)
            yield '<|wrong data|>'


    def task(self, row: dict):
        '''
        调用模型生成数据
        输入：转化为dict后的输入
        输出：[prompts列, 模型推理的response]列表
        '''
        # data = [
        #     {"role": "user",
        #         "content": row[self.config.query_column_name]},
        # ]
        if self.config.input_text_type == '':
            data = {
                "eval_task": self.config.eval_task,
                "config": {"url": self.config.url},
                "eval_mode": "single",
                "eval_data": [{
                    "max_generate": self.config.max_tokens,
                    "query": row[self.config.query_column_name],
                    "input_text": row[self.config.model_input_column_name],
                    "response": row[self.config.predict_column_name]
                }]
            }
        else:
            data = {
                "eval_task": self.config.eval_task,
                "config": {"url": self.config.url},
                "eval_mode": "single",
                "eval_data": [{
                    "max_generate": self.config.max_tokens,
                    "query": row[self.config.query_column_name],
                    "input_text_type": self.config.input_text_type,
                    "input_text": row[self.config.model_input_column_name],
                    "response": row[self.config.predict_column_name]
                }]
            }

        # assert self.config.client is not None and self.config.generate_config is not None, "client or generate_config is None"
        
        response = self.complete_response(
            messages=data, client=self.config.test_api, stream=False, **self.config.generate_config)
        if isinstance(response, Generator):
            response = ''.join(response)
        else:
            response = '<|wrong data|>'

        row[self.config.response_column_name] = response
        return row
    

    def chunk_dataframe(self, df):
        '''
        将输入数据分块
        输入：带prompts列的df文件，chunk_num
        输出：分块后的数据
        '''
        total_rows = df.shape[0]
        print("chunk size is", int(total_rows / self.config.chunk_num))
        assert int(total_rows / self.config.chunk_num) > 0, 'chunk size is too small'
        # 获取每个chunk块对应的index
        chunks = np.array_split(range(total_rows), self.config.chunk_num)
        for idx, current_chunk in zip(range(self.config.chunk_num), chunks):
            current_chunk = current_chunk.tolist()
            # 根据每个chunk块的第一个和最后一个index切分
            chunk_df = df.iloc[current_chunk[0]: (current_chunk[-1] + 1)]
            print(f'chunk_{idx},range({current_chunk[0]},{current_chunk[-1]})')
            yield chunk_df


    def format_time(self, seconds):
        if seconds < 60:
            return f"{seconds:.2f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.2f} minutes"
        else:
            hours = seconds / 3600
            return f"{hours:.2f} hours"


    def parser_model_response(self, merged_df) -> list:
        """
        评估专用解析
        """
        result_list = []
        user_list = merged_df[self.config.query_column_name].to_list()
        assistant_list = merged_df[self.config.response_column_name].to_list()
        for query,resp in zip(user_list,assistant_list):
            tmp_dict = {"query": query, "response": resp}
            result_list.append(tmp_dict)

        return result_list

    def parser_model_response_index(self, merged_df, index_name) -> list:
        """
        评估专用解析
        """
        result_list = []
        user_list = merged_df[self.config.query_column_name].to_list()
        assistant_list = merged_df[self.config.response_column_name].to_list()
        index_list = merged_df[index_name].to_list()
        for query, resp, index in zip(user_list, assistant_list, index_list):
            tmp_dict = {"query": query, "response": resp, "index":index}
            result_list.append(tmp_dict)

        return result_list

    def model_request(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        构建线程池，将输入数据分块后调用模型生成
        输入：带user-query列的df文件，chunk_num， thread_num
        输出：模型带推理的df_merge
        """
        start_time = time.time()
        chunk_generator = self.chunk_dataframe(df)
        
        # 创建一个线程池，最大线程数为..
        dfs = []
        with ThreadPoolExecutor(max_workers=self.config.thread_num) as executor:
            for idx,chunk_df in enumerate(chunk_generator):
                chunk_start_time = time.time()
                # 每个chunk块使用n个线程去推理
                results = executor.map(self.task, chunk_df.to_dict('records')) # 每行转为一个字典
                # 遍历结果
                final_results = []
                for result in tqdm(results,total=len(chunk_df)):
                    final_results.append(result)
                # 保存结果
                final_df = pd.DataFrame(final_results)
                dfs.append(final_df)
            df_merge = pd.concat(dfs)
        print(f'所有线程完成! time cost: {self.format_time(time.time()-start_time)}')

        return df_merge
    
