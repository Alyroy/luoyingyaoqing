import asyncio
import aiohttp
import json
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from asyncio import Semaphore
import time
import requests
from concurrent.futures import ThreadPoolExecutor

class RateLimiter:
    def __init__(self, rate):
        self.rate = rate
        self.current = 0
        self.last_check = datetime.now()

    async def wait(self):
        while True:
            now = datetime.now()
            passed_seconds = (now - self.last_check).total_seconds() # seconds
            if passed_seconds > 1:
                self.last_check = now
                self.current = 0
            if self.current < self.rate:
                self.current += 1
                return
            await asyncio.sleep(1)


class ZnyConfig:
    def __init__(self, url, model_name='gpt4o', temperature=0.9, max_retries=1, qps=2, max_concurrent=10, asyncio_flag=False, query_column_name='prompts', response_column_name='assistant'):
        self.url = url # 智能云GPT api
        self.model_name = model_name # gpt4o或gpt4
        self.temperature = temperature # llm输出温度，zny下的gpt4基本无效，因为是全球节点，还是会有随机性
        self.max_retries = max_retries # 调用gpt报错后最多重试 max_retries 次
        self.qps = qps # 多线程 or 异步多线程下，qps，不要超过5
        self.max_concurrent = max_concurrent # 异步多线程参数，一般10或者20，太大会接口超过qps
        self.asyncio_flag = asyncio_flag # True=异步多线程，只能python调用；False=普通多线程，Jupyter或者python均可
        self.query_column_name = query_column_name # llm模型输入列名
        self.response_column_name = response_column_name # llm模型输出列名


class CallLLMByZny(object):
    """
    支持多线程，及异步多线程两种方式调用智能云llm api 服务
    其中，gen_assistant_async 为异步多线程，只能通过python执行
    gen_assistant_threaded 为多线程，python和Jupyter都能只用
    """
    def __init__(self, config: ZnyConfig):
        self.config = config

    def make_chat_request_entry(self, messages):
        """
        """
        if self.config.model_name == "gpt4o":
            data_entry = {
                "messages": [{'role' : 'user', 'contents' : [{"type": "text","text": messages[i]}]} for i in range(len(messages))],
                "temperature": self.config.temperature
                }
        elif self.config.model_name == "gpt4" or self.config.model_name == "wenxin":
            data_entry = {
                "messages": [{"role": "user", "content": messages[i]} for i in range(len(messages))],
                "temperature": self.config.temperature
            }
        else:
            raise "当前仅支持old_gpt（gpt4/wenxin） 或者 new_gpt格式（gpt4o）"

        return data_entry

    async def request_chat_async(self, rate_limiter, semaphore, session, messages):
        """
        Async version of the request_chat function
        """
        if not isinstance(messages, list):
            messages = [messages]

        data_entry = self.make_chat_request_entry(messages)
        headers = {'Content-Type': 'application/json'}

        retries = 0
        while retries < self.config.max_retries:
            await rate_limiter.wait()  # 控制请求的发出速率
            async with semaphore:  # 限制同时处理的请求数量
                try:
                    async with session.post(self.config.url, json=data_entry, headers=headers) as response:
                        # content = await response.text() 
                        # print(content)
                        response_data = await response.json()
                    return response_data
                except Exception as e:
                    print(f'chatgpt api exception: {e}')
                    retries += 1
                    await asyncio.sleep(2)

        print('Maximum retry attempts reached, returning error')
        return {"error": "Maximum retry attempts reached, returning error"}
        

    async def process_prompts_chunk_async(self, rate_limiter, semaphore, session, prompts):
        """
        Async version of the process_prompts_chunk function
        """
        response = await self.request_chat_async(rate_limiter, semaphore, session, prompts)
        return [prompts, response]

    async def gen_assistant_async(self, prompts_ls):
        rate_limiter = RateLimiter(self.config.qps)
        semaphore = Semaphore(self.config.max_concurrent)  # 限制最大并发数为 max_concurrent，暂时无限制，可以根据自身需求调整大小

        async with aiohttp.ClientSession() as session:
            tasks = [self.process_prompts_chunk_async(rate_limiter, semaphore, session, prompts) for prompts in prompts_ls] # 逐一输入prompt
            responses = []

            for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
                response = await future
                responses.append(response)

        return responses

    def request_chat(self, messages):
        """
        单条调用gpt4
        max_retries: 报错时，最多重复调用次数
        """
        if not messages:
            return []

        if not isinstance(messages, list):
            return self.request_chat([messages])
        # print(messages)
        data_entry = self.make_chat_request_entry(messages)
        # print(data_entry)
        headers = {'Content-Type': 'application/json'}
        retries = 0  # 重置重试计数器
        while retries < self.config.max_retries:  # 最大重试次数限制
            try:
                response = requests.post(self.config.url, headers=headers, json=data_entry)
                response_data = json.loads(response.text)
                break  # 如果成功，就跳出while循环
            except Exception as e:
                print('chatgpt api 调用异常：{}'.format(e))
                time.sleep(2)  # 异常调用后休眠1秒
                retries += 1  # 如果失败，增加重试计数器
        if retries == self.config.max_retries:  # 如果重试次数达到最大值
            print('最大重试次数已达，返回空')
            return {"error": "最大重试次数已达，返回空"}
        # print(response_data)
        return response_data

    def process_prompts_chunk(self, prompts):
        response = self.request_chat(prompts)
        return [prompts, response]  # 返回prompts，支持后处理

    def gen_assistant_threaded(self, prompts_ls):
        with ThreadPoolExecutor(self.config.qps) as executor:
            futures = [executor.submit(self.process_prompts_chunk, prompts) for prompts in prompts_ls]
            response_ls = [future.result() for future in tqdm(futures)]

        return response_ls


    ### 解析gpt结果
    def parser_gpt_response(self, response_ls):
        """
        解析response_ls，转为assistant_df[prompts,assistant]
        """
        user_ls = []
        assistant_ls = []
        for response in response_ls:
            user = response[0][-1]
            user_ls.append(user)

            try:
                if self.config.model_name == 'gpt4o' or self.config.model_name == 'gpt4o':
                    assistant = response[1]['data']['choices'][0]['content']
                elif self.config.model_name == 'wenxin':
                    assistant = response[1]['data']['result']
                assistant_ls.append(assistant)
            except Exception as e:
                print(e)
                print(response[1]) # 返回第一个报错内容
                assistant_ls.append("<|wrong data|>")

        assistant_df = pd.DataFrame()
        assistant_df[self.config.query_column_name] = user_ls
        assistant_df[self.config.response_column_name] = assistant_ls

        return assistant_df


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


    def get_gpt4api_df(self, init_prompt_df):
        """
        init_prompt_df，必须包含prompt列，prompt列为模型输入
        """
        prompt_df = init_prompt_df.copy()

        final_reward_list = []
        while len(prompt_df) > 0:
            print("剩余case", len(prompt_df), '/',  len(init_prompt_df))
            prompts_ls = prompt_df[self.config.query_column_name].to_list()
            prompts_ls = [[prompt] for prompt in prompts_ls]

            if self.config.asyncio_flag:
                loop = asyncio.get_event_loop()
                response_ls = loop.run_until_complete(self.gen_assistant_async(prompts_ls))
            else:
                response_ls = self.gen_assistant_threaded(prompts_ls)

            assistant_df = self.parser_gpt_response(response_ls)
            final_reward_list.append(assistant_df[assistant_df[self.config.response_column_name]!='<|wrong data|>']) # 提取有效gpt生成内容
            prompt_df = assistant_df[assistant_df[self.config.response_column_name]=='<|wrong data|>']
            if self.config.max_retries == 0 and len(prompt_df) > 0:
                print("重复请求次数已达最大, 剩余", len(prompt_df), "条数据为空")
                break
            self.config.max_retries -= 1
            time.sleep(2) # 重试之前休眠2s

        final_reward_list.append(prompt_df)
        used_assistant_df = pd.concat(final_reward_list, ignore_index=True)
        merged_df = pd.merge(init_prompt_df, used_assistant_df, on=self.config.query_column_name, how='inner')

        return merged_df


    # def get_gpt4api_df(self, init_prompt_df, chunk_size: int = None, save_path: str = None):
    #     if chunk_size and save_path:
    #         # Make sure save path directory exists
    #         if not os.path.exists(save_path):
    #             os.makedirs(save_path)

    #         # Split the DataFrame into chunks
    #         chunks = [init_prompt_df[i:i + chunk_size] for i in range(0, len(init_prompt_df), chunk_size)]
    #         result_dfs = []

    #         # Process each chunk
    #         for i, chunk in enumerate(chunks):
    #             print(f"Processing chunk {i + 1}/{len(chunks)}")
    #             result_df = self._process_chunk(chunk)
    #             chunk_save_path = os.path.join(save_path, f"chunk_result_{i}.csv")
    #             result_df.to_csv(chunk_save_path, index=False)
    #             result_dfs.append(result_df)

    #         # Combine all chunk results
    #         merged_df = pd.concat(result_dfs, ignore_index=True)
    #         return merged_df

    #     else:
    #         return self._process_chunk(init_prompt_df)

    # def _process_chunk(self, chunk_df):
    #     """
    #     Process a chunk of the DataFrame.
    #     """
    #     prompt_df = chunk_df.copy()

    #     final_reward_list = []
    #     while len(prompt_df) > 0:
    #         print("剩余case", len(prompt_df), '/', len(chunk_df))
    #         prompts_ls = prompt_df[self.config.query_column_name].to_list()
    #         prompts_ls = [[prompt] for prompt in prompts_ls]

    #         if self.config.asyncio_flag:
    #             loop = asyncio.get_event_loop()
    #             response_ls = loop.run_until_complete(self.gen_assistant_async(prompts_ls))
    #         else:
    #             response_ls = self.gen_assistant_threaded(prompts_ls)

    #         assistant_df = self.parser_gpt_response(response_ls)
    #         final_reward_list.append(assistant_df[assistant_df[self.config.response_column_name] != '<|wrong data|>']) # 提取有效gpt生成内容
    #         prompt_df = assistant_df[assistant_df[self.config.response_column_name] == '<|wrong data|>']
    #         if self.config.max_retries == 0 and len(prompt_df) > 0:
    #             print("重复请求次数已达最大, 剩余", len(prompt_df), "条数据为空")
    #             break
    #         self.config.max_retries -= 1
    #         time.sleep(2) # 重试之前休眠2s

    #     final_reward_list.append(prompt_df)
    #     used_assistant_df = pd.concat(final_reward_list, ignore_index=True)
    #     merged_df = pd.merge(chunk_df, used_assistant_df, on=self.config.query_column_name, how='inner')

    #     return merged_df