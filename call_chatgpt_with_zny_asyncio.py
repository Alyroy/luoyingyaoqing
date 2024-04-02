import asyncio
import aiohttp
import json
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from asyncio import Semaphore
import time
import random
import requests
from concurrent.futures import ThreadPoolExecutor
import pandas as pd


class RateLimiter:
    def __init__(self, rate):
        self.rate = rate
        self.current = 0
        self.last_check = datetime.now()

    async def wait(self):
        while True:
            now = datetime.now()
            passed_seconds = (now - self.last_check).seconds
            if passed_seconds > 1:
                self.last_check = now
                self.current = 0
            if self.current < self.rate:
                self.current += 1
                return
            await asyncio.sleep(1)
            
            
def make_chat_request_entry(messages, message_type=''):
    """
    示例：
    单轮 system prompt
    # messages = [
    #    {"role": "system", "content": "你是理想同学"}
    #    {"role": "user", "content": "你是谁"},
    # ]
    单轮/多轮 user/assistant prompt
    # messages = [
    #    {"role": "user", "content": "周杰伦的代表作是什么"},
    #    {"role": "assistant", "content": "周杰伦是华语乐坛的重要人物，他的音乐作品多种多样，风格独特，深受广大听众的喜爱。以下是他的一些代表作：\n\n1.《青花瓷》：这首歌是周杰伦的代表作之一，歌曲以中国传统文化为背景，歌词优美，旋律悠扬。\n\n2.《简单爱》：这首歌是周杰伦早期的作品，歌词直接表达了对爱情的向往和追求，深受年轻人的喜爱。\n\n3.《不能说的秘密》：这首歌是周杰伦自编自导的电影《不能说的秘密》的主题曲，歌曲旋律优美，歌词深情。\n\n4.《七里香》：这首歌是周杰伦的经典作品之一，歌曲旋律悠扬，歌词描绘了一段深情的爱情故事。\n\n5.《稻香》：这首歌歌词深入人心，歌曲旋律优美，是周杰伦的代表作之一。\n\n6.《双截棍》：这首歌是周杰伦的代表作之一，歌曲以中国传统武术为主题，展现了周杰伦的音乐创新和实验精神。\n\n7.《夜曲》：这首歌是周杰伦的经典作品之一，歌曲旋律优美，歌词深情，展现了周杰伦的音乐才华。\n\n以上只是周杰伦的部分代表作，他的音乐作品还有很多，每一首都有其独特的魅力。"},
    #    {"role": "user", "content": "他哪一年出道的"}
    # ]
    """
    if message_type == "system":
        # 单轮 system prompt
        data_entry = {
            "messages": [{"role": ["system", "user"][i % 2], "content": messages[i]} for i in range(len(messages))]
        }
    else:
        # 单轮/多轮 user/assistant prompt
        data_entry = {
            "messages": [{"role": ["user", "assistant"][i % 2], "content": messages[i]} for i in range(len(messages))]
        }

    return data_entry



async def request_chat_async(rate_limiter, semaphore, session, messages, message_type='', max_retries=10, url = ""):
    """
    Async version of the request_chat function
    """
    if not isinstance(messages, list):
        return request_chat_async(rate_limiter, semaphore, session, [messages], message_type, max_retries, url)
    
    data_entry = make_chat_request_entry(messages,message_type)
    # url = "https://gpt4-example.fc.chj.cloud/gpt4/conversation" # 示例
    headers = {'Content-Type': 'application/json'}
    
    # Introduce a small random delay. do not work
    # delay = random.uniform(0.01, 0.1)
    # await asyncio.sleep(delay)

    retries = 0
    while retries < max_retries:
        await rate_limiter.wait()  # 控制请求的发出速率
        async with semaphore:  # 限制同时处理的请求数量
            try:
                async with session.post(url, json=data_entry, headers=headers) as response:
                    return await response.json()
            except Exception as e:
                print(f'chatgpt api exception: {e}')
                retries += 1
                await asyncio.sleep(1)
                
    print('Maximum retry attempts reached, returning error')
    return {"error": "Maximum retry attempts reached, returning error"}


async def process_prompts_chunk_async(rate_limiter, semaphore, session, prompts, message_type='', max_retries=10, url = ""):
    """
    Async version of the process_prompts_chunk function
    """
    response = await request_chat_async(rate_limiter, semaphore, session, prompts, message_type, max_retries, url)
    return [prompts, response]


async def gen_assistant_async(prompts_ls, message_type='', max_retries=10, qps=2, max_concurrent=20, output_assistant_path="", url = ""):
    """
    qps 最大为5，建议设置小于5
    max_concurrent 为并发数限制，文档没有要求，但是RateLimiter在异步时有时无法控制好qps，因此加此限制，具体数值可根据自身需要调整
    """
    rate_limiter = RateLimiter(qps)
    semaphore = Semaphore(max_concurrent)  # 限制最大并发数为 max_concurrent，暂时无限制，可以根据自身需求调整大小

    async with aiohttp.ClientSession() as session:
        tasks = [process_prompts_chunk_async(rate_limiter, semaphore, session, prompts, message_type, max_retries, url) for prompts in prompts_ls]
        responses = []

        # with open(output_assistant_path, 'a', encoding='utf-8') as f:
        #     for prompt, future in tqdm(zip(prompts_ls, asyncio.as_completed(tasks)), total=len(tasks)):
        #         response = await future
        #         f.write(json.dumps([prompt, response], ensure_ascii=False) + "\n")
        #         f.flush()  # ensure the data is written to disk after each iteration
        #         responses.append(response)
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            response = await future
            responses.append(response)

    return responses

### 单条数据
def request_chat(messages, message_type='', max_retries=10, url = ""):
    """
    单条调用gpt4
    max_retries: 报错时，最多重复调用次数
    """
    if not messages:
        return []
    
    if not isinstance(messages, list):
        return request_chat([messages], message_type, max_retries, url)
    
    # prepare data
    data_entry = make_chat_request_entry(messages, message_type)

    headers = {'Content-Type': 'application/json'}
    retries = 0  # 重置重试计数器
    while retries < max_retries:  # 最大重试次数限制
        try:
            response = requests.request("POST", url, headers=headers, json=data_entry)
            response_data = json.loads(response.text)
            break  # 如果成功，就跳出while循环
        except Exception as e:
            print('chatgpt api 调用异常：{}'.format(e))
            time.sleep(1)  # 异常调用后休眠1秒
            retries += 1  # 如果失败，增加重试计数器
    if retries == max_retries:  # 如果重试次数达到最大值
        print('最大重试次数已达，返回空')
        return {"error": "最大重试次数已达，返回空"}

    return response_data


def process_prompts_chunk(prompts, message_type='', max_retries=10, url = ""):
    """
    Process a chunk of prompts and handle retries
    """
    response = request_chat(prompts, message_type, max_retries, url)
    return [prompts, response]  # 返回prompts，支持后处理
            
            
def gen_assistant_threaded(prompts_ls, message_type='', max_retries=10, max_threads=2, url = ""):
    """
    多线程获取gpt结果
    df:需要包含prompts（带system prompt 和 user query）
    max_retries：最大重试次数
    max_threads: Maximum number of threads to use. gpt4 最大3，gpt3 最大20
    """
    with ThreadPoolExecutor(max_threads) as executor:
        futures = [executor.submit(process_prompts_chunk, prompts, message_type, max_retries, url) for prompts in prompts_ls]
        response_ls = [future.result() for future in tqdm(futures)]

    return response_ls


### 提取gpt数据
def parser_gpt_response_async(response_ls):
    """
    解析response_ls，转为assistant_df[prompts,assistant]
    """
    user_ls = []
    assistant_ls = []
    for response in response_ls:
        user = response[0][-1]
        user_ls.append(user)

        try:
            assistant = response[1]['choices'][0]['content']
            assistant_ls.append(assistant)
        except Exception as e:
            print(e)
            print(response[1])
            assistant_ls.append("<|wrong data|>")

    assistant_df = pd.DataFrame()
    assistant_df['prompts'] = user_ls
    assistant_df['assistant'] = assistant_ls

    return assistant_df


def get_gpt4api_df(init_prompt_df, message_type='', max_request_times=1, qps=2, max_concurrent=5, asyncio_flag=True, url=""):
    """
    max_request_times: 最多重复请求多少次gpt；gpt调用如果有问题，重复调用多少次
    qps: 最大为20，建议设置小于10
    max_concurrent：并发，无特殊限制，可根据自身需求设置
    asyncio_flag: 是否使用异步多线程 True = 异步多线程；False = 多线程
    url: 大模型调用API地址
    """
    prompt_df = init_prompt_df.copy()
    
    final_reward_list = []
    while len(prompt_df) > 0:
        print("剩余case", len(prompt_df), '/',  len(init_prompt_df))
        prompts_ls = prompt_df['prompts'].to_list()
        prompts_ls = [[prompt] for prompt in prompts_ls]
        
        timestamp = int(time.time())
        if asyncio_flag:
            loop = asyncio.get_event_loop()
            response_ls = loop.run_until_complete(gen_assistant_async(prompts_ls, max_request_times, qps, max_concurrent, output_assistant_path=f"asyncio_response_{max_request_times}_{timestamp}.json",url=url))
        else:
            response_ls = gen_assistant_threaded(prompts_ls, message_type=message_type, max_retries=max_request_times, max_threads=qps, url=url)

        assistant_df = parser_gpt_response_async(response_ls)
        final_reward_list.append(assistant_df[assistant_df['assistant']!='<|wrong data|>']) # 提取有效gpt生成内容
        prompt_df = assistant_df[assistant_df['assistant']=='<|wrong data|>']
        max_request_times -= 1
        if max_request_times == 0 and len(prompt_df) > 0:
            print("重复请求次数已达最大, 剩余", len(prompt_df), "条数据为空")
            break
        time.sleep(2) # 重试之前休眠2s
        
    final_reward_list.append(prompt_df)
    used_assistant_df = pd.concat(final_reward_list)
    merged_df = pd.merge(init_prompt_df, used_assistant_df, on='prompts', how='inner')
    merged_df.drop(columns=['prompts'], inplace=True)
    
    return merged_df


def get_prompts_df(df: pd.DataFrame, oneshot_prompt: str) -> pd.DataFrame:
    """
    df 需要包含 user observation
    oneshot_prompt 为构造好的prompt
    """
        
    df['prompts'] = df.apply(lambda row: oneshot_prompt + f"""
    ---
    下面是给出的实际问题：
    Background:
    ```{row['observation']}```
    Question:
    ```{row['user-query']}```
    Answer：
    """, axis=1)
    
    return df


if __name__ == "__main__":
    data = {
        "prompts": [
            ["你是理想同学", '你是谁'],
            ["天津的地标性建筑有天津之眼和瓷房子，美食有煎饼和狗不理包子", "去天津玩2天，怎么规划行程"]
        ],
        "single": [
            ['你是谁'],
            ["解释一下杜鹃花"],
            ["去天津玩2天，怎么规划行程"]
        ],
        "context": [
            ["刘德华的代表作是什么", "周杰伦的代表作有很多，其中最为人所知的可能是《青花瓷》、《七里香》、《不能说的秘密》等。", "我问的是刘德华"],
            ["周杰伦的代表作是什么", "周杰伦的代表作有很多，其中最为人所知的可能是《青花瓷》、《七里香》、《不能说的秘密》等。", "他是哪一年出道的"],
        ],
    }

    # prompt 调用方法
    # prompt_df = data['prompts']
    # message_type = "prompts"

    # 单轮 调用方法
    prompts_ls = data['single']
    message_type = ""
    
    # 多轮 调用方法
    # prompt_df = data['context']
    # message_type = ""
    
    response_ls = gen_assistant_threaded(prompts_ls, message_type='', max_retries=1, max_threads=2)
    assistant_df = parser_gpt_response_async(response_ls)
    print(assistant_df)
    
    
    loop = asyncio.get_event_loop()
    response_ls = loop.run_until_complete(gen_assistant_async(prompts_ls, message_type='', max_retries=1, qps=2, max_concurrent=20, output_assistant_path="asyncio_out.json"))
    assistant_df = parser_gpt_response_async(response_ls)
    print(assistant_df)
    
    
    
    folder = '/mnt/pfs-ssai-nlu/renhuimin/pro_qa/data/car/to_be_labeled_data/v20231011/'
    obs_sales_df = pd.read_csv(folder+'售后问答obs_title.csv').rename(columns={'assistant':'raw_assistant'})
    with open('prompt_conf/car_gen_prompt.txt', 'r') as file:
        car_prompt = file.read()

    init_prompt_df = get_prompts_df(obs_sales_df,car_prompt)[:3]
    merged_df = get_gpt4api_df(init_prompt_df, message_type='', max_request_times=1, qps=2, max_concurrent=20, asyncio_flag=False)
    print(merged_df['assitant'])
    
    init_prompt_df = get_prompts_df(obs_sales_df,car_prompt)[:3]
    merged_df = get_gpt4api_df(init_prompt_df, message_type='', max_request_times=1, qps=2, max_concurrent=20, asyncio_flag=True)
    print(merged_df['assitant'])