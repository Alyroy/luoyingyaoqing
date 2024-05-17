import json
import pandas as pd
import re
import csv
from datetime import datetime
import time
import random
import ast
from tqdm import tqdm
import copy
import os 
from utils import preprocess_df,create_directory


## csv --> sft 工具
def convert_api_list2str(apis: str) -> str:
    """
    把api字符串，转为sft数据输入格式,[{"APINAME":"MEDIASearch", "QUERY":"推荐一部张艺谋"}]
    Args:
        apis:str 可以转为list
    Return:
        api_content: str
    """
    apis = eval(apis)
    api_content = ''
    for api in apis:
        api_content += "<|api_start|>"
        for k,v in api.items():
            api_content += "<|kvs|>"
            api_content += k
            api_content += "=>"
            api_content += v
            api_content += "<|kve|>"
        api_content += "<|api_end|>"
    return api_content


def convert_api_str2dict(api: str) -> list:
    api_content = []
    apis = re.findall("<\|api_start\|>([\s\S]*?)<\|api_end\|>", api)
    for a in apis:
        kvs = re.findall("<\|kvs\|>([\s\S]*?)<\|kve\|>", a)
        api_dict = dict()
        for kv in kvs:
            try:
                api_dict[kv.split("=>")[0]] = kv.split("=>")[1]
            except:
                continue
        api_content.append(api_dict)
    return api_content


def convert_api_raw2sft(api: str) -> list:
    """
    把api字符串，转为sft数据输入格式
    Args:
        api:str需要切割的一长串字符
    Return:
        api_content: list,token切割
    """
    api_content = []
    apis = re.findall("<\|api_start\|>([\s\S]*?)<\|api_end\|>", api)
    for a in apis:
        api_content.append({"token": "<|api_start|>"})
        kvs = re.findall("<\|kvs\|>([\s\S]*?)<\|kve\|>", a)
        for kv in kvs:
            api_content.append({"token": "<|kvs|>"})
            api_content.append(kv.split("=>")[0])
            api_content.append({"token": "=>"})
            api_content.append(kv.split("=>")[1])
            api_content.append({"token": "<|kve|>"})
        api_content.append({"token": "<|api_end|>"})
    return api_content


def convert_observation_raw2sft(observation: str) -> list:
    """
    observation转为sft输入格式，observation字符串有token
    observation: str, after eval is list
    obtypes: list
    """
    observations = re.findall("<\|kvs\|>([\s\S]*?)<\|kve\|>", observation)
    observation_content = []
    for o in observations:
        observation_content.append({"token": "<|kvs|>"})
        observation_content.append(o)
        observation_content.append({"token": "<|kve|>"})

    if observation_content == []:
        raise "observation 解析为空"
    return observation_content


def convert_assistant_raw2sft(assistant: str, relevant_label:str) -> list:
    """
    将str型的assistant，转为sft需要的list数据
    """
    if assistant is None:
        raise ValueError("assistant 为空")  # 抛出一个异常类的实例
    assistant_content = []
    if '<|br|>' in assistant:
        # 使用split保留换行符
        ass_ls = assistant.split('<|br|>')
        for i, ass in enumerate(ass_ls):
            # 检查是否为空字符串（即使包含换行符）
            if ass.strip() or '\n' in ass:
                assistant_content.append(ass)
            if i < len(ass_ls) - 1:  # 如果不是最后一个元素，添加token
                assistant_content.append({"token": "<|br|>"})
    else:
        assistant_content = [assistant]
        
    if relevant_label == '不相关':
        assistant_content = [{"token": "<|irrelevant|>"}] + assistant_content
        
    return assistant_content


def convert_csv_to_sft(df:pd.DataFrame, api_flag=True, prompt='', prompt_ratio=0) -> pd.DataFrame:
    """
    从csv df生成sft_messages df格式数据
    df:['id', 'source', 'user-query', 'Thought', 'API', 'observation', 'assistant', 'relevant_label',
       'produce_source', 'task_name', 'create_time','update_time', 'create_user', 'update_user', 'is_reviewwd','update_content']
       其中observation = '["资料1","资料2","资料3"]'
    api_flag：boolean，True 生成有API的sft数据；False 生成无thought API observation的数据
    """
    messages_ls = []
    for i in range(len(df)):
        item = df.iloc[i]
        messages = []

        if item['user-query'][0:2] == "['" and item['user-query'][-2:] == "']":
            user = ast.literal_eval(item['user-query'])
        else:
            user = [item['user-query']]
        
        if random.random() < prompt_ratio:
            user = [user[0] + prompt]
        
        messages.append({"role": "user", "content": user})

        if not api_flag:
            thought = []
        else:
            if item['Thought'][0:2] == "['" and item['Thought'][-2:] == "']":
                thought = ast.literal_eval(item['Thought'])
            else:
                thought = [item['Thought']]
        messages.append({"role": "thought", "content": thought})

        body = {}
        body["api"] = item['API']

        if not api_flag:
            api_content = []
        else:
            api_content = convert_api_raw2sft(body["api"])
        messages.append({"role": "api", "content": api_content})

        observations = []
        obs_tmp_ls = ast.literal_eval(item['observation'])
        ob_types = re.findall(r'APINAME=>(.*?)(?=<|kve|>)', body["api"]) 

        # 处理API生成错误的问题
        if ob_types == []:
            for k in range(len(obs_tmp_ls)):
                json_observation = {'Results':ast.literal_eval(item['observation'])[k]}
                observaion_ = "<|kvs|>{}<|kve|>".format(json.dumps(json_observation, ensure_ascii=False))
                observations.append(observaion_)
        else:
            for k in range(min(len(ob_types),len(obs_tmp_ls))):
                json_observation = {ob_types[k]+'Results':ast.literal_eval(item['observation'])[k]}
                observaion_ = "<|kvs|>{}<|kve|>".format(json.dumps(json_observation, ensure_ascii=False))
                observations.append(observaion_)
                
        body["observation"] = ''.join(observations)
        if not api_flag:
            observation_content = []
        else:
            observation_content = convert_observation_raw2sft(body["observation"])
        messages.append({"role": "observation", "content": observation_content})

        assistant = item['assistant']
        try:
            relevant_label = item['relevant_label']
        except:
            relevant_label = '相关'
        assistant_content = convert_assistant_raw2sft(assistant,relevant_label)
        messages.append({"role": "assistant", "content": assistant_content})
        
        messages_ls.append(messages)
    
    # 初始化要创建的新DataFrame的字典
    new_df_data = {
        'id': df['id'].astype(str),  # 转换id列为字符串格式
        'source':df['source'],
        'messages':messages_ls
    }
    
    # 添加df中存在的可选列
    optional_columns = ['produce_source', 'create_time', 'update_time',
        'create_user', 'update_user', 'task_name', 'is_reviewed','update_content','system']

    # 使用for循环和条件检查将可选列添加到字典中
    for col in optional_columns:
        if col in df.columns:
            new_df_data[col] = df[col]

    # 使用字典来创建新的DataFrame
    new_df = pd.DataFrame(new_df_data)
    
    return new_df


def convert_csv_to_sft_tmp(df:pd.DataFrame, api_flag=True, prompt='', prompt_ratio=0) -> pd.DataFrame:
    messages_ls = []
    for i in range(len(df)):
        item = df.iloc[i]
        messages = []
    
        if item['user-query'][0:2] == "['" and item['user-query'][-2:] == "']":
            user = ast.literal_eval(item['user-query'])
        else:
            user = [f"已知检索结果为：{item['observation']}\n用户问题是：{item['user-query']}"] 
        
        if random.random() < prompt_ratio:
            user = [user[0] + prompt]
        
        messages.append({"role": "user", "content": user})
        messages.append({"role": "thought", "content": []})
        messages.append({"role": "api", "content": []})
        messages.append({"role": "observation", "content": []})
        assistant = item['assistant']
        messages.append({"role": "assistant", "content": assistant})
        
        messages_ls.append(messages)
    
    # 初始化要创建的新DataFrame的字典
    new_df_data = {
        'id': df['id'].astype(str),  # 转换id列为字符串格式
        'source':df['source'],
        'messages':messages_ls
    }
    
    # 使用字典来创建新的DataFrame
    new_df = pd.DataFrame(new_df_data)
    
    return new_df
    
def merge_multi_sft_data(df:pd.DataFrame) -> pd.DataFrame:
    """
    从df格式生成sft jsonl格式，包含单轮及多轮数据，按照id group by 合并messages后保存
    df:[id,source,messages]
    """
    # new_df = df.groupby('id').agg({'source': 'first', 'messages': list}).reset_index()

    # 定义函数来合并 messages 列为一维列表
    def merge_messages(group):
        return [message for sublist in group for message in sublist]
    
    # 根据 id 及 turn_id 做排序，防止多轮顺序生成错误
    df_sorted = df.sort_values(by=['id', 'turn_id'])

    # 根据 id 分组，保留 id 和 source 列，并合并 messages 列为一维列表
    new_df = df.groupby('id').apply(lambda group: pd.Series({
        'source': group['source'].iloc[0],
        'messages': merge_messages(group['messages']),
    })).reset_index()
    
    optional_columns = ['produce_source', 'create_time', 'update_time',
        'create_user', 'update_user', 'task_name', 'is_reviewed','update_content','system']

    # 使用for循环和条件检查将可选列添加到字典中
    for col in optional_columns:
        if col in df.columns:
            new_df[col] = df.groupby('id').apply(lambda group: pd.Series({
                            col: group[col].iloc[0]})).reset_index()
    
    return new_df


def user_prompt2query(row):
    """Combine 'user-query' with 'user_prompt' if 'user_prompt' is not null or empty."""
    user_query = row['user-query']
    user_prompt = row.get('user_prompt', None)  # 使用 get 以安全处理不存在的列
    
    # 检查 'user_prompt' 是否存在且不为空
    if pd.notna(user_prompt) and user_prompt.strip():
        return f"{user_query}\n{user_prompt.strip()}"
    return user_query


def gen_sft_data(input_path: str, output_path: str, api_flag: bool = True, multi_flag: bool = False, prompt='', prompt_ratio=0):
    """
    api_flag: True = 生成API assistant；False = 生成个无API assistant
    multi_flag: 是否按照ID 合并session
    """
    df = preprocess_df(input_path)
    df = df[~df['observation'].isin(['[]','[[]]'])] # 去掉obs为空的
    # df = df[df['update_time']!='2024/4/24']

    if 'source' not in df.columns:
        df['source'] = '无'
    if 'id' not in df.columns:
        df['id'] = '0'
        
    # 增加user-prompt
    if 'user_prompt' in df.columns:
        # 如果存在，则根据 'user_prompt' 更新 'user-query'
        df['user-query'] = df.apply(user_prompt2query, axis=1)

    sft_df = convert_csv_to_sft(df.copy(),api_flag, prompt, prompt_ratio)
    if multi_flag:
        sft_df = merge_multi_sft_data(sft_df)
    print('sft 数量：',len(sft_df))
    
    sft_df.to_json(output_path, orient='records', lines=True, force_ascii=False)

    
def gen_multi_turn(prev_paths: list[str], curr_path: str, output_path: str, output_num: float = None, min_turn_num: int = 1, max_turn_num: int = 3):
    """
    从其他人或自己的数据中拼多轮对话
    output_num 如果是小于1的 float --> 自身长度deratio; 如果为整数 --> 最终输出数量
    """
    prev_info_pool = []
    for prev_path in prev_paths:
        with open(prev_path, 'r') as in_file:
            for line in in_file:
                info = json.loads(line)
                if len(info['messages']) == 5: #and check_agent(info['messages'])
                    prev_info_pool.append(info)
    
    curr_info_pool = []
    with open(curr_path, 'r') as in_file:
        for line in in_file:
            info = json.loads(line)
            curr_info_pool.append(info)
    
    random.shuffle(curr_info_pool)
    if output_num is None:
        output_num = len(curr_info_pool)
    if output_num <= 1:
        output_num = int(len(curr_info_pool)*output_num)
    
    with open(output_path, 'w') as out_file:
        for output_index in tqdm(range(output_num), unit='output'):
            curr_info = copy.deepcopy(curr_info_pool[output_index % len(curr_info_pool)])
            curr_messages: List[Any] = curr_info['messages']
            prev_turn_num = random.randint(min_turn_num, max_turn_num)
            for _ in range(prev_turn_num):
                prev_info = random.choice(prev_info_pool)
                curr_messages = prev_info['messages'] + curr_messages
                
            # 无需处理，因为多轮只拼接user和assistant
            # if len(str(curr_messages)) > 6000: # 增加多轮拼接的长度限制，防止训练截断maxtoken
            #     continue
            curr_info['messages'] = curr_messages
            
            out_file.write(json.dumps(curr_info, ensure_ascii=False) + '\n')
            
            
def gen_multi_sft_data(output_load_folder,output_paths,category,output_num):
    # 同类同域拼多轮
    for i in range(len(output_paths)):
        multi_related_prev_paths = [output_paths[i]]
        multi_related_curr_path = output_paths[i]
        domain = output_paths[i].split('/')[-1].split('-')[0]
        create_directory(output_load_folder + '{}/multi/'.format(category))
        multi_related_output_path = output_load_folder + '{}/multi/{}-self-multi.jsonl'.format(category,domain)
        gen_multi_turn(multi_related_prev_paths, multi_related_curr_path, multi_related_output_path, 
                       output_num = output_num, min_turn_num = 2, max_turn_num = 7)

    # 同类不同域拼多轮
    for i in range(len(output_paths)):
        multi_related_prev_paths = output_paths
        multi_related_curr_path = output_paths[i]
        domain = output_paths[i].split('/')[-1].split('-')[0]
        multi_related_output_path = output_load_folder + '{}/multi/{}-sameCate-multi.jsonl'.format(category,domain)
        gen_multi_turn(multi_related_prev_paths, multi_related_curr_path, multi_related_output_path, 
                       output_num = output_num, min_turn_num = 2, max_turn_num = 15)
        
## sft --> csv 工具
def convert_observation_sft2raw(observation_content: list) -> list:
    """
    把observation 中的token去掉，存在多个observation的情况。
    """
    observations = [] 
    for item in observation_content:
        if isinstance(item, str):
            item = ast.literal_eval(item)
        elif isinstance(item, dict):
            item = item
        if 'AUTOSearchResults' in item:
            observations.append(item['AUTOSearchResults'])
        elif 'QASearchResults' in item:
            observations.append(item['QASearchResults'])
        elif 'MEDIASearchResults' in item:
            observations.append(item['MEDIASearchResults'])

    return observations


def convert_sptoken_sft2raw(assistant_content: list) -> str:
    """
    将带有special token的list型的api_content 或者 assistant_content，转为易读的str型数据
    """
    assistant = ''
    for item in assistant_content:
        if isinstance(item, str):
            assistant += item
        elif isinstance(item, dict) and 'token' in item and item['token'] != '<irr>':
            assistant += item['token']
    return assistant


def convert_sft_to_df(df:pd.DataFrame) -> pd.DataFrame:
    """
    从 sft df['messages'] 中提取元素，拆成单一df。如果是多轮数据，则顺序加行，并生成turn_id
    Args:
        df: load from sft
    return:
        new_df: add new columns [id, turn_id, source, user, thought, api, observation, assistant]
    """
    user_ls,thought_ls,api_ls,observation_ls,assistant_ls = [],[],[],[],[]
    source_ls = []
    id_ls = []
    turn_id_ls = []
    for i in range(len(df)):
        messages = df.iloc[i]['messages']
        num = int(len(messages)/5)
        id_ls.extend([df.iloc[i]['id']]*num)
        source_ls.extend([df.iloc[i]['source']]*num)
        turn_id_ls.extend(list(range(1,num+1)))
        for m in messages:
            if m['role'] == 'user':
                user = m['content'][0]
                user_ls.append(user)
            elif m['role'] == 'thought':
                if m['content'] != []:
                    thought = m['content'][0]
                    thought_ls.append(thought)
                else:
                    thought_ls.append('')
            elif m['role'] == 'api':
                if m['content'] != []:
                    api = convert_sptoken_sft2raw(m['content'])
                    api_ls.append(api)
                else:
                    api_ls.append('')
            elif m['role'] == 'observation':
                if m['content'] != []:
                    observations = convert_observation_sft2raw(m['content'])
                    observation_ls.append(observations)
                else:
                    observation_ls.append('')
            elif m['role'] == 'assistant':
                assistant = convert_sptoken_sft2raw(m['content'])
                assistant_ls.append(assistant)

    data = {
        'id': id_ls,
        'turn_id': turn_id_ls,
        'source': source_ls,
        'user-query': user_ls,
        'Thought': thought_ls,
        'API': api_ls,
        'observation': observation_ls,
        'assistant': assistant_ls
    }

    new_df = pd.DataFrame(data)

    return new_df


## 拆分api工具
def convert_api_sft2raw(api_content: list) -> [list,list,list,list]:
    """
    从sft api数据格式中提取单一元素
    """
    api_names,categorys,api_querys,api_tags = [],[],[],[]

    for i in range(len(api_content)):
        if api_content[i] == 'APINAME':
            api_names.append(api_content[i+2])
        elif api_content[i] == 'CATEGORY':
            categorys.append(api_content[i+2])
        elif api_content[i] == 'QUERY':
            api_querys.append(api_content[i+2])
        elif api_content[i] == 'TAG':
            api_tags.append(api_content[i+2])

    return api_names,categorys,api_querys,api_tags


def extract_api_contents(df: pd.DataFrame) -> pd.DataFrame:
    """
    提取 api中需要的元素，方便调取搜索
    """
    api_names_ls,api_categorys_ls,api_querys_ls,api_tags_ls = [],[],[],[]
    for api_content in df['API'].to_list():
        try:
            api_ls = convert_api_raw2sft(api_content)
            api_names,api_categorys,api_querys,api_tags = convert_api_sft2raw(api_ls)
            api_names_ls.append(api_names)
            api_categorys_ls.append(api_categorys)
            api_querys_ls.append(api_querys)
            api_tags_ls.append(api_tags)
        except Exception as e:
            print(e)
            # print(api_ls)
            api_names_ls.append([])
            api_categorys_ls.append([])
            api_querys_ls.append([])
            api_tags_ls.append([])
    
    df['API-NAME'] = api_names_ls
    df['API-CATEGORY'] = api_categorys_ls
    df['API-QUERY'] = api_querys_ls
    df['API-TAG'] = api_tags_ls
    
    return df


## 其他数据转换的工具
def judge_zongfen(text):
    """ 判断回复是否总分模版类型
    """
    pattern = re.compile(r'(：\n* *\n+1.)', re.IGNORECASE)
    flag = pattern.search(str(text))
    return flag

# 对某一列处理
def add_zongfen_flag(row):
    """ 基于assistant内容判断回复类型
    """
    flag = 0
    pattern = re.compile(r'(\d{4}年)')
    text = row['assistant']
    # 判断text是否包含"：\n\n1."
    if judge_zongfen(text) and text.find(" = ") == -1 and text.find("import") == -1 \
        and text.find("###") == -1 and (not pattern.search(str(text))):
        flag = 1
            
    return flag


def add_token_assistant(ori_lst):
    """ 给句子增加 br
    """
    new_lst = []
    for i in range(0, len(ori_lst)):
        if i != 0:
            new_lst.append(f'\n{i}.{ori_lst[i]}')
            if ori_lst[i].strip().endswith(":") or ori_lst[i].strip().endswith("："):
                return "<|br|>".join([])
        else:
            new_lst.append(ori_lst[i])
            
    # 保留总分“总”部分只有一段的情况，就是\n\n只出现一次的情况
    if new_lst[-1].find("\n\n") != -1 and len(new_lst[-1].split("\n\n")) == 2:
        tail_lst = new_lst[-1].split("\n\n")
        new_lst[-1] = tail_lst[0]
        new_lst.append(f'\n{tail_lst[1]}')
    else:
        new_lst = []
        
    return "<|br|>".join(new_lst)


def generate_format_assistant(x):
    """ 获取assistant，给其加分隔符
    """
    assistant = x["assistant"]
    new_assistant = []
    assistant_list = re.split(r'\n+\d+.', assistant)
    new_assistant = add_token_assistant(assistant_list)
    return new_assistant
  
# zongfen_df["assistant"] = zongfen_df.apply(generate_format_assistant, axis=1)

def get_structured_data(df,thought_tail='该问题请使用总分类模板回复。'):
    """
    总分类：该问题请使用总分类模板回复。
    描述类：该问题请使用描述类模板回复。
    比较类：该问题请使用比较类模板回复。
    时间线：该问题请使用时间线模板回复。
    服务专家：该问题涉及服务专家，请使用相应模板回复。
    """
    df['Thought'] = df['Thought'].apply(lambda x: x+thought_tail)
    if '总分' in thought_tail:
        df['assistant2'] = df['assistant'].apply(
        lambda x: '\n'.join([line + '<|br|>' if line.strip() != '' else line for line in str(x).split('\n+\d+.')]))

    df = df.astype(str)
    return df


# Applying the transformation
def flatten_and_number(lst):
    lst = eval(lst)
    counter = 1
    result = []
    for sublist in lst:
        for item in sublist:
            result.append(f'obs{counter} {item}')
            counter += 1
    return '\n\n'.join(result)

# df['observation2'] = df['observation'].apply(flatten_and_number)


# 日志数据转训练格式数据
def convert_format(df):
    def safe_literal_eval(s):
        try:
            return ast.literal_eval(s)
        except:
            return None

    def process_thought(thought_raw):
        try:
            return safe_literal_eval(thought_raw)[0]['thought']
        except (TypeError, IndexError, KeyError):
            return None

    def process_api(api_raw):
        try:
            return convert_api_list2str(api_raw)
        except:
            return None

    def process_observation(observation):
        try:
            return [list(item.values())[0] for item in safe_literal_eval(observation)]
        except:
            return None

    df['Thought'] = df['Thought_raw'].apply(process_thought)
    df['API'] = df['API_raw'].apply(process_api)
    df['observation'] = df['Observation'].apply(process_observation)
    df.drop(columns=['Thought_raw', 'API_raw', 'Observation'], inplace=True)
    return df


if __name__ == '__main__':
    # csv 转 jsonl
    data = {
    'id':['test-1'],
    'turn_id':[1],
    'source':['测试'],
    'user-query':['世界的第三高峰是哪个'],
    'Thought':['涉及事实问答，查询世界第三高峰的信息。'],
    'API':['<|api_start|><|kvs|>APINAME=>QASearch<|kve|><|kvs|>CATEGORY=>地理<|kve|><|kvs|>QUERY=>世界第三高峰<|kve|><|kvs|>TAG=>世界&第三高峰<|kve|><|api_end|>'],
    'observation':["[['一些内容','一些内容','一些内容']]"],
    'assistant':['世界的第三高峰是干城章嘉峰，位于喜马拉雅山脉，海拔高达8586米。'],
    'relevant_label':['相关']
    }
    df = pd.DataFrame(data)
    sft_df = convert_csv_to_sft(df.copy(),api_flag=True)
    sft_df
    
    # jsonl 转 csv