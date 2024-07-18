import os
import ast
import re
import json
import random
import copy
from tqdm import tqdm
from datetime import datetime
import pandas as pd

import sys 
sys.path.append("../../") 
from common import utils,utils_log

SYSTEM_PROMPT = '[unused0]system\n你是一个名字叫做理想同学的AI数字生命体。\n理想同学是一个可靠的智能家庭助手，由理想汽车智能空间部门创造。\n理想同学能够理解人类的指令和意图，并且给出合理的、切合问题的、没有歧视、中立的、安全的回复。\n\n请根据以下文本写一个合适的回复。[unused1]\n'


class API:
    def __init__(self, thought: str, api_dict: dict) -> None:
        self.thought = thought
        self.api_dict = api_dict

    def convert_api_str2dict(self, api: str) -> dict:
        """
        Params: api:str 单条API<|xxx|>ddd<|xxx|>
        Return: api_dict:dict,单个api dict
        如果多条API，只取第一组
        """
        # 定义匹配 <api_start> 和 <api_end> 之间内容的正则表达式模式
        api_pattern = re.compile(r'<\|api_start\|>(.*?)<\|api_end\|>', re.DOTALL)
        # 定义匹配 <kvs>键值对<kve> 的正则表达式模式
        kvs_pattern = re.compile(r'<\|kvs\|>(.*?)=>(.*?)<\|kve\|>', re.DOTALL)
        
        # 查找到所有 <api_start> 和 <api_end> 之间的内容
        api_matches = api_pattern.findall(api)
        
        # 只取第一个匹配
        content = api_matches[0]
        result_dict = {}
        for kvs_match in kvs_pattern.finditer(content):
            key, value = kvs_match.group(1).strip(), kvs_match.group(2).strip()
            result_dict[key] = value
        self.api_dict = result_dict
        return result_dict

    def convert_api_dict2str(self) -> str:
        """
        把api dict 转为 sft 数据输入格式
        Args:
            apis: dict, {"APINAME":"MEDIASearch", "QUERY":"推荐一部张艺谋"}
        Return:
            api_content: str <|xxx|>
        """
        api_content = "<|api_start|>"
        for k, v in self.api_dict.items():
            api_content += "<|kvs|>"
            api_content += k
            api_content += "=>"
            api_content += v
            api_content += "<|kve|>"
        api_content += "<|api_end|>"
        return api_content
    

class APIList:
    def __init__(self, api_list: list = [API]) -> None:
        """
        api_list: list of API, 继承class API()
        """
        if api_list is None:
            self.api_list = []
        else:
            self.api_list = api_list
    
    def add_api(self, api: API) -> None:
        """
        增加一个API对象到api_list中
        """
        self.api_list.append(api)

    def convert_api_log2dict_list(self, log_api: str) -> list[dict]:
        """
        日志格式转api list of dict
        Args:
            log_api: pro_input = '[unused0]api\n[unused4][unused2]APINAME[unused7]MEDIASearch[unused3][unused2]MEDIATYPE[unused7]新闻[unused3][unused2]QUERY[unused7]今天有什么新闻[unused3][unused2]PUBLISHTIME[unused7]今天[unused3][unused2]STARTTIME[unused7]{"norm":{"day":"+0"},"raw":"今天"}[unused3][unused5][unused1]'
        Return:
            api_dict: [{'APINAME': 'MEDIASearch','MEDIATYPE': '新闻', 'QUERY': '今天有什么新闻','PUBLISHTIME': '今天','STARTTIME': '{"norm":{"day":"+0"},"raw":"今天"}'}]
        """
        api_list = utils_log.clean_api_pattern(log_api)
        self.api_list = api_list
    
    def convert_api_str2dict_list(self, api_str: str) -> None:
        """
        把API字符串解析成多个API字典，并保存到api_list中
        """
        apis = re.findall("<\|api_start\|>([\s\S]*?)<\|api_end\|>", api_str)
        for api in apis:
            api_obj = API(thought="", api_dict={})
            api_obj.convert_api_str2dict("<|api_start|>" + api + "<|api_end|>")
            self.api_list.append(api_obj)
    
    def convert_api_list2str(self) -> str:
        """
        将api_list中的所有API转换为字符串
        Return:
            return_str: str
        """
        return_str = ""
        for api_obj in self.api_list:
            return_str += api_obj.convert_api_dict2str()
        return return_str
    
    def convert_api_str2sft_list(self, api: str) -> list:
        """
        把api字符串<|xxx|>，转为sft数据输入格式
        Args:
            api:str需要切割的一长串字符，带<|xxx|>
        Return:
            api_content: list,token切割，sft训练格式
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


class Observation:
    def __init__(self, id: str = "", content: str = "", extra: dict[str, str] = {}) -> None:
        self.id = id
        self.content = content
        # self.extra = extra
    
    def get_observation_dict(self):
        return {"id": self.id, "content": self.content}
        
    def get_random_format_observation_dict(self):
        split_text_list = self.content.split(" ")
        if len(split_text_list) > 1:
            title = split_text_list[0]
            if len(split_text_list[1:]) > 1:
                content = "".join(split_text_list[1:])
            else:
                content = split_text_list[1]
        else:
            title = ""
            content = self.content
        return {"id": self.id, "title": title, random.sample(["content", "chunk"], 1)[0]: content}


class ObservationList:
    def __init__(self, observation_list: list[Observation] = [Observation]) -> None:
        self.observation_list = observation_list

    def init_from_json(self, input_json: str) -> None:
        self.observation_list = json.loads(input_json)

    def init_from_16b_input(self, input_text: str) -> None:
        try:
            pattern = r"\[\'.*?\'\]|\[\".*?\"\]"
            results = re.findall(pattern, input_text)
            result_list = [eval(_) for _ in results]
            self.init_from_list(result_list)
        except:
            try:
                result_list = eval(input_text)
                self.init_from_list(result_list)
            except:
                pattern = r"\'.*?\'|\".*?\""
                results = re.findall(pattern, input_text)
                result_list = [eval(_) for _ in results]
                self.init_from_list(result_list)
        return self

    def init_from_list(self, input_list: list) -> None:
        flattened_list = list(chain.from_iterable(input_list))
        self.observation_list = [
            Observation(str(i+1), str(flattened_list[i])) for i in range(len(flattened_list))
        ]
        return self
    
    def init_from_observation_dict_list(self, observation_dict_list: list) -> None:
        self.observation_list = [Observation(str(_["id"]), str(_["content"]) if "content" in _.keys() else _["chunk"]) for _ in observation_dict_list]
        return self

    def get_content_list(self):
        content_list = [_.content for _ in self.observation_list]
        return content_list
    
    def get_id_list(self):
        id_list = [_.id for _ in self.observation_list]
        return id_list
    
    def get_observation_dict_list(self):
        observation_dict_list = [_.get_observation_dict() for _ in self.observation_list]
        return observation_dict_list
    
    def get_random_format_observation_dict_list(self, mode="reverse"):
        observation_dict_list = [_.get_random_format_observation_dict() for _ in self.observation_list]
        if mode == "reverse":
            observation_dict_list = observation_dict_list[::-1]
        elif mode == "random":
            observation_dict_list = random.sample(observation_dict_list, k=len(observation_dict_list))
        else:
            observation_dict_list = observation_dict_list

        self.init_from_observation_dict_list(observation_dict_list)
        return observation_dict_list

    
    def convert_observation_list2str(self, apis: list, observation_list:list) -> str:
        """
        把obs list of dict，转为sft数据str带special token 格式
        Args:
            apis: list of dict, [{"APINAME":"MEDIASearch", "QUERY":"推荐一部张艺谋"}] # 解析apiname 拼接只obs
            observation_list: 2D-list of obs content
        Return:
            obs_content: str <|kvs|>obs<|kve|>
        """
        # 兼容API数量与obs数量不一致的问题
        # observation_list = self.get_content_list()
        apinames = [api.get('APINAME', 'QASearch') for api in apis] # 确保如果字典中没有 APINAME 键，就返回默认值 QASearch

        observations = []
        if utils.is_2d_list(observation_list):
            if len(apinames) != len(observation_list):
                print(f'API数量与OBS数量不对应:{len(apinames)} vs {len(observation_list)}') # {apinames} {observation_list}
            for idx in range(min(len(apinames),len(observation_list))):
                json_observation = {apinames[idx] + 'Results': observation_list[idx]}
                observation_str = "<|kvs|>{}<|kve|>".format(json.dumps(json_observation, ensure_ascii=False))
                observations.append(observation_str)
            return ''.join(observations)
        else:
            raise 'observation_list 不是2d-list'
        
 
    def convert_observation_str2sft_list(self, observation_str: str) -> list:
        """
        observation转为sft输入格式，observation字符串有token
        observation: str, after eval is list
        obtypes: list
        """
        observations = re.findall("<\|kvs\|>([\s\S]*?)<\|kve\|>", observation_str)
        observation_content = []
        for o in observations:
            observation_content.append({"token": "<|kvs|>"})
            observation_content.append(o)
            observation_content.append({"token": "<|kve|>"})
        return observation_content


class Messages():
    def __init__(self, user: list, assistant: list) -> None:
        self.user = user
        self.assistant = assistant

    def convert_assistant_str2sft_list(self) -> list:
        """
        将str型的assistant，转为sft需要的list数据
        """
        assistant_content = []
        if '<|br|>' in self.assistant:
            # 使用split保留换行符
            ass_ls = self.assistant.split('<|br|>')
            for i, ass in enumerate(ass_ls):
                # 检查是否为空字符串（即使包含换行符）
                if ass.strip() or '\n' in ass:
                    assistant_content.append(ass)
                if i < len(ass_ls) - 1:  # 如果不是最后一个元素，添加token
                    assistant_content.append({"token": "<|br|>"})
        else:
            assistant_content = [self.assistant]
        return assistant_content


class DataFormat():
    """
    常见的格式转换：
    1. 飞书线上csv -> 训练数据jsonl
    2. 训练数据jsonl -> 可读性csv
    3. 日志格式 到 csv ？？
    """
    def __init__(self, api_flag: bool = True, multi_flag: bool = False):
        self.api_flag = api_flag
        self.multi_flag = multi_flag

    def validate_input(self, df: pd.DataFrame) -> bool:
        """
        检查必要字段是不是都存在
        """
        required_columns = ['id', 'turn_id', 'source', 'user-query', 'Thought', 'API', 'observation', 'assistant']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input should be a pandas DataFrame")

        return True
        
    def convert_csv_to_sft(self, df) -> pd.DataFrame:
        """
        从csv df生成sft_messages df格式数据
        df:['id', 'turn_id', 'source', 'user-query', 'Thought', 'API', 'observation', 'assistant', 'relevant_label',
           'produce_source', 'task_name', 'create_time','update_time', 'create_user', 'update_user', 'is_reviewwd','update_content']
           其中,user-query,Thought,assistant 为str，API 为 list of dict， obs为 2d-list
           observation = '[["资料1","资料2","资料3"]]'
        api_flag：boolean，True 生成有API的sft数据；False 生成无thought API observation的数据
        """
        self.validate_input(df)
        
        messages_ls = []
        for i in range(len(df)):
            item = df.iloc[i]
            messages = []

            # user
            user = item['user-query'] # str
            if not isinstance(user, str):
                raise ValueError(f"Expected 'user-query' to be str, but got {type(user)}")
            messages.append({"role": "user", "content": [user]})
            
            # thought
            thought = item['Thought']  # str(list) or list or str
            if not self.api_flag:
                thought_content = []
            else:
                if isinstance(thought, str) and thought.startswith("['") and thought.endswith("']"):
                    thought_content = ast.literal_eval(thought)
                elif isinstance(thought, str):
                    thought_content = [thought]
                elif isinstance(thought, list):
                    thought_content = thought
                else:
                    raise ValueError(f"Unexpected format for 'Thought': {type(thought)}")
            if not all(isinstance(t, str) for t in thought_content):
                raise ValueError("Each item in 'Thought' should be a string")
            messages.append({"role": "thought", "content": thought_content})
    
            # api
            api = item['API']
            if not self.api_flag:
                api_content = []
            else:
                if isinstance(api, str):
                    api_list = ast.literal_eval(api)
                elif isinstance(api, list):
                    api_list = api
                else:
                    raise ValueError(f"Unexpected format for 'API': {type(api)}")
                    
                api_objects = [API(thought="",api_dict=api_dict) for api_dict in api_list]
                api_list_obj = APIList(api_objects)
                api_str = api_list_obj.convert_api_list2str()
                api_content = api_list_obj.convert_api_str2sft_list(api_str)
            messages.append({"role": "api", "content": api_content})
    
            # observation
            observation = item['observation']  # str(2d-list)
            if not self.api_flag:
                observation_content = []
            else:
                if isinstance(observation, str):
                    observation_2d_list = ast.literal_eval(observation)
                elif isinstance(observation, list):
                    observation_2d_list = observation
                obs_list_obj = ObservationList()
                obs_str = obs_list_obj.convert_observation_list2str(api_list, observation_2d_list)
                observation_content = obs_list_obj.convert_observation_str2sft_list(obs_str)
            messages.append({"role": "observation", "content": observation_content})
    
            # assistant 
            assistant = item['assistant']  # str， assistant为空也会正常通过
            if not isinstance(assistant, str):
                raise ValueError(f"Expected 'assistant' to be str, but got {type(assistant)}")
            mes_obj = Messages(user='',assistant=assistant)
            assistant_content = mes_obj.convert_assistant_str2sft_list()
            messages.append({"role": "assistant", "content": assistant_content})
    
            # all
            messages_ls.append(messages)
        # 初始化要创建的新DataFrame的字典
        # id turn_id 可能没有在df中，需要增加默认值
        new_df_data = {
            'id': df['id'].astype(str),  # 转换id列为字符串格式
            'turn_id': df['turn_id'].astype(int),  # 转换turn_id列为字符串格式
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
    
    
    def merge_multi_sft_data(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        从df格式生成sft jsonl格式，包含单轮及多轮数据，按照id group by 合并messages后保存
        df:[id,source,messages]
        """    
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
                temp_df = df.groupby('id').apply(lambda group: pd.Series({
                                col: group[col].iloc[0]})).reset_index()
                new_df = pd.merge(new_df, temp_df, on='id', how='left')
        
        return new_df

    
    def gen_sft_data(self, df: pd.DataFrame, flag_16b_inputs=False) -> pd.DataFrame:
        """
        api_flag: True = 生成API assistant；False = 生成个无API assistant
        multi_flag: 是否按照ID 合并session
        """
        def user_prompt2query(row):
            """Combine 'user-query' with 'user_prompt' if 'user_prompt' is not null or empty."""
            user_query = row['user-query']
            user_prompt = row.get('user_prompt', None)  # 使用 get 以安全处理不存在的列
            
            # 检查 'user_prompt' 是否存在且不为空
            if pd.notna(user_prompt) and user_prompt.strip():
                return f"{user_query}\n{user_prompt.strip()}"
            return user_query
            
        if self.api_flag:
            df = df[~df['observation'].isin(['[]','[[]]'])] # 去掉obs为空的
    
        if 'source' not in df.columns:
            df['source'] = '无'
        if 'id' not in df.columns:
            df['id'] = '0'
        if 'turn_id' not in df.columns:
            df['turn_id'] = 1
        if 'system' not in df.columns:
            df['system'] = SYSTEM_PROMPT

        if flag_16b_inputs:
            if 'assistant' not in df.columns:
                df['assistant'] = '无'
            else:
                df['assistant'] = df['assistant'].fillna('无')
            
        # 增加user-prompt
        if 'user_prompt' in df.columns:
            # 如果存在，则根据 'user_prompt' 更新 'user-query'
            df['user-query'] = df.apply(user_prompt2query, axis=1)
    
        sft_df = self.convert_csv_to_sft(df.copy())
        if self.multi_flag:
            sft_df = self.merge_multi_sft_data(sft_df)
        print('sft 数量：',len(sft_df))

        if flag_16b_inputs:
            data = self.convert_to_train_data(sft_df)
            return data
        # 将 DataFrame 转化为 JSON 字符串
        # json_str = sft_df.to_json(orient='records', lines=True, force_ascii=False)
        
        # 将 JSON 字符串写入文件，并指定编码
        # with open(self.output_path, 'w', encoding='utf-8') as f:
        #     f.write(json_str)
        # sft_df.to_json(output_path, orient='records', lines=True, force_ascii=False)

        return sft_df


    def convert_to_train_data(self, df):
        """
        messages格式的df
        """
        df = df.copy()
        df["chatML_data"] = df.apply(convert_to_chatml_single_data,axis=1)
        df2 = df[df["chatML_data"]!="###worng data"]
        text_list= df2["chatML_data"].to_list()
        system_list = df2["system"].to_list()
        
        #convert to training data
        data = []
        for i in range(len(text_list)):
            res = text_list[i]
            system_prompt = system_list[i]
            if str(system_prompt) == "":
                system_prompt = SYSTEM_PROMPT
            # print(system_prompt)
            try:
                if "<None>"  not in res[-1][-1]:
                    context_list = []
                    for j in res[:-1]: # 多轮数据只拼接历史的user和assistant
                        # print(j)
                        context_list.append(j[0])
                        context_list.append(j[4])
                    for i in res[-1]:
                        context_list.append(i)
                    query = "\n".join(context_list[:-1])+"\n"
                    output = context_list[-1]
                    data.append({'instruction':f"[unused0]system\n{system_prompt}[unused1]\n{query}", 'output':output})
                    
            except IndexError as err:
                print("IndexError err")
                print(res)
    
        return data
    

    def convert_sft_to_df(self, df:pd.DataFrame) -> pd.DataFrame:
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
                        api = self.convert_sptoken_sft2str(m['content'])
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
                    assistant = self.convert_sptoken_sft2str(m['content'])
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

    def convert_sptoken_sft2str(self, content_ls: list) -> str:
        """
        将带有special token的list型的api_content 或者 assistant_content，转为易读的str型数据
        """
        assistant = ''
        for item in content_ls:
            if isinstance(item, str):
                assistant += item
            elif isinstance(item, dict) and 'token' in item and item['token'] != '<irr>':
                assistant += item['token']
        return assistant
        
        
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
        utils.create_directory(output_load_folder + '{}/multi/'.format(category))
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


def convert_to_chatml_single_data(x):
    signs = {"<|lc_start|>":"[unused0]","<|lc_end|>":"[unused1]","<|kvs|>":"[unused2]","<|kve|>":"[unused3]","<|api_start|>":"[unused4]","<|api_end|>":"[unused5]","<|eoa|>":"[unused6]","=>":"[unused7]", "<|br|>":"[unused8]", "<|irrelevant|>":"[unused9]"}
    l = x["messages"]
    text=[]
    ml = len(l)
    #print("ml:{}".format(ml))
    for index in range(ml):
        i = l[index]
        if i["role"] in ["user","thought","api","assistant","observation","observation "]:
            if len(i["content"])==0: #没内容的角色
                if i["role"]=="observation" and index==ml-2:
                    # 将[unused0]assistant接到输入之后
                    item = signs["<|lc_start|>"]+i["role"].rstrip()+"\n<None>"+signs["<|lc_end|>"]+"\n[unused0]assistant"
                else:
                    item = signs["<|lc_start|>"]+i["role"].rstrip()+"\n<None>"+signs["<|lc_end|>"]
                text.append(item)
            else: #有内容的角色
                item_list=[]
                for j in i["content"]:
                        if isinstance(j,dict):
                            if "token" not in j:
                                print(x)
                            item_list.append(signs[j["token"]])
                        else:
                            item_list.append(str(j))
                if i["role"]=="api":
                    item = signs["<|lc_start|>"]+i["role"].rstrip()+"\n"+"".join(item_list)+signs["<|eoa|>"]+signs["<|lc_end|>"]
                elif i["role"]=="observation" and index==ml-2:
                    # 将[unused0]assistant接到输入之后
                    item = signs["<|lc_start|>"]+i["role"].rstrip()+"\n"+"".join(item_list)+signs["<|lc_end|>"]+"\n[unused0]assistant"
                elif i["role"]=="assistant" and index==ml-1:
                    item = "".join(item_list).replace("[unused0]assistant\n", "").replace("[unused1]", "")+signs["<|lc_end|>"]
                elif i["role"]=="assistant" and index!=ml-1:
                    item = signs["<|lc_start|>"]+i["role"].rstrip()+"\n"+"".join(item_list)+signs["<|lc_end|>"]
                else:
                    item = signs["<|lc_start|>"]+i["role"].rstrip()+"\n"+"".join(item_list)+signs["<|lc_end|>"]
                text.append(item)
    # print(text)
    if len(text)%5 !=0 :
        return "###worng data"
    res = [text[i:i+5] for i in range(0, len(text), 5)]
    return res