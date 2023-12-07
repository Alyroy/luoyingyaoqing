import json
import requests
from tqdm import tqdm
import pandas as pd
import time
import random  
import copy
import ast

from utils_data_format_conversion import convert_api_raw2sft,convert_api_str2dict,convert_api_sft2raw


def gen_api(category: str, query: str) -> [str, str]:
    """
    调用1B模型服务，快速生成API
    Param:
        category:"QASearch","AUTOSearch","AIPainting","TaskMaster","MEDIASearch","other_vision","other_origin"
        query: str
    Returns:
        thought: str
        api: str （含special token）
    """
    # url = "http://172.21.194.26:8018/ligpt_with_api/search"
    url = "http://172.21.194.26:16666/ligpt_with_api/search"

    payload = json.dumps({
      "prompt_id": category,
      "query": [query]
    })
    headers = {
      'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    res_json = json.loads(response.text)
    
    try:
        thought = res_json['first_response']['thought']
        api = res_json['first_response']['api']
    except Exception as e:
        print(query, e)
        thought = ''
        api = ''
        
    return thought, api
    

# def convert_api_sft2raw(api_content: list) -> [list,list,list,list]:
#     """
#     从sft api数据格式中提取单一元素
#     """
    
#     api_names,categorys,api_querys,api_tags = [],[],[],[]

#     for i in range(len(api_content)):
#         if api_content[i] == 'APINAME':
#             api_names.append(api_content[i+2])
#         elif api_content[i] == 'CATEGORY':
#             categorys.append(api_content[i+2])
#         elif api_content[i] == 'QUERY':
#             api_querys.append(api_content[i+2])
#         elif api_content[i] == 'TAG':
#             api_tags.append(api_content[i+2])

#     return api_names,categorys,api_querys,api_tags


def get_api_df(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """
    提取 api中需要的元素，方便调取搜索
    """
    thought_ls, api_names_ls,api_categorys_ls,api_querys_ls,api_tags_ls = [],[],[],[],[]
    apis_ls = []
    for query in tqdm(df['user-query'].to_list()):
        thought, api_content = gen_api(category,query)
        apis_ls.append(api_content)
        thought_ls.append(thought)
        try:
            api_ls = convert_api_raw2sft(api_content)
            api_names,api_categorys,api_querys,api_tags = convert_api_sft2raw(api_ls)
        except Exception as e:
            print(e)
            api_names,api_categorys,api_querys,api_tags
        api_names_ls.append(api_names)
        api_categorys_ls.append(api_categorys)
        api_querys_ls.append(api_querys)
        api_tags_ls.append(api_tags)
    
    df['Thought'] = thought_ls
    df['API'] = apis_ls
    df['API-NAME'] = api_names_ls
    df['API-CATEGORY'] = api_categorys_ls
    df['API-QUERY'] = api_querys_ls
    df['API-TAG'] = api_tags_ls
    
    return df


def search_bing(query: str,top_k : int = 3) -> list:
    """
    逆序返回搜索结果，即top 3 => 321
    """
    url = "http://ks-dev-engine-server-inference.ssai-apis-staging.chj.cloud:80/cloud/inner/nlp/kg/knowledge-search-engine/search"

    payload = json.dumps({
      "search_query": query,
      "source": "Source_BING",
      "search_slots": {
        "tag": []
      },
      "searchBot": "kg-bot"
    })
    headers = {
      'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    res_json = json.loads(response.text)

    observation = []
    try:
        for _data in res_json["data"]:
            for bot_data in _data["bot_data"]:
                observation.append(bot_data["content"])
    except Exception as e:
        print('search_single_api error:')
        print(query, e)

    return observation[:top_k][::-1]

def postmansearch_single_api(user_query, api_query, tag, category, top_k=3, env='dev') -> list:
    """
    内部搜索接口
    """
    if env == 'dev':
        url = "http://ks-dev-engine-server-inference.ssai-apis-staging.chj.cloud:80/cloud/inner/nlp/kg/knowledge-search-engine/search"
    elif env == 'arch':
        url = 'http://ks-arch-engine-server-inference.ssai-apis-staging.chj.cloud:80/cloud/inner/nlp/kg/knowledge-search-engine/search'
    else:
        raise '目前只支持 dev arch环境'
    payload = json.dumps({
        "metadata": {
        "request_info": {
            "msgId": "postman-kgsearch-rhm"
        },
      "vehicle_info": {
      "vin": "LW433B126P1037555",
      "vehicle_config_code": "1,64,2,0,34,63,255,0,253,255,159,255,0,31,227,255,255",
      "hu_ota_version": "4.5.1-1.12.101",
      "hu_version": "4.1.12006",
      "spu": "",
      "user_manual_version": "7",
      "help_app_version": "2003000",
      "voice_version": "5.0.0.0",
      "vehicle_model": "VEHICLE_MODEL_X01"
        },
        "dynamic_info": {
          "speaker_seat": "RIGHT_SEAT_OF_SECOND_ROW",
          "night_mode": 2,
          "position": {
            "coordinate": "wgs84",
            "latitude": 22.993279,
            "longitude": 113.702286,
            "country": "",
            "city": ""
          }
        }
      },
      "orig_query": user_query,
      "search_query": api_query,
      "search_category": [category],
      "source": "Source_INNER",
      "search_slots": {
          "tag": tag
      },
      "searchBot": "kg-bot"
    })
    headers = { 'Content-Type': 'application/json'}

    response = requests.request("POST", url, headers=headers, data=payload)

    JsonResponse = json.loads(response.text)

    observation = []
    try:
        for i in range(top_k):
            observation += [JsonResponse['data'][0]['bot_data'][i]['content']]
    except Exception as e:
        print('search_single_api error: ')
        print(user_query, e)

    return observation[::-1] # 逆序


def get_media_slots(query: str) -> dict:
    url = "http://nlu-testone-lids-server-inference.ssai-apis-staging.chj.cloud:80/cloud/inner/nlp/lids/nlu_engine/parse_v2"

    payload = json.dumps({
      "input": {
        "text": query
      }
    })

    headers = {
      'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    res_json = json.loads(response.text)

    try:
        slots = res_json['dialog_acts'][0]['dass'][0]['slots']
    except:
        slots = ''
        
    return slots


def get_media_observation(query: str, slots: dict, top_k :int = 10) -> list:
    """
    通过query和slots返回 影视推荐相关内容
    """
    url = "http://ks-engine-server-inference.ssai-apis-staging.chj.cloud:80/cloud/inner/nlp/kg/knowledge-search-engine/media-search"

    payload = json.dumps({
      "metadata": {
        "request_info": {
          "msgId": "postman-dean-1reaHUeB9WEITBe9"
        }
      },
      "orig_query": "",
      "search_query": query,
      "slots": slots
    })
    headers = {
      'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    res_json = json.loads(response.text)
    
    observations = []
    try:
        for _data in res_json["data"]:
            observations.append(_data['content'])
    except Exception as e:
        print('media search error:')
        print(query, e)

    return observations[:top_k][::-1]
    # return random.sample(observations, min(10,len(observations)))  


def get_media_obs_df(df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """
    通过解析1B的API，提取slots，利用mediaSearch接口调取obs
    默认返回10个obs
    """
    api_ls = df['API'].to_list()
    slots_ls = []
    obs_ls = []
    for api in api_ls:
        slots = convert_api_str2dict(api)
        obs = []
        slots_copy = copy.deepcopy(slots)
        for slot in slots_copy:
            query = slot['QUERY']
            slot.pop('APINAME',None)
            slot.pop('QUERY',None)
            single_obs = get_media_observation(query,slot,top_k)
            obs.append(single_obs)
        slots_ls.append(slots)
        obs_ls.append(obs)

    df['slots'] = slots_ls
    df['observation'] = obs_ls
    
    return df


def get_all_observation(df: pd.DataFrame, top_k: int = 3, max_retries: int = 1, random_k_flag: bool = False, bing_flag: bool = False, env: str = 'dev') -> pd.DataFrame:
    """
    如果有多个api，则返回多个observation
    df: necessary columns = [API-QUERY,API-TAG,API-CATEGORY]，每个值由list呈现，例如“['人物']”
    return: df add new columns = ['observation']
    """
    print('env',env)
    observation_ls = []
    for i in tqdm(range(len(df))):
        if df.iloc[i]['API-QUERY'] == "[]":
            observation_ls.append([])
        else:
            user_query = df.iloc[i]['user-query']
            api_querys = ast.literal_eval(df.iloc[i]['API-QUERY'])
            tags_ = ast.literal_eval(df.iloc[i]['API-TAG'])
            tags = [tag.split('&') for tag in tags_]
            categorys = ast.literal_eval(df.iloc[i]['API-CATEGORY'])

            observations = []
            for k in range(len(api_querys)):
                retries = 0  # 重置重试计数器 
                if random_k_flag:
                    top_k = random.choice([2,3,4,5])
                while retries < max_retries:  # 最大重试次数限制
                    try:
                        if bing_flag:
                            observation = search_bing(user_query, top_k)
                        else:
                            if categorys[k] == '汽车':
                                observation = postmansearch_single_api(user_query, api_querys[k], tags[k], categorys[k], top_k, env)
                            else:
                                observation = postmansearch_single_api(api_querys[k], api_querys[k], tags[k], categorys[k], top_k, env)
                        break  # 如果成功，就跳出while循环
                    except Exception as e:
                        print('第{}个 search api 调用异常：{}'.format(i, e))
                        time.sleep(1) # 异常调用后休眠5秒
                        retries += 1  # 如果失败，增加重试计数器
                if retries == max_retries:  # 如果重试次数达到最大值
                    print('最大重试次数已达，返回空')
                    # observation = ['我目前无法提供准确的答案。']
                    observation = []
                observations.append(observation)
            observation_ls.append(observations)
            
    df['observation'] = observation_ls

    return df


def custom_order_2d(df: pd.DataFrame, order: list = [0,1,2,3,4]) -> pd.DataFrame:
    """
    后处理obs顺序
    """
    def reorder_observations(outer_list):
        return [[inner[i] for i in order if i < len(inner)] for inner in outer_list]

    df['observation'] = df['observation'].apply(reorder_observations)
    return df