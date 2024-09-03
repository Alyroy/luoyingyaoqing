import pandas as pd
import random
import requests
import json
import os
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
import argparse, logging
import time
import re
import sys 
# from utils import *
from .search_http_tool import SearchByHttpTool 

class BaseApiEle(object):
    def __init__(self, appname, category, normal_query, tag, slots, time_slots=None) -> None:
        self.appname = appname
        self.normal_query = normal_query
        self.category = category
        self.tag = tag
        self.slots = slots
        self.time_slots = time_slots    # api timeslot

    def to_json(self):
        return json.dumps(self, ensure_ascii=False, default=lambda o: o.__dict__, sort_keys=True, indent=4)

class ApiRecord(object):
    def __init__(self, user_query) -> None:
        self.user_query = user_query
        self.api_res_list : List[BaseApiEle] = []  
        self.api_name = ''      
        self.is_valid = 0  #0=invalid, 1 = valild
        self.invalid_desc = ""  # invalid desc
    def to_json(self):
        return json.dumps(self, ensure_ascii=False, default=lambda o: o.__dict__, sort_keys=True, indent=4)


def get_api_4query(user_query, input_api_objs)->ApiRecord:
    city_scene_slot_keys = ["COUNTY", "CITY", "TOWN","PROVINCE", "SCENE_TAG", "TIME","REGION","CROWD"]  # 城市景点推荐slot key
    api_record = ApiRecord(user_query)
    try:
        api_ele_list = []
        if isinstance(input_api_objs, dict):  # 如果是单json,转为[]
            input_api_objs = [input_api_objs]
        for one_api_obj in input_api_objs: # 多api
            api_name = get_v_4dict_by_keys(one_api_obj, ['APINAME','apiname'], '')
            api_name = str(api_name).lower()

            api_record.api_name = api_name

            api_query = get_v_4dict_by_keys(one_api_obj, ['QUERY','query'])
            api_tag =  get_v_4dict_by_keys(one_api_obj, ['TAG','tag'], '')    # mediasearch不一定有
            api_category = get_v_4dict_by_keys(one_api_obj, ['CATEGORY', 'category'],'')  #mediasearch 不一定有

            time_slots = get_v_4dict_by_keys(one_api_obj, ['timeSlots', 'timeslots'],'') 
            if len(time_slots) == 0:
                time_slots = None
            else:
                time_slots = json.loads(time_slots)   #json dict
                
            not_focus_slot_key = ['apiname', 'query','category','tag','starttime','timeslots']  # slot里面不需要解析的key
            if api_name in ['qasearch', 'autosearch'] :
                not_focus_slot_key = ['apiname', 'query','category','tag','starttime','timeslots']
            elif api_name in ['mediasearch']:
                not_focus_slot_key = ['apiname', 'query','tag','starttime','timeslots']
            else:
                api_record.invalid_desc = "un process for %s" % api_name
                print('invalid api type, one_api_str:%s' % (str(one_api_obj)))       

            slots = {}
            if api_category in ['城市景点推荐']:
                for slot_key in city_scene_slot_keys:
                    slots[slot_key.upper()] = {'item' : []}
            for slot_key in one_api_obj:
                slot_key_lw = str(slot_key).lower()
                slot_key_up = str(slot_key).upper()

                slot_value = one_api_obj[slot_key] 

                if slot_key_lw not in not_focus_slot_key:
                    if api_category in ['城市景点推荐'] :
                        slots[slot_key_up] = {'item' : [slot_value] } 
                    else:
                        slots[slot_key_up] = slot_value
                    
            api_ele_list.append(BaseApiEle(api_name, api_category, api_query, api_tag, slots, time_slots))
                
        api_record.api_res_list = api_ele_list
        api_record.is_valid = 1
    except Exception as e:
        print('exception:%s, api 结果输入:%s' % (str(e), str(input_api_objs)) )       
        api_record.invalid_desc = "api输入解析异常" 
        api_record.is_valid = 0
    return api_record
       

 # 单结果
def search_results(search_http_tool:SearchByHttpTool, api_record : ApiRecord):
    http_tool = search_http_tool 
    retry_num = 5   # 重试次数

    multi_llm_obs = []   #返回结果

    query = api_record.user_query
    if api_record.is_valid == 0:   # 不请求搜索
        print('query=%s, api is invalid' % (query) )
        return multi_llm_obs

    for api_res in api_record.api_res_list: #多api 
        for i in range(retry_num):
            if api_record.api_name.lower() in ['qasearch', 'autosearch']:
                tags = api_res.tag.split('&')
                category = str(api_res.category).replace('&',',').split(',',-1)[0] #可能有多个值,取第一个值
                _, origin_docs = http_tool.qa_search(origin_query=query, norm_query=api_res.normal_query, 
                                                    tags=tags, slots=api_res.slots, category=category, time_slots=api_res.time_slots)
            elif api_record.api_name.lower() in ['mediasearch']:
                _, origin_docs = http_tool.media_search(query, api_res.normal_query, api_res.slots, time_slots=api_res.time_slots)
            try:
                if len(origin_docs) > 0:
                    multi_llm_obs.append(origin_docs)
                    break
            except:
                multi_llm_obs.append('')
                break
    return multi_llm_obs


def get_obs(search_http_tool, query, input_api):
    api_record = get_api_4query(query, input_api)
    multi_llm_obs = search_results(search_http_tool, api_record)
    return multi_llm_obs


def get_v_4dict_by_keys(data, keys:List[str], default_value='none'):
    for key in keys:
        if key in data:
            value = data[key]
            if value == None or value == '' or value == '[]':
                continue
            return data[key]
    return default_value


def get_dict_value_bykey(dict_obj, key, dfv='')->str:
    if dict_obj and key in dict_obj:
        if isinstance(dict_obj[key],list):
            return json.dumps(dict_obj[key], ensure_ascii=False)
        else:
            return del_in_ch(str(dict_obj[key]))
    return dfv


# 去掉换行等无效字符
def del_in_ch(str_content: str) ->str:
    return str_content.replace('\r', '').replace('\n', '').replace('\"', '').replace('\\"','').replace('\t','')


def get_replaced_query(query:str, replace_parts:List[str]):
    query = query.strip()
    for replace_p in replace_parts:
        query = query.replace(replace_p, '')
    return query



if __name__ == '__main__':
    search_env = 'dev'     #调用的搜索环境，取值: arch | dev | liping | das | faq ... 
    search_env = 'arch'     #调用的搜索环境，取值: arch | dev | liping | das | faq ... 
    search_env = 'testtwo'     # testtwo
    search_env = 'app_dev'     #最近:手机app-dev

    control_param =  {      # 搜索需要的参数, 默认都不打开
        # "disable-bing-cache":"true",      # 是否不要bing cache
    }
    search_http_tool = SearchByHttpTool(search_env=search_env, 
                                        limit=-1,       # -1 则采用搜索内部默认, >0则手动指定 
                                        control_param=control_param)

    query = '王力宏是谁'
    input_api = [{"apiname": "QASearch", "category": "通用问答", "query": "王力宏的简介", "tag": "王力宏&简介"}]

    query = '周杰伦和王力宏谁厉害'
    input_api = [{"apiname": "QASearch", "category": "通用问答", "query": "周杰伦的简介", "tag": "周杰伦&简介"},
                 {"apiname": "QASearch", "category": "通用问答", "query": "王力宏的简介", "tag": "王力宏&简介"}]

    # query = '目前最先进的ai芯片是哪家公司的'
    # input_api = [{"STARTTIME":"{\"norm\":{\"now\":\"true\"},\"raw\":\"目前\"}","APINAME":"QASearch","QUERY":"目前最先进的AI芯片是哪家公司的","CATEGORY":"公司","TAG":"目前&最先进&AI芯片&公司&时效性","timeslots":"{\"type\":\"ABSOLUTE_TIME\",\"range\":\"POINT\",\"format\":\"yyyy-MM-dd HH:mm:ss\",\"start\":{\"raw\":\"目前\",\"time\":\"2024-05-24 21:01:56\",\"norm\":{\"now\":\"true\"}},\"end\":null,\"duration\":null,\"supplemental\":{\"raw\":\"\",\"time\":\"2024-05-24 21:01:56\",\"norm\":null},\"interval\":\"\"}"}]
    multi_llm_obs = get_obs(search_http_tool, query, input_api)
    print('finished')