# -*- coding: utf-8 -*-
import time
import pandas as pd
import requests
import random
import json
import os
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from urllib.parse import quote
from hashlib import md5
# from utils import *

class SearchByHttpTool(object):
    def __init__(self, search_env='dev', search_source='Source_INNER', limit=5, control_param={}):
        self.search_env = search_env
        self.limit = limit
        self.control_param = control_param
        self.search_source = search_source

    # media search
    def media_search(self,origin_query, norm_query, slots, time_slots=None):
        if self.search_env == 'app_dev':
            url = "http://ks-app-dev-inference.ssai-apis-staging.chj.cloud:80/cloud/inner/nlp/kg/knowledge-search-engine/media-search"
        elif self.search_env == 'testtwo':
            url = "http://ks-engine-server-inference.ssai-apis-staging.chj.cloud/cloud/inner/nlp/kg/knowledge-search-engine/media-search"
        else:
            url = "http://ks-{}-engine-server-inference.ssai-apis-staging.chj.cloud/cloud/inner/nlp/kg/knowledge-search-engine/media-search".format(self.search_env)

        req_json_ob = {
        "metadata": {
            "request_info": {
                "msgId": "postman-liping-1reaHUeB9WEITBe9"
            },
            "vehicle_info": {
                "vin": "LW433B126P1037555",
                "vehicle_config_code": "17,64,5,33,0,255,255,73,253,31,95,255,6,32,227,255,255",
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
        "orig_query": origin_query,
        "search_query": norm_query,
        "slots": slots
        }
        if self.limit > 0:
            req_json_ob['limit'] = self.limit
        if time_slots:
            req_json_ob['timeSlots'] = time_slots

        payload = json.dumps(req_json_ob)
        response = requests.request("POST", url, data=payload)

        fix_format_docs = []
        origi_docs_llm = [] #给大模型用，原样的doc返回

        # try:
        res_json = json.loads(response.text)
        for doc in res_json['data']:
            origi_docs_llm.append(doc)

            id = del_in_ch(get_v_4dict_by_keys(doc, ['id'], ''))
            title = del_in_ch(get_v_4dict_by_keys(doc, ['title'], ''))
            content =  del_in_ch(get_v_4dict_by_keys(doc, ['content','snippet'], ''))
            url = del_in_ch(get_v_4dict_by_keys(doc, ['source_link','url'], ''))
            if 'media_resources' in doc:
                url = get_dict_value_bykey(doc['media_resources'], 'play_url') 
            source = del_in_ch(get_v_4dict_by_keys(doc, ['source'], ''))
            extend_data = get_v_4dict_by_keys(doc, ['extend_data'], '') # extend_data is dict
            fix_format_docs.append({
                                'id': id, 
                                'title': title, 
                                'content': content,
                                'url': url,
                                'source': source,
                                'extend_data': extend_data,
                                })
        # except Exception as e:
        #     print('exception:%s, parse search results:%s' % (str(e), response.text)) 
        return fix_format_docs,origi_docs_llm

    # qasearch
    def qa_search(self, origin_query, norm_query,tags, slots, category, time_slots=None):
        if self.search_env == 'app_dev':
            url = "http://ks-app-dev-inference.ssai-apis-staging.chj.cloud:80/cloud/inner/nlp/kg/knowledge-search-engine/search"
        elif self.search_env == 'testtwo':
            url = "http://ks-engine-server-inference.ssai-apis-staging.chj.cloud:80/cloud/inner/nlp/kg/knowledge-search-engine/search"
        else:
            url = "http://ks-{}-engine-server-inference.ssai-apis-staging.chj.cloud:80/cloud/inner/nlp/kg/knowledge-search-engine/search".format(self.search_env)

        req_json_ob = {
        "metadata": {
            "request_info": {
                "msgId": "postman-liping-1reaHUeB9WEITBe9"
            },
            "vehicle_info": {
                "vin": "1c4dd333c1f111111", 
                "vehicle_config_code": "7,64,9,32,1,255,255,9,77,3,67,3,2,3,3,255,255", 
                "hu_ota_version": "4.5.1-1.12.101",
                "hu_version": "4.1.12006",
                "spu": "ss3pro",
                "user_manual_version": "7",
                "help_app_version": "2003000",
                "voice_version": "5.2.0.4",
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
        "orig_query": origin_query,
        "search_query": norm_query,
        "search_category": [category],
        "source": self.search_source,
        "search_slots": {
            "tag": tags,
            "slots":slots
        },
        "control_param": self.control_param, 
        "searchBot": "kg-bot"
        }
        if self.limit > 0:
            req_json_ob['limit'] = self.limit
        if time_slots :
            req_json_ob['timeSlots'] = time_slots
            

        # payload = json.dumps(req_json_ob, ensure_ascii=False)
        payload = json.dumps(req_json_ob)

        # response = requests.request("POST", url, headers=headers, data=payload) #header不需要，或者如果要，记得host跟着切换才行
        response = requests.request("POST", url, data=payload)

        fix_format_docs = []
        origi_docs_llm = [] #给大模型用，原样的doc返回
        try:
            res_json = json.loads(response.text)

            for doc in res_json['data'][0]['bot_data']:
                origi_docs_llm.append(doc)  #大模型用，原样的doc返回  

                id = del_in_ch(get_v_4dict_by_keys(doc, ['id'], ''))
                title = del_in_ch(get_v_4dict_by_keys(doc, ['title'], ''))
                content =  del_in_ch(get_v_4dict_by_keys(doc, ['content','snippet'], ''))
                url = del_in_ch(get_v_4dict_by_keys(doc, ['source_link','url'], ''))
                source = del_in_ch(get_v_4dict_by_keys(doc, ['source'], ''))
                extend_data = get_v_4dict_by_keys(doc, ['extend_data'], '')
                score = get_v_4dict_by_keys(doc, ['rel_score'], '')
                source_domain = get_v_4dict_by_keys(doc, ['source_domain'], '')
                fix_format_docs.append({
                                    'id': id, 
                                    'title': title, 
                                    'content': content,
                                    'url': url,
                                    'source': source,
                                    'score': score,
                                    'source_domain': source_domain,
                                    'extend_data': extend_data,
                                    })
        except Exception as e:
            print('exception:%s, parse search results:%s, query:%s' % (str(e), response.text, norm_query))       
        return fix_format_docs, origi_docs_llm

    def bing_native_search(self, norm_query):
        url = "https://csd-api-ontest.inner.chj.cloud/bcs-apihub-search-proxy-service/apihub/search/v1.0/bing-native"
        payload = {
            "q": norm_query,
            "responseFilter": "Webpages",
            "mkt": "zh-CN",
            "setLang": "zh-hans",
            "count": self.limit
        }
        headers = {
        'BCS-APIHub-RequestId': 'dean-loca-postman',
        'X-CHJ-GWToken': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJaNFJPWjBhSHRucUp4VUtQQjRtbjlIUUV3aEFlNmdoWiJ9.vPGBa8ngutKZkPhvbBNYZfHQnhn9cWRY6mdYyKbwDao'
        }
        response = requests.get(url, params=payload, headers=headers)
        _res_list = []
        try:
            res_json = json.loads(response.text)
            for bot_data in res_json['data']['webPages']['value']:
                date_published = get_v_4dict_by_keys(bot_data, ['datePublished'], 'none')
                title = get_v_4dict_by_keys(bot_data, ['name'], '').replace('\"', '').replace('\\"','')
                content = get_v_4dict_by_keys(bot_data, ['snippet'], '').replace('\"', '').replace('\\"','')
                _res_list.append({
                                    'title': title, 
                                    'content': title + " " + content,
                                    'datePublished' : date_published
                                    })
        except Exception as e:
            print('bing-native,except=%s, query=%s, response_text=%s' % (e, norm_query, response.text))
            _res_list = []
        return _res_list 

    # 定制接口，目标bing主要方式
    def bing_custom_search(self, norm_query, is_freshness):
        url = "https://csd-api-ontest.inner.chj.cloud/bcs-apihub-search-proxy-service/apihub/search/v1.0/bing-customization"
        payload = {
            "queryInfo": norm_query,
            "type": "Webpages",
            "mkt": "zh-CN",
            "setLang": "zh-hans",
            "count": self.limit
        }
        if is_freshness:
            payload['freshness']='Month'

        headers = {
        'BCS-APIHub-RequestId': 'dean-loca-postman',
        'X-CHJ-GWToken': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJaNFJPWjBhSHRucUp4VUtQQjRtbjlIUUV3aEFlNmdoWiJ9.vPGBa8ngutKZkPhvbBNYZfHQnhn9cWRY6mdYyKbwDao'
        }
        response = requests.get(url, params=payload, headers=headers)
        _res_list = []
        try:
            res_json = json.loads(response.text)
            for bot_data in res_json['data']['webPages']['value']:
                date_published = get_v_4dict_by_keys(bot_data, ['datePublished'], 'none')
                title = get_v_4dict_by_keys(bot_data, ['name'], '').replace('\"', '').replace('\\"','')
                content = get_v_4dict_by_keys(bot_data, ['snippet'], '').replace('\"', '').replace('\\"','')
                url = get_v_4dict_by_keys(bot_data, ['url'], '').replace('\"', '').replace('\\"','')
                _res_list.append({
                                    'title': title, 
                                    'content': content,
                                    'comb_content': title + " " + content,
                                    'url': url,
                                    'datePublished' : date_published
                                    })
        except Exception as e:
            print('bing-custom,except=%s, query=%s, response_text=%s' % (e, norm_query, response.text))
            _res_list = []
        return _res_list 

    # 实体链接服务
    def entity_link_http(self, query):
        if self.search_env == 'testtwo':
            url = "http://ks-engine-server-inference.ssai-apis-staging.chj.cloud:80/cloud/inner/nlp/kg/knowledge-search-engine/entity-link"
        else:
            url = "http://ks-{}-engine-server-inference.ssai-apis-staging.chj.cloud:80/cloud/inner/nlp/kg/knowledge-search-engine/entity-link".format(self.search_env)
        payload = {
            "query": query,
            "extra": {
                "disable-bing-cache": "",
                "outer-search-api": ""
            }
        }
        response = requests.post(url, json=payload)
        link_entitys = [] 
        unlink_entitys = []
        try:
            res_json = json.loads(response.text)
            if 'data' in res_json and len(res_json['data']) > 0 : 
                for data in res_json['data']:
                    if '人物' in data['general']['category']:
                        link_entitys.append(data['general']['mention'])

            if 'unlink_entity' in res_json and len(res_json['unlink_entity']) > 0 : 
                for data in res_json['unlink_entity']:
                    if 'PER' in data['category']:
                        unlink_entitys.append(data['mention'])
        except Exception as e:
            print('entity link exception=%s, query=%s, response_text=%s' % (e, query, response.text))
        return link_entitys, unlink_entitys 

    # 儿童百科问答
    def child_qa_http(self,query):
        url = "http://ssai-kid-ency-bot-lids-testtwo.ssai-apis-staging.chj.cloud/encyclopedia/kid/voice/query/release_test"
        payload = {
            "msgId": "2be44933c3244716ae302d91c0113f8e",
            "aiUserId": "LW433B126P1043369",
            "nlu": [
                {
                    "dass": [
                        {
                            "domain": "encyclopedia"
                        }
                    ]
                }
            ],
            "input": {
                "text": query
            },
            "timestamp": 1680831373032,
            "dialogStatus": "start",
            "communicationType": "fullDuplex"
        }
        response = requests.post(url, json=payload)
        domain = 'other'
        answer = 'unknown'
        try:
            res_json = json.loads(response.text)
            data_ret_obj = res_json['data']
            if 'speak' in data_ret_obj:
                domain = 'encyclopedia'
                answer = data_ret_obj['speak']
        except Exception as e:
            print('call-api,except=%s, query=%s, response_text=%s' % (e, query, response.text))

        return domain, answer

    # 360搜索    
    def get_360_http_1query(self, query):
        host_360 = 'https://openapi.m.so.com/v2/mwebsearch'
        keyword = query
        cid = 'lixiang_search'
        key = '219d9cb10c3f'
        tm = str(int(round(time.time() )))
        m = md5()
        m.update((cid + query + key + tm).encode("utf-8") )
        sign = m.hexdigest()[:16]

        # https://openapi.m.so.com/v2/mwebsearch?q=%E5%A4%A9%E7%A9%BA%E4%B8%BA%E4%BB%80%E4%B9%88%E6%98%AF%E8%93%9D%E8%89%B2%E7%9A%84&m=9aafc5d38c7f0319&t=1677917280&cid=lixiang_search
        url = f"{host_360}?q={keyword}&m={sign}&t={tm}&cid={cid}"
        print(url)

        response = requests.get(url)
        _res_list = []
        replace_parts = ['<b>','</b>']
        try:
            res_json = json.loads(response.text)
            if 'errno' not in res_json or res_json['errno'] != 0:
                print('call 360,get result error, query=%s' % query)
                return 'none', response.text, [] 
            for bot_data in res_json['items']:
                item_from = bot_data['from']
                item_type = bot_data['type'] 

                if item_from == 'onebox' and item_type == 'wenda_abstract':
                    title = get_v_4dict_by_keys(bot_data, ['title'], 'none')
                    content = get_v_4dict_by_keys(bot_data, ['abstract'], 'none')
                    url = bot_data['url']
                elif item_from != 'onebox':
                    title = get_v_4dict_by_keys(bot_data, ['title'], 'none')
                    content = get_v_4dict_by_keys(bot_data, ['summary','content_large','contentdtl'], 'none')
                    url = bot_data['url']
                else:       # 其他情况，不考虑
                    continue
    
                title = get_replaced_query(title, replace_parts)
                content = get_replaced_query(content, replace_parts)

                if title == 'none' or  content == 'none':
                    continue

                one_ans_json_obj = {
                'title': title, 
                'content': content,
                'comb_content': title + " " + content,
                    'url' : url 
                }
                _res_list.append(one_ans_json_obj)
                if len(_res_list) >= self.limit:
                    break
        except Exception as e:
            print('360 except=%s, query=%s, response_text=%s' % (e, query, response.text))
            _res_list = []
        print(_res_list)
        return url, response.text, _res_list 


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

