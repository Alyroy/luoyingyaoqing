import re
import json
import time


def parser_date(data: str):
    """日志解析用户提问时间"""
    user_date = re.findall(r"用户今天日期：(.*?)。用户现在时间", data.replace("\n", ""), re.S)
    return user_date[0] if len(user_date)>0 else time.strftime("%Y年%m月%d日")

def parser_loc(data: str):
    """日志解析用户提问地点"""
    user_address = re.findall(r"用户现在位置：(.*?)。", data.replace("\n", ""), re.S)
    return user_address[0] if len(user_address)>0 else "中国"


def parser_obs(pro_input: str) -> list[list]:
    """日志解析检索结果"""
    pro_input = pro_input.replace("[unused0] observation", "[unused0]observation")
    obs = re.findall("\[unused0\]observation(.*?)\[unused1\]", pro_input.replace("\n", ""))
    obs = clean_observation_pattern(obs)
    return obs


def parser_context_query(data:str) ->(list[dict],str):
    """
    获取上文对话 和 最后一轮query
    """
    pattern = re.compile(r'\[unused0\]user\n(.*?)\[unused0\]thought', re.DOTALL)
    match = pattern.findall(data)[0]
    match = match.replace('[unused8]','\n').replace('\nthought\n','')
    match = '[unused0]user\n'+match
    context,query = get_context_query(match)
    return context,query


def clean_observation_pattern(pattern_list):
    clean_pattern_list = []
    for item in pattern_list:
        observation_sub_list = re.findall("\[unused2\](.*?)\[unused3\]", item)
        for osl in observation_sub_list:
            try:
                osl = list(json.loads(osl).values())[0]
            except:
                osl = []
            clean_pattern_list.append(osl)
    return clean_pattern_list


def get_context_query(data):
    # 用正则表达式分割 data 为多个段落
    segments = re.split(r'\[unused[01]\]', data)
    
    # 删除空字符串和多余的空白字符
    segments = [seg.strip() for seg in segments if seg.strip()]
    
    result = []
    i = 0
    while i < len(segments):
        if segments[i].startswith("user"):
            user_text = re.sub(r'^user\n', '', segments[i])
            if i + 1 < len(segments) and segments[i + 1].startswith("assistant"):
                assistant_text = re.sub(r'^assistant\n', '', segments[i + 1])
                result.append({'user': user_text, 'assistant': assistant_text})
                i += 2  # 跳过 assistant 的段落
            else:
                result.append({'user': user_text})
                i += 1
        elif segments[i].startswith("assistant"):
            assistant_text = re.sub(r'^assistant\n', '', segments[i])
            result.append({'assistant': assistant_text})
            i += 1
    
    return result[:-1],result[-1]['user']