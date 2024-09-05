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


def get_query_result_from_16b_input(input_16b):
    pattern = "user\n(.*?)\[unused1\]"
    results = re.findall(pattern, input_16b, re.DOTALL)
    if len(results) > 0:
        return results[-1], len(results)
    else:
        return "", 0
    
def get_context_result_from_16b_input(input_16b):
    pattern_user = "user\n(.*?)\[unused1\]"
    pattern_assistant = "assistant\n(.*?)\[unused1\]"
    results_user = re.findall(pattern_user, input_16b, re.DOTALL)
    results_assistant = re.findall(pattern_assistant, input_16b)
    context_list = []
    if len(results_user) - len(results_assistant) == 1:
        for i in range(len(results_user)-1):
            context_list.append(
                {
                    "user": results_user[i],
                    "assistant": results_assistant[i]
                }
            )
    return context_list


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
