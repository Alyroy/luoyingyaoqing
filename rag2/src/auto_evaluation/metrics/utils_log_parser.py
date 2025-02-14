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


def content_parser_functioncall(content):
    '''
    Functioncall格式
    '''
    if not isinstance(content, str):
        content_clean = str(content).strip().replace("\n", "")
    else:
        content_clean = content.strip().replace("\n", "")
    content_str = re.findall("\[unused0\]user```function_call_result(.*?)```\[unused1\]", content_clean, re.S)

    if len(content_str) > 0:
        content_list = []
        try:
            content_jsons = json.loads(content_str[0])
        except:
            try:
                content_jsons = json.loads(content_str[0].replace('\'', '\"'))
            except:
                content_jsons = [{'content': []}]
                print('json error')
        
        # print(content_jsons)
        for content_item in content_jsons: # content_jsons[0]
            content_list += content_item["content"]
    else:
        print("No match found")
        content_list = ['']
    return content_list


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

def extract_history(content):
    pattern = r'\[unused0\]user(.*?)\[unused1\]'
    all_matches = re.findall(pattern, content, re.S)
    matches = [match for match in all_matches if '```' not in match]
    if len(matches) == 1:        #单轮对话说明没有历史对话
        return ""
    elif len(matches) > 1:       #说明是多轮对话
        pattern2 = r'\[unused0\]assistant(.*?)\[unused1\]'
        all_matches2 = re.findall(pattern2, content, re.DOTALL)
        matches2 = [match for match in all_matches2 if '```' not in match]
        history = ""
        for index, response in enumerate(matches2): #有response的说明是历史对话
            question = matches[index]
            history += '提问：{}\n'.format(question.strip().replace('\n', ' '))
            history += '回答：{}\n'.format(response.strip().replace('\n', ' '))
        return history
    else:
        return ""


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
