import pandas as pd
import numpy as np
import json
from typing import Dict, Any, List, Optional, TypeVar, Generic, Type, Tuple, Union
import re


def get_knowledge_search_msg(str_knowledge):
    knowledge_search_msg_list = []
    if str_knowledge == "":
        json_knowledge = []
    else:
        try:
            json_knowledge = json.loads(str_knowledge)
        except:
            json_knowledge = []

    for item in json_knowledge:
        knowledge_search_data = item.get("knowledge_search_data", {}).get("bot_data", [])
        knowledge_search_item_list = []
        for sub_item in knowledge_search_data:

            source = sub_item["source"]
            url = sub_item.get("source_link", "")
            title = sub_item.get("title", "")
            content = sub_item.get("content", "")
            extend_data = sub_item.get("extend_data", "")

            sub_dict = {
                "source": source,
                "url": url,
                "title": title,
                "content": content,
                "extend_data": extend_data
            }
            knowledge_search_item_list.append(sub_dict)
        knowledge_search_msg_list.append(knowledge_search_item_list)

    return knowledge_search_msg_list


# "<|kvs|>": "[unused2]",
# "<|kve|>": "[unused3]",
# "<|api_start|>": "[unused4]",
# "<|api_end|>": "[unused5]",
# "<|eoa|>": "[unused6]",
# "=>": "[unused7]",
def pre_process_input(request_text):
    """
    对T5的response进行分词处理
    """
    input_ = request_text.replace(" ", "")
    tokens = re.findall(r'\[((?:.|\n)*?)\]', input_)

    for x in tokens:
        x = "[" + x + "]"
        input_ = input_.replace(x, " ")
    input_ = input_.strip()
    slots = input_.split(" ")
    return tokens, slots


def clean_pattern(str_item):
    str_item = str_item.replace("[unused2]", "[")
    str_item = str_item.replace("[unused3]", " ")

    str_item = str_item.replace("[unused7]", "]")
    str_item = str_item.replace("[unused4]", " ")
    str_item = str_item.replace("[unused6]", " ")

    str_item = str_item.replace("[unused5]", " ")
    return str_item

def format_api_label(str_label):
    tokens, slots = pre_process_input(str_label)

    skill_domain_dict = dict()
    for index in range(min(len(tokens), len(slots))):
        skill_domain_dict[tokens[index]] = slots[index]

    return skill_domain_dict


def clean_api_pattern(pro_input):
    pro_input = pro_input.replace("[unused0] api", "[unused0]api")
    pattern_list = re.findall("\[unused0\]api(.*?)\[unused1\]", pro_input.replace("\n", "")) # 以防多API情况

    api_pattern_list = []
    for item in pattern_list:
        item = clean_pattern(item)
        if item.count("APINAME") > 1:
            # case:  [APINAME]AUTOSearch [CATEGORY]汽车 [QUERY]理想L9简介 [TAG]理想L9&简介   [APINAME]AUTOSearch [CATEGORY]汽车 [QUERY]理想ONE简介 [TAG]理想ONE&简介
            api_list = item.strip().split("[APINAME]")
            for api_item in api_list:
                api_item = format_api_label(api_item)
                if api_item != {}:
                    api_pattern_list.append(api_item)
        else:
            item = format_api_label(item)
            if item != {}:
                api_pattern_list.append(item)

    return api_pattern_list


def clean_thought_pattern(pattern_list):
    thought_pattern_list = []
    for item in pattern_list:
        item = clean_pattern(item)
        item = {"thought": item}
        thought_pattern_list.append(item)

    return thought_pattern_list


def clean_observation_pattern(pattern_list):
    clean_pattern_list = []
    for item in pattern_list:
        observation_sub_list = re.findall("\[unused2\](.*?)\[unused3\]", item)
        for osl in observation_sub_list:
            try:
                osl = json.loads(osl)
            except:
                osl = {}
            clean_pattern_list.append(osl)
    # re.findall("\[unused0\]thought(.*?)\[unused1\]", llm_input.replace("\n", ""))
    return clean_pattern_list


def extra_api_thought_observation(str_llm_input):
    str_thought, str_api, str_observation = [""], [""], [""]

    if "[unused0]thought" in str_llm_input:
        str_thought = re.findall("\[unused0\]thought(.*?)\[unused1\]", str_llm_input.replace("\n", ""))
        str_thought = clean_thought_pattern(str_thought)

    if "[unused0]api" in str_llm_input:
        str_api = re.findall("\[unused0\]api(.*?)\[unused1\]", str_llm_input.replace("\n", ""))
        str_api = clean_api_pattern(str_api)

    if "[unused0]observation" in str_llm_input:
        str_observation = re.findall("\[unused0\]observation(.*?)\[unused1\]", str_llm_input.replace("\n", ""))
        str_observation = clean_observation_pattern(str_observation)

    return str_thought, str_api, str_observation


def get_api_observation(str_prompts):

    try:
        prompts = str_prompts["llm_in_outs"]
    except:
        return [], [], []

    # 选择13b的结果
    llm_input_1b = ""
    llm_input_13b = ""
    for llm_str in prompts:
        if "你是一个名字叫做理想同学的AI机器人" in llm_str.get("llm_input", ""):
            llm_input_13b = llm_str["llm_input"]
        if "当前任务是根据当前请求和上文信息生成正确的API信息" in llm_str.get("llm_input", ""):
            llm_input_1b = llm_str["llm_output"].replace(" ", "").strip()

    thought_1b, api_1b, str_observation_1b = extra_api_thought_observation(llm_input_1b)
    thought_13b, api_13b, str_observation_13b = extra_api_thought_observation(llm_input_13b)

    # 如果13b没有结果, 选择1b的结果
    if llm_input_13b == "" or "CHARASearch" in str(api_1b):
        return thought_1b, api_1b, str_observation_1b
    else:
        return thought_13b, api_13b, str_observation_13b


def get_model_input_outputs(str_prompts, data_type):

    try:
        prompts = str_prompts["llm_in_outs"]
    except:
        return "", ""

    # 选择13b的结果
    llm_input_1b, llm_output_1b = "", ""
    llm_input_13b, llm_output_13b = "", ""
    for llm_str in prompts:
        if "当前任务是根据当前请求和上文信息生成正确的API信息" in llm_str.get("llm_input", ""):
            llm_input_1b = llm_str.get("llm_input", "")
            llm_output_1b = llm_str.get("llm_output", "")

        if "你是一个名字叫做理想同学的AI机器人" in llm_str.get("llm_input", ""):
            llm_input_13b = llm_str["llm_input"]
            llm_output_13b = llm_str["llm_output"]
        else:
            if llm_input_1b == "":
                llm_input_1b = llm_str.get("llm_input", "")
                llm_output_1b = llm_str.get("llm_output", "")

    if data_type == "1b":
        return llm_input_1b, llm_output_1b
    else:
        return llm_input_13b, llm_output_13


def extra_observation(data_observation):

    str_observation = str(data_observation)

    if str_observation in ["<None>", ""]:
        str_observation = ""
    else:
        if "TaskMaster" in str_observation:
            str_observation = ""
        elif "TODRequestResults" in str_observation:
            str_observation = ""
        else:
            str_observation = str_observation
            # print(str_observation)
    return str_observation