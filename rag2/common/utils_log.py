import requests
import json
import time
import re
import torch
import traceback
import pandas as pd

########################################
# log解析
########################################

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

def get_system_result_from_16b_input(input_16b):
    pattern = "system\n(.*?)\[unused1\]"
    results = re.findall(pattern, input_16b, re.DOTALL)
    if len(results) > 0:
        return results[-1]
    else:
        return ""

def get_system_data(input_16b):
    pattern = r"用户今天日期：(?P<Date>.*?)。\s*用户今天农历：(?P<LunarDate>.*?)。\s*用户现在时间：(?P<Time>.*?)。\s*用户现在位置：(?P<Location>.*?)。"
    match = re.search(pattern, input_16b)

    if match:
        user_date = match.group('Date')
        lunar_date = match.group('LunarDate')
        user_time = match.group('Time')
        user_location = match.group('Location')
        data_str = f'用户今天日期：{user_date}。用户今天农历：{lunar_date}。用户现在时间：{user_time}。用户现在位置：{user_location}。\n'
        return data_str
    else:
        return ""


########################################
# from 徐洋的武器库 llm_utils log解析相关
# "<|kvs|>": "[unused2]",
# "<|kve|>": "[unused3]",
# "<|api_start|>": "[unused4]",
# "<|api_end|>": "[unused5]",
# "<|eoa|>": "[unused6]",
# "=>": "[unused7]"
########################################

def clean_pattern(str_item):
    str_item = str_item.replace("[unused2]", "[")
    str_item = str_item.replace("[unused3]", " ")

    str_item = str_item.replace("[unused7]", "]")
    str_item = str_item.replace("[unused4]", " ")
    str_item = str_item.replace("[unused6]", " ")

    str_item = str_item.replace("[unused5]", " ")
    return str_item


def clean_api_pattern(pattern_list):
    api_pattern_list = []
    for item in pattern_list:
        item = clean_pattern(item)
        if item.count("APINAME") > 1:
            # case:  [APINAME]AUTOSearch [CATEGORY]汽车 [QUERY]理想L9简介 [TAG]理想L9&简介   [APINAME]AUTOSearch [CATEGORY]汽车 [QUERY]理想ONE简介 [TAG]理想ONE&简介
            api_list = item.strip().split("[APINAME]")
            for api_item in api_list:
                api_item = format_api_label_multi(api_item)
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
        if item != "<None>":
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


def format_api_label(str_label):
    tokens, slots = pre_process_input(str_label)

    skill_domain_dict = dict()
    for index in range(min(len(tokens), len(slots))):
        skill_domain_dict[tokens[index]] = slots[index]

    return skill_domain_dict

def format_api_label_multi(str_label):
    if str_label.replace(" ", "") == "":
        return {}
    tokens, slots = pre_process_input("[APINAME]"+str_label)

    skill_domain_dict = dict()
    for index in range(min(len(tokens), len(slots))):
        skill_domain_dict[tokens[index]] = slots[index]

    return skill_domain_dict


def extra_api_thought_observation(str_llm_input):
    str_thought, str_api, str_observation = [], [], []

    str_llm_input = str_llm_input.replace("[unused0] thought", "[unused0]thought")
    if "[unused0]thought" in str_llm_input:
        str_thought = re.findall("\[unused0\]thought(.*?)\[unused1\]", str_llm_input.replace("\n", ""))
        str_thought = clean_thought_pattern(str_thought)

    str_llm_input = str_llm_input.replace("[unused0] api", "[unused0]api")
    if "[unused0]api" in str_llm_input:
        str_api = re.findall("\[unused0\]api(.*?)\[unused1\]", str_llm_input.replace(" ", "").replace("\n", ""))
        str_api = clean_api_pattern(str_api)

    str_llm_input = str_llm_input.replace("[unused0] observation", "[unused0]observation")
    if "[unused0]observation" in str_llm_input:
        str_observation = re.findall("\[unused0\]observation(.*?)\[unused1\]", str_llm_input.replace("\n", ""))
        str_observation = clean_observation_pattern(str_observation)

    return str_thought, str_api, str_observation


def pre_process_input(request_text):
    input_ = request_text.replace(" ", "")
    tokens = re.findall(r'\[((?:.|\n)*?)\]', input_)

    for x in tokens:
        x = "[" + x + "]"
        input_ = input_.replace(x, " ")
    input_ = input_.strip()
    slots = input_.split(" ")
    return tokens, slots


def format_label(str_label):
    str_label = str_label.replace("[CLASSIFICATION]", "").replace("[CLS]", "")
    tokens, slots = pre_process_input(str_label)

    skill_domain_dict = dict()
    for index in range(min(len(tokens), len(slots))):
        skill_domain_dict[tokens[index]] = slots[index]

    return skill_domain_dict


def append_each_turn(sess, query, ans):
    if ans.strip().lower() not in ['', 'nan']:
        sess.insert(0, {
            "input_text": query, "output_text": ans,
            "session_type": 2,
            "safe_detect_info": {"query_status": 1, "response_status": 1},
            "type": "manu_append_from_llm"
        })
    return sess


def convert_input2session(model_input):
    model_sess = list()
    for ec in model_input.split('[unused0]user')[1:]:
        if "[unused0]assistant" not in ec:
            continue
        q = ec.split('[unused1]')[0].strip()
        a = ec.split('[unused0]assistant')[-1].split('[unused1]')[0].strip()
        model_sess = append_each_turn(model_sess, q, a)
    return model_sess


def format_session_context_list(log_sess, input_1b, input_13b):
    try:
        if '{' in log_sess:
            log_sess = json.loads(log_sess)
        else:
            log_sess = []
    except:
        # 异常log sess
        log_sess = []

    model_1b_sess = convert_input2session(input_1b)
    model_13b_sess = convert_input2session(input_13b)

    max_len = max(len(log_sess), len(model_1b_sess), len(model_13b_sess))
    if len(log_sess) == max_len:
        session_context_list = log_sess
    elif len(log_sess) == model_1b_sess:
        session_context_list = model_1b_sess
    else:
        session_context_list = model_13b_sess

    return json.dumps(session_context_list, ensure_ascii=False)

def get_api_thought_observation(str_model_1b_output, str_model_13b_input):
    # if "gpt" in str_domain:
        # API Observation Thought
    thought_1b, api_1b, observation_1b = extra_api_thought_observation(str_model_1b_output)
    thought_13b, api_13b, observation_13b = extra_api_thought_observation(str_model_13b_input)
    if "CHARASearch" in str(api_1b) or "MathQA" in str(api_1b) or "TODRequest" in str(api_1b):
        api = api_1b
        thought = thought_1b
        observation = observation_1b
    elif "AUTOSearch" in str(api_1b) or "QASearch" in str(api_1b) or "MEDIASearch" in str(api_1b):
        api = api_13b
        thought = thought_13b
        observation = observation_13b
    else:
        api = api_13b
        thought = thought_13b
        observation = observation_13b
    # else:
    #     api, observation, thought = [], [], []

    return api, observation, thought


def log2csv(df:pd.DataFrame,log_col='model_13b_input') -> pd.DataFrame:
    all_list = []
    for id, row in df.iterrows():
        pro_input = row[log_col]
        if pro_input == "" or pro_input == "nan" or pro_input == "NaN":
            pro_input = ""
        row['user-query'], num = get_query_result_from_16b_input(pro_input)
        row['user-query'] = re.sub("\[unused.*?\]", "\n", row['user-query'])
        api, observation, thought = get_api_thought_observation(pro_input, pro_input)
        row['api'] = api if api != [] else "[]"
        row['thought'] = thought[-1]["thought"] if thought != [] else ""
        row['thought'] = row['thought'].replace("该问题请使用总分类模板回复。", "")
        row['observation'] = [list(_.values())[0] for _ in observation] if observation != [] else "[]"
        if row['api'] != "[]" and row['observation'] != "[]" and row['thought'] == "":
            thought = "查询{user-query}相关信息。"
        row['raw_system'] = get_system_result_from_16b_input(pro_input).replace("[unused1]", "")
        context_list = get_context_result_from_16b_input(pro_input)
        row['context'] = str(context_list)

        system_time_prompt = get_system_data(pro_input)
        if len(system_time_prompt) > 0:
            row['system'] = \
                "你是一个名字叫做理想同学的AI数字生命体。\n理想同学是一个可靠的智能家庭助手，由理想汽车智能空间部门创造。理想同学使用了理想公司自研MindGPT大语言模型技术。\n理想同学能够理解人类的指令和意图，并且给出合理的、切合问题的、没有歧视、中立的、安全的回复。\n" \
                + system_time_prompt + "请根据以下文本写一个合适的回复。"
        else:
            row['system'] = row["raw_system"].replace("要求回复为markdown格式。", "")
        all_list.append(row)

    df = pd.DataFrame(all_list)
    return df