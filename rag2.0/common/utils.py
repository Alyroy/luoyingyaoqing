import pandas as pd
import numpy as np
import json
import os
from typing import Dict, Any, List, Optional, TypeVar, Generic, Type, Tuple, Union
import re
import uuid

# 新建文件夹
def create_directory(directory: str):
    """Creates a directory if it does not already exist."""
    os.makedirs(directory, exist_ok=True)
        

# 读数据
def get_df(input_f:str):
    '''
    读取输入文件
    '''
    if input_f.endswith(".csv"):
        df = pd.read_csv(input_f)
    elif input_f.endswith(".json"):
        df = pd.read_json(input_f)
    elif input_f.endswith(".jsonl"):
        df = pd.read_json(input_f, lines = True)
    print(f'文件总行数:{len(df)}')
    return df


def preprocess_df(f):
    df = pd.read_csv(f, skipinitialspace=True)
    cols = df.columns.tolist()
    for c in cols:
        if "Unnamed:" in c:
            df.drop(columns=c, inplace=True)
    return df


def read_txt(path: str) -> str:
    with open(path, 'r') as file:
        return file.read()

    
def read_json(path: str) -> Dict:
    """
    从json文件中读数据
    :param path:
    :return:
    """
    with open(path) as fi:
        return json.load(fi)


def read_lines(path: str) -> List[str]:
    with open(path) as fi:
        lines = fi.readlines()
    return [line.rstrip('\n') for line in lines]


def read_excel_sheet(file_name, sheet_name):
    df = pd.read_excel(file_name, sheet_name, keep_default_na=False)
    list_data = np.array(df).tolist()
    objs = []
    field_names = list(df.columns)
    for fields in list_data:
        obj = {}
        for i in range(len(field_names)):
            obj[field_names[i]] = fields[i]
        objs.append(obj)
    return objs


def read_pd_csv(file_name):
    df = pd.read_csv(file_name, keep_default_na=False)
    list_data = df.values.tolist()
    objs = []
    field_names = list(df.columns)
    print('filed_names=%s' % field_names)
    for fields in list_data:
        obj = {}
        for i in range(len(field_names)):
            obj[field_names[i]] = fields[i]
        objs.append(obj)
    return objs
    

# 写数据
def save_json(obj: Any, path: str) -> str:
    """
    写数据到json文件中
    :param obj:
    :param path:
    :return:
    """
    with open(path, 'w') as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2))
    return path


def save_csv(head_list, obj, path):
    data = pd.DataFrame(obj)
    data.to_csv(path, header=head_list, index=None)


def save_lines(obj, path):
    with open(path, 'w') as f:
        f.writelines(obj)
    return path



# 其他常用工具
def is_contain_chinese(strs):
    """
    判断字符串中是否包含中文字符
    """
    p = re.compile("[\u4e00-\u9fa5]")
    res_p = re.findall(p, strs)
    if len(res_p) == 0:
        return False
    else:
        return True

def generate_msg_id() -> str:
    """
    生成唯一的8位纯数字ID
    :return: 生成的ID
    """
    unique_id = str(uuid.uuid4())[:16]
    if len(unique_id) == 16:
        return "test_xy_" + unique_id
    return "test_xy_1234"


def extract_date_from_filename(filename):
    # 使用正则表达式从文件名中匹配日期格式 "YYYY-MM-DD"
    match = re.search(r'\d{4}-\d{2}-\d{2}', filename)
    if match:
        return match.group(0)
    else:
        return None


def is_2d_list(input_list) -> bool:
    """
    判断输入是否是一个二维列表
    Args:
        input_list: 任意类型的输入
    Return:
        bool: 如果输入是二维列表则返回True，否则返回False
    """
    # 检查输入列表
    if not isinstance(input_list, list):
        return False

    # 检查列表中的每个元素是否也是列表
    for element in input_list:
        if not isinstance(element, list):
            return False

    return True


def flatten_and_number(lst):
    lst = eval(lst)
    counter = 1
    result = []
    for sublist in lst:
        for item in sublist:
            result.append(f'obs{counter} {item}')
            counter += 1
    return '\n\n'.join(result)
