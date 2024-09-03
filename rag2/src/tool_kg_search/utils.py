# -*- coding: utf-8 -*-
import itertools 
from tqdm import tqdm
import re
import json
import random
from typing import List, Dict, Any, Tuple
import requests
# import grequests
import math
import pandas as pd
import os
from os.path import dirname, abspath
import csv
from collections import Counter   #引入Counter
import sys 


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

