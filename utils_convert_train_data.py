import json
import copy
from tqdm import tqdm
import random
import pandas as pd
from typing import List, Tuple, Union, Dict, Any
import re
import os
from utils import create_directory

# 加载数据
def load_data(folder_path:str)->pd.DataFrame:
    full_path=[]
    for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file != "sft_100w_20230614-without-conflict_single.jsonl" and file != "sft_100w_20230614-without-conflict_multi.jsonl":
                    part_path = os.path.join(root, file)
                    full_path.append(part_path)

    dl=[]
    for i in full_path:
        tmp  = pd.read_json(i,lines=True)
        tmp["owner"] = i.split("/")[7]
        tmp["folder"] = i.split("/")[8]
        tmp["dataset"] = i.split("/")[-1]
        dl.append(tmp)

    df = pd.concat(dl)
    print('数据总行数：',len(df))
    return df


def check_illegal_data(df):
    #检索不合法数据
    roles = ['user', 'thought', 'api', 'observation', 'assistant']

    keywords = ["moss", "chatbot", "chatgpt", "openai", "belle", "chatglm", "llama", "alpaca", "小智", "文心一言","ChatGLM2-6B",
            "gpt-4","gpt-3","chatgpt-3.5","gpt-3.5","AI助手","文本生成模型","文本模型","AI机器人","文本AI模型","人工智能语言模型",
            "ai公司", "复旦大学自然语言实验室", "上海人工智能实验室", "上海人工智能实验室", "复旦大学","2023年2月7日", "2021年9月","160亿",
            "8张A100","知识截止日期","截止到2021年","知识更新到2021年","知识更新（2021年）","截至2021年","截至到2021年","2021年","2022年1月","Mikhail","薇塔"]
    pattern1 = re.compile("|".join(keywords),re.IGNORECASE)
    pattern2 = re.compile(
                    r"(我[，是叫]|我的名字[是叫]|我(?:只)?是一[台种个位名款](?:中立的)?|作为|作为一[台种个位名款](?:中立的)?)"
                    r"\s*(?:\s*大?语言模型|大型语言模型|(?:AI)?人工智能(?:语言)?(?:模型)?|(?:智能)?\s*(?:AI)?\s*语言(?:处理)?模型|(?:智能)?\s*(?:AI)?\s*语言技术|"
                    r"(?:理想汽车)?(?:智能)?\s*(?:AI)?(?:智能)?\s*(?:算法)?(?:语音)?助手|AI\s*(?:模型)?|(?:AI)?\s*语料库|"
                    r"智能机器人|聊天机器人|智能助手程序|虚拟助手|模型|知识渊博的人工智能助手|自然语言处理模型|基于人工智能技术的语言模型|计算机程序)",
                    re.IGNORECASE)
    pattern3 = re.compile(r'(我作为老师，|我作为一[一-龥]+老师，|我作为教师，|我作为一[一-龥]+教师，|作为老师|作为一[一-龥]+老师，|作为教师|作为一[一-龥]+教师)', re.IGNORECASE)

    source_ls = ['通用问答-', '汽车问答-', '出游灵感-', '不能删的人设', '无API回复-']

    def detect_unlegal_data(x):
        text= x["messages"]
        if not pd.isnull(x["source"]):
            src = x["source"]
        else:
            src = "Unknown"
        owner = x["owner"]
        
        #检测数据角色数量，是不是都是5的倍数
        if not isinstance(text, list):
            return "###messages 格式不正确！###"
        
        #检测数据角色数量，是不是都是5的倍数
        if len(text)%5 != 0:
            return "###数据role 数量不正确！###"

        #检测每一组对话的5个角色是user->thought->api->observation->assistant
        sorted_roles = [d['role'] for d in text]
        for i in range(0, len(sorted_roles), 5):
            if sorted_roles[i:i+5] != roles:
                # if sorted_roles[i:i+5] != roles2:
                return "###数据role 顺序不正确！###"
            
        #检测每个role的content格式是否合法
        for i in text:
            if not isinstance(i["content"], list):
                return "###数据role的content 格式不是list！###"

            for j in i["content"]:
                if not isinstance(j, str) and not isinstance(j, dict) :
                    return "###数据role的content里的内容 格式不是list！###" 

        #检测assitant的内容是否有脏数据
        for i in text:
            if "assistant" == i["role"]:
                if len(i["content"])<1:
                    return "###数据assistant的content里的内回复为 空 数据！###"
                for j in i["content"]:
                    if not j:
                        return "###数据assistant的content里的内为 空字符串 数据！###"
                    if len(j) <100 and owner == "lisunzhu_general_sft":
                        return "###数据assistant的content里的内为 短数据！###"
                    #检测不合法人设
                    if pattern1.search(str(j)) or pattern2.search(str(j)) or pattern3.search(str(j)):
                        # print(src)
                        if not any(source in src for source in source_ls):
                            return "###不合法人设!###"
                        
                    # 检查 <|br|> <|irrelevant|>组合是否正确
                    if "<|br|>" in j or "<|irrelevant|>" in j:
                        return '##回复里有特殊token##'
        
        
        return "合格"
    
    df["vertify_messages"] = df.apply(detect_unlegal_data,axis=1)
    return df

    
def get_legal_data(input_file: str, output_file: str = None):
    df = pd.read_json(input_file,lines=True)
    df["vertify_messages"] = df.apply(detect_illegal_data,axis=1)
    illegal_num = len(df[df["vertify_messages"]!="合格"])
    if illegal_num > 0:
        print('不合格数据量：',illegal_num)
        df = df[df["vertify_messages"]=="合格"]
        if output_file:
            df.to_json(output_file, orient='records', lines=True)
    else:
        print('全部合格')
    
    return df[df["vertify_messages"]!="合格"]


#convert_to_chatml_data
def convert_to_chatml_single_data(x):
    signs = {"<|lc_start|>":"[unused0]","<|lc_end|>":"[unused1]","<|kvs|>":"[unused2]","<|kve|>":"[unused3]","<|api_start|>":"[unused4]","<|api_end|>":"[unused5]","<|eoa|>":"[unused6]","=>":"[unused7]","<|br|>":"[unused8]","<|irrelevant|>":"[unused9]"}
    l = x["messages"]
    text=[]
    for i in l:
        if i["role"] in ["user","thought","api","assistant","observation"]:
            #print(i["role"])
            if len(i["content"])==0: #没内容的角色
                item = signs["<|lc_start|>"]+i["role"].rstrip()+"\n<None>"+signs["<|lc_end|>"]
                text.append(item)
            else: #有内容的角色
                item_list=[]
                for j in i["content"]:
                        if isinstance(j,dict):
                            if "token" not in j:
                                print(x)
                            item_list.append(signs[j["token"]])
                        else:
                            item_list.append(str(j))
                if i["role"]=="api":
                    item = signs["<|lc_start|>"]+i["role"].rstrip()+"\n"+"".join(item_list)+signs["<|eoa|>"]+signs["<|lc_end|>"]
                else:
                    item = signs["<|lc_start|>"]+i["role"].rstrip()+"\n"+"".join(item_list)+signs["<|lc_end|>"]
                text.append(item)
    # print(text)
    if len(text)%5 !=0 :
        return "###worng data"
    res = [text[i:i+5] for i in range(0, len(text), 5)]
    return res


def convert_to_chatml_data(df: pd.DataFrame)->pd.DataFrame:
    df["chatML_data"] = df.apply(convert_to_chatml_single_data,axis=1)
    df2 = df[df["chatML_data"]!="###worng data"]
    text_list= df2["chatML_data"].to_list()

    #convert to training data
    l=[]
    for res in text_list:
        # print(res)
        try:
            if "<None>"  not in res[-1][-1]:
                # print(res)
                context_list = []
                for j in res[:-1]:
                    # print(j)
                    context_list.append(j[0])
                    context_list.append(j[4])
                for i in res[-1]:
                    context_list.append(i)
                query = "\n".join(context_list[:-1])+"\n"
                output = context_list[-1]
                l.append([query, output])
        except IndexError as err:
            print("IndexError err")
            print(res)

    df_all = pd.DataFrame(l)
    df_all= df_all.dropna()
    # df_all = df_all.sample(frac=1) # shuffle数据
    return df_all


def clean_data(df_all):
    def add_user(x):
        input_text =x
        return input_text

    def norm_output(x):
        norm_output  = x
        norm_output = norm_output.replace("您", "你") # 人设要求不能说您

        return norm_output

    df_all["norm_input"] =  df_all[0].apply(lambda x: add_user(x))
    df_all["norm_output"] =  df_all[1].apply(lambda x: norm_output(x))
    
    return df_all


def gen_trian_data(df_all :pd.DataFrame, output_folder: str):
    create_dir(output_folder)
    
    df_all = clean_data(df_all)
    # 假设数据框是df，将其分成8个块，尽量多分点，多少块就对应多少个线程
    chunk_size = len(df_all) // 8
    chunks = [df_all[i:i+chunk_size] for i in range(0, len(df_all), chunk_size)]

    # 将每个块保存为json文件
    for j in range(len(chunks)):
        train = chunks[j][["norm_input","norm_output"]].values.tolist()
        l=[]
        for i in train:
            data = {"instruction": i[0], "input":"","output":i[1] }
            l.append(data)
        file_out=output_folder+f"{j}.json"
        with open(file_out, 'w', encoding='utf-8') as out:
            json.dump(l, out, indent=4, ensure_ascii=False)
    print('finished!')

    
if __name__ == '__main__':
    # 校验数据合法性
    folder_path = "/mnt/pfs-ssai-nlu/renhuimin/pro_qa/data/sft_data/v20231109/renhuimin_assistant_sft/"
    df = load_data(folder_path)
    df = check_illegal_data(df)
    print(df[df['vertify_messages'] == '不合格'])
    
    # 生成训练数据
    illegal_df = get_legal_data(input_file=folder_path)
    df_all = convert_to_chatml_data(illegal_df)
    
    # 切分训练数据
    output_folder = '/mnt/pfs-ssai-nlu/renhuimin/pro_qa/data/sft_data/v20231109/train_data/'
    gen_trian_data(df_all,output_folder)