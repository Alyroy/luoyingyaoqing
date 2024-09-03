import os
import re
import sys
import json
import math
import pandas as pd

def get_full_path(input_folder):
    '''
    获取所有输入文件
    '''
    full_path=[]
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            #print(os.path.join(root, file))
            if file != "sft_100w_20230614-without-conflict_single.jsonl" and file != "sft_100w_20230614-without-conflict_multi.jsonl":
                part_path = os.path.join(root, file)
                if ".ipynb_checkpoints" not in  part_path and ".DS_Store" not in part_path and ".swp" not in part_path:
                    full_path.append(part_path)

    return full_path
                        
                        
                        

def load_data(input_folder):
    full_path=[]
    for root, dirs, files in os.walk(input_folder):
            for file in files:
                #print(os.path.join(root, file))
                if file != "sft_100w_20230614-without-conflict_single.jsonl" and file != "sft_100w_20230614-without-conflict_multi.jsonl":
                # if file != "cross_domain_unrelated_20230901_multi_turns.jsonl":
                    part_path = os.path.join(root, file)
                    if ".ipynb_checkpoints" not in  part_path and ".DS_Store" not in part_path and ".swp" not in part_path:
                        full_path.append(part_path)
    route_len = len(input_folder.split("/"))
    dl=[]
    #print(full_path)
    for i in full_path:
        print(i)
        tmp = pd.read_json(i,lines=True)
        tmp["owner"] = i.split("/")[route_len]
        tmp["folder"] = i.split("/")[route_len]
        tmp["dataset"] = i.split("/")[-1]
        tmp["producer"] = i.split("/")[route_len].split("_")[0]
        dl.append(tmp)

    df =pd.concat(dl)
    print('读入原始数据总数量',len(df))
    return df

def count_chinese_english_ratio(text):
    text = str(text)
    #匹配中英文
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
    english_pattern = re.compile(r'[A-Za-z]+')
    chinese_characters = re.findall(chinese_pattern, text)
    english_characters = re.findall(english_pattern, text)

    # 计算总数量
    chinese_count = sum(len(char) for char in chinese_characters)
    english_count = sum(len(char) for char in english_characters)

    # 计算总字符数量，以防止除以零
    total_count = chinese_count + english_count
    if total_count == 0:
        return 0, 0
 
    # 计算比例
    chinese_ratio = chinese_count / total_count
    english_ratio = english_count / total_count

    return english_ratio, chinese_count

def check_illegal_data(df):
    #检索不合法数据
    roles = ['user', 'thought', 'api', 'observation', 'assistant']
    # roles2 = ['role', 'thought', 'api', 'observation', 'assistant']
    #（无法/不能）（实时）（访问/浏览）互联网、（无法/不能）（访问/浏览）互联网、（无法/不能）（实时）联网
    keywords = ["gpt4", "gpt3", "moss", "chatbot", "chatgpt","chatGPT", "openai","opneAI", "belle", "chatglm", "llama", "alpaca", "小智", "文心一言","ChatGLM2-6B",
                "gpt-4","gpt-3","chatgpt-3.5","gpt-3.5","AI助手","文本生成模型","文本模型","AI机器人","文本AI模型","人工智能语言模型",
                "ai公司", "复旦大学自然语言实验室", "上海人工智能实验室", "上海人工智能实验室", "复旦大学",
                "2023年2月7日", "2021年9月","160亿","8张A100","知识截止日期","截止到2021年","知识更新到2021年","知识更新（2021年）",
                "截至2021年","截至到2021年","2021年","2022年","2022年1月","Mikhail","薇塔","虚拟助手","到2022年","至2022年","知识更新（2022年）","主人",
                "人工智能助手","计算机程序","语言模型AI","作为一名人工智能语言模型",
                "作为一个人工智能助手","作为一个计算机程序","作为一个语言模型AI","作为一个语言模型","作为一个AI助手","作为一个AI助理","作为一个AI语言模型","作为一位人工智能助手",
                "作为一个文本生成模型","作为文本模型","作为一个文本AI模型","作为一名人工智能语言模型","作为一个虚拟助手","作为一个智能机器人",
                "我是一个虚拟助手","我是一个基于文本的虚拟助手","我是OpenAI的虚拟助手","我主要是一个文本生成模型","我是一个文本生成模型",
                "我是OpenAI的GPT-4模型","我是一个人工智能语言模型","我是一个计算机程序","我是一个程序","我是一个计算机程序","作为一个机器模型","身为一个程序","作为一个机器模型",
                "我是一个虚拟助手","我只是一个文本生成模型","我是一个被训练的文本生成模型","我是OpenAI的GPT-4模型","我是基于GPT-4架构的大型文本生成模型",
                "由于我目前仅是一个文本生成模型","我是一个文本模型","我是一个纯文本模型","我是一个文本AI模型","是一个智能机器人","2021 年 9 月","无法实时访问互联网",
                "无法实时浏览互联网","不能实时访问互联网","不能实时浏览互联网","无法访问互联网","不能访问互联网","无法浏览互联网","不能浏览互联网","不能实时联网","无法实时联网","无法实时","不能实时",
                "作为一个机器模型","我是一个人工智能语言模型","我是一个计算机模型","作为一个人工智能语言模型","我是一个人工智能","我是一个大语言模型","我是一名语言模型","作为一个数据模型","我是一个智能语言模型","<\|wrong data\|>"
                ]
    # pattern1 = re.compile(r"(" + "|".join(keywords) + r")",re.IGNORECASE)
    #pattern1 = re.compile("|".join(keywords),re.DOTALL)
    pattern1 = re.compile("|".join(keywords),re.IGNORECASE)
    pattern2 = re.compile(
                    r"(我[，是叫]|我的名字[是叫]|我(?:只)?是一[台种个位名款](?:中立的)?|作为|作为一[台种个位名款](?:中立的)?)"
                    r"\s*(?:\s*大?语言模型|大型语言模型|(?:AI)?人工智能(?:语言)?(?:模型)?|(?:智能)?\s*(?:AI)?\s*语言(?:处理)?模型|(?:智能)?\s*(?:AI)?\s*语言技术|"
                    r"(?:理想汽车)?(?:智能)?\s*(?:AI)?(?:智能)?\s*(?:算法)?(?:语音)?助手|AI\s*(?:模型)?|(?:AI)?\s*语料库|"
                    r"智能机器人|聊天机器人|智能助手程序|虚拟助手|模型|知识渊博的人工智能助手|自然语言处理模型|基于人工智能技术的语言模型|计算机程序)",
                    re.DOTALL)
    pattern3 = re.compile(r'(我作为老师，|我作为一[一-龥]+老师，|我作为教师，|我作为一[一-龥]+教师，|作为老师|作为一[一-龥]+老师，|作为教师|作为一[一-龥]+教师)', re.DOTALL)
    patterns4 = re.compile(r"没有.{0,10}(情感|情绪|感情|感觉)", re.DOTALL)
    patterns5 = re.compile(r"^(对不起|不好意思)", re.DOTALL)
    patterns6 = re.compile(r".*(陪伴).{0,8}$", re.DOTALL)
    patterns7 = re.compile(r'(截止.*\d{4}年|到.*\d{4}年.*为止|截至.*\d{4}年)' , re.DOTALL)
    patterns8 = re.compile(r'((不能|无法).*(访问|浏览)互联网|(不能|无法).*(联网|更新))' , re.DOTALL)
    patterns9 = re.compile(r"^(对不起|抱歉|很抱歉)", re.DOTALL) # 只针对部分source=["lmsys-1m", "shareGPT"]
    patterns10 = re.compile(r"(lixiang|LW43|泊船瓜洲.*王之涣|王之涣.*泊船瓜洲|崔颢.*故人西辞黄鹤楼|故人西辞黄鹤楼.*崔颢)", re.DOTALL)
    patterns11 = re.compile(r"(lixiang|LW43)", re.DOTALL)
    patterns12 = re.compile(r"(chatgpt|openai|gpt4|gpt3|gpt-3|gpt-4)", re.IGNORECASE)
    
    source_ls = ['通用问答-', '汽车问答-', '出游灵感-', '不能删的人设', '无API回复-',"人设"]
    stop_sign=['.', '!', '？', '。', '?', '！',';']
    wrong_sign = [',','，',':','：']
    def detect_unlegal_data(x):
            text= x["messages"]
            src = "Unknown"
            if not pd.isnull(x["source"]):
                src = x["source"]
            else:
                src = "Unknown"
            owner = x["owner"]
            dataset = x["dataset"]
            reason = ""
            #检测数据角色数量，是不是都是5的倍数
            if not isinstance(text, list):
                return "###messages 格式不正确！###", reason
            #检测数据角色数量，是不是都是5的倍数
            if len(text)%5 != 0:
                return "###数据role 数量不正确！###", reason
            #检测每一组对话的5个角色是user->thought->api->observation->assistant
            sorted_roles = [d['role'] for d in text]
            for i in range(0, len(sorted_roles), 5):
                if sorted_roles[i:i+5] != roles:
                    # if sorted_roles[i:i+5] != roles2:
                    return "###数据role 顺序不正确！###", reason
            #检测每个role的content格式是否合法
            for i in text:
                if not isinstance(i["content"], list):
                    return "###数据role的content 格式不是list！###", reason
                for j in i["content"]:
                    if not isinstance(j, str) and not isinstance(j, dict) :
                        return "###数据role的content里的内容 格式不是list！###", reason
            last_assistant = text[-1]["content"][0]
            if last_assistant != "": # 增加对翻译脏数据的过滤
                english_ratio, chinese_count = count_chinese_english_ratio(last_assistant)
                if english_ratio > 0.9 and chinese_count > 0 and chinese_count <= 8:
                    return "###assistant不符合翻译规则!###", "回复中存在过多英文"
            #检测assitant的内容是否有脏数据
            for i in text:
                if "assistant" == i["role"]:
                    if len(i["content"])<1:
                        return "###数据assistant的content里的内回复为 空 数据！###", reason
                    for j in i["content"]:
                        if not j:
                            return "###数据assistant的content里的内为 空字符串 数据！###", reason
                        if len(j) <50 and owner == "lisunzhu_general_sft":
                            return "###数据assistant的content里的内为 短数据！###", reason
                        #检测不合法人设
                        str_j = str(j)
                        combined_pattern = re.compile("|".join([
                            pattern1.pattern,
                            pattern2.pattern,
                            pattern3.pattern,
                            patterns4.pattern,
                            patterns5.pattern,
                            patterns6.pattern,
                            patterns7.pattern,
                            patterns8.pattern
                        ]), re.IGNORECASE)
                        if combined_pattern.search(str_j):
                            if  "huimin" not in owner and "zhuyun" not in owner and "zhengxin" not in owner and "renshe" not in dataset and "yangliuyi" not in owner and "jiucuo" not in dataset and "zhangshuwen" not in owner and "guobao" not in owner and "songming" not in owner:
                                reason = combined_pattern.search(str(str_j)).group()
                                return "###assistant人设不合法!###", reason
                            
                        if patterns9.search(str_j) and src in ["lmsys-1m", "shareGPT", "MED", "min_length", "law", "metal", "renshe", "complex_inst", "wizardLM", "multi_turn", "CHINESE", "helpsteer", "sschatgpt"]:
                            reason = patterns9.search(str_j).group()
                            return "###道歉冲突数据!###", reason
                        if patterns10.search(str_j):
                            reason = patterns10.search(str_j).group()
                            return "###assistant不符公司安全及诗词脏数据!###", reason
                reason = ""
                # 对user进行过滤
                if "user" == i["role"]:
                    for j in i["content"]:
                        str_j = str(j)
                        if patterns11.search(str_j):
                            return "###user不符公司安全及诗词脏数据!###", reason
                        if patterns12.search(str_j) and "huimin" not in owner and "zhuyun" not in owner and "zhengxin" not in owner and "renshe" not in dataset and "yangliuyi" not in owner and "zhangshuwen" not in owner and "guobao" not in owner and "songming" not in owner:
                            reason = patterns12.search(str(str_j)).group()
                            return "###user人设不合法!###", reason
                                   
            return "合格", reason
    
    df[["vertify_messages", "reason"]] = df.apply(detect_unlegal_data, axis=1, result_type='expand')
    return df

def convert_to_chatml_single_data(x):
    signs = {"<|lc_start|>":"[unused0]","<|lc_end|>":"[unused1]","<|kvs|>":"[unused2]","<|kve|>":"[unused3]","<|api_start|>":"[unused4]","<|api_end|>":"[unused5]","<|eoa|>":"[unused6]","=>":"[unused7]", "<|br|>":"[unused8]", "<|irrelevant|>":"[unused9]"}
    l = x["messages"]
    if type(l) == str:
        l = eval(l)
    text=[]
    ml = len(l)
    #print("ml:{}".format(ml))
    for index in range(ml):
        i = l[index]
        if i["role"] in ["user","thought","api","assistant","observation","observation "]:
            if len(i["content"])==0: #没内容的角色
                if i["role"]=="observation" and index==ml-2:
                    # 将[unused0]assistant接到输入之后
                    item = signs["<|lc_start|>"]+i["role"].rstrip()+"\n<None>"+signs["<|lc_end|>"]+"\n[unused0]assistant"
                else:
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
                elif i["role"]=="observation" and index==ml-2:
                    # 将[unused0]assistant接到输入之后
                    item = signs["<|lc_start|>"]+i["role"].rstrip()+"\n"+"".join(item_list)+signs["<|lc_end|>"]+"\n[unused0]assistant"
                elif i["role"]=="assistant" and index==ml-1:
                    item = "".join(item_list).replace("[unused0]assistant\n", "").replace("[unused1]", "")+signs["<|lc_end|>"]
                elif i["role"]=="assistant" and index!=ml-1:
                    item = signs["<|lc_start|>"]+i["role"].rstrip()+"\n"+"".join(item_list)+signs["<|lc_end|>"]
                else:
                    item = signs["<|lc_start|>"]+i["role"].rstrip()+"\n"+"".join(item_list)+signs["<|lc_end|>"]
                text.append(item)
    # print(text)
    if len(text)%5 !=0 :
        return "###worng data"
    res = [text[i:i+5] for i in range(0, len(text), 5)]
    return res

def convert_to_chatml_data(df):
    df["chatML_data"] = df.apply(convert_to_chatml_single_data,axis=1)
    df2 = df[df["chatML_data"]!="###worng data"]
    text_list= df2["chatML_data"].to_list()
    if 'system' not in df2.columns:
        df2["system"] = ""
    system_list = df2["system"].to_list()

    #convert to training data
    l=[]
    data_len = len(text_list)
    for i in range(data_len):
        res = text_list[i]
        system_prompt = system_list[i]
        if str(system_prompt) == "nan":
            system_prompt = ""
            #print(f"system_prompt: {system_prompt}")
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
                l.append([system_prompt, query, output])
        except IndexError as err:
            print("IndexError err")
            print(res)

    df_all = pd.DataFrame(l)
    df_all = df_all.fillna("")
    df_all= df_all.dropna()
    df_all = df_all.sample(frac=1)
    return df_all


def clean_data(df_all):
    # pattern1 = re.compile(r'(ai\s*语言模型)', re.IGNORECASE)
    # pattern2 = re.compile(r'(ai\s*(?:公司)+)', re.IGNORECASE)
    # pattern3 = re.compile(r'(ai\s*大?语言模型|人工智能\s*大?语言模型|智能机器人|大型语言模型|智能助手程序|智能助手|聊天机器人|AI\s*助手|语言模型)', re.IGNORECASE)

    # pattern4 = re.compile(
    #     r"(我[，是叫]|我的名字[是叫]|我(?:只)?是一[台种个位名款](?:中立的)?|作为|作为一[台种个位名款](?:中立的)?)"
    #     r"\s*(?:\s*大?语言模型|大型语言模型|(?:AI)?人工智能(?:语言)?(?:模型)?|(?:智能)?\s*(?:AI)?\s*语言(?:处理)?模型|(?:智能)?\s*(?:AI)?\s*语言技术|"
    #     r"(?:理想汽车)?(?:智能)?\s*(?:AI)?(?:智能)?\s*(?:算法)?(?:语音)?助手|AI\s*(?:模型)?|(?:AI)?\s*语料库|"
    #     r"智能机器人|聊天机器人|智能助手程序|虚拟助手|模型)",
    #     re.IGNORECASE,
    # )

    # pattern5 = re.compile('对不起，|十分抱歉，|非常抱歉，|很抱歉，', re.IGNORECASE) # 人设不允许道歉
    
    def add_system(x):
        system_text =x
        return system_text
    
    def add_user(x):
        input_text =x
        return input_text

    def norm_output(x):
        norm_output  = x
        # norm_output = re.sub(pattern1, "语音助手", str(x))
        # norm_output = re.sub(pattern2, "理想汽车", norm_output)
        # norm_output = re.sub(pattern3, "理想汽车AI语音助手", norm_output)
        norm_output = norm_output.replace("您", "你") # 人设要求不能说您
        # norm_output = re.sub(pattern5, "", norm_output)
        # norm_output = re.sub(pattern4, "", norm_output)

        return norm_output
    
    df_all["system"] = df_all[0].apply(lambda x: add_system(x))
    df_all["norm_input"] =  df_all[1].apply(lambda x: add_user(x))
    df_all["norm_output"] =  df_all[2].apply(lambda x: norm_output(x))
    
    return df_all


def gen_train_data(df_all,output_folder, filename='', chunk=1):
    # 假设数据框是df，将其分成8个块，尽量多分点，多少块就对应多少个线程
    chunk_size = len(df_all) // chunk
    chunks = [df_all[i:i+chunk_size] for i in range(0, len(df_all), chunk_size)]
    system_default = "你是一个名字叫做理想同学的AI数字生命体。\n理想同学是一个可靠的智能家庭助手，由理想汽车智能空间部门创造。\n理想同学能够理解人类的指令和意图，并且给出合理的、切合问题的、没有歧视、中立的、安全的回复。\n\n请根据以下文本写一个合适的回复。"
    # 将每个块保存为json文件
    for j in range(len(chunks)):
        train = chunks[j][["system", "norm_input","norm_output"]].values.tolist()
        l=[]
        for i in train:
            system = i[0]
            instruction = ""
            if system == "":
                instruction = f"[unused0]system\n{system_default}[unused1]\n{i[1]}"
            else:
                instruction = f"[unused0]system\n{i[0]}[unused1]\n{i[1]}"
            data = {"instruction": instruction, "input":"","output":i[2] }
            l.append(data)
            
        file_out = f"{output_folder}/{filename}{j}.json"
        with open(file_out, 'w', encoding='utf-8', errors='ignore') as out:
            json.dump(l, out, indent=4, ensure_ascii=False)
    print('finished!')
    
    
def main(input_folder,output_folder):
    df = load_data(input_folder)
    print('原始加载数据量：',len(df))
    df = check_illegal_data(df)
    df_no = df[df["vertify_messages"]!="合格"]
    df_no.to_json("/mnt/pfs-ssai-nlu/pretrain/chatGPT/data/diff_data/vertify_data_no.json", orient='records', lines=True, force_ascii=False)
    print('不合格数据量：',len(df_no))
    print(f"不合格的数据：\n{df_no['producer'].unique().tolist()}")
    vertify_data_no_dir = "/mnt/pfs-ssai-nlu/pretrain/chatGPT/data/diff_data/vertify_data_no/" + output_folder.split("/")[-1]
    if not os.path.exists(vertify_data_no_dir):
        os.mkdir(vertify_data_no_dir)
    for producer in df_no['producer'].unique().tolist():
        tmp_df = df_no[df_no['producer'].isin([producer])]
        file_path = f"{vertify_data_no_dir}/vertify_data_no-{producer}.jsonl"
        print(file_path, f"{tmp_df.shape[0]}条")
        tmp_df.to_json(file_path, orient='records', lines=True, force_ascii=False)
    df = df[df["vertify_messages"]=="合格"]
    print('合格数据量：',len(df))
    #df.to_json("/mnt/pfs-ssai-nlu/pretrain/chatGPT/data/diff_data/vertify_data_ok.json", orient='records', lines=True, force_ascii=False)
    vertify_data_ok_dir = "/mnt/pfs-ssai-nlu/pretrain/chatGPT/data/diff_data/vertify_data_ok/" + output_folder.split("/")[-1]
    if not os.path.exists(vertify_data_ok_dir):
        os.mkdir(vertify_data_ok_dir)
    for producer in df['producer'].unique().tolist():
        tmp_df = df[df['producer'].isin([producer])]
        file_path = f"{vertify_data_ok_dir}/vertify_data_ok-{producer}.jsonl"
        print(file_path, f"{tmp_df.shape[0]}条")
        tmp_df.to_json(file_path, orient='records', lines=True, force_ascii=False)
    df_all = convert_to_chatml_data(df)
    df_all = clean_data(df_all)
    print('训练数据量：',len(df_all))
    gen_trian_data(df_all,output_folder)



def remove_keyword_from_values(data, keyword):
    """
    递归地检查JSON数据中的所有值，并去除出现的特定关键词。
    
    :param data: JSON数据，可以是字典或列表。
    :param keyword: 要去除的关键词。
    :return: 清理后的数据。
    """
    if isinstance(data, dict):
        # 字典类型，遍历每个键值对
        return {key: remove_keyword_from_values(value, keyword) for key, value in data.items()}
    elif isinstance(data, list):
        # 列表类型，遍历每个元素
        return [remove_keyword_from_values(element, keyword) for element in data]
    elif isinstance(data, str):
        # 字符串类型，直接处理字符串
        return data.replace(keyword, "\\n")
    else:
        # 其他类型，直接返回原值
        return data

def random_sample_from_json(json_path, sample_size):
    """从指定的JSON文件中随机采样指定数量的数据项"""
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    data = remove_keyword_from_values(data, "[unused8]")
    # 如果数据少于采样大小，则返回全部数据
    if len(data) <= sample_size:
        return data
    else:
        return random.sample(data, sample_size)

def json_split_chunks(all_samples, output_folder = "",k = 32):
    # 计算每个文件的大概条目数
    chunk_size = math.ceil(len(all_samples) / k)
    
    # 将数据切分成k份并写入新的JSON文件中
    for i in range(k):
        output_file_path = os.path.join(output_folder, f'{i}.json')  # 文件名格式化为三位数
        # 使用切片操作切分数据
        start_index = i * chunk_size
        end_index = start_index + chunk_size
        split_data = all_samples[start_index:end_index]
        
        # 将切分的数据写入到新文件中
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            json.dump(split_data, output_file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    input_folder = sys.argv[1]
    output_folder= sys.argv[2]
    main(input_folder,output_folder)