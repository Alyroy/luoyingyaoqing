import os
import re
import sys
import json
import pandas as pd


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
        tmp["folder"] = i.split("/")[route_len+1]
        tmp["dataset"] = i.split("/")[-1]
        dl.append(tmp)

    df =pd.concat(dl)
    return df

def check_illegal_data(df):
    #检索不合法数据
    roles = ['user', 'thought', 'api', 'observation', 'assistant']
    # roles2 = ['role', 'thought', 'api', 'observation', 'assistant']
    #（无法/不能）（实时）（访问/浏览）互联网、（无法/不能）（访问/浏览）互联网、（无法/不能）（实时）联网
    keywords = ["moss", "chatbot", "chatgpt","chatGPT", "openai","opneAI", "belle", "chatglm", "llama", "alpaca", "小智", "文心一言","ChatGLM2-6B",
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
    pattern1 = re.compile("|".join(keywords),re.DOTALL)
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
    patterns9 = re.compile(r"^(对不起|不好意思|抱歉|很抱歉)", re.DOTALL) # 只针对部分source=["lmsys-1m", "shareGPT"]
    patterns10 = re.compile(r"(lixiang|LW43|泊船瓜洲.*王之涣|王之涣.*泊船瓜洲|崔颢.*故人西辞黄鹤楼|故人西辞黄鹤楼.*崔颢)", re.DOTALL)
    patterns11 = re.compile(r"(lixiang|LW43)", re.DOTALL)
    
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
                        if len(j) <50 and owner == "lisunzhu_general_sft":
                            return "###数据assistant的content里的内为 短数据！###"
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
                        ]))
                        if combined_pattern.search(str_j):
                            if  "huimin" not in owner and "zhuyun" not in owner and "zhengxin" not in owner and "renshe" not in dataset and "yangliuyi" not in owner and "jiucuo" not in dataset:
                                return "###人设不合法!###" 
                        # if pattern1.search(str_j) or pattern2.search(str_j) or pattern3.search(str_j) or patterns4.search(str_j) or patterns5.search(str_j) or patterns6.search(str_j) or patterns7.search(str_j) or patterns8.search(str_j):
                        #     if  "huimin" not in owner and "zhuyun" not in owner and "zhengxin" not in owner and "renshe" not in dataset and "yangliuyi" not in dataset:
                        #         return "###人设不合法!###" 
                        # if pattern1.search(str(j)) or pattern2.search(str(j)) or pattern3.search(str(j)) or patterns4.search(str(j)) or patterns5.search(str(j)) or patterns6.search(str(j)) or patterns7.search(str(j)) or patterns8.search(str(j)):
                        #     if  "huimin" not in owner and "zhuyun" not in owner and "zhengxin" not in owner and "renshe" not in dataset and "yangliuyi" not in dataset:
                        #         return "###不合法人设!###"
                        if patterns9.search(str_j) and src in ["lmsys-1m", "shareGPT"]:
                            return "###道歉冲突数据!###"
                        if patterns10.search(str_j):
                            return "###assistant不符公司安全及诗词脏数据!###"
                if "user" == i["role"]:
                    for j in i["content"]:
                        str_j = str(j)
                        if patterns11.search(str_j):
                            return "###user不符公司安全及诗词脏数据!###"
                        
            return "合格"
    
    df["vertify_messages"] = df.apply(detect_unlegal_data,axis=1)
    return df

def convert_to_chatml_single_data(x):
    signs = {"<|lc_start|>":"[unused0]","<|lc_end|>":"[unused1]","<|kvs|>":"[unused2]","<|kve|>":"[unused3]","<|api_start|>":"[unused4]","<|api_end|>":"[unused5]","<|eoa|>":"[unused6]","=>":"[unused7]", "<|br|>":"[unused8]", "<|irrelevant|>":"[unused9]"}
    l = x["messages"]
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


def gen_trian_data(df_all,output_folder):
    # 假设数据框是df，将其分成8个块，尽量多分点，多少块就对应多少个线程
    chunk_size = len(df_all) // 32
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
        file_out=output_folder+f"/{j}.json"
        with open(file_out, 'w', encoding='utf-8') as out:
            json.dump(l, out, indent=4, ensure_ascii=False)
    print('finished!')
    
    
def main(input_folder,output_folder):
    df = load_data(input_folder)
    print('原始加载数据量：',len(df))
    df = check_illegal_data(df)
    df_no = df[df["vertify_messages"]!="合格"]
    #df_no.to_json(output_folder + "/vertify_data_no.json", orient='records', lines=True, force_ascii=False)
    df = df[df["vertify_messages"]=="合格"]
    print('合格数据量：',len(df))
    #df.to_json(output_folder + "/vertify_data_ok.json", orient='records', lines=True, force_ascii=False)
    df_all = convert_to_chatml_data(df)
    df_all = clean_data(df_all)
    print('训练数据量：',len(df_all))
    gen_trian_data(df_all,output_folder)
    
    
if __name__ == "__main__":
    input_folder = sys.argv[1]
    output_folder= sys.argv[2]
    main(input_folder,output_folder)
