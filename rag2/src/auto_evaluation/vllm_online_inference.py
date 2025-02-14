# encoding = utf-8

# import torch
import sys, os, re
import pandas as pd
import time
import argparse
import random
from tqdm import tqdm
import json
import numpy as np
# import torch.multiprocessing as mp
# from vllm import LLM, SamplingParams
# from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, AutoModelForCausalLM, AutoTokenizer
from litiktoken.tiktoken_tokenizer import Tokenizer

sys.path.append("../") 
from tool_rag_generation.data_format import DataFormat
from tool_llm_response.call_llm_with_vllm import CallLLMByPromptVllm, PromptVllmConfig
sys.path.append("../../") 
from common import utils_log,utils

# 定义设置随机种子的函数
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description='information')
    parser.add_argument('--gpus', dest='gpus', default="0,1,2,3,4,5,6,7", type=str, help='model path')
    parser.add_argument('--model', dest='model', default="/mnt/xxxxxx", type=str, help='model path')
    parser.add_argument('--tiktoken_path', dest='tiktoken_path', type=str, default=None, help='litiktoken path')
    parser.add_argument('--input_file', dest='input_file', type=str, help='input file path')
    parser.add_argument('--batch_num', dest='batch_number', type=int, default=100, help='inference batch number')
    parser.add_argument('--try_num', dest='try_num', type=int, default=1, help='repeated nums')
    parser.add_argument('--turn_mode', dest='turn_mode', type=str, default="13b", help='single or multi')
    parser.add_argument('--time_stamp', dest='time_stamp', type=str, help='time stamp')
    parser.add_argument('--output_path', dest='output_path', default="./infer_output/", type=str, help='output_path')
    parser.add_argument('--dosample_flag', dest='dosample_flag', action='store_true', help='do sample true')
    parser.add_argument('--no_dosample_flag', dest='dosample_flag', action='store_false', help='do sample false')
    parser.add_argument('--api_flag', dest='api_flag', action='store_true', help='Enable API flag')
    parser.add_argument('--no_api_flag', dest='api_flag', action='store_false', help='Disable API flag')
    parser.add_argument('--temperature', dest='temperature', type=float, default=0.9, help='temperature')
    parser.add_argument('--top_k', dest='top_k', type=float, default=50, help='temperature')
    parser.add_argument('--top_p', dest='top_p', type=float, default=0.95, help='temperature')
    parser.add_argument('--repetition', dest='repetition', type=float, default=1, help='repetition penalty')
    parser.add_argument('--max_tokens', dest='max_tokens', type=int, default=8000, help='max tokens')
    parser.add_argument('--eval_col', dest='eval_col', type=str, default="xx", help='column to be infered')
    parser.add_argument('--random', dest='random', action='store_false', default=True, help='random')
    parser.add_argument('--seed', dest='seed', type=int, default=9987, help='random seed')
    parser.add_argument('--url', dest='url', type=str, help='online model url')
    parser.add_argument('--model_name', dest='model_name', type=str, default="qwen", help='model name')
    parser.add_argument('--chunk_num', dest='chunk_num', type=int, default=1, help='chunk number')
    parser.add_argument('--thread_num', dest='thread_num', type=int, default=8, help='thread number')
    parser.set_defaults(dosample_flag=True)
    parser.set_defaults(api_flag=True)  # 默认为True
    args = parser.parse_args()
    print(args)
    return args

def get_version_transformers():
    import pkg_resources  
    try:  
        # 尝试获取transformers包的版本  
        version = pkg_resources.get_distribution("transformers").version  
        print(f"The version of transformers is: {version}")  
    except pkg_resources.DistributionNotFound:  
        print("The transformers package is not installed.")
    return version

def format_input_api(input_text):
    #"[unused0]user\n黄晓明是谁？[unused1]\n"
    if input_text.strip().endswith("assistant:"):
        input_text = input_text.rsplit("assistant:", 1)[0].strip() + "[unused1]\n"
    input_text = input_text.replace('\nuser:', "user:").replace("user:", "[unused1]\n[unused0]user\n")
    input_text = input_text.replace('\nassistant:', "assistant:").replace("assistant:", "[unused1]\n[unused0]assistant\n")
    input_text = input_text.lstrip(r"[unused1]").lstrip()
    if "[unused0]user" not in input_text:
        input_text = "[unused0]user\n" + input_text + "[unused1]\n"
    if not input_text.endswith("[unused1]\n"):
        input_text = input_text + "[unused1]\n"
    return input_text

def api_infer_core(infer_config, infer_call, infer_data, idx_key="infer_align_id"):
    pass

def api_infer_main_funtion(args):
    directory = os.path.join(args.output_path, args.time_stamp)
    if not os.path.exists(directory):
        os.makedirs(directory)

    input_file = args.input_file
    print("input_f:", input_file) 
    if os.path.isdir(input_file):
        file_list = [os.path.join(input_file, x) for x in os.listdir(input_file) if x.endswith(".csv")]
    elif os.path.isfile(input_file):
        file_list = [input_file]

    temp_num = args.batch_number
    idx_key = "infer_align_id"

    process_config = PromptVllmConfig(
        **{
            "url": args.url, # url
            "chunk_num": args.chunk_num,
            "thread_num": args.thread_num,
            "query_column_name": args.eval_col, # llm模型输入列名
            "response_column_name": "response"
        }
    )

    generate_config = PromptVllmConfig(
        **{
            "model": args.model_name,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition,
            "max_tokens": args.max_tokens, # 最大输出长度
            "stop": ['<|endoftext|>'],
            "skip_special_tokens": True,
        }
    )

    call_vllm = CallLLMByPromptVllm(process_config, generate_config)
    
    for one_file in file_list:
        print("inference_file:{}".format(one_file))

        test_dataset_file = one_file.strip('/').split('/')[-1].strip()
        result_file_prefix = test_dataset_file + "." + args.time_stamp
        output_pt = directory + "/" + result_file_prefix
        save_path = output_pt + ".csv"
        temp_path = output_pt + "-temp_data.jsonl"
        
        df =utils.preprocess_df(one_file)
        if args.eval_col in df.columns:
            df = df[~df[args.eval_col].isna()]
        
        dl = df.to_dict(orient="records")
        total_rows = len(dl)
        print('total rows is:', total_rows)
        
        for idx, item in enumerate(dl, start=1):
            item[idx_key] = idx

        result_ls = []
            # 加载缓存数据
        if os.path.exists(temp_path):
            try:
                result_ls = pd.read_json(temp_path, lines=True).to_dict(orient="records")
                result_ls = sorted(result_ls, key=lambda x: x[idx_key])
            except Exception as e:
                print(f"Failed to read temp data from {temp_path}: {e}")

        chunk_prompt_ls = []
        chunk_data_ls = []
        for data_item in tqdm(dl):
            if result_ls:
                if int(data_item[idx_key]) <= int(result_ls[-1][idx_key]):
                    continue

            prompt = {
                idx_key: data_item[idx_key],
                args.eval_col: f"<|im_start|>{data_item[args.eval_col]}"
            }

            chunk_prompt_ls.append(prompt)
            chunk_data_ls.append(data_item)
            
            if len(chunk_prompt_ls) >= temp_num:
                call_vllm.prompt_request_datalist(chunk_prompt_ls, show_tqdm=False)
                for chunk_data, chunk_prompt in zip(chunk_data_ls, chunk_prompt_ls):
                    if not chunk_data[idx_key] == chunk_prompt[idx_key]:
                        raise ValueError("infer list and input list do not match")
                
                    assistant = chunk_prompt["response"].replace("[unused1]", "").strip()
                    assistant = assistant.replace("[unused8]", "\n").strip()

                    # chunk_data.pop(idx_key, "none")
                    chunk_data["predict_output"] = assistant
                    chunk_data["full_output"] = chunk_prompt["response"]
                
                result_ls.extend(chunk_data_ls)
                save_data = chunk_data_ls[:temp_num]
                pd.DataFrame(save_data).to_json(temp_path, mode='a', lines=True, orient="records", force_ascii=False)

                chunk_data_ls = chunk_prompt_ls[temp_num:]
                chunk_prompt_ls = chunk_prompt_ls[temp_num:]
    
    if chunk_prompt_ls:
        call_vllm.prompt_request_datalist(chunk_prompt_ls, show_tqdm=False)
        for chunk_data, chunk_prompt in zip(chunk_data_ls, chunk_prompt_ls):
            if not chunk_data[idx_key] == chunk_prompt[idx_key]:
                raise ValueError("infer list and input list do not match")
        
            assistant = chunk_prompt["response"].replace("[unused1]", "").strip()
            assistant = assistant.replace("[unused8]", "\n").strip()

            # chunk_data.pop(idx_key, "none")
            chunk_data["predict_output"] = assistant
            chunk_data["full_output"] = chunk_prompt["response"]

        result_ls.extend(chunk_data_ls)
        save_data = chunk_data_ls
        pd.DataFrame(save_data).to_json(temp_path, mode='a', lines=True, orient="records", force_ascii=False)
    
    for data_item in result_ls:
        data_item.pop(idx_key)
    
    pd.DataFrame(result_ls).to_csv(save_path, index=None, encoding='utf_8_sig')
    os.remove(temp_path)
    return

if __name__ == "__main__":
    # 设置种子
    args = parse_args()
    if args.random:
        args.seed = random.randint(0, 10000)
    seed_everything(args.seed)

    start_time = time.time()
    if args.turn_mode in ["vllm_online"]:
        api_infer_main_funtion(args)
    else:
        print("illegal turn mode, pls check: ", args.turn_mode)
    print("--- %s seconds used---" % (time.time() - start_time))
