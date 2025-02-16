# encoding = utf-8

import torch
import sys, os, re
import pandas as pd
import time
import argparse
import random
import json
import numpy as np
import torch.multiprocessing as mp
from vllm import LLM, SamplingParams
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, AutoModelForCausalLM, AutoTokenizer
from litiktoken.tiktoken_tokenizer import Tokenizer

sys.path.append("../") 
from tool_rag_generation.data_format import DataFormat

sys.path.append("../../") 
from common import utils_log,utils
# """
# signs = {
#     "<|lc_start|>": "[unused0]",
#     "<|lc_end|>": "[unused1]",
#     "<|kvs|>": "[unused2]",
#     "<|kve|>": "[unused3]",
#     "<|api_start|>": "[unused4]",
#     "<|api_end|>": "[unused5]",
#     "<|eoa|>": "[unused6]",
#     "=>": "[unused7]"
# }
# """
# empty_str = "[unused0]thought\n<None>[unused1]\n[unused0]api\n<None>[unused1]\n[unused0]observation\n<None>[unused1]\n"
def parse_args():
    parser = argparse.ArgumentParser(description='information')
    parser.add_argument('--gpus', dest='gpus', default="0,1,2,3,4,5,6,7", type=str, help='model path')
    parser.add_argument('--model', dest='model', type=str, help='model path')
    parser.add_argument('--tiktoken_path', dest='tiktoken_path', type=str, default=None, help='litiktoken path')
    parser.add_argument('--input_file', dest='input_file', type=str, help='input file path')
    parser.add_argument('--batch_size', dest='batch_size', type=int, help='inference batch size')
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
    parser.add_argument('--num_beams', dest='num_beams', type=float, default=1, help='repetition penalty')
    parser.add_argument('--eval_col', dest='eval_col', type=str, default="xx", help='column to be infered')
    parser.set_defaults(dosample_flag=True)
    parser.set_defaults(api_flag=True)  # 默认为True
    args = parser.parse_args()
    print(args)
    return args

# 定义设置随机种子的函数
def seed_everything(seed=9987):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

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

def do_func_api_single(gpu_no, params, input_f, api_flag=True, bsize=20, loop=5, mode="moe", checkpoint='', directory='', total_gpus=[0], tiktoken_model_path=None):
    parallel_size = torch.cuda.device_count()

    # hf_version = get_version_transformers().split('.')
    # os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(total_gpus[gpu_no])
    # print('current gpu no is', total_gpus[gpu_no])
    
    # 按照以下方式导入模型和分词器
    if mode == "qwen":
        if tiktoken_model_path  != 'none':
            tokenizer = Tokenizer(tiktoken_model_path)
        # model = LLM(model=checkpoint, tokenizer=checkpoint, tensor_parallel_size=parallel_size)
        # model.llm_engine.tokenizer.padding_side = "left"

        # modify by lvjianwei
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, add_bos_token=False, trust_remote_code=True)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map="auto",trust_remote_code=True)

        print("input_f:{}".format(input_f))
        # sampling_params = SamplingParams(
        #     temperature=params.temperature,
        #     top_p=params.top_p,
        #     top_k=params.top_k,
        #     repetition_penalty=params.repetition,
        #     max_tokens=11384, #8000, #注意：这是生成的最大长度，不是输入的最大长度
        #     skip_special_tokens=True,
        # )
    
    # if int(hf_version[1]) > 28: 
    #     model.llm_engine.tokenizer.padding_side = "left"
    # else:
    #     model = LlamaForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map="auto", use_cache=True)   
    print("input_f:", input_f) 
    if os.path.isdir(input_f):
        file_list = [os.path.join(input_f, x) for x in os.listdir(input_f) if x.endswith(".csv")]
    elif os.path.isfile(input_f):
        file_list = [input_f]

    for one_file in file_list:
        print("inference_file:{}".format(one_file))
        sampling_params = SamplingParams(
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            repetition_penalty=params.repetition,
            max_tokens=2048, #注意：这是生成的最大长度，不是输入的最大长度
            skip_special_tokens=True,
        )
        
        df =utils.preprocess_df(one_file)
        if params.eval_col in df.columns:
            df = df[~df[params.eval_col].isna()]
        total_rows = df.shape[0]
        print('total rows is:', total_rows)
        chunks = np.array_split(range(total_rows), 1)
        current_chunk = chunks[gpu_no].tolist()
        print('current chunk is', gpu_no, 'with the range of: ', current_chunk[0], current_chunk[-1])
        df = df[current_chunk[0]:(current_chunk[-1] + 1)]
        #dev_data = df["input"].values.tolist()

        if params.eval_col in df.columns:
            dev_data = []
            for i in range(len(df)):
                input_dict = {}
                input_dict['instruction'] = df.iloc[i][params.eval_col]
                input_dict['output'] = '无[unused1]'
                dev_data.append(input_dict)
        else:
            dataformat_obj = DataFormat(api_flag,multi_flag=False)
            dev_data = dataformat_obj.gen_sft_data(df, flag_16b_inputs=True)
            
        count = 0
        for _ in range(loop):
            full_input = list()
            predict_text = list()
            predict_api = list()
            predict_tmp = list()
            predict_thought = list()
            full_output = list()

            for idx in range(0, len(dev_data), bsize):
                count += 1
                print(str(gpu_no), 'status', str(count*bsize), len(current_chunk))
                # print("================== batch:", count, " ==================")
                # 分别对输入文本进行批量推理
                input_text_list = list()
                if idx >= len(dev_data):
                    break
                for input_ in dev_data[idx: idx + bsize]:
                    input_text = "<|im_start|>%s" % input_['instruction']
                    # sys_prompt = "[unused0]system\n你是一个名字叫做理想同学的AI数字生命体。\n理想同学是一个可靠的智能家庭助手，由理想汽车智能空间部门创造。\n理想同学能够理解人类的指令和意图，并且给出合理的、切合问题的、没有歧视、中立的、安全的回复。\n\n请根据以下文本写一个合适的回复。[unused1]\n"
                    # input_text_list.append(sys_prompt + input_text + "[unused0]assistant\n")
                    input_text_list.append(input_text)

                ############################# dense vllm #############################
                for try_idx in range(10):
                    try:
                        # raw_outputs = model.generate(input_text_list, sampling_params)

                        # modify by lvjianwei
                        tok_input = tokenizer.batch_encode_plus(input_text_list, padding=True, truncation=True,
                                                                return_tensors='pt')
                        input_ids = tok_input['input_ids'].to('cuda')
                        attention_mask = tok_input['attention_mask'].to('cuda')
                        output = model.generate(input_ids=input_ids, attention_mask=attention_mask, temperature=params.temperature,
                                                top_k=params.top_k,
                                                top_p=params.top_p,num_beams=1,
                                                repetition_penalty=params.repetition, eos_token_id=tokenizer.eos_token_id,
                                                max_new_tokens=2048,
                                                do_sample=True
                                                )
                        # 解码生成的文本输出
                        raw_outputs = tokenizer.batch_decode(output,skip_special_tokens=True)

                        break

                    except Exception as e:
                        print(e)
                        continue           
                
                for i in range(len(input_text_list)):
                    # output_text = raw_outputs[i].outputs[0].text
                    # output_text_false = raw_outputs_false[i].outputs[0].text
                    # modify by lvjianwei
                    output_text = raw_outputs[i]
                    if output_text is None:
                        full_input.append(input_text_list[i])
                        full_output.append(input_text_list[i])
                        continue

                    output_assist = output_text.split(input_text_list[i].strip())[-1]  # 生成的输出内容
                    assistant = output_assist.split("[unused1]")[0].strip()
                    assistant = assistant.replace("[unused8]", "\n").strip()

                    # assistant = assistant.split("assistant")[-1].strip()
                    print("assistant:", assistant)
                    print("raw output:", output_text)
                    print("-------------")

                    output_assist_false = output_text_false.split(input_text_list[i].strip())[-1]  # 生成的输出内容
                    assistant_false = output_assist_false.split("[unused1]")[0].strip()
                    assistant_false = assistant_false.replace("[unused8]", "\n").strip()

                    predict_text.append(assistant)
                    # predict_api.append(api)
                    # predict_thought.append(thought)
                    full_input.append(input_text_list[i])
                    full_output.append(input_text_list[i]+output_text)

            # 保存数据
            df.insert(loc=df.shape[1], column="predict_output", value=predict_text)
            # df.insert(loc=df.shape[1], column="api", value=predict_api)
            # df.insert(loc=df.shape[1], column="thought", value=predict_thought)
            df.insert(loc=df.shape[1], column="full_input", value=full_input)
            df.insert(loc=df.shape[1], column="full_output", value=full_output)
        df = df.rename(columns={'assistant':'correct_output'})

        test_dataset_file = one_file.strip('/').split('/')[-1].strip()
        result_file_prefix = test_dataset_file + "." + params.time_stamp
        output_pt = directory + "/" + result_file_prefix
        pd.DataFrame(df).to_csv(output_pt + ".csv", index=None, encoding='utf_8_sig')

def func_api_single(params, input_f, api_flag=True, bsize=20, loop=5, mode="moe", tiktoken_path=None):
    args = parse_args()
    directory = os.path.join(args.output_path, args.time_stamp)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # n_process = len(args.gpus.split(','))
    n_process = 1
    # do_func_api_single(0, params, input_f, api_flag, bsize, loop, mode, args.model, output_pt, args.gpus.split(','), tiktoken_path)
    mp.spawn(do_func_api_single, nprocs=n_process, args=(params, input_f, api_flag, bsize, loop, mode, args.model, directory, args.gpus.split(','), tiktoken_path))

    # test_dataset_file = args.input_file.strip('/').split('/')[-1].strip()
    # result_file_prefix = test_dataset_file + "." + args.time_stamp
    # output_pt = directory + "/" + result_file_prefix
    # # merge files
    # df = pd.concat([pd.read_csv(output_pt + '_' + str(i) + '.csv') for i in range(n_process)], ignore_index=True)
    # [os.remove(output_pt + '_' + str(i) + '.csv') for i in range(n_process)]
    # pd.DataFrame(df).to_csv(output_pt + ".csv", index=None, encoding='utf_8_sig')
    print('finished')

if __name__ == "__main__":
    # 设置种子
    # seed_everything()
    args = parse_args()
    start_time = time.time()
    if args.turn_mode in ["qwen"]:
        func_api_single(args, args.input_file, api_flag=args.api_flag, bsize=args.batch_size, loop=args.try_num, mode=args.turn_mode, tiktoken_path=args.tiktoken_path)
    # elif args.turn_mode in ["1b"]:
    #     func_api_single(args, args.input_file, api_flag=args.api_flag, bsize=args.batch_size, loop=args.try_num, mode="1b")
    else:
        print("illegal turn mode, pls check: ", args.turn_mode)
    print("--- %s seconds used---" % (time.time() - start_time))
