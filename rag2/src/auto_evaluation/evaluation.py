import sys
import os
import argparse
import time
from utils.vote_strategy import vote_strategy
from base.base_eval import BaseModelEval
from metrics import (
    authenticityEval,
    relResponseEval,
    authenticityTestAPIEval,
    relResponseTestAPIEval,
    richnessEval,
    richnessTestAPIEval
)

sys.path.append("..")
BASE_PATH = os.path.abspath(os.path.dirname(__file__))

sys.path.append("../../")
from common import utils


METRIC_DICT = {
    "relevance":{
        "function":relResponseEval,
    },
    "authenticity":{
        "function":authenticityEval,
    },
    "authenticity_test_api_eval":{
        "function":authenticityTestAPIEval,
    },
    "relevance_test_api_eval":{
        "function":relResponseTestAPIEval,
    },
    "richness": {
        "function": richnessEval,
    },
    "richness_test_api_eval": {
        "function": richnessTestAPIEval,
    }
}


def evaluate(model_list: list[str], url_list: list[str], metric: str, eval_column_list: list[str], save_column: str, input_file: str, output_dir: str, prompt_path: str, thread_num: int, chunk_num: int, temperature: float, eval_mode:str='user_obs_ans_concat', max_tokens=8000, input_text_type: str='default', eval_task='richness_eval_v1'):
    
    df = utils.get_df(input_file)
    df = df[~df[eval_column_list[-1]].isna()] # 回复为空不评估
    if 'user-query' in df.columns and 'query' not in df.columns:
        df['query'] = df['user-query']
    # df = df.iloc[:20]
    result_list = []
    for i in range(len(model_list)):
        model = model_list[i]
        assert model in ['qwen', 'qwen25-72b', 'qwen2-72b', 'llama3-70b', 'qwen2_72b', 'qwen1.5_72b', 'qwen1.5_110b', 'autoj', 'deepseek', 'gpt4o', 'gpt4', 'mindgpt','wenxin'], "model name is not right!!!"
        print("---------------------------------\n正在使用", model, "进行评估\n---------------------------------")
        url = url_list[i]
        assert metric in METRIC_DICT, "metric name is not right!!!"
        func = METRIC_DICT[metric]['function']
        assert isinstance(func, BaseModelEval), "func type is wrong!!!"
        if metric in ['authenticity_test_api_eval', 'relevance_test_api_eval', 'richness_test_api_eval']:
            result, reason = func.main_eval(
                model = model,
                url = url, 
                eval_column_list = eval_column_list, 
                df = df, 
                output_dir = output_dir, 
                prompt_path = prompt_path, 
                thread_num = thread_num, 
                chunk_num = chunk_num, 
                temperature = temperature,
                eval_mode = eval_mode,
                max_tokens = max_tokens,
                input_text_type = input_text_type,
                eval_task = eval_task
            )
        elif metric in ['authenticity', 'relevance']:
            result, reason = func.main_eval(
                model = model,
                url = url, 
                eval_column_list = eval_column_list, 
                df = df, 
                output_dir = output_dir, 
                prompt_path = prompt_path, 
                thread_num = thread_num, 
                chunk_num = chunk_num, 
                temperature = temperature,
                eval_mode = eval_mode,
                max_tokens = max_tokens            
            )
        else:
            result, reason = func.main_eval(
                model = model,
                url = url, 
                eval_column_list = eval_column_list, 
                df = df, 
                output_dir = output_dir, 
                prompt_path = prompt_path, 
                thread_num = thread_num, 
                chunk_num = chunk_num, 
                temperature = temperature,
                eval_mode = eval_mode
            )
        assert len(result) == len(df), "result length is wrong!"
        
        # 将每个模型的输出分数和打分原因存入df
        df[model + '_eval_response'] = reason
        df[model + '_' + metric] = result
        
        show_dic = {}
        show_dic['model'] = model
        show_dic['task_name'] = metric
        show_dic['result'] = result
        show_dic['reason'] = reason
        result_list.append(show_dic)
        print("---------------------------------\n", model, "评估完毕\n---------------------------------")
        # 当前仅支持单个模型推理，多个模型vote待优化
        cnt_1 = result.count(1)
        cnt_0 = result.count(0)
        # Calculate the total count of valid evaluations
        total_valid_evaluations = cnt_1 + cnt_0
        
        # Check to avoid division by zero
        if total_valid_evaluations == 0:
            positive_ratio = 0
        else:
            positive_ratio = cnt_1 / total_valid_evaluations
        
        # Print the formatted output
        print(f"{input_file}, 正例数量：{cnt_1}, 测试集数量：{len(df)}, 有效评估数量:{total_valid_evaluations}, {metric}, 正例占比：{positive_ratio}")

    
    # 对多模型评估结果进行投票打分
    
    final_scores = vote_strategy(df, result_list)
    # cnt_1 = final_scores.count(1)
    # print(f"{input_file}, 正例数量：{cnt_1}, 测试集数量：{len(df)}, {metric}, 正例占比：{cnt_1/len(df)}")
    
    # 最终评估结果存入自定义列
    df[save_column] = final_scores
    
    # 存储结果
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename, _ = os.path.splitext(os.path.basename(input_file))
    df.to_csv(os.path.join(output_dir, f'{filename}_auto_eval_{metric}.csv'), index=False, encoding='utf-8-sig')
    return result_list, final_scores


parser  =  argparse.ArgumentParser(description = 'information')
parser.add_argument("--model_list", nargs = '+', type = str, 
                    default = ["gpt4o", "wenxin"], help = "eval model list")
parser.add_argument("--url_list", nargs = '+', type = str, 
                    default = ["http://172.24.136.32:8008/v1", "http://172.24.136.254:8001/v1", "http://172.24.136.254:8000/v1"], help = "model url list")
parser.add_argument("--eval_column_list", nargs = 3, type = str,
                    default = ["user_query", "observation", "predict_output"], help = "columns name of query, obs and ans")
parser.add_argument("--save_column", type = str,
                    default = "eval_output", help = "column name of saved eval output")
parser.add_argument("--metric", type = str,
                    default = "authenticity", help = "metric")
parser.add_argument("--input_dir", type = str, 
                    default = "./", help = "input data path")
parser.add_argument("--output_dir", type = str,
                    default = "./", help = "output data path")
parser.add_argument("--prompt_path", type = str, 
                    default = "./prompts/authenticity-prompts-rag.txt", help = "prompt path")
parser.add_argument("--thread_num", type = int, default = 10, help = "thread num")
parser.add_argument("--chunk_num", type = int, default = 2, help = "chunk num")
parser.add_argument("--temperature", type = float, default = 0.7, help = "temperature")
parser.add_argument("--eval_mode", type = str, 
                    default = "user_obs_ans_concat", help = "user_obs_ans_concat,model_13b_log,with_prompt")
parser.add_argument("--input_text_type", type = str, 
                    default = "default", help = "default,function_call")
parser.add_argument("--max_tokens", type = int, default = 8000, help = "max_tokens")
parser.add_argument("--eval_task", type = str, default = 'richness_eval_v1', help = "richness_eval_v1,richness_eval,repeat_eval")
args  =  parser.parse_args()
print(args)


if __name__ == "__main__":
    file = args.input_dir
    # 如果是文件夹
    # try:
    if os.path.isdir(file):
        files = [file+f for f in os.listdir(file) if '.ipynb_checkpoints' not in f and '.done' not in f and '.csv' in f]
        for input_file in files:
            print("--------------------------------------------\n正在评估 ", input_file, "\n--------------------------------------------")
            results, final_scores = evaluate(
                model_list = args.model_list,
                url_list = args.url_list,
                metric = args.metric,
                eval_column_list = args.eval_column_list,
                save_column = args.save_column,
                input_file = input_file,
                output_dir = args.output_dir,
                prompt_path = args.prompt_path,
                thread_num = args.thread_num,
                chunk_num = args.chunk_num,
                temperature = args.temperature,
                eval_mode = args.eval_mode,
                input_text_type = args.input_text_type,
                eval_task = args.eval_task
            )
            print("--------------------------------------------\n", input_file, "评估完毕\n--------------------------------------------")
    # 如果是文件
    else:
        print("--------------------------------------------\n正在评估", file, "\n--------------------------------------------")
        results, final_scores = evaluate(
            model_list = args.model_list,
            url_list = args.url_list,
            metric = args.metric,
            eval_column_list = args.eval_column_list,
            save_column = args.save_column,
            input_file = file,
            output_dir = args.output_dir,
            prompt_path = args.prompt_path,
            thread_num = args.thread_num,
            chunk_num = args.chunk_num,
            temperature = args.temperature,
            eval_mode = args.eval_mode,
            input_text_type = args.input_text_type,
            eval_task = args.eval_task
        )
        print("-------------------------------------\n", file, "文件评估完毕\n-------------------------------------") 
    # except Exception as e:
    #     print(e)