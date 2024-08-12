import sys
import os
import argparse
import os
import concurrent.futures
from utils.consistency_strategy import consistency_strategy
from base.base_eval import BaseModelEval
from metrics import (
    correctFilter,
    styleFilter
)

sys.path.append("..")
BASE_PATH = os.path.abspath(os.path.dirname(__file__))

sys.path.append("../../")
from common import utils

METRIC_DICT = {
    "correct_ans_filter":{
        "function":correctFilter
    },
    "speech_style_filter":{
        "function":styleFilter
    }
}

def eval_model(model, url, metric, eval_column_list, df, output_dir, prompt_path, thread_num, chunk_num, temperature, concat_prompt_flag):
    assert model in ['llama3_70b','qwen','qwen2_72b', 'qwen1.5_72b', 'qwen1.5_110b', 'autoj', 'deepseek', 'gpt4o', 'gpt4', 'mindgpt','wenxin'], "model name is not right!!!"
    print(f"---------------------------------\n正在使用 {model} 进行评估\n---------------------------------")
    assert metric in METRIC_DICT, "metric name is not right!!!"
    func = METRIC_DICT[metric]['function']
    assert isinstance(func, BaseModelEval), "func type is wrong!!!"
    result, reason = func.main_eval(
        model=model,
        url=url,
        eval_column_list=eval_column_list,
        df=df,
        output_dir=output_dir,
        prompt_path=prompt_path,
        thread_num=thread_num,
        chunk_num=chunk_num,
        temperature=temperature,
        concat_prompt_flag=concat_prompt_flag
    )
    assert len(result) == len(df), "result length is wrong!"

    print(f"---------------------------------\n {model} 评估完毕\n---------------------------------")
    return model, result, reason


def evaluate(model_list, url_list, metric, eval_column_list, save_column, input_file, output_dir, prompt_path, thread_num, chunk_num, temperature, concat_prompt_flag=True):
    df = utils.get_df(input_file)
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(model_list)) as executor:
        future_to_model = {
            executor.submit(
                eval_model,
                model,
                url_list[i],
                metric,
                eval_column_list,
                df,
                output_dir,
                prompt_path,
                thread_num,
                chunk_num,
                temperature,
                concat_prompt_flag
            ): model for i, model in enumerate(model_list)
        }
        
        for future in concurrent.futures.as_completed(future_to_model):
            model = future_to_model[future]
            try:
                model, result, reason = future.result()
                df[model + '_eval_response'] = reason
                df[model + '_' + metric] = result

                results.append({
                    'model': model,
                    'task_name': metric,
                    'result': result,
                    'reason': reason
                })
            except Exception as exc:
                print(f"{model} generated an exception: {exc}")

    # Consistency strategy to aggregate the results
    final_scores = consistency_strategy(df, results)

    df[save_column] = final_scores

    # Save the results
    df_1 = df[df[save_column] == 1]
    df_0 = df[df[save_column] == 0]
    df_minus_1 = df[df[save_column] == -1]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename, _ = os.path.splitext(os.path.basename(input_file))
    df.to_csv(os.path.join(output_dir, f'{filename}_{metric}.csv'), index=False)
    df_1.to_csv(os.path.join(output_dir, f'{filename}_{metric}_保留.csv'), index=False)
    df_0.to_csv(os.path.join(output_dir, f'{filename}_{metric}_废弃.csv'), index=False)
    df_minus_1.to_csv(os.path.join(output_dir, f'{filename}_{metric}_送标.csv'), index=False)

    print_output = f"{metric} 原始数据量：{len(df)},保留量：{len(df_1)},废弃量：{len(df_0)},送标量：{len(df_minus_1)}，送标比例：{len(df_minus_1)}/{len(df)}={len(df_minus_1)/len(df):.4f}"
    print(print_output)

    return results, final_scores


parser  =  argparse.ArgumentParser(description = 'information')
parser.add_argument("--model_list", nargs = '+', type = str, 
                    default = ["autoj", "deepseek", "qwen"], help = "eval model list")
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
parser.add_argument('--concat_prompt_flag', dest='concat_prompt_flag', action='store_true', help='concat prompt true')
parser.add_argument('--no_concat_prompt_flag', dest='concat_prompt_flag', action='store_false', help='concat prompt false')
parser.set_defaults(concat_prompt_flag=True)
args  =  parser.parse_args()
print(args)


if __name__ == "__main__":
    file = args.input_dir
    # 如果是文件夹
    if os.path.isdir(file):
        files = [file+f for f in os.listdir(file) if '.ipynb_checkpoints' not in f]
        for input_file in files:
            try:
                print("------------------------\n正在评估 ", input_file, "\n---------------------------------------")
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
                    concat_prompt_flag = args.concat_prompt_flag
                )
                print("---------------------------------\n", input_file, "评估完毕\n-----------------------------------")
            except Exception as e:
                print(e)
                continue

    # 如果是文件
    else:
        print("---------------------------------------\n正在评估", file, "\n--------------------------------------------")
        try:
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
                concat_prompt_flag = args.concat_prompt_flag
            )
            print("-------------------------------------\n", file, "文件评估完毕\n-------------------------------------") 
        except Exception as e:
            print(e)