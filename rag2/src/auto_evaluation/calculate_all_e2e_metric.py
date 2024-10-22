import re
import pandas as pd
import os
import sys
import argparse


def get_score(result:list):
    cnt_1 = result.count(1)
    cnt_0 = result.count(0)
    score = cnt_1/(cnt_1+cnt_0)
    formatted_score = format(score, ".4f")  # 保留四位小数
    return formatted_score,cnt_1,cnt_0


def cal_auth_rel(eval_folder,job_name,eval_model_name='qwen25-72b',auth_metric='authenticity_test_api_eval',rel_metric='relevance_test_api_eval'):
    input_files = ['手机APP_泛化集_2024-08-23T21_34_46.379.csv','APP同分布开发集16b评估-0821.csv']

    all_score_dict = {}
    all_cnt1_auth = 0
    all_cnt0_auth = 0
    all_cnt1_rel = 0
    all_cnt0_rel = 0
    for file in input_files:
        path_auth = os.path.join(eval_folder,'真实性打分',f'{file}.{job_name}_auto_eval_{auth_metric}.csv')
        df_auth = pd.read_csv(path_auth)
        result_auth = df_auth[eval_model_name + '_' + auth_metric].to_list()
        formatted_score_auth,cnt_1_auth,cnt_0_auth = get_score(result_auth)
        all_cnt1_auth+=cnt_1_auth
        all_cnt0_auth+=cnt_0_auth
        all_score_dict[f'{file}-真实性'] = formatted_score_auth
    
        path_rel = os.path.join(eval_folder,'相关性打分',f'{file}.{job_name}_auto_eval_{rel_metric}.csv')
        df_rel = pd.read_csv(path_rel)
        result_rel = df_rel[eval_model_name + '_' + rel_metric].to_list()
        formatted_score_rel,cnt_1_rel,cnt_0_rel = get_score(result_rel)
        all_cnt1_rel+=cnt_1_rel
        all_cnt0_rel+=cnt_0_rel
        all_score_dict[f'{file}-相关性'] = formatted_score_rel

    overall_auth = format(all_cnt1_auth/(all_cnt1_auth+all_cnt0_auth),".4f") 
    overvall_rel = format(all_cnt1_rel/(all_cnt1_rel+all_cnt0_rel),".4f") 
    all_score_dict['整体真实性得分'] = overall_auth
    all_score_dict['整体相关性得分'] = overvall_rel
    # print(all_score_dict)

    result_items = ['整体真实性得分', '整体相关性得分', '手机APP_泛化集_2024-08-23T21_34_46.379.csv-真实性','手机APP_泛化集_2024-08-23T21_34_46.379.csv-相关性',
                    'APP同分布开发集16b评估-0821.csv-真实性','APP同分布开发集16b评估-0821.csv-相关性']
    result_value = []
    with open(f"{eval_folder}/result_all.txt", "w") as f:
        for item in result_items:
            value = str(all_score_dict[item])
            result_value.append(value)
        print("\t".join(result_items))
        print("\t".join(result_value))
        f.write("\t".join(result_items)+"\n")
        f.write("\t".join(result_value)+"\n")

    return all_score_dict



if __name__ == "__main__":
    parser  =  argparse.ArgumentParser(description = 'calculate all e2e score')
    parser.add_argument("--eval_folder", type = str, default = "eval_output", help = "evaluation results folder,with job name")
    parser.add_argument("--job_name", type = str, default = "job_name", help = "job name")
    parser.add_argument("--eval_model_name", type = str, default = "qwen", help = "eval_model_name")
    parser.add_argument("--auth_metric", type = str, default = "authenticity_test_api_eval", help = "auth_metric")
    parser.add_argument("--rel_metric", type = str, default = "relevance_test_api_eval", help = "rel_metric")
    args  =  parser.parse_args()
    print(args)

    cal_auth_rel(args.eval_folder,args.job_name,args.eval_model_name,args.auth_metric,args.rel_metric)