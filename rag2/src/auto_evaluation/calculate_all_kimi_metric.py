import re
import pandas as pd
import os
import sys
import argparse
import traceback

sys.path.append('../../')
from common import utils

def get_score(result: list):
    cnt_1 = result.count(1)
    cnt_0 = result.count(0)

    if cnt_1 + cnt_0 == 0:
        # 如果 cnt_1 和 cnt_0 的总和为 0，意味着 result 中既没有 1 也没有 0
        return 0, cnt_1, cnt_0

    score = cnt_1 / (cnt_1 + cnt_0)
    formatted_score = format(score, ".4f")  # 保留四位小数
    return formatted_score, cnt_1, cnt_0


def evaluate_metric(eval_folder, file, job_name, eval_model_name, metric, metric_type):
    path = os.path.join(eval_folder, f'{metric_type}打分', f'{file}.{job_name}_auto_eval_{metric}.csv')
    print(path)
    if not os.path.exists(path):
        print(f'{file}-{metric_type} file not found, setting score to 0.')
        return 0, 0, 0, 0
    
    df = pd.read_csv(path)
    wrong_num = len(df[df['qwen_eval_response'].str.contains('wrong data')])
    df = df[~df['qwen_eval_response'].str.contains('wrong data')]
    result = df[f'{eval_model_name}_{metric}'].to_list()
    formatted_score, cnt_1, cnt_0 = get_score(result)
    
    return formatted_score, cnt_1, cnt_0, wrong_num


def cal_auth_rel(eval_folder, job_name, file='select-new-1231-lvfc_form-2k_final_input.csv1113.out.csv', eval_model_name='qwen', auth_metric='authenticity_test_api_eval', rel_metric='relevance_test_api_eval', rich_metric='richness_test_api_eval'):
    # input_files = ['lvfc_form-2k_final_input.csv1113.out.csv']
    if os.path.isdir(file):
        input_files = utils.get_full_path(file)
    else:
        input_files = [file]

    all_score_dict = {}
    all_cnt1_auth, all_cnt0_auth = 0, 0
    all_cnt1_rel, all_cnt0_rel = 0, 0
    all_cnt1_rich, all_cnt0_rich = 0, 0

    for file in input_files:
        # Authenticity scoring
        formatted_score_auth, cnt_1_auth, cnt_0_auth, auth_wrong_num = evaluate_metric(eval_folder, file, job_name, eval_model_name, auth_metric, '真实性')
        all_cnt1_auth += cnt_1_auth
        all_cnt0_auth += cnt_0_auth
        all_score_dict[f'{file}-真实性'] = formatted_score_auth
        all_score_dict[f'{file}-真实性-wrong-num'] = auth_wrong_num
        
        # Relevance scoring
        formatted_score_rel, cnt_1_rel, cnt_0_rel, rel_wrong_num = evaluate_metric(eval_folder, file, job_name, eval_model_name, rel_metric, '相关性')
        all_cnt1_rel += cnt_1_rel
        all_cnt0_rel += cnt_0_rel
        all_score_dict[f'{file}-相关性'] = formatted_score_rel
        all_score_dict[f'{file}-相关性-wrong-num'] = rel_wrong_num
        
        # Richness scoring
        formatted_score_rich, cnt_1_rich, cnt_0_rich, rich_wrong_num = evaluate_metric(eval_folder, file, job_name, eval_model_name, rich_metric, '丰富性')
        all_cnt1_rich += cnt_1_rich
        all_cnt0_rich += cnt_0_rich
        all_score_dict[f'{file}-丰富性'] = formatted_score_rich
        all_score_dict[f'{file}-丰富性-wrong-num'] = rich_wrong_num

        # Repeat scoring
        formatted_score_rich, cnt_1_rich, cnt_0_rich, rich_wrong_num = evaluate_metric(eval_folder, file, job_name, eval_model_name, rich_metric, '重复性')
        all_cnt1_rich += cnt_1_rich
        all_cnt0_rich += cnt_0_rich
        all_score_dict[f'{file}-重复性'] = formatted_score_rich
        all_score_dict[f'{file}-重复性-wrong-num'] = rich_wrong_num
    
        # markdown rule-base
        try:
            path = os.path.join(eval_folder, f'{file}.{job_name}.csv')
            df = pd.read_csv(path)
            df = df.fillna('')
            df['out_len'] = df['predict_output'].apply(lambda x: len(x))
            utils.cal_md_ratio(df,out_col='predict_output')
            all_score_dict[f'{file}-回复平均字数-中位数'] = df['out_len'].mean()-df['out_len'].median()
            all_score_dict[f'{file}-宽松MD-MD'] = format(df['is_md_loose'].mean()-df['is_md'].mean(), ".4f")
            all_score_dict[f'{file}-宽松list-list'] = format(df['is_md_list_loose'].mean()-df['is_md_list'].mean(), ".4f")
            all_score_dict[f'{file}-回复超过2k条数'] = len(df[df['out_len']>2000])
            all_score_dict[f'{file}-回复平均字数'] = format(df['out_len'].mean(), ".4f")
            all_score_dict[f'{file}-回复中位数'] = format(df['out_len'].median(), ".4f")
            all_score_dict[f'{file}-回复最大数'] = format(df['out_len'].max(), ".4f")
            all_score_dict[f'{file}-MD比例'] = format(df['is_md'].mean(), ".4f")
            all_score_dict[f'{file}-MD-loss比例'] = format(df['is_md_loose'].mean(), ".4f")
            all_score_dict[f'{file}-list比例'] = format(df['is_md_list'].mean(), ".4f")
            all_score_dict[f'{file}-list-loss比例'] = format(df['is_md_list_loose'].mean(), ".4f")
            all_score_dict[f'{file}-加粗比例'] = format(df['is_md_bold'].mean(), ".4f")
            all_score_dict[f'{file}-二级标题比例'] = format(df['is_md_2nd'].mean(), ".4f")
            
        except Exception as exc:
            traceback.print_exc()

    # Assuming `get_score` and computations for overall statistics exist
    print(all_score_dict)

    result_items = [
        f'{file}-真实性',
        f'{file}-相关性',
        f'{file}-丰富性',
        f'{file}-重复性',
        f'{file}-回复平均字数-中位数',
        f'{file}-宽松MD-MD',
        f'{file}-宽松list-list',
        f'{file}-回复超过2k条数',
        f'{file}-回复平均字数',
        f'{file}-回复中位数',
        f'{file}-回复最大数',
        f'{file}-MD比例',
        f'{file}-MD-loss比例',
        f'{file}-list比例',
        f'{file}-list-loss比例',
        f'{file}-加粗比例',
        f'{file}-二级标题比例',
        f'{file}-真实性-wrong-num',
        f'{file}-相关性-wrong-num',
        f'{file}-丰富性-wrong-num',
        f'{file}-重复性-wrong-num'
    ]
    
    result_value = [str(all_score_dict[item]) for item in result_items]
    print("\t".join(result_items))
    print("\t".join(result_value))
    
    with open(f"{eval_folder}/result_all.txt", "w") as f:
        f.write("\t".join(result_items) + "\n")
        f.write("\t".join(result_value) + "\n")

    return all_score_dict

if __name__ == "__main__":
    parser  =  argparse.ArgumentParser(description = 'calculate all e2e score')
    parser.add_argument("--eval_folder", type = str, default = "eval_output", help = "evaluation results folder,with job name")
    parser.add_argument("--job_name", type = str, default = "job_name", help = "job name")
    parser.add_argument("--file", type = str, default = "lvfc_form-2k_final_input.csv1113.out.csv", help = "eval file name")
    parser.add_argument("--eval_model_name", type = str, default = "qwen", help = "eval_model_name")
    parser.add_argument("--auth_metric", type = str, default = "authenticity_test_api_eval", help = "auth_metric")
    parser.add_argument("--rel_metric", type = str, default = "relevance_test_api_eval", help = "rel_metric")
    args  =  parser.parse_args()
    print(args)

    cal_auth_rel(args.eval_folder,args.job_name, args.file, args.eval_model_name,args.auth_metric,args.rel_metric)