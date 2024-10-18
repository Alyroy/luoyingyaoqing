import pandas as pd
from tqdm import tqdm
import os
import re
import traceback
import glob
import argparse

def load_data(input_folder):
    dl = []
    files = [f for f in os.listdir(input_folder) if '_保留.csv' in f and '.ipynb_checkpoints' not in f]
    for file in files: 
        df_ = pd.read_csv(input_folder+file)
        if 'task_name' not in df_.columns:
            df_['task_name'] = 'gpt4泛化query'
        dl.append(df_)
    df = pd.concat(dl)
    return df



def cal_stats(target_date,log_type,single_rag_type,infolder):
    out_df = pd.DataFrame()
    output_csv_file = f"{infolder}/log_distillation_stats_{log_type}_{single_rag_type}.csv"
    
    filter_input_folder = f"{infolder}/{target_date}/{log_type}/{single_rag_type}/correct_filter_output/"
    filter_df = load_data(filter_input_folder)
    filter_df = filter_df[~filter_df['parser_gpt4'].isna()]
    if len(filter_df) == 0:
        raise f"{target_date} {log_type} 蒸馏数据为空"

    raw_df = pd.read_csv(f'{infolder}/{target_date}/{log_type}/{single_rag_type}/{target_date}_log_data.csv')
    raw_df = raw_df.rename(columns={'task-name':'task_name'})
    raw_total_count = len(raw_df)
    raw_assistant_logic_bad_count = raw_df[raw_df['task_name'].isin(['逻辑性中','逻辑性差'])]['user-query'].count()
    raw_assistant_relevance_bad_count = raw_df[raw_df['task_name'].isin(['相关性中','相关性差'])]['user-query'].count()

    filter_total_count = len(filter_df)
    filter_assistant_logic_bad_count = filter_df[filter_df['task_name'].isin(['逻辑性中','逻辑性差'])]['user-query'].count()
    filter_assistant_relevance_bad_count = filter_df[filter_df['task_name'].isin(['相关性中','相关性差'])]['user-query'].count()

    filter_df['assistant_length'] = filter_df['parser_gpt4'].apply(len)
    assistant_length_mean = filter_df['assistant_length'].mean()
    assistant_length_q10 = filter_df['assistant_length'].quantile(0.1)
    assistant_length_q90 = filter_df['assistant_length'].quantile(0.9)
    
    new_data = {
        "时间": target_date,
        "原始query总量": raw_total_count, 
        "原始逻辑性低或中数量": raw_assistant_logic_bad_count, 
        "原始相关性低或中数量": raw_assistant_relevance_bad_count, 
        "数据蒸馏筛选后总量": filter_total_count, 
        "数据蒸馏筛选后逻辑性低或中数量": filter_assistant_logic_bad_count, 
        "数据蒸馏筛选后相关性低或中数量": filter_assistant_relevance_bad_count, 
        "蒸馏回复平均长度": int(assistant_length_mean), 
        "蒸馏回复长度10分位": int(assistant_length_q10), 
        "蒸馏回复长度90分位": int(assistant_length_q90)
    }
    
    # 确定是否存在已有的CSV文件
    if os.path.exists(output_csv_file):
        existing_df = pd.read_csv(output_csv_file, encoding='utf-8-sig')
        out_df = existing_df._append(new_data, ignore_index=True)
    else:
        out_df = pd.DataFrame()._append(new_data, ignore_index=True)
    
    out_df.to_csv(output_csv_file, header=True, index=False, encoding='utf-8-sig')
    return out_df


if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='report for distillaiton')
    
    # 添加参数
    parser.add_argument('--target_date', type=str, help='统计指标的目标文件日期，格式为 YYYY-MM-DD')
    parser.add_argument('--log_type', type=str, default='raw', help='raw or extension')
    parser.add_argument('--single_rag_type', type=str, default='single_True_rag_True', help='single rag label 格式single_True_rag_True')
    parser.add_argument('--infolder', type=str, default='/mnt/pfs-guan-ssai/nlu/renhuimin/data/log_distillation/', help='log distillation root folder')
    args = parser.parse_args()
    print(args)

    cal_stats(args.target_date,args.log_type,args.single_rag_type,args.infolder)