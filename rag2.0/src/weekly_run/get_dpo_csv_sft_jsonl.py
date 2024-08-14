import argparse
import pandas as pd
import os
from datetime import datetime
import sys
sys.path.append('../')
from tool_rag_generation.data_format import DataFormat
sys.path.append('../../')
from common import utils


def remove_br_template(row):
    if pd.isna(row['thought']) or row['thought'] == '':  # 检查是否为空或缺失值
        return f"涉及知识问答，查询{row['user-query']}"
    else:
        return row['thought'].replace('该问题请使用总分类模板回复。', '')


def gen_dpo_df(df,date,folder):
    usecols = ['user-query','api','thought','observation','assistant','parser_gpt4','system']
    dpo_human_df = df[(df['assistant_relevance'].isin(['中','低']))|(df['assistant_logic'].isin(['中','低']))]
    dpo_human_df = dpo_human_df[usecols].rename(columns={'assistant':'modelA','parser_gpt4':'modelB'})
    dpo_human_df['thought'] = dpo_human_df.apply(remove_br_template, axis=1)  # thought 去掉总分模板
    dpo_human_df['query是否有效'] = ''
    dpo_human_df['哪个回复好'] = ''
    dpo_human_df['落败原因'] = ''
    dpo_human_df['observation2'] = dpo_human_df['observation'].apply(utils.flatten_and_number)
    dpo_human_df.to_csv(folder+f'{date}_dpo.csv',index=False)
    return dpo_human_df


def gen_sft_df(df,date,folder):
    sft_df = df[~((df['assistant_relevance'].isin(['中', '低'])) | (df['assistant_logic'].isin(['中', '低'])))]
    usecols = ['user-query','api','thought','observation','parser_gpt4','system']
    sft_df = sft_df[usecols].rename(columns={'parser_gpt4':'assistant'})
    sft_df['thought'] = sft_df.apply(remove_br_template, axis=1)  # thought 去掉总分模板
    
    # 转训练数据
    now = datetime.now()    
    create_date = now.strftime('%Y-%m-%d')
    sft_df['source'] = '手机日志'
    sft_df['relevant_label'] = '相关'
    sft_df['produce_source'] = 'GPT4'
    sft_df['task-name'] = '手机日志-'+date
    sft_df['create_time'] = create_date
    sft_df['update_time'] = create_date
    sft_df['create_user'] = 'renhuimin'
    sft_df['update_user'] = 'renhuimin'
    sft_df['is_reviewed'] = '否'
    sft_df['update_content'] = ''
    sft_df['id'] = ['applog-'+date+'-'+str(i) for i in range(len(sft_df))]
    sft_df['turn_id'] = 0

    sft_df.to_csv(folder+f'{date}_sft.csv',index=False)
    return sft_df


def get_merged_sft_jsonl_dpo_csv(start_date_str,end_date_str,out_folder,dpo_outpt,sft_outpt):
    dates = utils.generate_date_list(start_date_str, end_date_str)
    dpo_df_ls = []
    sft_df_ls = []
    for date in dates:
        folder = f'/mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/rag2.0/data/distillation_data/log_data/{date}/'
        cates = ['raw/','extension/']
        dl = []
        for cate in cates:
            files = [f for f in os.listdir(folder+cate+'correct_filter_output/') if '保留.csv' in f and '.ipynb_checkpoints' not in f]
            for file in files:
                df_ = pd.read_csv(folder+cate+'correct_filter_output/'+file)
                df_.columns = [col.lower() for col in df_.columns]
                dl.append(df_)
        df = pd.concat(dl)
        
        dpo_df = gen_dpo_df(df,date,folder)
        sft_df = gen_sft_df(df,date,folder)
    
        dpo_df_ls.append(dpo_df)
        sft_df_ls.append(sft_df)
    
    merged_dpo_df = pd.concat(dpo_df_ls)
    merged_sft_df = pd.concat(sft_df_ls)
    
    # 保存dpo数据
    utils.create_directory(out_folder)
    merged_dpo_df.to_csv(out_folder+dpo_outpt,index=False, encoding='utf-8-sig')
    
    # 保存sft数据
    merged_sft_df.to_csv(out_folder+sft_outpt+'.csv',index=False, encoding='utf-8-sig')
    dataformat_obj = DataFormat(api_flag=True,multi_flag=False)
    merged_sft_jsonl = dataformat_obj.gen_sft_data(merged_sft_df)
    merged_sft_jsonl.to_json(out_folder+sft_outpt+'.jsonl', orient='records', lines=True, force_ascii=False)

    print(f'dpo数量：{len(merged_dpo_df)}, sft数量：{len(merged_sft_df)}')


if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='merge dpo and sft data from app log.')
    
    # 添加参数
    parser.add_argument('--start_date', type=str, help='文件日期，格式为 YYYY-MM-DD')
    parser.add_argument('--end_date', type=str, help='文件日期，格式为 YYYY-MM-DD')
    parser.add_argument('--out_folder', type=str, help='数据存储文件夹')
    parser.add_argument('--dpo_outpt', type=str, help='dpo文件名')
    parser.add_argument('--sft_outpt', type=str, help='sft文件名无后缀')
    args = parser.parse_args()
    print(args)

    get_merged_sft_jsonl_dpo_csv(args.start_date,args.end_date,args.out_folder,args.dpo_outpt,args.sft_outpt)