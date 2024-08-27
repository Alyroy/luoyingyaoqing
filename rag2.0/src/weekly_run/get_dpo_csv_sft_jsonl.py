import argparse
import pandas as pd
import os
import traceback
from datetime import datetime
import sys
sys.path.append('../')
from tool_rag_generation.data_format import DataFormat
sys.path.append('../../')
from common import utils


def remove_br_template(row):
    """
    移除 thought 列中的特定模板内容
    """
    if pd.isna(row['thought']) or row['thought'] == '':
        return f"涉及知识问答，查询{row['user-query']}"
    return row['thought'].replace('该问题请使用总分类模板回复。', '')


def get_dpo_songbiao_df(df,date,outfolder):
    usecols = ['user-query','api','thought','observation','assistant','parser_gpt4','system','task_name','task-name','source','uid','context']
    available_cols = df.columns.tolist()
    cols_to_load = [col for col in usecols if col in available_cols]
    dpo_human_df = df[cols_to_load].rename(columns={'assistant':'modelA','parser_gpt4':'modelB'})
    
    dpo_human_df['thought'] = dpo_human_df.apply(remove_br_template, axis=1)  # thought 去掉总分模板
    dpo_human_df['query是否有效'] = ''
    dpo_human_df['哪个回复好'] = ''
    dpo_human_df['落败原因'] = ''
    dpo_human_df['observation2'] = dpo_human_df['observation'].apply(utils.flatten_and_number)
    dpo_human_df.to_excel(outfolder+f'/{date}_dpo_送标.xlsx',index=False)
    return dpo_human_df


def get_sft_songbiao_df(df,date,outfolder):
    usecols = ['user-query','api','thought','observation','parser_gpt4','system','task_name','task-name','source','uid','context']
    available_cols = df.columns.tolist()
    cols_to_load = [col for col in usecols if col in available_cols]
    sft_df = df[cols_to_load].rename(columns={'parser_gpt4':'assistant'})
    
    sft_df['thought'] = sft_df.apply(remove_br_template, axis=1)  # thought 去掉总分模板
    sft_df['observation2'] = sft_df['observation'].apply(utils.flatten_and_number)
    sft_df['query是否有效'] = ''
    sft_df['obs是否相关'] = ''
    sft_df['回复是否保留'] = ''
    sft_df['废弃原因'] = ''
    
    sft_df.to_excel(outfolder+f'/{date}_sft_送标.xlsx',index=False)
    return sft_df
    

def get_sft_train_df(df,date,outfolder):
    usecols = ['user-query','api','thought','observation','parser_gpt4','system','task_name','task-name','source','uid','context']
    available_cols = df.columns.tolist()
    cols_to_load = [col for col in usecols if col in available_cols]
    sft_df = df[cols_to_load].rename(columns={'parser_gpt4':'assistant'})
    sft_df['thought'] = sft_df.apply(remove_br_template, axis=1)  # thought 去掉总分模板
    
    # 转训练数据
    now = datetime.now()    
    create_date = now.strftime('%Y-%m-%d')
    sft_df['relevant_label'] = '相关'
    sft_df['produce_source'] = 'GPT4'
    sft_df['create_time'] = create_date
    sft_df['update_time'] = create_date
    sft_df['create_user'] = 'renhuimin'
    sft_df['update_user'] = 'renhuimin'
    sft_df['is_reviewed'] = '否'
    sft_df['update_content'] = ''
    sft_df['id'] = ['log-'+date+'-'+str(i) for i in range(len(sft_df))]
    sft_df['turn_id'] = 0

    sft_df.to_csv(outfolder+f'/{date}_sft_train.csv',index=False,encoding='utf-8-sig')
    return sft_df


def merge_dataframes(dates, in_folder, raw_extension_type, single_rag_type, source):
    dpo_songbiao_list = []
    sft_songbiao_list = []
    sft_train_list = []
    
    for date in dates:
        print(date)
        folder = os.path.join(in_folder, date)
        try:
            files = [f for f in os.listdir(os.path.join(folder, raw_extension_type, single_rag_type, "correct_filter_output")) 
                     if '保留.csv' in f and '.ipynb_checkpoints' not in f]
            
            df_list = [pd.read_csv(os.path.join(folder, raw_extension_type, single_rag_type, "correct_filter_output", file)) for file in files]
            combined_df = pd.concat(df_list)
            combined_df.columns = [col.lower() for col in combined_df.columns]
            combined_df['source'] = source + '_' + raw_extension_type + '_' + single_rag_type
            
            if 'raw' in raw_extension_type:
                dpo_df = get_dpo_songbiao_df(combined_df, date, os.path.join(folder, raw_extension_type, single_rag_type))
                dpo_songbiao_list.append(dpo_df)
            
            sft_train_df = get_sft_train_df(combined_df, date, os.path.join(folder, raw_extension_type, single_rag_type))
            sft_train_list.append(sft_train_df)
        except Exception as exc:
            traceback.print_exc()
        
        try:
            files = [f for f in os.listdir(os.path.join(folder, raw_extension_type, single_rag_type, "correct_filter_output")) 
                     if '送标.csv' in f and '.ipynb_checkpoints' not in f]
            
            df_list = [pd.read_csv(os.path.join(folder, raw_extension_type, single_rag_type, "correct_filter_output", file)) for file in files]
            combined_df = pd.concat(df_list)
            combined_df.columns = [col.lower() for col in combined_df.columns]
            combined_df['source'] = source + '_' + raw_extension_type + '_' + single_rag_type
            
            sft_songbiao_df = get_sft_songbiao_df(combined_df, date, os.path.join(folder, raw_extension_type, single_rag_type))
            sft_songbiao_list.append(sft_songbiao_df)
        except Exception as exc:
            traceback.print_exc()
    
    return dpo_songbiao_list, sft_songbiao_list, sft_train_list
    

def get_merged_sft_jsonl_dpo_csv(start_date_str, end_date_str, in_folder, out_folder, raw_extension_type, single_rag_type, dpo_outpt, sft_outpt, source='livis日志回流'):
    dates = utils.generate_date_list(start_date_str, end_date_str)
    
    dpo_songbiao_list, sft_songbiao_list, sft_train_list = merge_dataframes(dates, in_folder, raw_extension_type, single_rag_type, source)
    
    merged_dpo_df_songbiao = pd.concat(dpo_songbiao_list) if dpo_songbiao_list else pd.DataFrame()
    merged_sft_songbiao = pd.concat(sft_songbiao_list)
    merged_sft_df_train = pd.concat(sft_train_list)
    
    utils.create_directory(out_folder)
    
    if not merged_dpo_df_songbiao.empty:
        merged_dpo_df_songbiao.to_excel(os.path.join(out_folder, dpo_outpt+'.xlsx'), index=False)
    
    if not merged_sft_songbiao.empty:
        merged_sft_songbiao.to_excel(os.path.join(out_folder, f'{sft_outpt}_送标.xlsx'), index=False)
    
    merged_sft_df_train.to_csv(os.path.join(out_folder, f'{sft_outpt}_train.csv'), index=False, encoding='utf-8-sig')
    
    dataformat_obj = DataFormat(api_flag=True, multi_flag=False)
    merged_sft_jsonl = dataformat_obj.gen_sft_data(merged_sft_df_train)
    merged_sft_jsonl.to_json(os.path.join(out_folder, f'{sft_outpt}.jsonl'), orient='records', lines=True, force_ascii=False)

    print(f'dpo送标数量：{len(merged_dpo_df_songbiao)} , sft送标数量：{len(merged_sft_songbiao)}, sft训练数量：{len(merged_sft_df_train)}')
    

if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='merge dpo and sft data from app log.')
    
    # 添加参数
    parser.add_argument('--start_date', type=str, help='文件日期，格式为 YYYY-MM-DD')
    parser.add_argument('--end_date', type=str, help='文件日期，格式为 YYYY-MM-DD')
    parser.add_argument('--in_folder', type=str, default='/mnt/pfs-guan-ssai/nlu/renhuimin/data/log_distillation/', help='数据读取文件夹，截止到日期前')
    parser.add_argument('--out_folder', type=str, default='/mnt/pfs-guan-ssai/nlu/renhuimin/rag_tool/data/routine_label_data/', help='数据存储文件夹')
    parser.add_argument('--raw_extension_type', type=str, default='raw', help='raw or extension')
    parser.add_argument('--single_rag_type', type=str, default='single_True_rag_True', help='single rag 类型')
    parser.add_argument('--dpo_outpt', type=str, help='dpo文件名')
    parser.add_argument('--sft_outpt', type=str, help='sft文件名无后缀')
    parser.add_argument('--source', type=str, default='livis日志回流', help='livis car feishu')
    args = parser.parse_args()
    print(args)

    get_merged_sft_jsonl_dpo_csv(args.start_date, args.end_date,args.in_folder, args.out_folder, args.raw_extension_type, args.single_rag_type, args.dpo_outpt, args.sft_outpt, args.source)