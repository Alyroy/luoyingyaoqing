import argparse
import pandas as pd
import os
import json
import traceback
import ast

def create_directory(directory: str):
    """Creates a directory if it does not already exist."""
    os.makedirs(directory, exist_ok=True)

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
    dl=[]
    for i in full_path:
        print(i)
        tmp = pd.read_json(i,lines=True)
        dl.append(tmp)

    df = pd.concat(dl)
    print('读入原始数据总数量',len(df))
    return df


def get_user(df: pd.DataFrame) -> list:
    query_ls = []
    for i in range(len(df)):
        row = df.iloc[i]
        try:
            msg = row['messages']
            if isinstance(msg, str):
                msg = ast.literal_eval(msg)
            query = msg[len(msg)-5]['content'][0] # 取多轮最后一轮query
            query_ls.append(query)
        except Exception as exc:
            traceback.print_exc()
            print(row['messages'])

    return query_ls


if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='distillate gpt4 for app log.')
    
    # 添加参数
    parser.add_argument('--input_folder', type=str, help='input文件夹')
    parser.add_argument('--output_folder', type=str, default='/mnt/pfs-guan-ssai/nlu/renhuimin/data/trained_querys/', help='output文件夹')
    args = parser.parse_args()
    print(args)
    create_directory(args.output_folder)

    df = load_data(args.input_folder)
    query_ls = get_user(df)

    output_file_path = args.output_folder + 'queries.jsonl'
    with open(output_file_path, 'w') as file:
        # 遍历列表中的每个元素，将其按照JSONL格式写入文件
        for item in query_ls:
            file.write(json.dumps({"user": item}, ensure_ascii=False) + "\n")