import argparse
import sys
import os
import subprocess
import traceback
import pandas as pd
import numpy as np
import json
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
from data_processing import DataFilter
from base_distillation import BaseDistillation
sys.path.append('../')
from tool_kg_search.get_api_obs import get_api_df, get_obs_df
from tool_llm_response.call_llm_with_zny import CallLLMByZny, ZnyConfig
sys.path.append('../../')
from common import utils


def get_free_gpu():
    """
    获取当前可用显存最多的 GPU 设备号。
    """
    # 使用 nvidia-smi 命令获取 GPU 信息
    nvidia_smi_output = subprocess.check_output(
        "nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader", 
        shell=True
    )
    # 解析输出，获取每个 GPU 的可用显存
    gpu_memory = [int(x) for x in nvidia_smi_output.decode('utf-8').split('\n') if x != '']
    # 找到可用显存最多的 GPU 索引
    gpu_id = gpu_memory.index(max(gpu_memory))
    return gpu_id
os.environ["CUDA_VISIBLE_DEVICES"] = str(get_free_gpu())


class GPT4DistillationExtended(BaseDistillation):
    def __init__(self, config):
        super().__init__(config)

    def split_you_time_command_queries(self,query_ls:list) -> (list, list[dict]):
        """
        检查query里是否包含时效性、任务型、人设，去掉这类query，以免污染模型
        """
        clean_queries, you_command_time_queries = [], []
        time_word_list = ['今天', '昨天', '明天', '本周', '上周', '今年', '明年', '去年', '最近', '目前', '现在']
        command_pattern = "^(播放|我想听|我想看|我要听|我要看|打开|关闭|找一下|找一家|搜一下|看一下|放个|停止|开始)"

        for query in query_ls:
            bad_query_dict = {'query': query}
    
            # 检查时效性
            if any(time_word in query for time_word in time_word_list):
                bad_query_dict['tag'] = '时效性'
            # 检查人设
            elif '你' in query:
                bad_query_dict['tag'] = '人设你'
            # 检查任务型
            elif re.search(command_pattern, query):
                bad_query_dict['tag'] = '任务型'
            else:
                clean_queries.append(query)
                continue
                
            you_command_time_queries.append(bad_query_dict)

        return clean_queries, you_command_time_queries
            
        
        
    def get_query_ls(self, raw_query_ls):
        query_ls = []
        for q in raw_query_ls:
            query_ls.extend(q.split('\n'))
        query_ls = [q for q in query_ls if "-" not in q and "**" not in q and q != "" and "问题：" not in q]
        query_ls = list(set(query_ls))
        return query_ls

    def get_new_query_ls(self, df):
        query_ls1 = self.get_query_ls(df['generalized_question_from_user'].to_list())
        query_ls2 = self.get_query_ls(df['generalized_question_from_assistant'].to_list())
        query_ls = query_ls1 + query_ls2
        return query_ls

    def get_unique_query(self, query_ls, batch_size=32):
        # 1. 文本表示
        BGE_MODEL_PATH = self.config['embedding_model_path']
        model = SentenceTransformer(BGE_MODEL_PATH).cuda()
        
        # 分批次处理
        all_embeddings = []
        for i in range(0, len(query_ls), batch_size):
            batch_queries = query_ls[i:i + batch_size]
            batch_embeddings = model.encode(batch_queries, convert_to_tensor=True)
            all_embeddings.append(batch_embeddings.detach().cpu().numpy())
        embeddings = np.vstack(all_embeddings)
        
        # 2. 计算相似度
        similarity_matrix = cosine_similarity(embeddings)
    
        # 3. 使用聚类算法来识别不相似的query
        threshold = 0.9  # 设定相似度阈值，小于这个值的query相似的
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1-threshold,  # 将距离阈值设为1-相似度阈值
            metric='cosine',
            linkage='average'
        )
        labels = clustering.fit_predict(embeddings)
        
        # 取每个簇第一个元素
        unique_queries = []
        for label in np.unique(labels):
            index = np.where(labels == label)[0][0]
            unique_queries.append(query_ls[index])
            
        return unique_queries

    def get_distillation_data(self, date):
        print(f"Processing distillation data for date: {date}")

        is_single_flag = self.config['is_single_flag']
        is_rag_flag = self.config['is_rag_flag']
        
        infolder = os.path.join(self.config['base_input_path'], date)
        outfolder = os.path.join(self.config['base_output_path'], date, f'extension/single_{str(is_single_flag)}_rag_{str(is_rag_flag)}')

        print(f"Input folder: {infolder}")
        print(f"Output folder: {outfolder}")
        utils.create_directory(outfolder)

        files = [f'prod/{date}_rule_labeled.csv.gpt_labeled.csv']
        dl = []

        for file in files:
            file_path = os.path.join(infolder, file)
            print(f"Reading input file: {file_path}")
            try:
                df = pd.read_csv(file_path)
                df = df[~df['user-query'].isna()]
                filter_df = DataFilter.filter_bad_df(df, self.config)
                filter_df = DataFilter.get_task_usecols(filter_df)
                dl.append(filter_df)
            except Exception as exc:
                print(f"Exception occurred while processing file: {file_path}")
                traceback.print_exc()

        if not dl:
            print("Data list is empty, no valid dataframes read.")
            return

        df = pd.concat(dl)
        df.to_csv(os.path.join(outfolder, f'{date}_log_data.csv'), index=False)

        query_ls = self.get_new_query_ls(df)
        clean_queries, you_command_time_queries_dict = self.split_you_time_command_queries(query_ls)

        # 将坏查询写入 jsonl 文件
        output_jsonl_file = os.path.join(outfolder, f'{date}_时效人设任务型query.jsonl') 
        with open(output_jsonl_file, 'w', encoding='utf-8') as file:
            for item in you_command_time_queries_dict:
                file.write(json.dumps(item, ensure_ascii=False) + '\n')

        # 符合要求查询，继续生成api thought obs gpt4
        unique_queries = self.get_unique_query(clean_queries,batch_size=64)
        query_df = pd.DataFrame(unique_queries, columns=['user-query'])
        df_api = get_api_df(query_df, col_query='user-query', output_file=os.path.join(outfolder, f'{date}_api.csv'), url=self.config['api_url'])
        df_api = df_api[df_api['api'] != '[]']
        df_obs = get_obs_df(df_api, length_limit=20000, output_path=os.path.join(outfolder, f'{date}_obs.csv'))
        df_obs = pd.read_csv(os.path.join(outfolder, f'{date}_obs.csv'))

        df_obs = df_obs[~df_obs['observation'].isin(['[[]]', '[]'])]
        system = ("你是一个名字叫做理想同学的AI数字生命体。\n"
                  "理想同学是一个可靠的智能家庭助手，由理想汽车智能空间部门创造。"
                  "理想同学使用了理想公司自研MindGPT大语言模型技术。\n"
                  "理想同学能够理解人类的指令和意图，并且给出合理的、切合问题的、没有歧视、"
                  "中立的、安全的回复。\n请根据以下文本写一个合适的回复。")
        df_obs['system'] = system
        df_obs['task-name'] = 'gpt4泛化query'
        utils.split_and_save_df(df_obs, chunk_size=500, outfolder=os.path.join(outfolder, 'split_data'))

        prompt_path = self.get_prompt_path()
        self.check_and_generate_gpt4_files(outfolder, prompt_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distillate GPT-4 for app extended log.')
    parser.add_argument('--date', type=str, help='文件日期，格式为 YYYY-MM-DD')
    parser.add_argument('--api_url', type=str, help='API调用url')
    parser.add_argument('--embedding_model_path', type=str, default='/mnt/pfs-guan-ssai/nlu/wangxiaoyuan/bge-base-zh',help='bge路径')
    parser.add_argument('--is_single_flag', dest='is_single_flag', action='store_true', help='是否为单轮数据')
    parser.add_argument('--no_is_single_flag', dest='is_single_flag', action='store_false', help='是否为多轮数据')
    parser.add_argument('--is_rag_flag', dest='is_rag_flag', action='store_true', help='是否为rag数据')
    parser.add_argument('--no_is_rag_flag', dest='is_rag_flag', action='store_false', help='不是rag数据')
    parser.add_argument('--model_url', type=str, help='模型URL')
    parser.add_argument('--model_name', type=str, help='模型名称')
    parser.add_argument('--prompt_path', type=str, help='提示路径')
    parser.add_argument('--base_output_path', type=str, default='/mnt/pfs-guan-ssai/nlu/renhuimin/data/log_distillation/', help='基础输出路径')

    parser.set_defaults(is_single_flag=True, is_rag_flag=True)
    args = parser.parse_args()
    print(f"Arguments: {args}")

    config = {
        'is_single_flag': args.is_single_flag,
        'is_rag_flag': args.is_rag_flag,
        'model_name': args.model_name,
        'api_url': args.api_url,
        'embedding_model_path': args.embedding_model_path,
        'prompt_path': args.prompt_path,
        'zny_config': ZnyConfig(
            url=args.model_url,
            model_name=args.model_name,
            temperature=0.5,
            max_retries=5,
            qps=3,
            max_concurrent=10,
            asyncio_flag=False,
            query_column_name='prompts',
            response_column_name='gpt4response'
        ),
        'base_input_path': '/mnt/pfs-guan-ssai/nlu/wangxiaoyuan/online-data/',
        'base_output_path': args.base_output_path
    }

    distillator = GPT4DistillationExtended(config)
    distillator.get_distillation_data(args.date)