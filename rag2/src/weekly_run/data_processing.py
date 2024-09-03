import pandas as pd
import numpy as np
import re
import os
import subprocess
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer

"""
公共类，数据筛选及prompt拼接
"""

def get_free_gpu():
    """
    获取当前可用显存最多的 GPU 设备号。
    """
    nvidia_smi_output = subprocess.check_output(
        "nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader", 
        shell=True
    )
    gpu_memory = [int(x) for x in nvidia_smi_output.decode('utf-8').split('\n') if x != '']
    gpu_id = gpu_memory.index(max(gpu_memory))
    return gpu_id

os.environ["CUDA_VISIBLE_DEVICES"] = str(get_free_gpu())


class DataFilter:
    # uid 真实性 obs事实歧义
    @staticmethod
    def get_task_usecols(filter_df):
        # 检查列是否存在，将存在的条件加入条件列表中
        task_conditions = []
        task_choices = []
        
        if 'assistant_truthfulness' in filter_df.columns:
            task_conditions.extend([
                (filter_df['assistant_truthfulness'] == '低'),
                (filter_df['assistant_truthfulness'] == '中')
            ])
            task_choices.extend([
                '真实性差',
                '真实性中'
            ])
            
        if 'assistant_relevance' in filter_df.columns:
            task_conditions.extend([
                (filter_df['assistant_relevance'] == '低'),
                (filter_df['assistant_relevance'] == '中')
            ])
            task_choices.extend([
                '相关性差',
                '相关性中'
            ])
            
        if 'assistant_logic' in filter_df.columns:
            task_conditions.extend([
                (filter_df['assistant_logic'] == '低'),
                (filter_df['assistant_logic'] == '中')
            ])
            task_choices.extend([
                '逻辑性差',
                '逻辑性中'
            ])
            
        if 'observation_has_truthfulness_ambiguity' in filter_df.columns:
            task_conditions.append((filter_df['observation_has_truthfulness_ambiguity'] == 1))
            task_choices.append('obs事实矛盾')

        if 'assiatant_has_hallucination_based_on_obs' in filter_df.columns:
            task_conditions.append((filter_df['assiatant_has_hallucination_based_on_obs'] == 1))
            task_choices.append('回复幻觉基于obs编造')

        if 'assiatant_has_faulty_construction' in filter_df.columns:
            task_conditions.append((filter_df['assiatant_has_faulty_construction'] == 1))
            task_choices.append('回复结构缺陷')
            
        filter_df['task_name'] = np.select(task_conditions, task_choices, default='正常数据')
        
        desired_cols = ['user-query', 'assistant', 'api', 'thought', 'observation', 'system', 'context',
                        'generalized_question_from_user', 'generalized_question_from_assistant',
                        'task_name','uid']
    
        available_cols = filter_df.columns.tolist()
        cols_to_load = [col for col in desired_cols if col in available_cols]
        filter_df = filter_df[cols_to_load]
    
        return filter_df

    @staticmethod
    def filter_bad_df(df, config):
        and_conditions = [
            "(is_chara == 0) | (is_chara == -1)",
            "(is_math == 0) | (is_math == -1)",
            "(is_child == 0) | (is_child == -1)",
            "(is_simplified == 0) | (is_simplified == -1)",
            "(is_guidance == 0) | (is_guidance == -1)",
            "(is_realtime == 0) | (is_realtime == -1)",
            "(query_has_you == 0) | (query_has_you == -1)",
            "(query_has_relative_time == 0) | (query_has_relative_time == -1)",
            "(query_has_command == 0) | (query_has_command == -1)",
            "(source == 'real')",
            "(is_valid_llm == 1)",
            "(is_too_similar_to_train == 0)"
        ]

        if config['is_rag_flag']:
            and_conditions.append("(is_rag == 1) | (is_rag == -1)")
        else:
            and_conditions.append("(is_rag == 0)")

        if config['is_single_flag']:
            and_conditions.append("(is_single_turn == 1) | (is_single_turn == -1)")
        else:
            and_conditions.append("(is_single_turn == 0)")

        or_conditions = [
            "assistant_relevance.isin(['中', '低'])",
            "assistant_logic.isin(['中', '低'])",
            "assistant_truthfulness.isin(['中','低'])",
            "(observation_has_truthfulness_ambiguity == 1)",
            "(assiatant_has_hallucination_based_on_obs == 1)",
            "(assiatant_has_faulty_construction == 1)"
        ]

        existing_columns = set(df.columns)
        def extract_column_names(condition):
            column_regex = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)")
            matches = column_regex.findall(condition)
            return matches

        def is_condition_valid(condition):
            column_names = extract_column_names(condition)
            column_names = [column for column in column_names if 'isin' not in column]
            return all(column in existing_columns for column in column_names)

        valid_and_conditions = ["({})".format(cond) for cond in and_conditions if is_condition_valid(cond)]
        valid_or_conditions = ["({})".format(cond) for cond in or_conditions if is_condition_valid(cond)]

        if valid_and_conditions:
            and_condition_str = " & ".join(valid_and_conditions)
        else:
            and_condition_str = "True"

        if valid_or_conditions:
            or_condition_str = " | ".join(valid_or_conditions)
        else:
            or_condition_str = "True"


        filter_df = df.query(f"({and_condition_str}) & ({or_condition_str})")

        if len(filter_df) == 0:
            target_df = df.query(and_condition_str)
            filter_df = target_df.sample(n=min(len(target_df),100)) # 如果没有有问题的数据，随机返回100条
            
        return filter_df

    @staticmethod
    def get_unique_query(query_ls:list[dict], batch_size=32, BGE_MODEL_PATH='', query_col_name='user-query'):
        """
        计算模型相似性
        query_ls: [{'user-query':''}]
        """
        model = SentenceTransformer(BGE_MODEL_PATH).cuda()

        # 提取 query 文本
        query_texts = [q[query_col_name] for q in query_ls]

        # 分批次处理
        all_embeddings = []
        for i in range(0, len(query_texts), batch_size):
            batch_queries = query_texts[i:i + batch_size]
            batch_embeddings = model.encode(batch_queries, convert_to_tensor=True)
            all_embeddings.append(batch_embeddings.detach().cpu().numpy())
        embeddings = np.vstack(all_embeddings)

        similarity_matrix = cosine_similarity(embeddings)

        threshold = 0.9
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1-threshold,
            metric='cosine',
            linkage='average'
        )
        labels = clustering.fit_predict(embeddings)

        unique_queries = []
        for label in np.unique(labels):
            index = np.where(labels == label)[0][0]
            unique_queries.append(query_ls[index])

        return unique_queries


class PromptConstructor:
    @staticmethod
    def construct_prompt(row, oneshot_prompt, config):
        if config['is_single_flag'] and config['is_rag_flag']:
            return oneshot_prompt + f"""
            ---
            下面是给出的实际问题：
            system:
            {row['system']}
            Observation:
            {row['observation']}
            Question:
            {row['user-query']}
            Answer：
            """
        elif config['is_single_flag'] and not config['is_rag_flag']:
            return oneshot_prompt + f"""
            ---
            下面是给出的实际问题：
            system:
            {row['system']}
            Question:
            {row['user-query']}
            Answer：
            """
        elif not config['is_single_flag'] and config['is_rag_flag']:
            return oneshot_prompt + f"""
            ---
            下面是给出的实际问题：
            system:
            {row['system']}
            Context:
            {row['context']}
            Observation:
            {row['observation']}
            Question:
            {row['user-query']}
            Answer：
            """
        elif not config['is_single_flag'] and not config['is_rag_flag']:
            return oneshot_prompt + f"""
            ---
            下面是给出的实际问题：
            system:
            {row['system']}
            Context:
            {row['context']}
            Question:
            {row['user-query']}
            Answer：
            """