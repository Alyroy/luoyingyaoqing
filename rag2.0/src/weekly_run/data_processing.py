import pandas as pd
import numpy as np
import re


"""
公共类，数据筛选及prompt拼接
"""

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
            
        filter_df['task_name'] = np.select(task_conditions, task_choices, default=None)
        
        desired_cols = ['user-query', 'assistant', 'api', 'thought', 'observation', 'system', 'context',
                        'generalized_question_from_user', 'generalized_question_from_assistant','query_synonym',
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
            "(observation_has_truthfulness_ambiguity == 1)"
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

        return df.query(f"({and_condition_str}) | ({or_condition_str})")


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