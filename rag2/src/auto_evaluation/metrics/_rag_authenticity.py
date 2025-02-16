# -*-coding:utf-8-*-
from metrics.utils_log_parser import parser_date, parser_loc, parser_obs, get_query_result_from_16b_input, get_context_result_from_16b_input, extract_history, content_parser_functioncall
import re
import sys
import random
import string
import traceback
sys.path.append('../')
from base.base_eval import BaseModelEval
sys.path.append('../../')
from tool_llm_response.call_llm_with_zny import CallLLMByZny,ZnyConfig
from tool_llm_response.call_llm_with_vllm import CallLLMByVllm,VllmConfig
import warnings

class AuthenticityEval(BaseModelEval):
    def __init__(self, task_name):
        super().__init__(task_name)


    def read_prompt(self, path):
        with open(path, "r") as f:
            self.prompt = f.read().strip()
        assert self.prompt not in ['', 'nan'],"prompt is none!"


    def concat_prompt(self, df, eval_column_list, prompt_path): #生成prompts，并存入df prompts列
        '''
        将query, obs, predict_output三列拼入prompt，并存入prompts列
        '''
        self.read_prompt(prompt_path)
        query, obs, ans = eval_column_list
        instruction = "以上就是模型生成的内容，请你根据给出的问题和参考资料，将其切分为多个最小粒度的原子信息并按照相关性划分等级然后进行准确性评估。"
        # 使用apply函数拼接prompt与三列的数据，并存入prompts列
        df['llm_prompts'] = df.apply(lambda row: f"{self.prompt}\n问题：{row[query]}\n参考资料：{row[obs]}\n生成内容：{row[ans]}\n{instruction}", axis=1)
        return df


    def add_log_prompt(self, df, eval_column_list, prompt_path):
        '''
        解析log to date_time, location, context(user), observation, assistant
        1. 提取各个元素
        2. 拼接prompt
        '''
        self.read_prompt(prompt_path)
        log_input_col, query_col, ans_col = eval_column_list
        prompts_ls = []
        for i in range(len(df)):
            request = df.iloc[i][log_input_col]
            date_info = parser_date(request) # 提取时间
            address_info = parser_loc(request) # 提取地点
            common_prompt_info = self.prompt.replace("$$$date$$$",date_info).replace("$$$pos$$$",address_info) # 替换时间地点
            query = df.iloc[i][query_col] # get_query_result_from_16b_input(request)
            context = extract_history(request) # get_context_result_from_16b_input(request)
            obs = str(content_parser_functioncall(request)) # parser_obs(request)
            response = df.iloc[i][ans_col]
            # instruction = "以上就是模型生成的内容，请你根据给出的问题和参资料，将其切分为多个最小粒度的原子信息并按照相关性划分等级然后进行准确性评估。"
            instruction = "以上就是模型生成的内容，请你根据给出的问题和参考答案进行准确性评估。"
            if context != "":
                # prompt = "\n".join([common_prompt_info + "\n", "历史对话：",context, "问题：", query, '参考资料：', obs, '生成内容：', response.strip(), instruction])
                prompt = f"{common_prompt_info}\n历史对话：{context}\n问题：{query}\n参考答案：{obs}\n生成内容：{response.strip()}\n{instruction}"
            else:
                # prompt = f"{common_prompt_info}\n问题：{query}\n参考答案：{obs}\n生成内容：{response.strip()}\n{instruction}"
                prompt = "\n".join([common_prompt_info + "\n", "问题：", str(query), '参考答案：', str(obs), '生成内容：', str(response).strip(), instruction])
            prompts_ls.append(prompt)
        df['llm_prompts'] = prompts_ls
        return df
        

    def result_parse(self,response):
        """
        解析真实性打分
        """
        try:
            # print(response)
            eval_data = re.search("准确性(.*?)兜底", response.replace("\n","").strip())
            # print(eval_data.group(1))
            eval_score = re.findall("【(.*?)】",eval_data.group(1))
            # print(eval_score)
            eval_result = [int(float(x)) for x in eval_score]
            ret = 1
            for ele in eval_result:
                if ele != 1:
                    ret = 0
                    break
            return ret
        except Exception as exc:
            traceback.print_exc()
            return -1
    
    def result_truth_parse(self, response):
        """
        解析真实性打分
        """
        if "<|wrong data|>" in response:
            warnings.warn("注意：当前例子真实性评估不满足要求，为<wrong data>，请检查原因，是否Qwen部署或调用失败。可能导致结果偏低", UserWarning)
            print("注意：当前例子真实性评估不满足要求，为<wrong data>，请检查原因，是否Qwen部署或调用失败。可能导致结果偏低")
            return [-2]

        eval_data = re.search("准确性(.*?)兜底", response.replace("\n","").strip())
        if eval_data == None:
            warnings.warn('警告，Qwen进行当前例子真实性评估无法被解析，请检查原因，常见原因1：max_generate设置过短，评估没有完成；2：评估没有遵循指令。可能导致最终结果偏低', UserWarning)
            print('警告，Qwen进行当前例子真实性评估无法被解析，请检查原因，常见原因1：max_generate设置过短，评估没有完成；2：评估没有遵循指令。可能导致最终结果偏低')
            return [-3]
        
        # 正常情况
        try:
            eval_score = re.findall("【(\d*?)】",eval_data.group(1))
            eval_result = [int(float(x)) for x in eval_score]
            return eval_result
        except Exception as exc:
            traceback.print_exc()
            return []

    def result_truth_parse_todo(self, response):
        """
        解析待定状态的真实性打分
        """
        if "<|wrong data|>" in response:
            warnings.warn("注意：评估不满足要求，为<wrong data>，请检查原因，是否Qwen部署或调用失败。可能导致结果偏低", UserWarning)
            print("注意：评估不满足要求，为<wrong data>，请检查原因，是否Qwen部署或调用失败。可能导致结果偏低")
            return [-2]

        eval_data = re.search("准确性(.*?)兜底", response.replace("\n","").strip())
        if eval_data == None:
            warnings.warn('警告，Qwen评估无法被解析，请检查原因，常见原因1：max_generate设置过短，评估没有完成；2：评估没有遵循指令。可能导致最终结果偏低', UserWarning)
            print('警告，Qwen评估无法被解析，请检查原因，常见原因1：max_generate设置过短，评估没有完成；2：评估没有遵循指令。可能导致最终结果偏低')
            return [-3]
        
        # 正常情况
        try:
            eval_score = re.findall("「(\d*?)」",eval_data.group(1))
            eval_result = [int(float(x)) for x in eval_score]
            return eval_result
        except Exception as exc:
            traceback.print_exc()
            return []

    def parse_backup(self, response):
        """
        解析兜底数据
        """
        if "<|wrong data|>" in response:
            return -2
        try:
            patterns = [r'原子信息：(.*?)是否为兜底回复：', r'是否为兜底回复：【(.*?)】']
            match_backup = re.search(patterns[1], response.replace("\n", ""))
            result = int(match_backup.group(1))
        except Exception as exc:
            traceback.print_exc()
            return -2
        return result
        

    def result_sorted(self, input_with_prompt, responses):
        '''
        根据原始输入文件的prompts列顺序，检查并纠正模型评估的response顺序
        '''
        try:
            prompt_index = {v: k for k, v in enumerate(input_with_prompt)}
            query_list = [x['query'] for x in responses]
            response_list = [x['response'] for x in responses]
            responses_index = [prompt_index[x] for x in query_list]
            response_dict = dict(zip(responses_index, response_list))
            response_sorted = sorted(response_dict.items(), key=lambda x: x[0], reverse=False)
            response_sorted_list = [x[1] for x in response_sorted]
        except Exception as exc:
            traceback.print_exc()
            print("error while result sorted:",e)
        return response_sorted_list
    
    def result_sorted_byindex(self, responses):
        try:
            prompt_index = {x["index"]: x for x in responses}
            response_sorted = sorted(prompt_index.values(), key=lambda x: x["index"], reverse=False)
            response_sorted_list = [x["response"] for x in response_sorted]
        except Exception as exc:
            traceback.print_exc()
            print("error while result sorted:",e)
        return response_sorted_list
    
    def main_eval(self, model: str, url: str, eval_column_list: list[str], df, output_dir: str, prompt_path: str, thread_num: int, chunk_num: int, temperature: float, eval_mode:str = 'user_obs_ans_concat', max_tokens=8000):
        '''
        主评估函数
        '''
        try:
            if eval_mode=='user_obs_ans_concat':
                # 读取待评估文件并与prompt进行拼接
                df_with_prompts = self.concat_prompt(df, eval_column_list, prompt_path)
                query_column_name = "llm_prompts"
            elif eval_mode=='with_prompt':
                df_with_prompts = df
                query_column_name = eval_column_list[0] # 如果不拼已有的prompt默认取第一个做eval
            elif eval_mode=='model_13b_log':
                df_with_prompts = self.add_log_prompt(df, eval_column_list, prompt_path)
                query_column_name = "llm_prompts"
            else:
                raise "目前仅支持user_obs_ans_concat(输入user-query, observation, assistant列后拼接prompt), model_13b_log(输入13b output 后处理拼接prompt), with_prompt(已拼接好prompt)"
            

            index_name = "eval_index-" +  ''.join(random.choice(string.ascii_lowercase) for _ in range(8)) 
            df_with_prompts[index_name] = [i+1 for i in range(df_with_prompts.shape[0])]

            if(model in ["gpt4","gpt4o","wenxin"]):
                config = ZnyConfig(
                    url = url, # 智能云GPT api
                    model_name = model, # system:转为system和user, '':输入的message都是user
                    temperature = temperature, # llm输出温度，zny下的gpt4基本无效，因为是全球节点，还是会有随机性
                    max_retries = 5, # 调用gpt报错后最多重试 max_retries 次
                    qps = 5, # 多线程 or 异步多线程下，qps，不要超过5
                    max_concurrent = 5, # 异步多线程参数，一般10或者20，太大会接口超过qps
                    asyncio_flag = False, # True=异步多线程，只能python调用；False=普通多线程，Jupyter或者python均可
                    query_column_name = query_column_name, # llm模型输入列名
                    response_column_name = 'assistant_89757' # llm模型输出列名
                )
                call_zny = CallLLMByZny(config)
                merged_df = call_zny.get_gpt4api_df(df_with_prompts)
                responses = call_zny.parser_model_response_index(merged_df, index_name)
            # 其余模型
            else:
                config = VllmConfig(
                    model = model,
                    url = url, # 智能云GPT api
                    temperature = temperature,
                    top_p = 0.9,
                    max_tokens = max_tokens, # 最大输出长度
                    chunk_num = chunk_num,
                    thread_num = thread_num,
                    query_column_name = query_column_name, # llm模型输入列名
                    response_column_name = 'assistant_89757'
                )
                call_vllm = CallLLMByVllm(config)
                responses_tmp = call_vllm.model_request(df_with_prompts)
                responses = call_vllm.parser_model_response_index(responses_tmp, index_name) # 格式转化
                
            # 根据原始输入文件，检查responses是否顺序一致，不一致则按顺序对responses进行重排
            # response_sorted_list = self.result_sorted(df_with_prompts[query_column_name].to_list(), responses)
            response_sorted_list = self.result_sorted_byindex(responses)
            df_with_prompts = df_with_prompts.drop(columns=[index_name])
            
            truth_result_list = []
            for resp in response_sorted_list:
                result_truth = self.result_truth_parse(resp) # 解析模型回复
                result_truth_todo = self.result_truth_parse_todo(resp)
                truth_score = 1 if 0 not in result_truth and 0 not in result_truth_todo and result_truth!=[-2] and result_truth!=[-3] and result_truth_todo!=[-2] and result_truth_todo!=[-3] else 0
                backup_score = self.parse_backup(resp) # 兜底回复均为0
                truth_result_tmp = 0 if  truth_score == 0 or backup_score==1 else 1
                # if truth_backup == 1:
                #     truth_result_tmp = 0
                # else:
                #     truth_result_tmp = truth_score
                # truth_result_tmp = 0 if truth_score == 0 or truth_backup==1 or truth_backup==-2 else 1
                truth_result_list.append(truth_result_tmp)
            truth_reason = response_sorted_list
        except Exception as exc:
            traceback.print_exc()
            truth_result_list = [-1 for _ in range(len(df))]
            truth_reason = ['nan' for _ in range(len(df))]
        return truth_result_list, truth_reason

authenticityEval = AuthenticityEval("authenticity_eval")
