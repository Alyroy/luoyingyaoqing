# -*-coding:utf-8-*-
import re
import sys
sys.path.append('../')
from base.base_eval import BaseModelEval

sys.path.append('../../')
from tool_llm_response.call_llm_with_zny import CallLLMByZny,ZnyConfig
from tool_llm_response.call_llm_with_vllm import CallLLMByVllm,VllmConfig

class RelevanceEval(BaseModelEval):
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
        
        # 使用apply函数拼接prompt，并存入prompts列
        df['llm_prompts'] = df.apply(lambda row: f"{self.prompt}【数据输入】\n【query】\n{row[query]}\n【response】\n{row[ans]}\n【结果输出】", axis=1)
        return df

    def result_parse(self,response):
        if "<|wrong data|>" in response:
            return -2
        try:
            responses_parser = re.findall(r"{{(.*?)}}", response, re.S)
            score_str = responses_parser[0] if len(responses_parser) == 3 else -1
            score = float(score_str)
            if score == -1:
                rel_pred = -1
            elif score > 3:
                rel_pred = 1
            else:
                rel_pred = 0
            # rel_pred = 1 if score > 3 else 0 # 评分为1-5，大于3则评估为相关，否则不相关
        except:
            self.logger.error("[rel-eval]:结果解析失败！")
            rel_pred = -1
        return rel_pred


    def parse_backup(self,response):
        """
        解析兜底数据
        """
        if "<|wrong data|>" in response:
            return -2
        try:
            patterns = [r'原子信息：(.*?)是否为兜底回复：', r'是否为兜底回复：【(.*?)】']
            match_backup = re.search(patterns[1], response.replace("\n", ""))
            result = int(match_backup.group(1))
        except:
            return -2
        return result
        

    def result_sorted(self, input_with_prompt, responses):
        try:
            prompt_index = {v: k for k, v in enumerate(input_with_prompt)}
            query_list = [x['query'] for x in responses]
            response_list = [x['response'] for x in responses]
            responses_index = [prompt_index[x] for x in query_list]
            response_dict = dict(zip(responses_index, response_list))
            response_sorted = sorted(response_dict.items(), key=lambda x: x[0], reverse=False)
            response_sorted_list = [x[1] for x in response_sorted]
        except Exception as e:
            self.logger.error("error while result sorted:",e)
        return response_sorted_list

    def main_eval(self, model: str, url: str, eval_column_list: list[str], df, output_dir: str, prompt_path: str, thread_num: int, chunk_num: int, temperature: float, concat_prompt_flag:bool = True):
        '''
        主评估函数
        '''
        try:
            if concat_prompt_flag:
                # 读取待评估文件并与prompt进行拼接
                df_with_prompts = self.concat_prompt(df = df, eval_column_list = eval_column_list, prompt_path = prompt_path)
                query_column_name = "llm_prompts"
            else:
                df_with_prompts = df
                query_column_name = eval_column_list[0] # 如果不拼已有的prompt默认取第一个做eval
                
            if(model in ["gpt4","gpt4o","wenxin"]):
                config = ZnyConfig(
                    url = url, # 智能云GPT api
                    model_name = model, # system:转为system和user, '':输入的message都是user
                    temperature = temperature, # llm输出温度，zny下的gpt4基本无效，因为是全球节点，还是会有随机性
                    max_retries = 5, # 调用gpt报错后最多重试 max_retries 次
                    qps = 5, # 多线程 or 异步多线程下，qps，不要超过5
                    max_concurrent = 20, # 异步多线程参数，一般10或者20，太大会接口超过qps
                    asyncio_flag = False, # True=异步多线程，只能python调用；False=普通多线程，Jupyter或者python均可
                    query_column_name = query_column_name, # llm模型输入列名
                    response_column_name = 'assistant_89757' # llm模型输出列名
                )
                call_zny = CallLLMByZny(config)
                merged_df = call_zny.get_gpt4api_df(df_with_prompts)
                responses = call_zny.parser_model_response(merged_df)
            # 其余模型
            else:
                config = VllmConfig(
                    model = model,
                    url = url, # 智能云GPT api
                    temperature = temperature,
                    top_p = 0.9,
                    max_tokens = 2048, # 最大输出长度
                    chunk_num = chunk_num,
                    thread_num = thread_num,
                    query_column_name = query_column_name, # llm模型输入列名
                    response_column_name = 'assistant_89757'
                )
                call_vllm = CallLLMByVllm(config)
                responses_tmp = call_vllm.model_request(df_with_prompts)
                responses = call_vllm.parser_model_response(responses_tmp) # 格式转化
                
            # 根据原始输入文件，检查responses是否顺序一致，不一致则按顺序对responses进行重排
            response_sorted_list = self.result_sorted(df_with_prompts[query_column_name].to_list(), responses)
            # rel_result = [self.result_parse(resp) for resp in response_sorted_list]
            rel_result = []
            for resp in response_sorted_list:
                rel_score = self.result_parse(resp) # 解析模型回复
                rel_backup = self.parse_backup(resp) # 兜底回复均为0
                rel_result_tmp = 0 if rel_score == 0 or rel_backup==1 else 1
                rel_result.append(rel_result_tmp)
            rel_reason = response_sorted_list
        except Exception as e:
            rel_result = [-1 for _ in range(len(df))]
            rel_reason = ['nan' for _ in range(len(df))]
            self.logger.error("error while relevance eval:{}".format(e))
        return rel_result,rel_reason

relResponseEval = RelevanceEval("relevance_eval")