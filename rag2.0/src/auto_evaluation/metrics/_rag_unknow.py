# -*-coding:utf-8-*-
import re
import sys
sys.path.append('../')
from base.base_eval import BaseModelEval

sys.path.append('../../')
from tool_llm_response.call_llm_with_zny import CallLLMByZny,ZnyConfig
from tool_llm_response.call_llm_with_vllm import CallLLMByVllm,VllmConfig

class UnknowEval(BaseModelEval):
    """
    评估基类：
    @read_prompt:prompt加载
    @main_eval:评估主函数体
    @result_parse:评估结果解析
    """
    def __init__(self, task_name):
        super().__init__(task_name)


    def read_prompt(self, path):
        # 读取prompt
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
        df['llm_prompts'] = df.apply(lambda row: f"{self.prompt}\n---\nQuestion: {row[query]}\nBackground:{row[obs]} \nAnswer: {row[ans]}\n你的回答：", axis=1)
        return df

    
    def result_parse(self, response):
        result = -1
        if "<|wrong data|>" in response:
            return result
        
        try:
            # 使用正则表达式匹配相关性、真实性和准确性得分
            relevance_match = re.search(r"兜底得分: \{\{(.*?)\}\}", response, re.S)
            if relevance_match:
                result = float(relevance_match.group(1))
            
        except Exception as e:
            self.logger.error(f"[rel-eval]:结果解析失败！错误信息: {e}")
        
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
        except Exception as e:
            self.logger.error("error while result sorted:",e)
        return response_sorted_list


    def main_eval(self, model: str, url: str, eval_column_list: list[str], df, output_dir: str, prompt_path: str, thread_num: int, chunk_num: int, temperature: float):
        '''
        主评估函数
        '''
        try:
            # 读取待评估文件并与prompt进行拼接
            df_with_prompts = self.concat_prompt(df = df, eval_column_list = eval_column_list, prompt_path = prompt_path)
            # 读取待评估文件并与prompt进行拼接
            # 如果是gpt4
            # 由于将拼接好的prompt存入了prompts列，因此取query_column_name = "llm_prompts"
            if(model == "gpt4" or model == "wenxin"):
                responses = get_gpt4api_df(df_with_prompts, query_column_name = "prompts", message_type='', max_request_times = 5, qps=5, max_concurrent=10, asyncio_flag=False, url=url)
            # 其余模型
            else:
                responses = model_request(
                    model = model, 
                    url = url,
                    df = df_with_prompts,
                    thread_num = thread_num,
                    chunk_num = chunk_num,
                    temperature = temperature
                )
            # 根据原始输入文件，检查responses是否顺序一致，不一致则按顺序对responses进行重排
            response_sorted_list = self.result_sorted(df_with_prompts['prompts'].to_list(), responses)
            rel_result = [self.result_parse(resp) for resp in response_sorted_list] # 只保留准确性
            rel_reason = response_sorted_list
        except Exception as e:
            rel_result = [-1 for _ in range(len(df))]
            rel_reason = ['nan' for _ in range(len(df))]
            self.logger.error("error while unkown eval:{}".format(e))
        return rel_result,rel_reason
        
unknowEval = UnknowEval("unknow_eval")