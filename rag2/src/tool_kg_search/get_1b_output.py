import json
import requests
import ast

class LinluResult(object):
    def __init__(self,query:str,url:str='http://nlu-lids-server-50-inference.ssai-apis-staging.chj.cloud/cloud/inner/nlp/lids/nlu_engine/parse_v2') -> None:
        """
        输入query，可输出落域分类
        """
        self.query = query
        self.url = url

    def gen_domain(self) -> dict:
        """
        获取linlu服务内容
        """
        payload = json.dumps({
          "input": {
            "text": self.query
          },
          "metadata": {
            "voiceVersion": "5.0.0.0"
          },
          "sessionContextsV2": [
            {
              "inputText": "",
              "actResults": [],
              "messageIds": [],
              "thoughtChains": [],
              "structDisplayDatas": []
            }
          ],
          "debugInfo": {},
          "latestConversationLogs": [],
          "sessionContexts": [],
          "staticsCarData": {},
          "signalInfos": [],
          "nluClarifies": []
        })
        headers = {
          'Content-Type': 'application/json'
        }
        
        response = requests.request("POST", self.url, headers=headers, data=payload)
        res_json = json.loads(response.text)
        return res_json
        

    def get_api_name(self) -> str:
        """
        解析api_name
        """
        res_json = self.gen_domain()
        raw_domain = res_json['dialog_acts'][0]['dass'][0]['domain']
        domain_dict = {'qa':'QASearch','autoqa':'AUTOSearch','mediaqa':'MEDIASearch'}
        domain_value = domain_dict.get(raw_domain, '')
        return domain_value


class ApiResult(object):
    def __init__(self, category: str, query: list, url: str = "http://172.24.139.47:16073/ligpt_with_api/search") -> None:
        """
        category:"QASearch","AUTOSearch","AIPainting","TaskMaster","MEDIASearch","other_vision","other_origin"
        query: list
        """
        self.category = category
        self.query = query
        self.url = url
        
    def gen_api(self) -> dict:
        """
        调用1B模型服务，快速生成API
        Param:
            category:"QASearch","AUTOSearch","AIPainting","TaskMaster","MEDIASearch","other_vision","other_origin"
            query: str
        Returns:
            res_json: 调接口返回的原始内容
        """
        payload = json.dumps({
          "prompt_id": self.category,
          "query": self.query
        })
        headers = {
          'Content-Type': 'application/json'
        }
        
        response = requests.request("POST", self.url, headers=headers, data=payload)
        res_json = json.loads(response.text)
        
        return res_json


def get_thought_api(query, url):
    """
    解析1b接口
    """
    if isinstance(query, str):
        query = [query]
    try:
        # 获取apiname
        linlu_tool = LinluResult(query[-1])
        category = linlu_tool.get_api_name()

        # 获取1b结果
        api_tool = ApiResult(category, query, url)
        response = api_tool.gen_api()
        assistant = response['first_response']['assistant']
        parser_result = response.get('first_response', {}).get('assistant', None)
        
        if not parser_result:
            raise ValueError("No valid response from API")
        
        parser_result = ast.literal_eval(parser_result)

        thought_ls = []
        api_ls = []

        for value in parser_result:
            thought_ls.append(value['thought'])
            api_dict = value['arguments']
            api_dict['name'] = value['name']
            api_ls.append(api_dict)
        return thought_ls, api_ls, response, assistant
    except Exception as e:
        print(f"Error occurred: {e}")
        return [], [], [], []
