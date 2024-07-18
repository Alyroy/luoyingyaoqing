import json
import requests
import ast

class ApiResult(object):
    def __init__(self, category: str, query: str, url: str = "http://172.24.139.95:16073/ligpt_with_api/search") -> None:
        """
        category:"QASearch","AUTOSearch","AIPainting","TaskMaster","MEDIASearch","other_vision","other_origin"
        query: str
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
          "query": [self.query]
        })
        headers = {
          'Content-Type': 'application/json'
        }
        
        response = requests.request("POST", self.url, headers=headers, data=payload)
        res_json = json.loads(response.text)
        
        return res_json


def get_thought_api(query, category, url):
    """
    解析1b接口
    """
    try:
        api_tool = ApiResult(category, query, url)
        response = api_tool.gen_api()
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
        return thought_ls, api_ls
    except Exception as e:
        print(f"Error occurred: {e}")
        return [], []
    
