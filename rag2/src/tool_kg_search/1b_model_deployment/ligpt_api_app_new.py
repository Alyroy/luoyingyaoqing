import traceback

from flask import Flask, request, jsonify, Response
import re
import json
from collections import OrderedDict
import torch
from ligpt_inference_func import *

import logging
from time import strftime, localtime
import math
from math import *

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_MIMETYPE'] = "application/json;charset=utf-8"

logging.basicConfig(
    level=logging.DEBUG,  # 控制台打印的日志级别
    filename='log/ligpt_api-{}.log'.format(strftime('%Y-%m-%d_%H', localtime())),  # 将日志写入.log文件中
    filemode='a',  # 模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志；a是追加模式，默认如果不写的话，就是追加模式
    format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'  # 日志格式
)


def extract_function_call(text):
    # 使用正则表达式匹配从```function_call到```之间的内容
    if "<None>" in text:
        return "<None>"
    pattern = r'```function_call\n(.*?)\n```'
    match =  matches = re.findall(pattern, text, re.DOTALL)
    return match[0]

@app.route('/ligpt_with_api/search', methods=['POST', 'GET'])
def search():
    query = ''
    prompt_id = ''
    try:
        if request.method == 'POST':
            if request.content_type.startswith('application/json'):
                query = request.json.get('query')
                prompt_id = request.json.get('prompt_id')
                # last_func = request.json.get('last_func')
                # last_func_answer = request.json.get('last_func_answer')
            else:
                query = request.values.get('query')
                prompt_id = request.values.get('prompt_id')
                # last_func = request.json.get('last_func')
                # last_func_answer = request.json.get('last_func_answer')
        if request.method == 'GET':
            query = request.args.get('query')

        prompt_pre_text = prompt_map.get(prompt_id, prompt_map.get("other_origin"))
        final_response = OrderedDict()
        if query:
            if request.method == 'POST' and type(query) == list:
                if len(query) >= 1:
                    input_text = ''
                    for i in range(len(query)):
                        temp = query[i].replace('\\n', '\n')
                        if i % 2 == 0:
                            input_text += '<|lc_start|>user\n{}<|lc_end|>\n'.format(temp)
                        else:
                            input_text += '<|lc_start|>assistant\n{}<|lc_end|>\n'.format(temp)
                else:
                    input_text = '没获取到输入'
                    input_text = '<|lc_start|>user\n{}<|lc_end|>\n'.format(input_text)
            else:
                input_text = query.replace('\\n', '\n')
                input_text = '<|lc_start|>user\n{}<|lc_end|>\n'.format(input_text)

            input_text_list = [input_text]
            
            input_prompt_text_list = []
            for text in input_text_list:
                text = prompt_pre_text + text
                for key, value in key_to_token.items():
                    text = text.replace(key, value)
                input_prompt_text_list.append(text)
            
            # # 首次inference
            # if last_func:
            #     input_text = input_prompt_text_list[0]
            #     function_call = "```function_call\n{0}\n```\n".format(json.dumps(last_func, ensure_ascii=False))
            #     function_call_res = "```function_call_result\n{0}\n```\n".format(json.dumps(last_func_answer, ensure_ascii=False))
            #     input_text = input_text + "[unused0]assistant\n{0}[unused1]\n".format(function_call)
            #     # input_text = input_text+"[unused0]user\n{0}[unused1]\n[unused0]assistant\n".format(function_call_res)
            #     input_text = input_text+"[unused0]user\n{0}[unused1]\n[unused0]\n".format(function_call_res)
            # else:
            #     input_text = input_prompt_text_list[0]
            #     input_prompt_text_list = [input_text+"[unused0]assistant\n"]
            
            # print(input_text_list)
            print("模型的输入######",input_prompt_text_list)
            response = inference(input_text_list, input_prompt_text_list, max_length=5024)
            print("模型输出¥¥¥¥¥",response)
            response_new = response
            for key, value in key_to_token.items():
                response_new = response_new.replace(value, key)
            logging.info('初始结果======\n{}'.format(response_new))

            res1 = OrderedDict()
            res1['prompt_ids'] = list(prompt_map.keys())
            res1['model_prompt_input'] = input_prompt_text_list
            if re.findall('<\|lc_start\|>user[\n]*([\s\S]*?)<\|lc_end\|>', response_new):
                res1['user'] = re.findall('<\|lc_start\|>user[\n]*([\s\S]*?)<\|lc_end\|>', response_new)[-1]
            if re.findall('<\|lc_start\|>thought[\n]*([\s\S]*?)<\|lc_end\|>', response_new):
                res1['thought'] = re.findall('<\|lc_start\|>thought[\n]*([\s\S]*?)<\|lc_end\|>', response_new)[0]
            if re.findall('<\|lc_start\|>api[\n]*([\s\S]*?)<\|lc_end\|>', response_new):
                res1['api'] = re.findall('<\|lc_start\|>api[\n]*([\s\S]*?)<\|lc_end\|>', response_new)[0]

            thought = res1.get('thought', '<None>')
            if '<|lc_start|>api' in response_new and len(re.findall('<\|lc_start\|>user', response_new)) != len(
                    re.findall('<\|lc_start\|>assistant', response_new)):
                if re.findall('<\|lc_start\|>api[\n]*([\s\S]*?)<\|eoa\|>', response_new):
                    api = re.findall('<\|lc_start\|>api[\n]*([\s\S]*?)<\|eoa\|>', response_new)[0]
                # if re.findall('<\|lc_start\|>api[\n]*([\s\S]*?)<\|lc_end\|>', response_new):
                #     api = re.findall('<\|lc_start\|>api[\n]*([\s\S]*?)<\|lc_end\|>', response_new)[0]
                else:
                    api = '<None>'
                logging.info(api)
                if re.findall('APINAME=>([\s\S]*?)<\|kve\|>', api):
                    api_name = re.findall('APINAME=>([\s\S]*?)<\|kve\|>', api)[0].strip()
                else:
                    api_name = ''
                logging.info(api_name)
                # 交互式手动输入
                # observation_input = input('请输入【{}】API observation的结果=>'.format(search_query))
                search_results = None
                if api_name == 'QASearch':
                    observation_response = {}
                    try:
                        category = re.findall('CATEGORY=>([\s\S]*?)<\|kve\|>', api)[0].strip()
                        logging.info(category)
                        search_query = re.findall('QUERY=>([\s\S]*?)<\|kve\|>', api)[0].strip()
                        logging.info(search_query)
                        tag = re.findall('TAG=>([\s\S]*?)<\|kve\|>', api)[0].strip()
                        tag = tag.split('&')
                        logging.info(tag)
                        # google search结果页top
                        # observation_response = request_google_search(search_query)
                        # if '人物' in category:
                        logging.info('>>>>请求知识搜索引擎结果......')
                        logging.info(input_text)
                        raw_query = re.findall('<\|lc_start\|>user[\n]*([\s\S]*?)<\|lc_end\|>', input_text)[0].strip()
                        # observation_response = request_knowledge_engine(raw_query, category, search_query, tag)
                        observation_response = {}
                        logging.info(
                            '>>>>observation结果=={}'.format(observation_response.get('observation', '<None>')))
                        observation = '<|lc_start|>observation\n<|kvs|>%s<|kve|><|lc_end|>\n' % (
                            observation_response.get('observation', '<None>'))
                        # else:
                        #     thought = '<None>'
                        #     api = '<None>'
                        #     logging.info('>>>>非人物问题......')
                        #     observation = '<|lc_start|>observation\n<None><|lc_end|>\n'
                    except:
                        logging.info('>>>>请求搜索引擎结果执行异常')
                        observation = '<|lc_start|>observation\n<None><|lc_end|>\n'

                    search_results = observation_response.get('search_result', [])
                elif api_name == 'Calculate':
                    # 计算器
                    try:
                        key = re.findall('\[KEY\](.*?)<eo', api)[0].strip()
                        logging.info(key)
                        logging.info('>>>>请求计算器结果......')
                        observation_response = '{APIResults=>%s}' % eval(key)
                    except:
                        logging.info('>>>>请求计算器结果执行异常')
                        observation_response = ''

                    logging.info('>>>>observation结果=={}'.format(observation_response))
                    observation = 'observation:{}<eoo>\n'.format(observation_response)
                else:
                    observation = '<|lc_start|>observation\n<None><|lc_end|>\n'

                # first_input = 'user:' + re.findall('user:([\s\S]*)<eou>', response)[
                #     0] + '<eou>\n' + thought + '\n' + api + '\n'
                # first_input = '<|lc_start|>user\n' + re.findall('<\|lc_start\|>user\n([\s\S]*?)<\|lc_end\|>\n', response)[0]

                first_input = response if response.endswith('\\n') else response + '\n'
                api_input = first_input + observation
                # 二次inference
                input_text_list = [api_input]
                input_prompt_text_list = []
                for text in input_text_list:
                    # text = prompt_pre_text + text
                    for key, value in key_to_token.items():
                        text = text.replace(key, value)
                    input_prompt_text_list.append(text)
                logging.info('input_prompt_text_list:{}'.format(input_prompt_text_list))
                response = inference(input_text_list, input_prompt_text_list, max_length=4096)
                for key, value in key_to_token.items():
                    response = response.replace(value, key)
                logging.info('API增强后结果======\n{}'.format(response))
                res = OrderedDict()
                if re.findall('<\|lc_start\|>user[\n]*([\s\S]*?)<\|lc_end\|>', response):
                    res['user'] = re.findall('<\|lc_start\|>user[\n]*([\s\S]*?)<\|lc_end\|>', response)[-1]
                if re.findall('<\|lc_start\|>thought[\n]*([\s\S]*?)<\|lc_end\|>', response):
                    res['thought'] = re.findall('<\|lc_start\|>thought[\n]*([\s\S]*?)<\|lc_end\|>', response)[0]
                if re.findall('<\|lc_start\|>api[\n]*([\s\S]*?)<\|lc_end\|>', response):
                    res['api'] = re.findall('<\|lc_start\|>api\n*([\s\S]*?)<\|lc_end\|>', response)[0]
                if re.findall('<\|lc_start\|>observation[\n]*([\s\S]*?)<\|lc_end\|>', response):
                    res['observation'] = re.findall('<\|lc_start\|>observation[\n]*([\s\S]*?)<\|lc_end\|>', response)[0]
                res['search_results'] = search_results
                # if re.findall('assistant:([\s\S]*?)<eo.>', response):
                #     res['assistant'] = re.findall('assistant:([\s\S]*?)<eo.>', response)[0] + '<eor>'
                if re.findall('<\|lc_start\|>assistant[\n]*([\s\S]*?)<\|lc_end\|>', response):
                    ans = re.findall('<\|lc_start\|>assistant[\n]*([\s\S]*?)<\|lc_end\|>', response)[-1]
                    if "function_call" in ans:
                        ans = extract_function_call(ans)
                    res['assistant'] = ans
                    final_response['answer'] = ans.strip()
                final_response['api_tools_response'] = res
            else:
                if re.findall('<\|lc_start\|>observation[\n]*([\s\S]*?)<\|lc_end\|>', response_new):
                    res1['observation'] = \
                        re.findall('<\|lc_start\|>observation[\n]*([\s\S]*?)<\|lc_end\|>', response_new)[0]
                if re.findall('<\|lc_start\|>assistant[\n]*([\s\S]*?)<\|lc_end\|>', response_new):
                    ans = re.findall('<\|lc_start\|>assistant[\n]*([\s\S]*?)<\|lc_end\|>', response_new)[
                        -1]
                    if "function_call" in ans:
                        ans = extract_function_call(ans)
                    res1['assistant'] = ans
                    
            final_response['first_response'] = res1
            # input_text_list = ['user:{}\nthought:<eot>\napi:<eoa>\nobservation:<eoo>\nassistant:'.format(
            #     final_response['first_response'].get('user', ''))]
            # input_prompt_text_list = []
            # for text in input_text_list:
            #     text = prompt_pre_text + text
            #     input_prompt_text_list.append(text)
            # logging.info('input_text_list:{}'.format(input_text_list))
            # response = inference(input_text_list, input_prompt_text_list, max_length=1024)
            # logging.info('无API的最终结果======\n{}'.format(response))
            # res = OrderedDict()
            # if re.findall('user:([\s\S]*?)<eo.>', response):
            #     res['user'] = re.findall('user:([\s\S]*?)<eo.>', response)[0] + '<eou>'
            # if re.findall('thought:([\s\S]*?)<eo.>', response):
            #     res['thought'] = re.findall('thought:([\s\S]*?)<eot>', response)[0] + '<eot>'
            # if re.findall('api:([\s\S]*?)<eo.>', response):
            #     res['api'] = re.findall('api:([\s\S]*?)<eo.>', response)[0] + '<eoa>'
            # if re.findall('observation:([\s\S]*?)<eo.>', response):
            #     res['observation'] = re.findall('observation:([\s\S]*?)<eo.>', response)[0] + '<eoo>'
            # if re.findall('assistant:([\s\S]*?)<eo.>', response):
            #     res['assistant'] = re.findall('assistant:([\s\S]*?)<eo.>', response)[0] + '<eor>'
            # final_response['second_response'] = res
            # if re.findall('assistant:([\s\S]*?)<eo.>', response):
            #     ans = response.split('assistant:')[-1]
            #     final_response['answer'] = ans.strip().replace('<eor>', '')
        logging.info(json.dumps(final_response, ensure_ascii=False, indent=2))
        return Response(json.dumps(final_response, ensure_ascii=False), mimetype='application/json')
    except Exception as e:
        traceback.print_exc()
        return Response(json.dumps({'error_msg': '服务异常'}, ensure_ascii=False), mimetype='application/json')


if __name__ == '__main__':
    app.run(port=16073, host="0.0.0.0", debug=False)
