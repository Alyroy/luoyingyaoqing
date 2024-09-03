import os
import re
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaTokenizer, LlamaForCausalLM
import requests
import json

key_to_token = {
    "<|lc_start|>": "[unused0]",
    "<|lc_end|>": "[unused1]",
    "<|kvs|>": "[unused2]",
    "<|kve|>": "[unused3]",
    "<|api_start|>": "[unused4]",
    "<|api_end|>": "[unused5]",
    "<|eoa|>": "[unused6]",
    "=>": "[unused7]",
    "<|br|>": "[unused8]"
}

# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

#prompt_test1 = "<|lc_start|>system\n你现在进行的是一个语言理解任务，需要根据上下文信息和当前请求进行APINAME的识别和相应槽位的提取。如果是涉及通用问答的请求，需要生成QASearch的API；如果是涉及汽车问答的请求，需要生成AUTOSearch的API；如果是涉及绘画大师的请求，需要生成AIPainting的API；如果是涉及到媒体搜索的请求，需要生成MEDIASearch的API；如果是涉及到任务型对话的请求，需要生成TODRequest的API；如果涉及到非安全的请求，需要生成SafeReject的API；其余请求则不生成API。\n请针对以下请求内容生成正确的结果。<|lc_end|>\n"
#domain_prompt = "[unused0]system\n你现在进行的是一个语言理解任务，需要根据上下文信息和当前请求进行APINAME的识别和相应槽位的提取。如果是涉及通用问答的请求，需要生成QASearch的API；如果是涉及汽车问答的请求，需要生成AUTOSearch的API；如果是涉及绘画大师的请求，需要生成AIPainting的API；如果是涉及到媒体搜索的请求，需要生成MEDIASearch的API；如果是涉及到任务型对话的请求，需要生成TODRequest的API；如果涉及到人设问答的请求，需要生成CHARASearch的API；其余请求则不生成API。\n请针对以下请求内容生成正确的结果。[unused1]\n"
#prompt_other_origin = "[unused0]system\n你现在进行的是一个语言理解任务，需要根据上下文信息和当前请求进行APINAME的识别和相应槽位的提取。如果是涉及到任务型对话的请求，需要生成TODRequest的API；如果涉及到非安全的请求，需要生成SafeReject的API；其余请求则不生成API。\n请针对以下请求内容生成正确的结果。[unused1]\n"
#prompt_other_vision = "[unused0]system\n当前任务是用户看向乘客，判断是否需要拒识。 请针对以下请求内容生成正确的结果。[unused1]\n"
#prompt_mediasearch = "[unused0]system\n你现在进行的是一个语言理解任务，当前请求是一个媒体搜索的请求，需要生成媒体搜索对应的API。\n请针对以下请求内容生成正确的结果。[unused1]\n"
#prompt_autosearch = "[unused0]system\n你现在进行的是一个语言理解任务，当前请求是一个汽车问答的请求，需要生成汽车问答对应的API。\n请针对以下请求内容生成正确的结果。[unused1]\n"

# domain_prompt="[unused0]system\n你是一个名字叫做理想同学的AI数字生命体，由理想汽车智能空间部门创造。\n\n# Tools\n## Functions\n{\"name\": \"QASearch\", \"description\": \"查询关于通用百科的知识\", \"parameters\": {\"type\": \"object\", \"properties\": {\"CATEGORY\": {\"type\": \"string\", \"description\": \"用户请求所属的领域\"}, \"QUERY\": {\"type\": \"string\", \"description\": \"用户请求改写后的结果\"}, \"TAG\": {\"type\": \"string\", \"description\": \"用户请求关键词提取后的结果\"}}, \"required\": [\"CATEGORY\", \"QUERY\", \"TAG\"]}}\n\n{\"name\": \"AIPaiting\", \"description\": \"绘画大师相关的请求\", \"parameters\": {\"type\": \"object\", \"properties\": {\"KEYPHRASE\": {\"type\": \"string\", \"description\": \"用户请求的摘要\"}, \"CATEGORY\": {\"type\": \"string\", \"description\": \"用户请求的领域分类，默认为其他\"}, \"STYLE\": {\"type\": \"string\", \"description\": \"用户请求的绘画风格，默认为空\"}, \"POSITION\": {\"type\": \"string\", \"description\": \"用户请求对绘画位置的要求\"}, \"SCREENNAME\": {\"type\": \"string\", \"description\": \"用户请求对绘画屏幕的要求\"}}, \"required\": [\"KEYPHRASE\", \"CATEGORY\", \"STYLE\"]}}\n\n{\"name\": \"TODRequest\", \"description\": \"任务型对话相关的请求\", \"parameters\": {\"type\": \"object\", \"properties\": {\"DOMAIN\": {\"type\": \"string\", \"description\": \"用户请求的领域\"}, \"ACTION\": {\"type\": \"string\", \"description\": \"用户请求的动作\"}}, \"required\": [\"DOMAIN\"]}}[unused1]\n"
# prompt_mediasearch = "[unused0]system\n你是一个名字叫做理想同学的AI数字生命体，由理想汽车智能空间部门创造。\n\n# Tools\n## Functions\n\"name\"\n\n\"description\"\n\n\"parameters\"[unused1]\n"
# prompt_autosearch = "[unused0]system\n你是一个名字叫做理想同学的AI数字生命体，由理想汽车智能空间部门创造。\n\n# Tools\n## Functions\n{\"name\": \"AUTOSearch\", \"description\": \"查询关于汽车相关的知识\", \"parameters\": {\"type\": \"object\", \"properties\": {\"CATEGORY\": {\"type\": \"string\", \"description\": \"用户请求所属的领域\"}, \"QUERY\": {\"type\": \"string\", \"description\": \"用户请求进行改写后的结果\"}, \"TAG\": {\"type\": \"string\", \"description\": \"用户请求关键词提取后的结果\"}}, \"required\": [\"CATEGORY\", \"QUERY\", \"TAG\"]}}[unused1]\n"

#domain_prompt = """[unused0]system\n你要进行一个语言理解任务。\n\n# Tools\n\n## Functions\n\n{"name":"qasearch"}\n{"name":"aipainting"}\n{"name":"autosearch"}\n{"name":"charasearch"}\n{"name":"todrequest"}\n{"name":"mathqa"}[unused1]\n"""
#domain_prompt =  """[unused0]system\n你要进行一个语言理解任务。\n\n# Tools\n\n## Functions\n\n{"name":"qasearch"}\n{"name":"aipainting"}\n{"name":"autosearch"}\n{"name":"mediasearch"}\n{"name":"charasearch"}\n{"name":"todrequest"}\n{"name":"mathqa"}[unused1]\n"""
#domain_prompt = """[unused0]system\n你要进行一个语言理解任务。\n\n# Tools\n\n## Functions\n\n{"name":"qasearch"}\n{"name":"aipainting"}\n{"name":"autosearch"}\n{"name":"mediasearch"}\n{"name":"charasearch"}\n{"name":"todrequest"}\n{"name":"mathqa"}\n{"name":"comparenumber"}[unused1]\n"""
domain_prompt = """[unused0]system\n你要进行一个语言理解任务。\n\n# Tools\n\n## Functions\n\n{"name":"qasearch"}\n{"name":"aipainting"}\n{"name":"autosearch"}\n{"name":"mediasearch"}\n{"name":"charasearch"}\n{"name":"todrequest"}\n{"name":"mathqa"}\n{"name":"comparenumber"}\n{"name":"visionqa"}[unused1]\n"""
prompt_mediasearch = """[unused0]system\n你要进行一个语言理解任务。\n\n# Tools\n\n## Functions\n\n{"name":"mediasearch"}[unused1]\n"""
prompt_autosearch =  """[unused0]system\n你要进行一个语言理解任务。\n\n# Tools\n\n## Functions\n\n{"name":"autosearch"}[unused1]\n"""

# prompt_map = {
#     "other_origin": "<|lc_start|>system\n当前任务是根据当前请求和上文信息生成正确的API信息。\n请针对以下请求内容生成正确的结果。<|lc_end|>\n",
#     "other_vision": "<|lc_start|>system\n当前面部朝向为乘客。当前任务是根据当前请求和上文信息生成正确的API信息。\n请针对以下请求内容生成正确的结果。<|lc_end|>\n",
#     "QASearch": "<|lc_start|>system\n当前请求是一个通用领域百科知识的意图，需要生成通用知识查询对应的API结构化内容。\n请针对以下请求内容生成正确的结果。<|lc_end|>\n",
#     "AUTOSearch": "<|lc_start|>system\n当前请求是一个汽车领域知识的意图，需要生成汽车知识查询对应的API结构化内容。\n请针对以下请求内容生成正确的结果。<|lc_end|>\n",
#     "AIPainting": "<|lc_start|>system\n当前请求是一个绘画生成的意图，需要生成绘画对应的API结构化内容。\n请针对以下请求内容生成正确的结果。<|lc_end|>\n",
#     "TaskMaster": "<|lc_start|>system\n当前请求是一个任务大师的意图，需要生成任务大师对应的API结构化内容。\n请针对以下请求内容生成正确的结果。<|lc_end|>\n",
#     "MEDIASearch": "<|lc_start|>system\n当前请求是一个影视推荐的意图，需要生成媒体搜索对应的API结构化内容。\n请针对以下请求内容生成正确的结果。<|lc_end|>\n",
# }


prompt_map = {
    "other_origin": domain_prompt,
    "other_vision": domain_prompt,
    "QASearch": domain_prompt,
    "AUTOSearch": prompt_autosearch,
    "AIPainting": domain_prompt,
    "TaskMaster": domain_prompt,
    "MEDIASearch": prompt_mediasearch,
}


#prompt_map = {
#    "other_origin": "<|lc_start|>system\n当前任务是根据当前请求和上文信息生成正确的API信息。\n请针对以下请求内容生成正确的结果。<|lc_end|>\n",
#    "other_vision": "<|lc_start|>system\n当前面部朝向为乘客。当前任务是根据当前请求和上文信息生成正确的API信息。\n请针对以下请求内容生成正确的结果。<|lc_end|>\n",
#    "QASearch": "<|lc_start|>system\n当前请求是一个通用领域百科知识的意图，需要生成通用知识查询对应的API结构化内容。\n请针对以下请求内容生成正确的结果。<|lc_end|>\n",
#    "AUTOSearch": "<|lc_start|>system\n当前请求是一个汽车领域知识的意图，需要生成汽车知识查询对应的API结构化内容。\n请针对以下请求内容生成正确的结果。<|lc_end|>\n",
#    "AIPainting": "<|lc_start|>system\n当前请求是一个绘画生成的意图，需要生成绘画对应的API结构化内容。\n请针对以下请求内容生成正确的结果。<|lc_end|>\n",
#    "TaskMaster": "<|lc_start|>system\n当前请求是一个任务大师的意图，需要生成任务大师对应的API结构化内容。\n请针对以下请求内容生成正确的结果。<|lc_end|>\n",
#    "MEDIASearch": "<|lc_start|>system\n当前请求是一个影视推荐的意图，需要生成媒体搜索对应的API结构化内容。\n请针对以下请求内容生成正确的结果。<|lc_end|>\n",
#}

# model = '/mnt/pfs-ssai-nlu/chenyabo/sft_model_1b/20230824-1b-multi-prompt'
# model = '/mnt/pfs-ssai-nlu/xinhongsheng/work_dir/1b_sft_model/v20240305_online_llm2llm_0412/'
model = "/mnt/pfs-guan-ssai/nlu/xinhongsheng/stanford_alpaca_dynamics_2048_dev/output_model/v20240729_api_to_function_call_8_tool"
# 按照以下方式导入模型和分词器也可以
# tokenizer = AutoTokenizer.from_pretrained(model, max_length=1024)
# model = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto", device_map="auto")
tokenizer = LlamaTokenizer.from_pretrained(model)
# unused_tokens = []
# for i in range(100):
#     unused_tokens.append("[unused" + str(i) + "]")
# tokenizer.add_tokens(unused_tokens, special_tokens=True)

tokenizer.padding_side = "left"
model = LlamaForCausalLM.from_pretrained(model, torch_dtype=torch.float16, device_map="auto")


# model.resize_token_embeddings(len(tokenizer))

# tmp_input = tokenizer.batch_encode_plus(
#     ['[unused0]', '[unused1]', '[unused2]', '[unused3]', '[unused4]', '[unused5]', '[unused6]', '[unused7]', ' ', '\n'],
#     padding=True, truncation=True, return_tensors='pt')
# tmp_input_ids = tmp_input['input_ids'].to('cuda')
# print(tmp_input_ids)
# #
# tmp_input = tokenizer.batch_encode_plus(
#     ['[unused0]system\n你是一个 机器人[unused1]\n[unused0]user\n你好吗\n'],
#     padding=True, truncation=True, return_tensors='pt')
# tmp_input_ids = tmp_input['input_ids'].to('cuda')
# print(tmp_input_ids)
#
# output_text_list = tokenizer.batch_decode([1404, 38815, 38816, 38817, 38818, 38819, 38820, 38821, 38822, 13, 29876],
#                                           skip_special_tokens=True)
# print(output_text_list)

# torch.save(model.state_dict(), model_path)

def inference(input_text_list, input_prompt_text_list, max_length=4096):
    startTime = time.time()
    # ------分别对两个输入文本进行批量推理
#print("模型推理输入==={}".format(input_prompt_text_list))
    new_list = []
    for sub_list in input_prompt_text_list:
        sub_list = sub_list.replace("<s>","")
        sub_list = sub_list+"[unused0]assistant\n"
        new_list.append(sub_list.strip("\n"))
    input_prompt_text_list = new_list
    print("模型原始输入:",input_prompt_text_list[0])
    # print(input_prompt_text_list[0])
    tok_input = tokenizer.batch_encode_plus(input_prompt_text_list, padding=True, truncation=True, return_tensors='pt')
    input_ids = tok_input['input_ids'].to('cuda')
    attention_mask = tok_input['attention_mask'].to('cuda')
    print("input token length:========{}".format(len(input_ids[0])))
    output = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                            max_length=max_length,
                            top_k=1,
                            top_p=0.95,
                            repetition_penalty=1,
                            temperature=1e-05, do_sample=False)

    # ------解码生成的文本输出
    output_text_list = tokenizer.batch_decode(output, skip_special_tokens=False)
    # print(output_text_list)
#print('模型原始输出=========={}'.format(output_text_list))
    # 去掉输入部分
    predict_text = []
    for i in range(len(output_text_list)):
        out_text = (output_text_list[i].split("</s>"))[0]
        out_text = out_text.replace('] ', ']')
        out_text = out_text.replace('[unused1][unused0]', '[unused1]\n[unused0]')
        out_text = out_text.replace('api[unused4]', 'api\n[unused4]')
        out_text = out_text.replace('observation[unused2]', 'observation\n[unused2]')

        # print(out_text)
        # out_text = out_text.split(input_prompt_text_list[i])[-1]
        predict_text.append(out_text)

    useTime1 = (time.time() - startTime)
#    print('cost time:{}'.format(useTime1))
    return out_text


def request_google_search(query):
    url = "http://10.240.200.29:6699/google_search"
    payload = json.dumps({
        "query": query
    })
    headers = {
        'Content-Type': 'application/json'
    }
    google_response = requests.request("POST", url, headers=headers, data=payload)
    # print(google_response.text)
    return json.loads(google_response.text)


def request_knowledge_engine(raw_query, category, query, tag: list):
    url = "http://ks-dev-engine-server-inference.ssai-apis-staging.chj.cloud:80/cloud/inner/nlp/kg/knowledge-search-engine/search"

    payload = json.dumps({
        "metadata": {
            "vin": "postman-dean-test-zhangcy-service",
            "msgId": "postman-dean-test-zhangcy-service"
        },
        "orig_query": raw_query,
        "search_query": query,
        "search_category": [category],
        "search_slots": {
            "tag": tag
        },
        "searchBot": "kg-bot"
    })
    headers = {
        'Content-Type': 'application/json'
    }
    print('QA搜索结果==========')
    print(json.dumps(json.loads(payload), ensure_ascii=False))
    response = requests.request("POST", url, headers=headers, data=payload)
    print(response.text)
    response = json.loads(response.text)

    res_data = response.get("data", None)
    observation = '<None>'
    if res_data and len(res_data) > 0:
        res_data = res_data[0]
        bot_data = res_data.get("bot_data", [])
        if bot_data:
            if len(bot_data) > 3:
                bot_data = bot_data[:3]
            res = [item.get('content', '') for item in bot_data]
            # print(res)
            if len(res) > 0:
                observation = json.dumps({"QASearchResults": res}, ensure_ascii=False)

    # print(observation)
    observation_response = {'query': query, 'search_result': response, 'observation': observation}
    return observation_response


if __name__ == '__main__':
    # while 1:
    #
    #     input_text = input('请输入请求：')
    #     if input_text in ['exit', '退出', '结束']:
    #         break
    #
    #     query = input_text.strip()
    #
    #     input_text = "<|lc_start|>user\n{}<|lc_end|>\n".format(query)
    #     input_text = input_text.replace('\\n', '\n')
    #     input_text = input_text.replace('\\r', '\r')
    #
    #     for key, value in key_to_token.items():
    #         input_text = input_text.replace(key, value)
    #
    #     input_text_list = [input_text]
    #     # 首轮
    #     # prompt_pre_text = '<|lc_start|>system\n今天是2023年8月16日，你是一个名字叫做理想同学的AI机器人.\n理想同学是一个可靠的大语言模型，由理想汽车智能空间部门创造。\n理想同学能够理解人类的指令和意图，并且给出合理的、切合问题的、没有歧视、中立的、安全的回复。\n\n请根据以下文本写一个合适的回复。<|lc_end|>\n'
    #     prompt_pre_text = '<|lc_start|>system\n当前请求是一个汽车领域知识的意图，需要生成汽车知识查询对应的API结构化内容。\n请针对以下请求内容生成正确的结果。<|lc_end|>\n'
    #     # 非首轮有上文
    #     # prompt_pre_text = '<|lc_start|>system\n你是一个名字叫做理想同学的AI机器人.\n理想同学是一个可靠的大语言模型，由理想汽车智能空间部门创造。\n理想同学能够理解人类的指令和意图，并且给出合理的、切合问题的、没有歧视、中立的、安全的回复。\n\n请根据以下文本写一个合适的回复。<|lc_end|>\n<|lc_start|>user\n李寻欢是谁<|lc_end|>\n<|lc_start|>assistant\n李寻欢，古龙笔下的著名重要虚构武林人物，其小李飞刀百发百中，从不落空，曾杀死金钱帮帮主上官金虹，在百晓生的兵器谱上排名第三。 他的传人为叶开。 首见于作品《多情剑客无情剑》之中。 其后虽不再在小说登场，却仍在其他古龙小说中被别的人物提及，名字可见于《边城浪子》、《天涯‧明月‧刀》、《飞刀，又见飞刀》等作品。<|lc_end|>\n'
    #     # prompt_pre_text = '<|lc_start|>system\n你是一个名字叫做理想同学的AI对话机器人。\n你能够理解人类的指令和意图，并给出正确的、合理的、切合问题的、安全的回复。\n为了确保回复的准确性、时效性和相关性，你可以通过生成API调用来获取有效信息帮助你做出回复，无需API辅助时不用生成API。\n你支持的API有以下几类：（1）通用知识搜索QASearch，当涉及到通用事实类知识或时效性问答时你可以调用它，主观评价类或开放类问题不需要调用它；（2）汽车知识搜索AUTOSearch，当涉及到汽车领域知识问答时，你可以调用它。\n\n请根据以下对话历史或用户请求做出合理的API生成。\n\n请根据以下文本写一个合适的回复。<|lc_end|>\n'
    #     for key, value in key_to_token.items():
    #         prompt_pre_text = prompt_pre_text.replace(key, value)
    #     input_prompt_text_list = []
    #     for text in input_text_list:
    #         text = prompt_pre_text + text
    #         input_prompt_text_list.append(text)
    #
    #     # inference
    #     print(input_prompt_text_list)
    #     response = inference(input_text_list, input_prompt_text_list, max_length=4096)
    #
    #     for key, value in key_to_token.items():
    #         response = response.replace(value, key)
    #     print('结果===\n{}'.format(response))

    # 批量文件测试
    writer = open("./test_car_1.output", "w", encoding="utf-8")
    with open("./test_car_1.input", "r", encoding="utf-8") as fr:
        for line in fr:
            query = line.strip()

            input_text = "<|lc_start|>user\n{}<|lc_end|>\n".format(query)

            for key, value in key_to_token.items():
                input_text = input_text.replace(key, value)

            input_text_list = [input_text]
            # 首轮
            # prompt_pre_text = '<|lc_start|>system\n你是一个名字叫做理想同学的AI机器人.\n理想同学是一个可靠的大语言模型，由理想汽车智能空间部门创造。\n理想同学能够理解人类的指令和意图，并且给出合理的、切合问题的、没有歧视、中立的、安全的回复。\n\n请根据以下文本写一个合适的回复。<|lc_end|>\n'
            # prompt_pre_text = '<|lc_start|>system\n当前请求是一个汽车领域知识的意图，需要生成汽车知识查询对应的API结构化内容。\n请针对以下请求内容生成正确的结果。<|lc_end|>\n'
            prompt_id = "AUTOSearch"
            prompt_pre_text = prompt_map.get(prompt_id, prompt_map.get("other_origin"))
            # 非首轮有上文
            # prompt_pre_text = '<|lc_start|>system\n你是一个名字叫做理想同学的AI机器人.\n理想同学是一个可靠的大语言模型，由理想汽车智能空间部门创造。\n理想同学能够理解人类的指令和意图，并且给出合理的、切合问题的、没有歧视、中立的、安全的回复。\n\n请根据以下文本写一个合适的回复。<|lc_end|>\n<|lc_start|>user\n李寻欢是谁<|lc_end|>\n<|lc_start|>assistant\n李寻欢，古龙笔下的著名重要虚构武林人物，其小李飞刀百发百中，从不落空，曾杀死金钱帮帮主上官金虹，在百晓生的兵器谱上排名第三。 他的传人为叶开。 首见于作品《多情剑客无情剑》之中。 其后虽不再在小说登场，却仍在其他古龙小说中被别的人物提及，名字可见于《边城浪子》、《天涯‧明月‧刀》、《飞刀，又见飞刀》等作品。<|lc_end|>\n'
            for key, value in key_to_token.items():
                prompt_pre_text = prompt_pre_text.replace(key, value)
            # prompt_pre_text = ''
            input_prompt_text_list = []
            for text in input_text_list:
                text = prompt_pre_text + text
                input_prompt_text_list.append(text)

            # inference
            print(input_prompt_text_list)
            response = inference(input_text_list, input_prompt_text_list, max_length=4096)

            for key, value in key_to_token.items():
                response = response.replace(value, key)
            print('结果===\n{}'.format(response))
            response = response.replace("thought\n", "\t")
            response = response.replace("<|lc_end|>\n<|lc_start|>api\n<|api_start|>", "\t")
            response = response.replace("<|lc_end|>\n<|lc_start|>api\n", "\t")



            response = response.replace("<|api_start|>", "\t")
            response = response.replace("<|lc_start|>", "\t")
            response = response.replace("<|lc_end|>", "")
            response = response.replace("<|api_end|>", "\t")
            response = response.replace("<|eoa|>", "")
            response = response.replace("<|kvs|>", "")
            response = response.replace("<|kve|>", "\t")
            response = response.replace("APINAME=>", "\t")
            response = response.replace("CATEGORY=>", "\t")
            response = response.replace("QUERY=>", "\t")
            response = response.replace("TAG=>", "\t")
            response = response.replace("\n", "")
            response = response.replace("\t\t\t", "\t")
            response = response.replace("\t\t", "\t")

            writer.write(query + "\t" + response.replace("\n", "\\n") + "\n")
            writer.flush()
    writer.close()
