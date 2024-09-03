import pandas as pd
import re
import os
from sklearn.metrics import precision_score, recall_score, f1_score

def result_truth_parse(response):
    """
    解析真实性打分
    """
    try:
        # print(response)
        eval_data = re.search("准确性评估(.*?)兜底回复", response.replace("\n","").strip())
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
    except:
        return -1


def result_rel_parse(response):
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
        # self.logger.error("[rel-eval]:结果解析失败！")
        rel_pred = -1
    return rel_pred


def log_relevance_parse(eval_res):
    """
    解析log评估，评估的角度必须全部为5
    """
    try:
        eval_data = re.findall("【(.*?)】",eval_res.replace("\n","").strip())
        eval_score = [int(float(x)) for x in eval_data]
        # print(eval_score)
        eval_result = [False if x!=5 else True for x in eval_score]
        # print(eval_result)
        if False in eval_result:
            return 0
        else:
            return 1
    except:
        return -1

def parse_backup(response):
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

# 统计方法
def update_score(test_data,eval_col='gpt4_eval_response',eval_metric='真实性'):
    total = 0
    pos = 0
    rels = []
    test_data[eval_col] = test_data[eval_col].fillna('<|wrong data|>')
    for idx, row in test_data.iterrows():
        eval_res = row[eval_col]
        if eval_metric=='真实性':
            score =result_truth_parse(eval_res)
        elif eval_metric=='相关性':
            score =log_relevance_parse(eval_res)
        # print(truth_score)
        backup = parse_backup(eval_res)
        result = 0 if score == 0 or backup==1 else 1
        rels.append(result)
        pos += result
        total += 1

    if eval_metric == '真实性':
        test_data['真实性打分'] = rels
    elif eval_metric == '相关性':
        test_data['相关性打分'] = rels

    return test_data

def main(folder, eval_col='gpt4_eval_response', eval_metric='真实性'):
    files = sorted([f for f in os.listdir(folder) if 'checkpoint' not in f])
    for file in files:
        df = pd.read_csv(folder+file)
        df = update_score(df,eval_col,eval_metric)
        score = df[eval_metric+'打分'].mean()
        print(file,eval_metric+'打分',score)