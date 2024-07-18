# 真实性打分多模型投票策略
def consistency_strategy(df, result_list):
    '''
    评估策略：
    1. 从多个模型看，只有所有模型打分结果都为1时，final_score = 1；
       只有所有模型打分结果都为0时，final_score = 0；
       其他情况下，final_score = -1

    2. 从每个模型看
    真实性：由三部分组成，分别为相关性等级1, 2, 3, 其权重占比递减。因此，相关性等级1占决定性作用。为此制定如下策略：
    · 只有：相关性等级1为1，相关性等级2为1，真实性才为1。
    · 其余情况下，真实性为0

    相关性：若相关性分数大于3，则相关，返回1，否则为0

    如果要修改单个模型的真实性打分策略，请在_rag_authenticity.py的result_parse()函数里进行修改
    相关性同理，在_rag_relevance.py里修改

    输入：数据df文件与打分结果list
    输出：所有数据最终评分结果list
    show_dic = {}
    show_dic['model'] = model
    show_dic['task_name'] = metric
    show_dic['result'] = result
    show_dic['reason'] = reason
    result_list.append(show_dic)
    '''
    # 判断模型打分结果条数是否与数据条数一致
    assert len(df) == len(result_list[0]['result']), "result length is wrong!"
    final_scores = []
    # 遍历每条记录
    for idx in range(len(df)):
        record_scores = []
        # 遍历每个模型对每条记录的打分结果
        for eval_model in result_list:
            model_score = eval_model['result'][idx]
            record_scores.append(model_score)
        # 判断所有模型的打分结果
        if all(score == 1 for score in record_scores):
            final_score = 1
        elif all(score == 0 for score in record_scores):
            final_score = 0
        else:
            final_score = -1
        final_scores.append(final_score)
    return final_scores
