import pandas as pd
import ast
# 读取CSV文件
df = pd.read_csv('wenxin_safety_reflect_w_answer.csv')

# 过滤掉任意列包含空列表[]的行

########一层中括号为空

# df_filtered = df[~df.apply(lambda x: x.astype(str).str.contains('^\[\]$')).any(axis=1)]
# df_filtered.to_csv('obs_0115_filter.csv', index=False,encoding='utf-8-sig')


########两层中括号为空

df_filtered = df[~df.apply(lambda x: x.astype(str).str.contains(r'^\[\]$|\[\[\]\]$')).any(axis=1)]
df_filtered.to_csv('wenxin_safety_reflect_w_answer_filter.csv', index=False,encoding='utf-8-sig')
# 对列表类型的字段取第一个元素
# def get_first_element(x):
#     try:
#         # 将字符串形式的列表转换为Python列表
#         lst = ast.literal_eval(x)
#         # 如果是列表类型，返回第一个元素
#         if isinstance(lst, list) and len(lst) > 0:
#             return lst[0]
#     except:
#         pass
#     return x

# # 对每一列应用取第一个元素的操作
# df_filtered = df_filtered.apply(lambda x: x.apply(get_first_element))
# # 保存结果
# df_filtered.to_csv('filter_wenxin_safety300_reflect_w_answer.csv', index=False,encoding='utf-8-sig')