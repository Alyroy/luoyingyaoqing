import pandas as pd
import ast
# 读取CSV文件
df_filtered = pd.read_csv('obs_0115_filter.csv')

# 过滤掉任意列包含空列表[]的行

df_filtered = df[~df.apply(lambda x: x.astype(str).str.contains('^\[\]$')).any(axis=1)].copy()
list_columns = ['user-query', 'thought', 'observation','api']
# 对列表类型的字段取第一个元素
def get_first_element(x):
    try:
        # 将字符串形式的列表转换为Python列表
        lst = ast.literal_eval(x)
        # 如果是列表类型，返回第一个元素
        if isinstance(lst, list) and len(lst) > 0:
            return lst[0]
    except:
        pass
    return x
for column in list_columns:
    if column in df_filtered.columns:
        df_filtered[column] = df_filtered[column].apply(get_first_element)
# 对每一列应用取第一个元素的操作
#df_filtered = df_filtered.apply(lambda x: x.apply(get_first_element))
# 保存结果
#df_filtered.to_csv('obs_300_filtered.csv', index=False,encoding='utf-8-sig')
new_data = {
    'query': df_filtered['user-query'].tolist(),
    'response':['']*len(df_filtered),
    'thought': df_filtered['thought'].tolist(),
    'api_name': [],
    'category': [],
    'search_tags': [],
    'doc': df_filtered['observation'].tolist(),
    'search_query': [],
    'is_relative': ['True'] * len(df_filtered),
    'create_user': ['jiajuntong'] * len(df_filtered)
}
# 解析api字段
for api_str in df_filtered['api']:
    #print(api_str)
    try:
        api_dict = dict(api_str)
        #print(api_dict)
        new_data['api_name'].append(api_dict.get('apiname', ''))
        new_data['category'].append(api_dict.get('category', ''))
        new_data['search_tags'].append(api_dict.get('tag', ''))
        new_data['search_query'].append(api_dict.get('query', ''))
    except:
        new_data['api_name'].append('')
        new_data['category'].append('')
        new_data['search_tags'].append('')
        new_data['search_query'].append('')

# 创建新的DataFrame
df_new = pd.DataFrame(new_data)
df_new_ = df_new[~df_new.apply(lambda x: x.astype(str).str.contains('^\[\]$')).any(axis=1)].copy()
#df_new=df_new.dropna(how='any',inplace=False)
# 保存为新的CSV文件
df_new_.to_csv('/mnt/pfs-guan-ssai/nlu/jiajuntong/data/sft/sft_input/sft_input_data_filter.csv', index=False,encoding='utf-8')
#df_new.to_csv('./sft_input_data_filter.csv', index=False)

