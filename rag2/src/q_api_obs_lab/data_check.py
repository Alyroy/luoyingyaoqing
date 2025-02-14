import pandas as pd
import re

def convert_chinese_punctuation(text):
    """转换中文标点符号为英文标点符号"""
    if not isinstance(text, str):
        return text
        
    punctuation_map = {
        '。': '.',
        '，': ',',
        '！': '!',
        '？': '?',
        '"': '"',
        '"': '"',
        ''': "'",
        ''': "'",
        '：': ':',
        '；': ';',
        '（': '(',
        '）': ')',
        '【': '[',
        '】': ']',
        '《': '<',
        '》': '>',
        '、': ',',
        '～': '~',
        '…': '...',
        '—': '-'
    }
    
    for ch, en in punctuation_map.items():
        text = text.replace(ch, en)
    return text

def clean_text(text):
    """清理单个文本"""
    if not isinstance(text, str):
        return text
    
    # 1. 基础清理
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)  # 多个空格替换为单个
    text = text.replace('\n', ' ')     # 替换换行
    
    # 2. 转换中文标点
    text = convert_chinese_punctuation(text)
    
    # 3. 清理连续的标点符号
    text = re.sub(r'[.,!?]+', '.', text)  # 连续的句号、逗号等替换为单个句号
    text = re.sub(r'\.{2,}', '...', text) # 处理省略号
    
    return text

def clean_csv_file(input_file='./query_0115.csv', output_file='./query_0115_cleaned.csv'):
    """清理CSV文件"""
    try:
        # 读取CSV文件
        df = pd.read_csv(input_file, encoding='utf-8')
        
        # 显示原始数据信息
        print("原始数据信息：")
        print(f"行数: {len(df)}")
        print(f"重复行数: {df.duplicated().sum()}")
        
        # 数据清理步骤
        # 1. 删除重复行
        df = df.drop_duplicates()
        
        # 2. 清理文本
        df['user-query'] = df['user-query'].apply(clean_text)
        
        # 3. 删除空值
        df = df.dropna()
        
        # 4. 删除空字符串
        df = df[df['user-query'].str.len() > 0]
        
        # 显示清理后的数据信息
        print("\n清理后数据信息：")
        print(f"行数: {len(df)}")
        
        # 保存清理后的数据
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n数据已保存到: {output_file}")
        
        return df
        
    except Exception as e:
        print(f"处理出错: {str(e)}")
        return None

def check_problematic_lines(df):
    """检查问题数据"""
    print("\n检查问题数据：")
    
    # 检查中文标点
    chinese_punct_pattern = r'[。，！？""''：；（）【】《》、～…—]'
    chinese_punct = df[df['user-query'].str.contains(chinese_punct_pattern, regex=True)]
    if not chinese_punct.empty:
        print(f"\n仍包含中文标点的行数: {len(chinese_punct)}")
        print("示例：")
        print(chinese_punct.head())
    
    # 检查特殊字符
    special_chars = df[df['user-query'].str.contains(r'[^\w\s\.,!?:;\(\)\[\]<>\'\"~-]', regex=True)]
    if not special_chars.empty:
        print(f"\n包含特殊字符的行数: {len(special_chars)}")
        print("示例：")
        print(special_chars.head())
    
    # 检查过长的文本
    long_text = df[df['user-query'].str.len() > 100]
    if not long_text.empty:
        print(f"\n超长文本数量: {len(long_text)}")
        print("示例：")
        print(long_text.head())

if __name__ == "__main__":
    # 执行数据清理
    
    cleaned_df = clean_csv_file()
    
    if cleaned_df is not None:
        # 检查问题数据
        check_problematic_lines(cleaned_df)