import json
import csv

# 输入和输出文件路径
input_file = "/mnt/pfs-guan-ssai/nlu/jiajuntong/data/alignment/pol.st.3361.jsonl"
output_file = "/mnt/pfs-guan-ssai/nlu/jiajuntong/code/rag_tool/rag2/src/auto_llm_distillation/query_content.csv"

# 存储提取的内容
contents = []

# 读取jsonl文件并提取content
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line)
            messages = data['messages']
            for message in messages:
                if message['role'] == 'user':
                    # 由于content是列表，我们取第一个元素
                    content = message['content'][0]
                    contents.append([content])
        except json.JSONDecodeError:
            continue

# 写入CSV文件
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['query'])  # 写入表头
    writer.writerows(contents)

print(f"内容已成功提取并保存到 {output_file}")