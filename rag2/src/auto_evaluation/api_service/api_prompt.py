from langchain_openai import ChatOpenAI
import time


llm = ChatOpenAI(
    temperature=0,
    model="qwen2-72b",
    base_url="http://172.24.168.15:8012/v1",  # 2 停
    openai_api_key="none",
    streaming=True
)
query = "写一段快速排序的代码"

prompt = [
    # {"role": "system", "content": system_prompt},
    {"role": "user", "content": query}
]

response = llm.invoke(prompt)  # 前排下车后座椅自动设为坐姿二 # 写一段快速排序的代码
print(response.content)
