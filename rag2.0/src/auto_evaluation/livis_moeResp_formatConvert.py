import pandas as pd

if __name__ == "__main__":    
    rawdata_path = "/mnt/pfs-guan-ssai/nlu/gongwuxuan/code/rag_tool/data/output_data/moe_0723_0.2_10_0.2/手机app_必过集_自动化标注_2024-07-16T14_57_49.876_GPT4turbo.csv.moe_0723_0.2_10_0.2_0.csv"
    pdframe = pd.read_csv(rawdata_path)