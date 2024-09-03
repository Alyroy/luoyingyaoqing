import re
import pandas as pd

if __name__ == "__main__":    
    rawdata_path = "/mnt/pfs-guan-ssai/nlu/gongwuxuan/code/rag_tool/data/output_data/v6_0709_0.2_10_0.2/手机app_必过集_自动化标注_2024-07-16T14_57_49.876_GPT4turbo.csv.v6_0709_0.2_10_0.2.csv"
    pdframe = pd.read_csv(rawdata_path)
    eval_cols = ["full_output"] # ["full_input", "full_output"]
    
    eval_dict = dict()
    for col in eval_cols:
        eval_dict[col] = list()


    for i in range(pdframe.shape[0]):
        for col in eval_cols:
            coldata = pdframe.iloc[i][col]
            eval_dict[col].append(coldata)

            temp_coldata = (coldata.split("</s>"))[0]

            user = temp_coldata.split("[unused0] user")[-1].split("[unused1]")[0].strip()
            thought = temp_coldata.split("[unused0] thought")[-1].split("[unused1]")[0].strip()
            api = temp_coldata.split("[unused0] api")[-1].split("[unused1]")[0].strip()
            api = re.sub("\[unused[0-9]+\]", "", api).strip()
            observation = temp_coldata.split("[unused0] observation")[-1].split("[unused1]")[0].strip()
            api = re.sub("\[unused[0-9]+\]", "", api).strip()
            #assistant = tmp.split("[unused1]")[0].strip()
            if "[unused0] assistant" in temp_coldata:
                assistant = temp_coldata.split("[unused0] assistant")[-1].split("[unused1]")[0].strip()
            else:
                assistant = ""
            if api.strip() == "APINAME Silentwait":
                assistant = "#拒识#"

            print("user: ", user)
            print("thought: ", thought)
            print("api: ", api)
            print("observation: ", observation)
            print("assisstant: ", assistant)
            print()

        