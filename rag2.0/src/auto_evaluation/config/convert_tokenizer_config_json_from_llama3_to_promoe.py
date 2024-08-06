import sys,json

fin=open(sys.argv[1])

js=json.load(fin)

for i in range(100):
	key_ori=str(128010+i)
	new_content="[unused"+str(i)+"]"
	js["added_tokens_decoder"][key_ori]["content"]=new_content


f=open(sys.argv[2],"w+")
json.dump(js,f,indent=4)