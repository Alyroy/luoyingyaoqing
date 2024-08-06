import sys,json

fin=open(sys.argv[1])

js=json.load(fin)

for i in range(100):
	new_content="[unused"+str(i)+"]"
	index=128010+i
	for j in range(len(js["added_tokens"])):
		if js["added_tokens"][j]["id"]==index:
			js["added_tokens"][j]["content"]=new_content
			break

f=open(sys.argv[2],"w+")
json.dump(js,f,indent=4)
