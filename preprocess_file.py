import json
import tqdm
me_train_dir ='./baiduQA_corpus/me_train.json'
filehandle = open("./QA_data_evidences.txt","w",encoding='utf8');

path = "./QA_data_evidences.txt"
def json_load(s:str):
    return json.load(open(s))
data = json_load(me_train_dir)

for w in list(data.values()):
    lq =  w['question']
    for e in w['evidences'].values():
        if e['answer'][0] != 'no_answer':
            # print(e['answer'])
            la = e['answer'][0]
            lp = e['evidence']
            lq = lq+" "+la+" "+lp+"\n"
            filehandle.write(lq)
            break
    if e['answer'][0] == 'no_answer':
        la = e['answer'][0]
        lq = lq + " " + la + "\n"
        filehandle.write(lq)
    # word_pairs.append(pair)
    # print(pair[1])
    # print(pair)
    # pair = []
filehandle.close()