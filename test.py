import json



def json_load(s: str):
    return json.load(open(s))

path =  'C:/MyQA/baiduQA_corpus/me_test.ann.json'
data = json_load(path)
print(data)