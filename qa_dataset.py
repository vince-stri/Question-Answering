import json
#from torch.utils.data import DataLoader, IterableDataset
import pandas as pd


# file = open(json_file)
# lines = file.readlines(3)
# print(lines)
# with open(json_file) as file:
#     for sample_line in file:
#         sample = json.loads(sample_line)
#     print(json.dumps(sample, indent=4, sort_keys=True))

#[question, data, start, end]
def extract_short_answer(data):
    result = []
    for element in data:
        if len(element['short_answers'])>0:
            #print(element['short_answers'][0])
            result.append(element['short_answers'][0]['start_token'])
            result.append(element['short_answers'][0]['end_token'])
            #print(result)
            return result
    #result = [-1, -1]
    print(result)
    return result

def purge_data(n, json_file):
    #with open('data_purged.jsonl', 'w') as dest_file:
    with open(json_file, 'r') as source_file:
        i = 0
        for line in source_file:
            result = []
            if i > n:
                break
            element = json.loads(line.strip())
            if 'question_text' in element:
                result.append(element['question_text'])
            tokens = ""
            for token in element['document_tokens']:
                if token['html_token'] == 0:
                    tokens+= token['token'] + ' '
            short_answers = extract_short_answer(element['annotations'])
            #result.append(tokens)
            result.append(short_answers[0])
            result.append(short_answers[1])
            print(result)
            #dest_file.write(json.dumps(element))
            i+=1
    print("data purged")

def get_question(json_file):
    with open(json_file, 'r') as data_file:
        for line in data_file:
            element = json.loads(line)
            if 'question_text' in element:
                print(element['question_text'])

                

json_file = './v1.0-simplified_nq-dev-all.jsonl'
data = purge_data(50, json_file)
#get_question('data_purged.jsonl')

#dataset = JsonDataset(['data/1.json', 'data/2.json', ...])
#dataloader = DataLoader(dataset, batch_size=32)

#for batch in dataloader:
#    y = model(batch)