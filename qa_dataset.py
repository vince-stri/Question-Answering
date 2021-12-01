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


def purge_data(n, json_file):
    with open('dest_file.json', 'w') as dest_file:
        with open(json_file, 'r') as source_file:
            i = 0
            for line in source_file:
                if i > n:
                    break
                element = json.loads(line.strip())
                if 'document_html' in element:
                    del element['document_html']
                if 'long_answer_candidates' in element:
                    del element['long_answer_candidates']
                if 'question_tokens' in element:
                    del element['question_tokens']
                dest_file.write(json.dumps(element))
                i+=1
    print("data purged")



json_file = '../v1.0-simplified_nq-dev-all.jsonl'
data = purge_data(500, json_file)


#dataset = JsonDataset(['data/1.json', 'data/2.json', ...])
#dataloader = DataLoader(dataset, batch_size=32)

#for batch in dataloader:
#    y = model(batch)