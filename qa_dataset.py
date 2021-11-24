import json
#from torch.utils.data import DataLoader, IterableDataset
import pandas as pd

json_file = './v1.0-simplified_nq-dev-all.jsonl'
file = open(json_file)
lines = file.readlines(3)
print(lines)
# with open(json_file) as file:
#     for sample_line in file:
#         sample = json.loads(sample_line)
#     print(json.dumps(sample, indent=4, sort_keys=True))


def extract_yes_no_question(json_file):
    

#dataset = JsonDataset(['data/1.json', 'data/2.json', ...])
#dataloader = DataLoader(dataset, batch_size=32)

#for batch in dataloader:
#    y = model(batch)