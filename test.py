from transformers import BertConfig, BertModel
model = BertModel.from_pretrained('./bert-qa-vacila-agauthier-mlemaire')

n = 0
for epcoh in range(n):
    question = ''
    text = ''
    start_position = 0
    end_position = 0
    outputs = model(question, text, start_position, end_position)
    loss = outputs[0]