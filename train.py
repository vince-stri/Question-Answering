import sys
import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
from transformers import AdamW
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

class NQDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

def get_raw(param):
    " ".join(param)
    return param # dict{text, question, start_position, end_position}

# Main function
if __name__ == "__main__":
    # System parameters
    k = 1
    if len(sys.argv) == 2:
        k = int(sys.argv[1])
        if k < 1:
            print("Error: the number of epochs must be positive")
            exit
    elif len(sys.argv) > 2:
        print("Usage:", sys.argv[0], "number_of_epochs")
        exit

    # Import pretrained model and tokenizer by HuggingFace from the HuggingFace library
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    
    # TO DEFINE
    param = ""
    raw = get_raw(param)
    # - - -

    batch_tokens = tokenizer(raw['question'], raw['text'])

    batch_tokens.update({'start_position': raw.start_position, 'end_position': raw.end_position})
    train_dt = NQDataset(batch_tokens)
    loader = torch.utils.data.DataLoader(train_dt, batch_size=16, shuffle=True)
    
    # Tokenization
    model.train()
    optim = AdamW(model.parameters(), lr=5e-5)

    allloss = []

    for epoch in range(k):
        loop = tqdm(loader)
        for batch in loop:
            optim.zero_grad()
            input_id = batch['input_ids']
            attention_mask = batch['attention_mask']
            start_position = batch['start_position']
            end_position = batch['end_position']

            outputs = model(input_id, attention_mask=attention_mask, start_position=start_position, end_position=end_position)
            loss = outputs[0]
            loss.backward()
            optim.step()

            allloss.append(loss.item())

            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())

    plt.plot(allloss)
    plt.show()
    exit
    model.save_pretrained('./bert-qa-vacila-agauthier-mlemaire')