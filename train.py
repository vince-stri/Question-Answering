import sys
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
from transformers import AdamW

def get_raw(param):
    " ".join(param)
    return param # dict {text, question, si_la, ei_la, si_sa, ei_sa, }

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
    
    # Tokenization
    batch_tokens = tokenizer(raw['question'], raw['text'])

    model.train()
    optim = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(k):
        optim.zero_grad()
        outputs = model()

        loss = outputs[0]
        loss.backward()
        optim.step()
    
    exit
    model.save_pretrained('./bert-qa-vacila-agauthier-mlemaire')