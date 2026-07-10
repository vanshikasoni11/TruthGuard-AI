import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizerFast.from_pretrained("bert_model")
model = DistilBertForSequenceClassification.from_pretrained("bert_model")
model.eval()

def predict(text):

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]

    fake_prob = float(probs[1])  
    real_prob = float(probs[0])   


    if real_prob > 0.75:
        label = "REAL"
    elif fake_prob > 0.75:
        label = "FAKE"
    else:
        label = "UNCERTAIN"

    return label, real_prob, fake_prob
