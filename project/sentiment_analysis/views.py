from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from .model_utils import model, tokenizer
import torch
from .SetModel import Model
from .VniAcronym import Acronym

def analyze_sentiment(request):
    if request.method == 'POST':
        text = request.POST['text']
        tokenized_text = tokenizer.encode(text, add_special_tokens=True)
        input_ids = torch.tensor(tokenized_text).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_ids)
            predictions = torch.argmax(outputs.logits, dim=-1)
            sentiment = "Positive" if predictions.item() == 1 else "Negative"

        return render(request, 'sentiment_analysis/result.html', {'text': text, 'sentiment': sentiment})

    return render(request, 'sentiment_analysis/analyze.html')


def analyze_sentiment1(request):
    M = Model('fine_tuned_model_best')
    A = Acronym()
    if request.method == 'POST':
        input_text = request.POST['text']
        tokenized_text = A.Solve_Acr(input_text)

        label, cof = M.Predict(input_text)
        with open("HisFeedBack/"+label+'.txt','a',encoding = 'utf8') as f:
            f.write(input_text+'\n')

        return render(request, 'sentiment_analysis/result.html', {'text': input_text, 'sentiment': label})

    return render(request, 'sentiment_analysis/analyze.html')