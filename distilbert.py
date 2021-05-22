#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) 2021 Grant Hadlich
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE. 
from sentimentmodel import SentimentModel
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from tqdm.auto import tqdm

classes = ["Negative", "Positive"]

class DistilBertModel(object):

    def __init__(self, path="./distilbert_100.pb"):

        self.classifier = torch.load(path, map_location=torch.device('cpu'))
        self.classifier.eval()

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.classifier.to(self.device)

        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.labels = torch.tensor([1]).unsqueeze(0).to(self.device)

    def name(self):
        return "DistilBert"

    def model_name_long(self):
        return "trained #DistilBert model from #huggingface"

    def get_classifier(self):
        return self.classifier

    def predict_batch(self, input, batch_size=25, disabletqdm=False):
        predict_ret = []
        scores_ret = []

        chunked_input = SentimentModel.chunks(input, batch_size)

        for text in tqdm(chunked_input, total=len(chunked_input), position=0, leave=True, unit_scale=batch_size, disable=disabletqdm):
            inputs = self.tokenizer(text,
                                    truncation=True,
                                    padding=True,
                                    return_tensors="pt").to(self.device)

            labels = torch.tensor([1 for _ in range(len(text))]).unsqueeze(0).to(self.device)
            outputs = self.classifier(**inputs, labels=labels)
            loss, logits = outputs[:2]

            # Move logits and labels to CPU
            predictions = logits.to("cpu").max(1).indices
            
            # Convert these logits to list of predicted labels values.
            predictions_text = [classes[pred] for pred in predictions]
            predict_ret += predictions_text

            scores = torch.softmax(logits.to(torch.device('cpu')), dim=1).tolist()

            for i in range(len(predictions)):
                scores_ret += [scores[i][predictions[i]]]

        return list(zip(predict_ret, scores_ret))

    def predict(self, text):
        prediction, score = self.predict_batch([text], batch_size=1, disabletqdm=True)[0]

        return prediction, score

    def create_text(self, data):
        positive = sum(data["Positive"].values())
        negative = abs(sum(data["Negative"].values()))
        total = positive + negative
        pos_percent = str(round(100*positive/total,1)) + "%"
        neg_percent = str(round(100*negative/total,1)) + "%"

        total_str = str(total)

        text = f"I analyzed the sentiment on the last {total_str} tweets from my home feed using a {self.model_name_long()}. "
        if (positive>negative):
            text += f"A majority ({pos_percent}) were classified as positive."
        elif (negative>positive):
            text += f"A majority ({neg_percent}) were classified as negative."
        else:
            text += f"There were an equal amount of positive and negative tweets."

        text += "\n#Python #NLP #PyTorch #Sentiment #GrantBot"

        return text

if __name__ == "__main__":

    model = DistilBertModel()

    SentimentModel.eval_model(model)