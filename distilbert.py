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
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import sys

sys.path.append("../sentimentmodels")

classes = ["Negative", "Positive"]

class DistilBertModel(object):

    def __init__(self, path="./distilbert.pb"):
        self.device = torch.device('cpu')
        self.classifier = torch.load(path, map_location=self.device)
        self.classifier.eval()

        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.labels = torch.tensor([1]).unsqueeze(0)

    def name(self):
        return "DistilBert"

    def model_name_long(self):
        return "trained #DistilBert model from #huggingface"

    def get_classifier(self):
        return self.classifier

    def predict(self, text):
        inputs = self.tokenizer([text],
                                truncation=True,
                                padding=True,
                                return_tensors="pt")
        outputs = self.classifier(**inputs, labels=self.labels)
        loss, logits = outputs[:2]
        prediction = logits.max(1).indices
        score = torch.softmax(logits, dim=1).tolist()[0]

        return classes[prediction], score[prediction]

    def create_text(self, data):
        positive = sum(data["Positive"].values())
        negative = abs(sum(data["Negative"].values()))
        neutral = abs(sum(data["Neutral"].values()))
        total = positive + negative + neutral
        pos_percent = str(round(100*positive/total,1)) + "%"
        neg_percent = str(round(100*negative/total,1)) + "%"
        neu_percent = str(round(100*neutral/total,1)) + "%"

        total_str = str(total)

        text = f"I analyzed the sentiment on the last {total_str} tweets from my home feed using a {self.model_name_long()}. "
        if (positive>(negative+neutral)):
            text += f"A majority ({pos_percent}) were classified as positive with {neu_percent} neutral and {neg_percent} negative."
        elif (positive>negative and positive>neutral):
            text += f"A plurality ({pos_percent}) were classified as positive with {neu_percent} neutral and {neg_percent} negative."
        elif (negative>(positive+neutral)):
            text += f"A majority ({neg_percent}) were classified as negative with {neu_percent} neutral and {pos_percent} positive."
        elif (negative>positive and negative>neutral):
            text += f"A plurality ({neg_percent}) were classified as negative with {neu_percent} neutral and {pos_percent} positive."
        elif (neutral>(positive+negative)):
            text += f"A majority ({neu_percent}) were classified as neutral with {pos_percent} positive and {neg_percent} negative."
        elif (neutral>positive and neutral>negative):
            text += f"A plurality ({neu_percent}) were classified as neutral with {pos_percent} positive and {neg_percent} negative."
        else:
            text += f"There were an equal amount of positive, neutral, and negative tweets."

        text += "\n#Python #NLP #PyTorch #Sentiment #GrantBot"

        return text

if __name__ == "__main__":

    model = DistilBertModel()
    text = "Sold! Enjoying with an ice cold @GuinnessIreland right now; Much love, Happy St. Patrick’s Day, and many thanks, folks!"

    pred, score = model.predict(text)

    print("Text: \"" + text + "\" is " + pred + " with a score of " + str(score))

    from nltk.corpus import twitter_samples

    positive_tweets = twitter_samples.strings('positive_tweets.json')
    negative_tweets = twitter_samples.strings('negative_tweets.json')

    correct = 0

    from tqdm.auto import tqdm

    for text in tqdm(positive_tweets, total=len(positive_tweets), position=0, leave=True):
        pred, score = model.predict(text)

        if (pred == "Positive"):
            correct += 1

    print("Accuracy on Positive: " + str(correct/len(positive_tweets)))

    correct = 0

    for text in tqdm(negative_tweets, total=len(negative_tweets), position=0, leave=True):
        pred, score = model.predict(text)

        if (pred == "Negative"):
            correct += 1

    print("Accuracy on Negative: " + str(correct/len(negative_tweets)))
