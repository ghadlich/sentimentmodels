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
import flair
import sys
import torch
from tqdm.auto import tqdm

from sentimentmodel import SentimentModel

import re

class FlairModel(object):

    def __init__(self):
        flair.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.classifier = flair.models.TextClassifier.load('en-sentiment')

    def name(self):
        return "Flair"

    def model_name_long(self):
        return "pretrained #Flair model from #ZalandoSE"

    def get_classifier(self):
        return self.classifier

    def predict_batch(self, inputs, batch_size=1):
        predict_ret = []
        scores_ret = []

        chunked_input = SentimentModel.chunks(inputs, batch_size)

        for i in range(len(chunked_input)):
            for j in range(len(chunked_input[i])):
                chunked_input[i][j] = flair.data.Sentence(chunked_input[i][j])

        for input in tqdm(chunked_input, total=len(chunked_input), position=0, leave=True, unit_scale=batch_size):
            result = self.classifier.predict(input)

            for s in input:
                labels = str(s.labels[0])

                if "POSITIVE" in labels:
                    label = "Positive"
                else:
                    label = "Negative"
                predict_ret.append(label)
                confidence = float(re.findall("\d*\.\d*", labels)[0])
                scores_ret.append(confidence)

        return list(zip(predict_ret, scores_ret))

    def predict(self, text):
        s = flair.data.Sentence(text)
        self.classifier.predict(s)

        labels = str(s.labels[0])

        confidence = float(re.findall("\d*\.\d*", labels)[0])

        if "POSITIVE" in labels:
            label = "Positive"
        elif "NEGATIVE" in labels:
            label = "Negative"
        else:
            label = None

        return label, confidence

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

    model = FlairModel()
    SentimentModel.eval_model(model)