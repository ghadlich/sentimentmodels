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
from tqdm.auto import tqdm
from textblob import TextBlob

class TextBlobModel(object):

    def __init__(self):
        pass

    def name(self):
        return "TextBlob"

    def model_name_long(self):
        return "pretrained #TextBlob model"

    def get_classifier(self):
        return self.classifier

    def predict_batch(self, inputs):
        predict_ret = []
        scores_ret = []

        for input in tqdm(inputs, total=len(inputs), position=0, leave=True, unit_scale=1):
            predict, score = self.predict(input)
            predict_ret.append(predict)
            scores_ret.append(score)

        return list(zip(predict_ret, scores_ret))

    def predict(self, input_tweet):
        result = TextBlob(input_tweet)

        if result.sentiment.polarity > 0:
            prediction = "Positive"
        else:
            prediction = "Negative"

        return prediction, result.sentiment.polarity

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

        text += "\n#Python #NLP #Classification #Sentiment #GrantBot"

        return text

if __name__ == "__main__":

    model = TextBlobModel()
    SentimentModel.eval_model(model)
