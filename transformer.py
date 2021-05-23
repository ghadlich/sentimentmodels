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
from transformers import pipeline
from sentimentmodel import SentimentModel
import torch
from tqdm.auto import tqdm

class TransformerModel(object):

    def __init__(self):

        self.init = False
        # self.classifier = pipeline('sentiment-analysis')

    def name(self):
        return "Transformer"

    def model_name_long(self):
        return "pretrained #BERT model from #huggingface"

    def get_classifier(self):
        return self.classifier

    def _init(self):
        self.init = True
        torch.cuda.empty_cache()
        device = 0 if torch.cuda.is_available() else -1
        self.classifier = pipeline('sentiment-analysis', framework='pt', device=device)

    def predict_batch(self, inputs, batch_size=20):
        if (not self.init):
            self._init()
        predict_ret = []
        scores_ret = []

        chunked_input = SentimentModel.chunks(inputs, batch_size)

        for input in tqdm(chunked_input, total=len(chunked_input), position=0, leave=True, unit_scale=batch_size):
            results = self.classifier(input)
            for result in results:
                if result['label'] == "POSITIVE":
                    predict_ret.append("Positive")
                else:
                    predict_ret.append("Negative")
                scores_ret.append(result['score'])

        return list(zip(predict_ret, scores_ret))

    def predict(self, input_tweet):
        if (not self.init):
            self._init()
        result = self.classifier(input_tweet)
        if result[0]['label'] == "POSITIVE":
            result[0]['label'] = "Positive"
        else:
            result[0]['label'] = "Negative"

        return result[0]['label'], result[0]['score']

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

    model = TransformerModel()
    SentimentModel.eval_model(model)
