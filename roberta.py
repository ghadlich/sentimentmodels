#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) 2022 Grant Hadlich
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
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm.auto import tqdm
import gc

classes = ["Negative", "Neutral", "Positive"]

class RoBertaModel(object):

    def __init__(self):

        model_name = "cardiffnlp/twitter-roberta-base-sentiment"
        self.classifier = AutoModelForSequenceClassification.from_pretrained(model_name)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.init = False

    def _init(self):
        gc.collect()
        self.init = True
        torch.cuda.empty_cache()
        self.classifier.to(self.device)
        self.labels = torch.tensor([1]).unsqueeze(0).to(self.device)

    def name(self):
        return "RoBERTa"

    def model_name_long(self):
        return "trained #roBERTa model from #huggingface"

    def get_classifier(self):
        return self.classifier

    def predict_batch_extended(self, input, batch_size=20, disabletqdm=False):
        if (not self.init):
            self._init()

        predict_ret = []
        scores_ret = []

        chunked_input = SentimentModel.chunks(input, batch_size)

        for text in tqdm(chunked_input, total=len(chunked_input), position=0, leave=True, unit_scale=batch_size, disable=disabletqdm):
            try:
                inputs = self.tokenizer(text,
                                        padding=True,
                                        truncation=True,
                                        max_length=128,
                                        return_tensors="pt").to(self.device)

                outputs = self.classifier(**inputs)

                logits = outputs[0]

                # Move logits and labels to CPU
                predictions = logits.to("cpu").max(1).indices
                
                scores = torch.softmax(logits.to(torch.device('cpu')), dim=1).tolist()

                # Convert these logits to list of predicted labels values.
                predictions_text = [classes[pred] for pred in predictions]
                predict_ret += predictions_text

                for i in range(len(predictions)):
                    scores_ret += [scores[i]]
            except Exception as e:
                print("Error with GPU: " + str(e))
                torch.cuda.memory_summary()
                continue

        return list(zip(predict_ret, scores_ret))

    def predict_batch(self, input, batch_size=20, disabletqdm=False):

        result = self.predict_batch_extended(input, batch_size, disabletqdm)

        # Remove the raw score list with score of the prediction
        # result[0] = prediction, score[3]
        result_ret = []
        for i in range(len(result)):
            prediction = result[i][0]
            index = classes.index(result[i][0])
            score = result[i][1][index]

            result_ret.append([prediction, score])

        return result_ret

    def predict(self, text):
        if (not self.init):
            self._init()

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

    model = RoBertaModel()

    SentimentModel.eval_model(model)