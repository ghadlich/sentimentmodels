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
from nltk.corpus import twitter_samples
import sys
sys.path.append("../sentimentmodels")

class SentimentModel(object):

    def __init__(self):
        pass

    @staticmethod
    def chunks(l, n):
        n = max(1, n)
        return [l[i:i+n] for i in range(0, len(l), n)]

    @staticmethod
    def eval_model(model):
        text = "Sold! Enjoying with an ice cold @GuinnessIreland right now; Much love, Happy St. Patrick’s Day, and many thanks, folks!"

        pred, score = model.predict(text)

        print("Text: \"" + text + "\" is " + pred + " with a score of " + str(score))

        

        positive_tweets = twitter_samples.strings('positive_tweets.json')
        negative_tweets = twitter_samples.strings('negative_tweets.json')

        correct = 0

        results = model.predict_batch(positive_tweets)

        for result in results:
            pred, score = result

            if (pred == "Positive"):
                correct += 1

        print("Accuracy on Positive: " + str(correct/len(positive_tweets)))

        correct = 0

        results = model.predict_batch(negative_tweets)

        for result in results:
            pred, score = result

            if (pred == "Negative"):
                correct += 1

        print("Accuracy on Negative: " + str(correct/len(negative_tweets)))


if __name__ == "__main__":
    
    print("Sentiment Model")
