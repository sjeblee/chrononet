#!/usr/bin/python3

import random

from models.model_base import ModelBase, ModelFactory

class RandomOrderFactory(ModelFactory):

    def get_model():
        return RandomOrderModel()

class RandomOrderModel(ModelBase):

    def fit(self, X, Y):
        pass # Does nothing because this model doesn't need training

    def predict(self, X): # TODO: add bin ratio for grouping events?
        ranks = []
        for n in range(len(X)):
            doc_ranks = []
            #num = math.ceil(len(events[n])*bin_ratio)
            for x in range(len(X[n])):
                doc_ranks.append(random.random())
            ranks.append(doc_ranks)
        return ranks

class MentionOrderFactory(ModelFactory):

    def get_model():
        return RandomOrderModel()

class MentionOrderModel(ModelBase):

    def fit(self, X, Y):
        pass # Does nothing because this model doesn't need training

    def predict(self, X):
        ranks = []
        for record in X:
            rank = 1
            doc_ranks = []
            for entry in record:
                doc_ranks.append(rank)
                rank += 1
        return ranks
