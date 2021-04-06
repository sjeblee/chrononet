#!/usr/bin/python3

import numpy

# local imports
from models.model_base import ModelBase, ModelFactory

class RandomFactory(ModelFactory):

    requires_dim = True

    def get_model(dim, params):
        params['input_size'] = dim
        return RandomModel(**params)

class RandomModel(ModelBase):
    model = None

    def __init__(self, num_classes=None, input_size=None, epochs=10):
        self.model = None
        self.prior = []

    def fit(self, X, Y):
        Y = Y.tolist()
        self.labels = list(set(Y))
        num_classes = len(self.labels)
        print('num classes:', str(num_classes))
        print('classes:', str(self.labels))
        self.prior = [1] * num_classes
        for label in self.labels:
            self.prior[int(label)] = Y.count(label)/float(len(Y))
        print('Learned distribution:', self.prior)

    def predict(self, X):
        #num = len(X) # size of the test set
        pred_labels = []
        for x in X:
            lab = numpy.random.choice(self.labels, p=self.prior)
            pred_labels.append(lab)
        return pred_labels
