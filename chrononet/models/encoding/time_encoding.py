#!/usr/bin/python3

# local imports
from models.model_base import ModelBase, ModelFactory
from models.ordering.pytorch_models import TimeEncoder

class TimeEncodingFactory(ModelFactory):

    requires_dim = True

    def get_model(dim, params):
        params['input_size'] = dim
        return TimeEncodingModel(**params)

class TimeEncodingModel(ModelBase):
    model = None

    def __init__(self, input_size, time_encoding_size, epochs=10, dropout=0.1):
        self.input_size = int(input_size)
        self.epochs = int(epochs)
        self.model = TimeEncoder(self.input_size, time_encoding_size, dropout_p=float(dropout))

    def fit(self, X, Y):
        self.model.fit(X, Y, self.epochs)

    def predict(self, X):
        return self.model.predict(X)
