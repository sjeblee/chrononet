#!/usr/bin/python3

# local imports
from models.model_base import ModelBase, ModelFactory
from .pytorch_models import GRU_GRU

class NeuralOrderFactory(ModelFactory):

    requires_dim = True

    def get_model(dim):
        return NeuralOrderModel(dim)

class NeuralOrderModel(ModelBase):
    model = None
    input_size = 0
    encoding_size = 64
    hidden_size = 32
    output_size = 1
    dropout = 0.1
    epochs = 10

    def __init__(self, input_size):
        self.input_size = input_size+2 # Add 2 for the target phrase flag and polarity flag
        self.model = GRU_GRU(self.input_size, self.encoding_size, self.hidden_size, self.output_size, dropout_p=self.dropout)

    def fit(self, X, Y):
        self.model.fit(X, Y, num_epochs=self.epochs)

    def predict(self, X):
        return self.model.predict(X)
