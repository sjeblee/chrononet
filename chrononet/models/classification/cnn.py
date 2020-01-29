#!/usr/bin/python3

# local imports
from models.model_base import ModelBase, ModelFactory
from .pytorch_classification import ElmoCNN, MatrixCNN, ElmoRNN, MatrixRNN, OrderRNN

class CNNFactory(ModelFactory):

    requires_dim = True

    def get_model(dim, params):
        params['input_size'] = dim
        return ElmoCNNModel(**params)

class ElmoCNNModel(ModelBase):
    model = None

    def __init__(self, input_size, num_classes, epochs=10, dropout=0.1, kernels=50, ksizes=5, pad_size=10):
        self.input_size = int(input_size)
        self.epochs = int(epochs)
        self.model = ElmoCNN(self.input_size, int(num_classes), num_epochs=self.epochs, dropout_p=float(dropout), kernel_num=int(kernels),
                             kernel_sizes=int(ksizes), pad_size=int(pad_size))

    def fit(self, X, Y):
        self.model.fit(X, Y)

    def predict(self, X):
        return self.model.predict(X)


class MatrixCNNFactory(ModelFactory):
    requires_dim = True

    def get_model(dim, params):
        params['input_size'] = dim
        return MatrixCNNModel(**params)

class MatrixCNNModel(ModelBase):
    model = None

    def __init__(self, input_size, num_classes, epochs=10, dropout=0.1, kernels=50, ksizes=5, pad_size=10):
        self.input_size = int(input_size)
        self.epochs = int(epochs)
        self.model = MatrixCNN(self.input_size, int(num_classes), num_epochs=self.epochs, dropout_p=float(dropout), kernel_num=int(kernels),
                               kernel_sizes=int(ksizes), pad_size=int(pad_size))

    def fit(self, X, Y):
        self.model.fit(X, Y)

    def predict(self, X):
        return self.model.predict(X)


class RNNFactory(ModelFactory):
    requires_dim = True

    def get_model(dim, params):
        params['input_size'] = dim
        return ElmoRNNModel(**params)

class ElmoRNNModel(ElmoCNNModel):
    model = None

    def __init__(self, input_size, hidden_size, num_classes, encoding_size=32, time_encoding_size=32, epochs=10, dropout=0.1, kernels=50, ksizes=5, pad_size=10, encoder_file=None):
        self.input_size = int(input_size)
        self.epochs = int(epochs)
        #self.model = ElmoRNN(self.input_size, int(hidden_size), int(num_classes), num_epochs=self.epochs, dropout_p=float(dropout))
        self.model = OrderRNN(self.input_size, int(hidden_size), int(num_classes), int(encoding_size), int(time_encoding_size), num_epochs=self.epochs, encoder_file=encoder_file, dropout_p=float(dropout))

    def fit(self, X, Y, Y2):
        self.model.fit(X, Y, Y2)


class MatrixRNNFactory(ModelFactory):
    requires_dim = True

    def get_model(dim, params):
        params['input_size'] = dim
        return MatrixRNNModel(**params)

class MatrixRNNModel(ElmoCNNModel):
    model = None

    def __init__(self, input_size, hidden_size, num_classes, epochs=10, dropout=0.1, kernels=50, ksizes=5, pad_size=10):
        self.input_size = int(input_size)
        self.epochs = int(epochs)
        self.model = MatrixRNN(self.input_size, int(hidden_size), int(num_classes), num_epochs=self.epochs, dropout_p=float(dropout))
