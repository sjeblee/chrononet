#!/usr/bin/python3

import os

from hyperopt import hp, fmin, tpe, space_eval

# local imports
from evaluation import ordering_metrics
from models.model_base import ModelBase, ModelFactory
from .pytorch_models import SetToSequence, SetToSequenceGroup, GRU_GRU, OrderGRU
from .transformer_models import SetOrder

class SetOrderFactory(ModelFactory):

    requires_dim = True

    def get_model(dim, params):
        params['input_size'] = dim
        print('model params:', params)
        return SetOrderModel(**params)

class SetOrderModel(ModelBase):
    model = None
    input_size = 0
    output_size = 1
    #invert = False # True for VA, False for THYME

    def __init__(self, input_size, encoding_size=16, time_encoding_size=8, hidden_size=40, dropout=0.1, encoder_file=None, read_cycles=48, group_thresh=0.001, sigma=1, epochs=15, invert=False, use_autoencoder=False, ae_file=None, checkpoint_dir=None):
        self.input_size = int(input_size)+0 # Add 2 for the target phrase flag and polarity flag
        self.group_thresh = float(group_thresh)
        self.epochs = int(epochs)
        use_ae = (str(use_autoencoder) == 'True')
        ae_file = os.path.join(checkpoint_dir, 'autoencoder.model')
        self.model = SetOrder(self.input_size, int(encoding_size), int(time_encoding_size), int(hidden_size), output_size=self.output_size,
                                  dropout_p=float(dropout),
                                  encoder_file=encoder_file,
                                  use_autoencoder=use_ae, autoencoder_file=ae_file,
                                  checkpoint_dir=checkpoint_dir)

    def fit(self, X, Y):
        self.model.fit(X, Y, num_epochs=self.epochs)

    def predict(self, X, group_thresh=0.001, return_encodings=False):
        if group_thresh is not None:
            self.model.group_thresh = group_thresh
        return self.model.predict(X, return_encodings=return_encodings)

class NeuralOrderFactory(ModelFactory):

    requires_dim = True

    def get_model(dim, params):
        params['input_size'] = dim
        print('model params:', params)
        return NeuralOrderModel(**params)

class NeuralOrderModel(ModelBase):
    model = None
    input_size = 0
    output_size = 1
    group_thresh = None
    #invert = False # True for VA, False for THYME

    def __init__(self, input_size, encoding_size=16, time_encoding_size=8, hidden_size=40, dropout=0.1, read_cycles=48, group_thresh=0.01, sigma=1, epochs=15, invert=False, encoder_file=None, use_autoencoder=False, ae_file=None, checkpoint_dir=None):
        self.input_size = int(input_size)+0 # Add 2 for the target phrase flag and polarity flag
        self.group_thresh = float(group_thresh)
        self.epochs = int(epochs)
        use_ae = (str(use_autoencoder) == 'True')
        #ae_file = '/h/sjeblee/research/data/thyme/chrono/orders2s_thyme_128_32/autoencoder.model'
        ae_file = os.path.join(checkpoint_dir, 'autoencoder.model')
        #self.model = GRU_GRU(self.input_size, self.encoding_size, self.hidden_size, self.output_size, dropout_p=self.dropout)
        self.model = SetToSequenceGroup(self.input_size, int(encoding_size), int(encoding_size), int(time_encoding_size), int(hidden_size), self.output_size,
                                        dropout_p=float(dropout), read_cycles=int(read_cycles), group_thresh=self.group_thresh,
                                        invert_ranks=bool(invert), sig=float(sigma), time_encoder_file=encoder_file, use_autoencoder=use_ae,
                                        autoencoder_file=ae_file, checkpoint_dir=checkpoint_dir)

    def fit(self, X, Y):
        self.model.fit(X, Y, num_epochs=self.epochs)

    def predict(self, X, group_thresh=0.1, return_encodings=False):
        if group_thresh is not None:
            self.model.group_thresh = group_thresh
        return self.model.predict(X, return_encodings=return_encodings)

# Linear neural model (GRU_GRU)
class NeuralLinearFactory(ModelFactory):

    requires_dim = True

    def get_model(dim, params):
        params['input_size'] = dim
        return NeuralLinearModel(**params)

class NeuralLinearModel(ModelBase):
    model = None
    input_size = 0
    output_size = 1
    invert = False # True for VA, False for THYME

    def __init__(self, input_size, encoder_file, encoding_size=32, time_encoding_size=2, hidden_size=32, dropout=0.2, read_cycles=48, epochs=10, use_autoencoder=False, ae_file=None, checkpoint_dir=None):
        self.input_size = int(input_size) # Add 2 for the target phrase flag and polarity flag
        self.epochs = int(epochs)
        use_ae = (str(use_autoencoder) == 'True')
        ae_file = os.path.join(checkpoint_dir, 'autoencoder.model')
        self.model = OrderGRU(self.input_size, int(encoding_size), int(time_encoding_size), int(hidden_size), self.output_size, encoder_file,
                              dropout_p=float(dropout), use_autoencoder=use_ae, ae_file=ae_file, checkpoint_dir=checkpoint_dir)
        #self.model = SetToSequence(self.input_size, self.encoding_size, self.time_encoding_size, self.hidden_size, self.output_size,
        #                           dropout_p=self.dropout, read_cycles=self.read_cycles, group_thresh=self.group_thresh, invert_ranks=self.invert)

    def fit(self, X, Y):
        self.model.fit(X, Y, num_epochs=self.epochs)

    def predict(self, X, return_encodings=False):
        return self.model.predict(X, return_encodings=return_encodings)

# HYPEROPT

class HyperoptNeuralOrderFactory(ModelFactory):

    requires_dim = True

    def get_model(dim):
        return HyperoptNeuralOrder(dim)

class HyperoptNeuralOrder(ModelBase):

    trainX = None
    trainY = None
    testX = None
    testY = None

    def __init__(self, input_size):
        self.input_size = input_size+0 # Add 2 for the target phrase flag and polarity flag

    def fit(self, X, Y):
        self.trainX = X
        self.trainY = Y

    def predict(self, testX, testY):
        # Set up data file references
        self.testX = testX
        self.testY = testY

        objective = self.obj_neural
        space = {
            'encoding_size': hp.choice('encoding_size', [16, 24, 32]),
            'time_size': hp.choice('time_size', [8, 16, 24, 32]),
            'read_cycles': hp.uniform('read_cycles', 5, 50),
            'epochs': hp.uniform('epochs', 10, 30),
            'group_thresh': hp.uniform('group_thresh', 0.001, 0.5),
            'sigma': hp.uniform('sigma', 0.5, 3),
            'dropout': hp.uniform('dropout', 0.0, 0.4),
            'invert_ranks': hp.choice('invert_ranks', [('yes', True), ('no', False)])
        }

        # Run hyperopt
        print("space: ", str(space))
        best = fmin(objective, space, algo=tpe.suggest, max_evals=50)
        print(best)
        return self.testY

    def obj_neural(self, params):
        output_size = 1
        encoding_size = int(params['encoding_size'])
        time_encoding_size = int(params['time_size'])
        hidden_size = (2*encoding_size) + time_encoding_size
        dropout = float(params['dropout'])
        sigma = float(params['sigma'])
        group_thresh = float(params['group_thresh'])
        read_cycles = int(params['read_cycles'])
        epochs = int(params['epochs'])
        invert = params['invert_ranks'][0]

        # Create and train the model
        self.model = SetToSequenceGroup(self.input_size, encoding_size, time_encoding_size, hidden_size, output_size,
                                   dropout_p=dropout, read_cycles=read_cycles, group_thresh=group_thresh, invert_ranks=invert, sig=sigma)
        self.model.epochs = epochs
        self.model.fit(self.trainX, self.trainY)

        # Test the model
        test_pred = self.model.predict(self.testX)

        # Score the model (Tau?)
        score = ordering_metrics.kendalls_tau(self.testY, test_pred)
        print('Tau:', score, 'Parameters: encoding:', encoding_size, 'time:', time_encoding_size, 'hidden:', hidden_size, 'sigma:', sigma,
              'group_thresh:', group_thresh, 'dropout:', dropout, 'invert_ranks:', invert, 'read_cycles:', read_cycles, 'epochs:', epochs)

        # Return a score to minimize
        return 1 - score
