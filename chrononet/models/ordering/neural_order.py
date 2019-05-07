#!/usr/bin/python3

from hyperopt import hp, fmin, tpe, space_eval

# local imports
from evaluation import ordering_metrics
from models.model_base import ModelBase, ModelFactory
from .pytorch_models import SetToSequence, SetToSequenceGroup, GRU_GRU

class NeuralOrderFactory(ModelFactory):

    requires_dim = True

    def get_model(dim):
        return NeuralOrderModel(dim)

class NeuralOrderModel(ModelBase):
    model = None
    input_size = 0
    encoding_size = 32
    time_encoding_size = 8
    hidden_size = encoding_size
    #hidden_size = (encoding_size*2) + time_encoding_size# + 2 + 2 # hidden_size needs to be equal to the final encoding size (encoding_size*2) + time_encoding
    output_size = 1
    dropout = 0.1
    epochs = 20
    read_cycles = 48
    group_thresh = 0.01 # Should be None for linear model
    sigma = 1
    invert = True # True for VA, False for THYME

    def __init__(self, input_size):
        self.input_size = input_size+0 # Add 2 for the target phrase flag and polarity flag
        #self.model = GRU_GRU(self.input_size, self.encoding_size, self.hidden_size, self.output_size, dropout_p=self.dropout)
        self.model = SetToSequenceGroup(self.input_size, self.encoding_size, self.time_encoding_size, self.hidden_size, self.output_size,
                                   dropout_p=self.dropout, read_cycles=self.read_cycles, group_thresh=self.group_thresh, invert_ranks=self.invert, sig=self.sigma)

    def fit(self, X, Y):
        self.model.fit(X, Y, num_epochs=self.epochs)

    def predict(self, X, group_thresh=0.01):
        if group_thresh is not None:
            self.model.group_thresh = group_thresh
        return self.model.predict(X)

class NeuralLinearFactory(ModelFactory):

    requires_dim = True

    def get_model(dim):
        return NeuralLinearModel(dim)

class NeuralLinearModel(ModelBase):
    model = None
    input_size = 0
    encoding_size = 32
    time_encoding_size = 2
    hidden_size = 32 # hidden_size needs to be equal to the final encoding size (encoding_size*2) + time_encoding
    output_size = 1
    dropout = 0.2
    epochs = 10
    read_cycles = 48
    group_thresh = None # Should be None for linear model
    invert = False # True for VA, False for THYME

    def __init__(self, input_size):
        self.input_size = input_size # Add 2 for the target phrase flag and polarity flag
        self.model = GRU_GRU(self.input_size, self.encoding_size, self.hidden_size, self.output_size, dropout_p=self.dropout)
        #self.model = SetToSequence(self.input_size, self.encoding_size, self.time_encoding_size, self.hidden_size, self.output_size,
        #                           dropout_p=self.dropout, read_cycles=self.read_cycles, group_thresh=self.group_thresh, invert_ranks=self.invert)

    def fit(self, X, Y):
        self.model.fit(X, Y, num_epochs=self.epochs)

    def predict(self, X):
        return self.model.predict(X)


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
