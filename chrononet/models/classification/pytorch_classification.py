# @author sjeblee@cs.toronto.edu

import numpy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.modules.elmo import Elmo, batch_to_ids
from torch import optim

from models.ordering.pytorch_models import GRU_GRU, OrderGRU

numpy.set_printoptions(threshold=numpy.inf)
debug = True
tdevice = 'cpu'
use_cuda = torch.cuda.is_available()
if use_cuda:
    tdevice = torch.device('cuda:3')

options_file = "/u/sjeblee/research/data/elmo/weights/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "/u/sjeblee/research/data/elmo/weights/elmo_2x4096_512_2048cnn_2xhighway_weights_PubMed_only.hdf5"

######################################################################
# Convolutional Neural Network
# ----------------------------
#
# 5 convlutional layers with max pooling, followed by a fully connected network
# Arguments:
#	embed_dim	: dimension of a word vector
#	class_num	: number of classes
#	kernel_num	: number of channels produced by the convolutions
#	kernel_sizes	: size of convolving kernels
#	dropout		: dropout to prevent overfitting
#	ensemble	: if true, used as input of RNNClassifier
#			  if false, used independently and make prediction
#	hidden_size	: number of nodes in hidden layers
#
#
class ElmoCNN(nn.Module):

    def __init__(self, input_size, num_classes, num_epochs=10, dropout_p=0.1, kernel_num=100, kernel_sizes=5, loss_func='crossentropy', pad_size=10, reduce_size=40):
        super(ElmoCNN, self).__init__()
        #options_file = "/u/sjeblee/research/data/elmo/weights/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        #weight_file = "/u/sjeblee/research/data/elmo/weights/elmo_2x4096_512_2048cnn_2xhighway_weights_PubMed_only.hdf5"
        self.elmo = Elmo(options_file, weight_file, 1, dropout=0).to(tdevice)

        D = input_size
        C = num_classes
        Ci = 1
        self.Co = kernel_num
        self.Ks = kernel_sizes
        self.epochs = num_epochs
        self.loss_func = loss_func
        self.pad_size = pad_size
        self.reduce_size = reduce_size

        if self.reduce_size > 0:
            self.gru = nn.GRU(input_size, int(self.reduce_size/2), bidirectional=True, dropout=dropout_p, batch_first=True)
            D = self.reduce_size

        self.convs = []
        for kn in range(self.Ks):
            self.convs.append(nn.Conv2d(Ci, self.Co, (kn+1, D)))

        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(self.Co*self.Ks, C) # Use this layer when train with only CNN model, i.e. No ensemble
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        #print('x:', str(x))
        batch_size = len(x)
        character_ids = batch_to_ids(x).to(tdevice)
        embeddings = self.elmo(character_ids)['elmo_representations']
        #print('elmo embeddings:', embeddings[0].size())
        X = embeddings[0].view(batch_size, -1, 1024) # (N, W, D)

        # Pad to 10 words
        if X.size(1) > self.pad_size:
            X = X[:, 0:self.pad_size, :]
        elif X.size(1) < self.pad_size:
            pad = self.pad_size - X.size(1)
            zero_vec = torch.zeros(X.size(0), pad, X.size(2), device=tdevice)
            X = torch.cat((X, zero_vec), dim=1)

        if self.reduce_size > 0:
            X, hn = self.gru(X, None)

        x = X.unsqueeze(1)  # (N, Ci, W, D)]
        print('x size:', x.size())
        x_list = []
        for conv in self.convs:
            x_list.append(self.conv_and_pool(x, conv))
        x = torch.cat(x_list, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)

        logit = self.fc1(x)  # (N, C)
        return logit

    ''' Create and train a CNN model
        Hybrid features supported - pass structured feats as X2
        Does NOT support joint training yet
        returns: the CNN model
    '''
    def fit(self, X, Y):
        # Train the CNN, return the model
        #st = time.time()

        # Params
        #Xarray = numpy.asarray(X).astype('float')
        #dim = Xarray.shape[-1]
        #num_labels = Y.shape[-1]
        batch_size = 1
        learning_rate = 0.001

        if use_cuda:
            self = self.to(tdevice)
            for conv in self.convs:
                conv.to(tdevice)

        # Train final model
        self.train(X, Y, learning_rate, batch_size)

    def train(self, X, Y, learning_rate, batch_size):
        #Xarray = numpy.asarray(X).astype('float')
        Yarray = Y.astype('int')
        X_len = len(X)
        print('X len:', X_len)
        print('Y numpy shape:', str(Yarray.shape))
        steps = 0
        st = time.time()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        if self.loss_func == 'crossentropy':
            loss = nn.CrossEntropyLoss()
        else:
            print('ERROR: unrecognized loss function name')

        for epoch in range(self.epochs):
            print('epoch', str(epoch))
            i = 0
            numpy.random.seed(seed=1)
            perm = torch.from_numpy(numpy.random.permutation(X_len))
            permutation = perm.long()
            perm_list = perm.tolist()
            Xiter = [X[i] for i in perm_list]
            #Xiter = X[permutation]
            Yiter = Yarray[permutation]

            while i+batch_size < X_len:
                batchX = Xiter[i:i+batch_size]
                batchY = Yiter[i:i+batch_size]
                #Xtensor = torch.from_numpy(batchX).float()
                Ytensor = torch.from_numpy(batchY).long()
                if use_cuda:
                    #Xtensor = Xtensor.cuda()
                    Ytensor = Ytensor.to(tdevice)

                optimizer.zero_grad()
                logit = self(batchX)

                loss_val = loss(logit, Ytensor)
                #print('loss: ', loss_val.data.item())
                loss_val.backward()
                optimizer.step()
                steps += 1
                i = i+batch_size

            # Print epoch time
            ct = time.time() - st
            unit = "s"
            if ct > 60:
                ct = ct/60
                unit = "m"
            print("time so far: ", str(ct), unit)
            print('loss: ', loss_val.data.item())

    def predict(self, testX, testids=None, labelencoder=None, collapse=False, threshold=0.1, probfile=None):
        y_pred = [] # Original prediction if threshold is not in used for ill-defined.
        #new_y_pred = [] # class prediction if threshold for ill-difined is used.

        for x in range(len(testX)):
            input_row = testX[x]

            icd = None
            if icd is None:
                icd_var = self([input_row])
                # Softmax and log softmax values
                icd_vec = self.logsoftmax(icd_var).squeeze()
                #print('pred vector:', icd_vec.size(), icd_vec)
                #print('argmax:', torch.argmax(icd_vec))
                #icd_vec_softmax = softmax(icd_var)
                cat = torch.argmax(icd_vec).item()
                if x == 0:
                    print('cat:', cat)
                #icd_code = cat

            y_pred.append(cat)
        #print "Probabilities: " + str(probs)

        return y_pred  # Uncomment this line if threshold is not in used.
        #return new_y_pred  # Comment this line out if threshold is not in used.


class MatrixCNN(nn.Module):

    def __init__(self, input_size, num_classes, num_epochs=10, dropout_p=0.1, kernel_num=100, kernel_sizes=5, loss_func='crossentropy', pad_size=10):
        super(MatrixCNN, self).__init__()

        D = input_size
        self.dim = input_size
        C = num_classes
        Ci = 1
        self.Co = kernel_num
        self.Ks = kernel_sizes
        self.epochs = num_epochs
        self.loss_func = loss_func
        self.pad_size = pad_size

        self.convs = []
        for kn in range(self.Ks):
            self.convs.append(nn.Conv2d(Ci, self.Co, (kn+1, D)))#.double())

        self.dropout = nn.Dropout(dropout_p)#.double()
        self.fc1 = nn.Linear(self.Co*self.Ks, C)#.double() # Use this layer when train with only CNN model, i.e. No ensemble
        self.logsoftmax = nn.LogSoftmax(dim=1)#.double()

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        print('x input size:', x[0].size(), x)
        batch_size = len(x)
        print('batch size:', batch_size)
        #character_ids = batch_to_ids(x).to(tdevice)
        #embeddings = self.elmo(character_ids)['elmo_representations']
        #print('elmo embeddings:', embeddings[0].size())
        X = x[0].view(batch_size, -1, self.dim) # (N, W, D)

        # Pad to 10 words
        if X.size(1) > self.pad_size:
            X = X[:, 0:self.pad_size, :]
        elif X.size(1) < self.pad_size:
            pad = self.pad_size - X.size(1)
            zero_vec = torch.zeros(X.size(0), pad, X.size(2), device=tdevice)#.double()
            X = torch.cat((X, zero_vec), dim=1)

        x = X.unsqueeze(1)  # (N, Ci, W, D)]
        print('x pad size:', x.size())
        x_list = []
        for conv in self.convs:
            x_list.append(self.conv_and_pool(x, conv))
        x = torch.cat(x_list, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)

        logit = self.fc1(x)  # (N, C)
        return logit

    ''' Create and train a CNN model
        Hybrid features supported - pass structured feats as X2
        Does NOT support joint training yet
        returns: the CNN model
    '''
    def fit(self, X, Y):
        # Train the CNN, return the model
        #st = time.time()

        # Params
        #Xarray = numpy.asarray(X).astype('float')
        #dim = Xarray.shape[-1]
        #num_labels = Y.shape[-1]
        batch_size = 1
        learning_rate = 0.001

        if use_cuda:
            self = self.to(tdevice)
            for conv in self.convs:
                conv.to(tdevice)

        # Train final model
        self.train(X, Y, learning_rate, batch_size)

    def train(self, X, Y, learning_rate, batch_size):
        #Xarray = numpy.asarray(X).astype('float')
        Yarray = Y.astype('int')
        X_len = len(X)
        print('X len:', X_len)
        print('Y numpy shape:', str(Yarray.shape))
        steps = 0
        st = time.time()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        if self.loss_func == 'crossentropy':
            loss = nn.CrossEntropyLoss()
        else:
            print('ERROR: unrecognized loss function name')

        for epoch in range(self.epochs):
            print('epoch', str(epoch))
            i = 0
            numpy.random.seed(seed=1)
            perm = torch.from_numpy(numpy.random.permutation(X_len))
            permutation = perm.long()
            perm_list = perm.tolist()
            Xiter = [X[i] for i in perm_list]
            #Xiter = X[permutation]
            Yiter = Yarray[permutation]

            while i+batch_size < X_len:
                batchX = Xiter[i:i+batch_size]
                batchY = Yiter[i:i+batch_size]
                #Xtensor = torch.from_numpy(batchX).float()
                Ytensor = torch.from_numpy(batchY).long()
                if use_cuda:
                    #Xtensor = Xtensor.cuda()
                    Ytensor = Ytensor.to(tdevice)

                optimizer.zero_grad()
                logit = self(batchX)

                loss_val = loss(logit, Ytensor)
                print('loss: ', loss_val.data.item())
                loss_val.backward(retain_graph=True)
                optimizer.step()
                steps += 1
                i = i+batch_size

            # Print epoch time
            ct = time.time() - st
            unit = "s"
            if ct > 60:
                ct = ct/60
                unit = "m"
            print("time so far: ", str(ct), unit)
            print('loss: ', loss_val.data.item())

    def predict(self, testX, testids=None, labelencoder=None, collapse=False, threshold=0.1, probfile=None):
        y_pred = [] # Original prediction if threshold is not in used for ill-defined.
        #new_y_pred = [] # class prediction if threshold for ill-difined is used.

        for x in range(len(testX)):
            input_row = testX[x]

            icd = None
            if icd is None:
                icd_var = self([input_row])
                # Softmax and log softmax values
                icd_vec = self.logsoftmax(icd_var).squeeze()
                #print('pred vector:', icd_vec.size(), icd_vec)
                #print('argmax:', torch.argmax(icd_vec))
                #icd_vec_softmax = softmax(icd_var)
                cat = torch.argmax(icd_vec).item()
                if x == 0:
                    print('cat:', cat)
                #icd_code = cat

            y_pred.append(cat)
        #print "Probabilities: " + str(probs)

        return y_pred  # Uncomment this line if threshold is not in used.


####################################
# ELMo RNN model
####################################

class ElmoRNN(nn.Module):

    def __init__(self, input_size, hidden, num_classes, num_epochs=10, dropout_p=0.1, loss_func='crossentropy', batch_size=1):
        super(ElmoRNN, self).__init__()
        self.elmo = Elmo(options_file, weight_file, 1, dropout=0).to(tdevice)

        C = num_classes
        self.epochs = num_epochs
        self.loss_func = loss_func
        self.batch_size = batch_size
        self.hidden_size = hidden
        print('GRU model: input:', input_size, 'hidden:', hidden, 'output:', C)

        self.gru = nn.GRU(input_size=int(input_size), hidden_size=int(self.hidden_size/2), batch_first=True, bidirectional=True, dropout=dropout_p)
        self.fc1 = nn.Linear(self.hidden_size, C)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        hn = None
        batch_size = len(x)
        character_ids = batch_to_ids(x).to(tdevice)
        embeddings = self.elmo(character_ids)['elmo_representations']
        #print('elmo embeddings:', embeddings[0].size())
        X = embeddings[0].view(batch_size, -1, 1024) # (N, W, D)

        output, hn = self.gru(X, hn)
        output = output[:, -1, :].view(batch_size, -1) # Save only the last timestep
        #print('output:', output.size())
        out = self.fc1(output)
        return out

    ''' Create and train a CNN model
        Hybrid features supported - pass structured feats as X2
        Does NOT support joint training yet
        returns: the CNN model
    '''
    def fit(self, X, Y):
        # Train the CNN, return the model
        #batch_size = 16
        learning_rate = 0.001

        if use_cuda:
            self = self.to(tdevice)

        # Train final model
        self.train(X, Y, learning_rate)

    def train(self, X, Y, learning_rate):
        #Xarray = numpy.asarray(X).astype('float')
        Yarray = Y.astype('int')
        X_len = len(X)
        print('X len:', X_len)
        print('Y numpy shape:', str(Yarray.shape))
        print('batch_size:', self.batch_size)
        steps = 0
        st = time.time()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        if self.loss_func == 'crossentropy':
            loss = nn.CrossEntropyLoss()
        else:
            print('ERROR: unrecognized loss function name')

        for epoch in range(self.epochs):
            print('epoch', str(epoch))
            i = 0
            numpy.random.seed(seed=1)
            perm = torch.from_numpy(numpy.random.permutation(X_len))
            permutation = perm.long()
            perm_list = perm.tolist()
            Xiter = [X[i] for i in perm_list]
            #Xiter = X[permutation]
            Yiter = Yarray[permutation]

            while i+self.batch_size < X_len:
                batchX = Xiter[i:i+self.batch_size]
                batchY = Yiter[i:i+self.batch_size]
                #Xtensor = torch.from_numpy(batchX).float()
                Ytensor = torch.from_numpy(batchY).long()
                if use_cuda:
                    Ytensor = Ytensor.to(tdevice)

                optimizer.zero_grad()
                logit = self(batchX)
                #print('logit:', logit, 'ytensor:', Ytensor)

                loss_val = loss(logit, Ytensor)
                print('loss: ', loss_val.data.item())
                loss_val.backward(retain_graph=True)
                optimizer.step()
                steps += 1
                i = i+self.batch_size

            # Print epoch time
            ct = time.time() - st
            unit = "s"
            if ct > 60:
                ct = ct/60
                unit = "m"
            print("time so far: ", str(ct), unit)
            print('loss: ', loss_val.data.item())

    def predict(self, testX, testids=None, labelencoder=None, collapse=False, threshold=0.1, probfile=None):
        y_pred = [] # Original prediction if threshold is not in used for ill-defined.

        for x in range(len(testX)):
            input_row = testX[x]

            icd = None
            if icd is None:
                icd_var = self([input_row])
                # Softmax and log softmax values
                icd_vec = self.logsoftmax(icd_var).squeeze()
                cat = torch.argmax(icd_vec).item()
                if x == 0:
                    print('cat:', cat)

            y_pred.append(cat)

        return y_pred


class MatrixRNN(ElmoRNN):

    def __init__(self, input_size, hidden, num_classes, num_epochs=10, dropout_p=0.1, loss_func='crossentropy', batch_size=1):
        super(ElmoRNN, self).__init__()
        #self.elmo = Elmo(options_file, weight_file, 1, dropout=0).to(tdevice)

        C = num_classes
        self.epochs = num_epochs
        self.loss_func = loss_func
        self.batch_size = batch_size
        self.hidden_size = hidden
        self.dim = input_size
        print('GRU model: input:', input_size, 'hidden:', hidden, 'output:', C)

        self.gru = nn.GRU(input_size=int(input_size), hidden_size=int(self.hidden_size/2), batch_first=True, bidirectional=True, dropout=dropout_p)#.double()
        self.fc1 = nn.Linear(self.hidden_size, C)#.double()
        self.logsoftmax = nn.LogSoftmax(dim=1)#.double()

    def forward(self, x):
        hn = None
        batch_size = len(x)
        X = x[0].view(batch_size, -1, self.dim) # (N, W, D)
        #character_ids = batch_to_ids(x).to(tdevice)
        #embeddings = self.elmo(character_ids)['elmo_representations']
        #print('elmo embeddings:', embeddings[0].size())
        #X = embeddings[0].view(batch_size, -1, 1024) # (N, W, D)
        #print('rnn input:', X)

        output, hn = self.gru(X, hn)
        output = output[:, -1, :].view(batch_size, -1) # Save only the last timestep
        #print('output:', output.size())
        out = self.fc1(output)
        return out


####################################
# ELMo RNN model
####################################

class OrderRNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes, encoding_size=64, time_encoding_size=16, output_size=1, num_epochs=10, dropout_p=0.1, loss_func='crossentropy', batch_size=1, encoder_file=None):
        super(OrderRNN, self).__init__()
        #self.elmo = Elmo(options_file, weight_file, 1, dropout=0).to(tdevice)
        self.order_model = OrderGRU(input_size, int(encoding_size), int(time_encoding_size), hidden_size, output_size, encoder_file, dropout_p=0.1)

        C = num_classes
        self.epochs = num_epochs
        self.loss_func = loss_func
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        print('Joint order/classify model: input:', input_size, 'hidden:', hidden_size, 'output:', C, 'batch_size:', self.batch_size)

        #self.gru = nn.GRU(input_size=self.hidden_size, hidden_size=int(self.hidden_size/2), batch_first=True, bidirectional=True, dropout=dropout_p)
        self.fc1 = nn.Linear(self.hidden_size*2, C)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        hn = None
        batch_size = len(x)
        #print('x:', x)
        ranks, X = self.order_model.forward(x[0])

        #output, hn = self.gru(X, hn)
        output = X[:, -1, :].view(-1, self.hidden_size*2) # Save only the last timestep
        print('output:', output.size())
        out = self.fc1(output)
        return out, ranks

    ''' Create and train an RNN model
        Hybrid features supported - pass structured feats as X2
        Does NOT support joint training yet
        returns: the CNN model
    '''
    def fit(self, X, Y, Y2=None):
        # Train the CNN, return the model
        #batch_size = 16
        learning_rate = 0.001

        if use_cuda:
            self = self.to(tdevice)

        # Train final model
        self.train(X, Y, learning_rate, Y2)

    def train(self, X, Y, learning_rate, Y2=None):
        #Xarray = numpy.asarray(X).astype('float')
        Yarray = Y.astype('int')
        X_len = len(X)
        print('X len:', X_len)
        print('Y numpy shape:', str(Yarray.shape))
        print('Y2', len(Y2))
        print('batch_size:', self.batch_size)
        steps = 0
        st = time.time()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        if self.loss_func == 'crossentropy':
            loss = nn.CrossEntropyLoss()
        else:
            print('ERROR: unrecognized loss function name')

        loss2 = nn.MSELoss()

        for epoch in range(self.epochs):
            print('epoch', str(epoch))
            i = 0
            numpy.random.seed(seed=1)
            perm = torch.from_numpy(numpy.random.permutation(X_len))
            permutation = perm.long()
            perm_list = perm.tolist()
            Xiter = [X[i] for i in perm_list]
            #Xiter = X[permutation]
            Yiter = Yarray[permutation]
            if Y2 is not None:
                Y2iter = Y.astype('int')[permutation]

            while i+self.batch_size < X_len:
                batchX = Xiter[i:i+self.batch_size]
                batchY = Yiter[i:i+self.batch_size]
                #Xtensor = torch.from_numpy(batchX).float()
                Ytensor = torch.from_numpy(batchY).long()
                if use_cuda:
                    Ytensor = Ytensor.to(tdevice)

                if Y2 is not None:
                    batchY2 = Y2iter[i:i+self.batch_size]
                    Y2tensor = torch.from_numpy(batchY2).float()
                    if use_cuda:
                        Y2tensor = Y2tensor.to(tdevice)

                optimizer.zero_grad()
                output, ranks = self(batchX)

                # Rank loss
                #print('ranks:', ranks)
                #rank_loss = loss2(ranks, Y2tensor)
                #print('rank loss:', rank_loss)
                #rank_loss.backward(retain_graph=True)

                # Classification loss
                loss_val = loss(output, Ytensor)
                print('class loss: ', loss_val.data.item())
                loss_val.backward()
                optimizer.step()
                steps += 1
                i = i+self.batch_size

            # Print epoch time
            ct = time.time() - st
            unit = "s"
            if ct > 60:
                ct = ct/60
                unit = "m"
            print("time so far: ", str(ct), unit)
            print('loss: ', loss_val.data.item())

    def predict(self, testX, testids=None, labelencoder=None, collapse=False, threshold=0.1, probfile=None):
        y_pred = [] # Original prediction if threshold is not in used for ill-defined.

        for x in range(len(testX)):
            input_row = testX[x]

            icd = None
            if icd is None:
                icd_var, ranks = self([input_row])
                # Softmax and log softmax values
                icd_vec = self.logsoftmax(icd_var).squeeze()
                cat = torch.argmax(icd_vec).item()
                if x == 0:
                    print('cat:', cat)

            y_pred.append(cat)

        return y_pred
