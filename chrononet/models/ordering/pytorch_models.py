#!/usr/bin/python3

import numpy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

numpy.set_printoptions(threshold=numpy.inf)
debug = True
tdevice = 'cpu'
use_cuda = torch.cuda.is_available()
if use_cuda:
    tdevice = torch.device('cuda:1')

# GRU with GRU encoder, input: (conversations (1), utterances, words, embedding_dim)
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_p=0.1):
        super(GRU_GRU, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        #self.encoding_size = encoding_size
        self.output_size = output_size
        self.dropout = dropout_p
        #self.gru0 = nn.GRU(input_size, int(encoding_size/2), bidirectional=True, dropout=dropout_p, batch_first=True)
        self.gru1 = nn.GRU(input_size, hidden_size, bidirectional=True, dropout=dropout_p, batch_first=True)
        self.linear = nn.Linear(hidden_size*2, output_size)
        self.softmax = nn.Softmax(dim=2)

    ''' Input is a list of lists of numpy arrays
    '''
    def forward(self, input, X2=None):
        # input expected as (events, words, embedding_dim)
        encodings = []
        hn = None
        index = 0
        extra_size = 0
        for row in input:
            # Create a tensor
            uttXnp = numpy.asarray(row).astype('float')
            uttX = torch.tensor(uttXnp, dtype=torch.float, device=tdevice)
            uttX = uttX.view(1, -1, self.input_size) # should be (1, #words, input_dim)
            #print('input tensor:', uttX)
            encoding, hn = self.gru0(uttX, hn) # should be (1, #words, encoding_dim)
            enc = encoding[:, -1, :].view(self.encoding_size) # Only keep the output of the last timestep
            #if debug: print('enc:', str(enc.size()), enc)

            # Add other features
            if X2 is not None:
                x2_np = numpy.asarray([X2[index]]).astype('float')
                x2_vec = torch.tensor(x2_np, dtype=torch.float, device=tdevice)
                if debug: print('x2:', x2_vec.size())
                extra_size = x2_vec.size()[0]
                enc = torch.cat((enc, x2_vec), dim=0)
            encodings.append(enc)
            index += 1

        conversation = torch.stack(encodings)
        conversation = conversation.view(1, -1, self.encoding_size+extra_size) # Treat whole conversation as a batch
        print('conversation:', conversation.size())
        output, hn = self.gru1(conversation, None)
        print('output:', output.size())
        #out1 = self.linear(output)
        out1 = self.softmax(self.linear(output))
        print('final out:', out1.size(), out1)
        return out1

    def initHidden(self, N):
        return torch.randn(1, N, self.hidden_size, device=tdevice).double()

    ''' Creates and trains a recurrent neural network model. Supports SimpleRNN, LSTM, and GRU
        X: a list of training data
        Y: a list of training labels
        WARNING: Currently you can't use the encoding layer and use_prev_labels at the same time
    '''
    def fit(self, X, Y, activation='relu', num_epochs=10, batch_size=1, loss_function='l1', X2=None):
        start = time.time()

        # Parameters
        hidden_size = self.hidden_size
        encoding_size = self.encoding_size
        dropout = self.dropout
        output_dim = self.output_size
        learning_rate = 0.1
        print_every = 100
        #teacher_forcing_ratio = 0.9

        print("hidden_size:", str(hidden_size), "dropout:", str(dropout), "epochs:", str(num_epochs))
        print("encoding size:", str(encoding_size), "(0 means utterances are already encoded and input should be 3 dims)")

        if batch_size > 1:
            if type(X) is list and batch_size > 1:
                X = numpy.asarray(X)
            if type(Y) is list and batch_size > 1:
                Y = numpy.asarray(Y)
            X = X.astype('float')
            Y = Y.astype('float')

            num_examples = X.shape[0]
            input_dim = X.shape[-1]
            max_length = Y.shape[1]
            #output_dim = Y.shape[-1]
            print("X:", str(X.shape), "Y:", str(Y.shape))
            #print("max_length: ", str(max_length))

        else: # Leave X and Y as lists
            num_examples = len(X)
            if encoding_size > 0:
                print("X 000:", str(type(X[0][0][0])), "Y 00:", str(type(Y[0][0])))
                input_dim = X[0][0][0].shape[0]
                #output_dim = Y[0][0].shape[0]
            else:
                input_dim = len(X[0][0])
                #output_dim = len(Y[0][0])
            print("X list:", str(len(X)), "Y list:", str(len(Y)))

        if X2 is not None:
            print("X2 list:", len(X2))

        print("input_dim: ", str(input_dim))
        print("output_dim: ", str(output_dim))

        # Set up optimizer and loss function
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        if loss_function == 'cosine':
            criterion = nn.CosineEmbeddingLoss()
        elif loss_function == 'crossentropy':
            criterion = nn.CrossEntropyLoss()
        elif loss_function == 'mse':
            criterion = nn.MSELoss()
        elif loss_function == 'l1':
            criterion = nn.L1Loss()
        else:
            print("WARNING: need to add loss function!")

        if use_cuda:
            self = self.to(tdevice)
            #criterion = criterion.cuda()

        # Train the model
        for epoch in range(num_epochs):
            print("epoch", str(epoch))
            i = 0
            while (i+batch_size) < num_examples:
                if i % print_every == 0:
                    print("batch i=", str(i))

                # Make sure the data is in the proper numpy array format
                if batch_size == 1:
                    batchXnp = X[i]
                    batchYnp = Y[i]
                    if debug: print("batchX len:", str(len(batchXnp)), "batchY len:", str(len(batchYnp)))
                    batchX2 = None
                    if X2 is not None:
                        batchX2np = X2[i]
                        if debug:
                            print("batchX2:", len(batchX2np), batchX2np)
                    #if type(batchXnp) is list:
                    #    batchXnp = numpy.asarray(batchXnp)
                    if type(batchYnp) is list:
                        batchYnp = numpy.asarray(batchYnp)
                    #print("batchX shape:", str(batchXnp.shape), "batchY shape;", str(batchYnp.shape))
                    if debug: print("batchY shape:", str(batchYnp.shape))
                else:
                    batchXnp = X[i:i+batch_size]
                    batchYnp = Y[i:i+batch_size]
                    if X2 is not None:
                        batchX2np = X2[i:i+batch_size]

                if encoding_size > 0:
                    batchX = batchXnp
                    batchY = batchYnp
                    if X2 is not None:
                        batchX2 = batchX2np
                    else:
                        batchX2 = None

                # Convert to tensors
                #batchXnp = batchXnp.astype('float')
                #    batchX = torch.cuda.FloatTensor(batchXnp)
                batchYnp = batchYnp.astype('float')
                batchY = torch.tensor(batchYnp, dtype=torch.float, device=tdevice)

                #print("batchX size:", str(batchX.size()), "batchY size:", str(batchY.size()))
                if debug: print("batchX[0]:", str(batchX[0]))
                if debug: print("batchY size:", str(batchY.size()))

                labels = batchY.view(batch_size, -1, output_dim)
                max_length = labels.size(1)
                if debug: print("max_length:", str(max_length))
                #if debug: print("batchX:", str(batchX.size()), "batchY:", str(batchY.size()))

                # Forward + Backward + Optimize
                optimizer.zero_grad()  # zero the gradient buffer
                loss = 0

                if encoding_size > 0:
                    outputs = self(batchX, batchX2).view(max_length, -1)
                else:
                    print('ERROR: Encoding size is 0 and I dont know what to do')
                    #outputs = self(samples).view(max_length, -1)
                #if debug: print("outputs:", str(outputs.size()))
                if loss_function == 'crossentropy':
                    for b in range(batch_size):
                        true_labels = torch.zeros(max_length).long()
                        if use_cuda:
                            true_labels = true_labels.cuda()
                        print('true_labels size:', str(true_labels.size()))
                        print('labels[b]', str(len(labels[b])))
                        for y in range(len(labels[b])):
                            true_label = labels[b][y].data
                            #print("true_label:", str(true_label.size()))
                            true_index = torch.max(true_label, 0)[1].long()
                            #print("true_index", str(true_index.size()))
                            true_labels[y] = true_index[0]
                        true_var = true_labels
                        print("true_var", str(true_var.size()))
                        loss = criterion(outputs, true_var)
                        loss.backward()
                        optimizer.step()
                else:
                    labels = labels.view(max_length, 1)

                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                if (i) % print_every == 0:
                    if debug: print('outputs:', outputs.size(), 'labels:', labels.size())
                    if debug: print('outputs:', outputs, 'labels:', labels)
                    print('Epoch [%d/%d], Loss: %.4f' %(epoch, num_epochs, loss.data.item()))
                i = i+batch_size

                del batchX
                del batchY
        print("GRU_GRU training took", str(time.time()-start), "s")

    def predict(self, testX, X2=None, batch_size=1, keep_list=True):
        # Test the Model
        print_every = 1000
        pred = []
        i = 0
        length = len(testX)# .shape[0]
        if debug: print("testX len:", str(len(testX)))
        while i < length:
            if i % print_every == 0:
                if debug: print("test batch", str(i))
            if (i+batch_size) > length:
                batch_size = length-i
            if keep_list:
                if batch_size == 1:
                    samples = testX[i]
                    if X2 is not None:
                        x2_batch = X2[i]
                else:
                    samples = testX[i:i+batch_size]
                    if X2 is not None:
                        x2_batch = X2[i:i+batch_size]
                if debug: print("samples:", str(len(samples)))
            else:
                x_array = numpy.asarray(testX[i:i+batch_size]).astype('float')
                if debug: print("test x_array:", str(x_array.shape))
                samples = torch.tensor(x_array, dtype=torch.float, device=tdevice)

            if X2 is not None:
                outputs = self(samples, x2_batch)
            else:
                outputs = self(samples)
            print("test outputs:", str(outputs.size()))
            num_items = outputs.size()[1]
            predicted = outputs.view(num_items).tolist()
            #_, predicted = torch.max(outputs.data, -1) # TODO: fix this
            print('predicted:', predicted)
            pred.append(predicted)
            if not keep_list:
                del sample_tensor
            i = i+batch_size
        return pred


# GRU with GRU encoder, input: (conversations (1), utterances, words, embedding_dim)
class GRU_GRU(nn.Module):
    def __init__(self, input_size, encoding_size, hidden_size, output_size, dropout_p=0.1):
        super(GRU_GRU, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.encoding_size = encoding_size
        self.output_size = output_size
        self.dropout = dropout_p
        self.gru0 = nn.GRU(input_size, int(encoding_size/2), bidirectional=True, dropout=dropout_p, batch_first=True)
        self.gru1 = nn.GRU(encoding_size, hidden_size, bidirectional=True, dropout=dropout_p, batch_first=True)
        self.linear = nn.Linear(hidden_size*2, output_size)
        self.softmax = nn.Softmax(dim=2)

    ''' Input is a list of lists of numpy arrays
    '''
    def forward(self, input, X2=None):
        # input expected as (events, words, embedding_dim)
        encodings = []
        hn = None
        index = 0
        extra_size = 0
        for row in input:
            # Create a tensor
            uttXnp = numpy.asarray(row).astype('float')
            uttX = torch.tensor(uttXnp, dtype=torch.float, device=tdevice)
            uttX = uttX.view(1, -1, self.input_size) # should be (1, #words, input_dim)
            #print('input tensor:', uttX)
            encoding, hn = self.gru0(uttX, hn) # should be (1, #words, encoding_dim)
            enc = encoding[:, -1, :].view(self.encoding_size) # Only keep the output of the last timestep
            #if debug: print('enc:', str(enc.size()), enc)

            # Add other features
            if X2 is not None:
                x2_np = numpy.asarray([X2[index]]).astype('float')
                x2_vec = torch.tensor(x2_np, dtype=torch.float, device=tdevice)
                if debug: print('x2:', x2_vec.size())
                extra_size = x2_vec.size()[0]
                enc = torch.cat((enc, x2_vec), dim=0)
            encodings.append(enc)
            index += 1

        conversation = torch.stack(encodings)
        conversation = conversation.view(1, -1, self.encoding_size+extra_size) # Treat whole conversation as a batch
        print('conversation:', conversation.size())
        output, hn = self.gru1(conversation, None)
        print('output:', output.size())
        out1 = self.linear(output)
        #out1 = self.softmax(self.linear(output))
        print('final out:', out1.size(), out1)
        return out1

    def initHidden(self, N):
        return torch.randn(1, N, self.hidden_size, device=tdevice).double()

    ''' Creates and trains a recurrent neural network model. Supports SimpleRNN, LSTM, and GRU
        X: a list of training data
        Y: a list of training labels
        WARNING: Currently you can't use the encoding layer and use_prev_labels at the same time
    '''
    def fit(self, X, Y, activation='relu', num_epochs=10, batch_size=1, loss_function='mse', X2=None):
        start = time.time()

        # Parameters
        hidden_size = self.hidden_size
        encoding_size = self.encoding_size
        dropout = self.dropout
        output_dim = self.output_size
        learning_rate = 0.01
        print_every = 100
        #teacher_forcing_ratio = 0.9

        print("hidden_size:", str(hidden_size), "dropout:", str(dropout), "epochs:", str(num_epochs))
        print("encoding size:", str(encoding_size), "(0 means utterances are already encoded and input should be 3 dims)")

        if batch_size > 1:
            if type(X) is list and batch_size > 1:
                X = numpy.asarray(X)
            if type(Y) is list and batch_size > 1:
                Y = numpy.asarray(Y)
            X = X.astype('float')
            Y = Y.astype('float')

            num_examples = X.shape[0]
            input_dim = X.shape[-1]
            max_length = Y.shape[1]
            #output_dim = Y.shape[-1]
            print("X:", str(X.shape), "Y:", str(Y.shape))
            #print("max_length: ", str(max_length))

        else: # Leave X and Y as lists
            num_examples = len(X)
            if encoding_size > 0:
                print("X 000:", str(type(X[0][0][0])), "Y 00:", str(type(Y[0][0])))
                input_dim = X[0][0][0].shape[0]
                #output_dim = Y[0][0].shape[0]
            else:
                input_dim = len(X[0][0])
                #output_dim = len(Y[0][0])
            print("X list:", str(len(X)), "Y list:", str(len(Y)))

        if X2 is not None:
            print("X2 list:", len(X2))

        print("input_dim: ", str(input_dim))
        print("output_dim: ", str(output_dim))

        # Set up optimizer and loss function
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        if loss_function == 'cosine':
            criterion = nn.CosineEmbeddingLoss()
        elif loss_function == 'crossentropy':
            criterion = nn.CrossEntropyLoss()
        elif loss_function == 'mse':
            criterion = nn.MSELoss()
        elif loss_function == 'l1':
            criterion = nn.L1Loss()
        else:
            print("WARNING: need to add loss function!")

        if use_cuda:
            self = self.to(tdevice)
            #criterion = criterion.cuda()

        # Train the model
        for epoch in range(num_epochs):
            print("epoch", str(epoch))
            i = 0
            while (i+batch_size) < num_examples:
                if i % print_every == 0:
                    print("batch i=", str(i))

                # Make sure the data is in the proper numpy array format
                if batch_size == 1:
                    batchXnp = X[i]
                    batchYnp = Y[i]
                    if debug: print("batchX len:", str(len(batchXnp)), "batchY len:", str(len(batchYnp)))
                    batchX2 = None
                    if X2 is not None:
                        batchX2np = X2[i]
                        if debug:
                            print("batchX2:", len(batchX2np), batchX2np)
                    #if type(batchXnp) is list:
                    #    batchXnp = numpy.asarray(batchXnp)
                    if type(batchYnp) is list:
                        batchYnp = numpy.asarray(batchYnp)
                    #print("batchX shape:", str(batchXnp.shape), "batchY shape;", str(batchYnp.shape))
                    if debug: print("batchY shape:", str(batchYnp.shape))
                else:
                    batchXnp = X[i:i+batch_size]
                    batchYnp = Y[i:i+batch_size]
                    if X2 is not None:
                        batchX2np = X2[i:i+batch_size]

                if encoding_size > 0:
                    batchX = batchXnp
                    batchY = batchYnp
                    if X2 is not None:
                        batchX2 = batchX2np
                    else:
                        batchX2 = None

                # Convert to tensors
                #batchXnp = batchXnp.astype('float')
                #    batchX = torch.cuda.FloatTensor(batchXnp)
                batchYnp = batchYnp.astype('float')
                batchY = torch.tensor(batchYnp, dtype=torch.float, device=tdevice)

                #print("batchX size:", str(batchX.size()), "batchY size:", str(batchY.size()))
                if debug: print("batchX[0]:", str(batchX[0]))
                if debug: print("batchY size:", str(batchY.size()))

                labels = batchY.view(batch_size, -1, output_dim)
                max_length = labels.size(1)
                if debug: print("max_length:", str(max_length))
                #if debug: print("batchX:", str(batchX.size()), "batchY:", str(batchY.size()))

                # Forward + Backward + Optimize
                optimizer.zero_grad()  # zero the gradient buffer
                loss = 0

                if encoding_size > 0:
                    outputs = self(batchX, batchX2).view(max_length, -1)
                else:
                    print('ERROR: Encoding size is 0 and I dont know what to do')
                    #outputs = self(samples).view(max_length, -1)
                #if debug: print("outputs:", str(outputs.size()))
                if loss_function == 'crossentropy':
                    for b in range(batch_size):
                        true_labels = torch.zeros(max_length).long()
                        if use_cuda:
                            true_labels = true_labels.cuda()
                        print('true_labels size:', str(true_labels.size()))
                        print('labels[b]', str(len(labels[b])))
                        for y in range(len(labels[b])):
                            true_label = labels[b][y].data
                            #print("true_label:", str(true_label.size()))
                            true_index = torch.max(true_label, 0)[1].long()
                            #print("true_index", str(true_index.size()))
                            true_labels[y] = true_index[0]
                        true_var = true_labels
                        print("true_var", str(true_var.size()))
                        loss = criterion(outputs, true_var)
                        loss.backward()
                        optimizer.step()
                else:
                    labels = labels.view(max_length, 1)

                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                if (i) % print_every == 0:
                    if debug: print('outputs:', outputs.size(), 'labels:', labels.size())
                    if debug: print('outputs:', outputs, 'labels:', labels)
                    print('Epoch [%d/%d], Loss: %.4f' %(epoch, num_epochs, loss.data.item()))
                i = i+batch_size

                del batchX
                del batchY
        print("GRU_GRU training took", str(time.time()-start), "s")

    def predict(self, testX, X2=None, batch_size=1, keep_list=True):
        # Test the Model
        print_every = 1000
        pred = []
        i = 0
        length = len(testX)# .shape[0]
        if debug: print("testX len:", str(len(testX)))
        while i < length:
            if i % print_every == 0:
                if debug: print("test batch", str(i))
            if (i+batch_size) > length:
                batch_size = length-i
            if keep_list:
                if batch_size == 1:
                    samples = testX[i]
                    if X2 is not None:
                        x2_batch = X2[i]
                else:
                    samples = testX[i:i+batch_size]
                    if X2 is not None:
                        x2_batch = X2[i:i+batch_size]
                if debug: print("samples:", str(len(samples)))
            else:
                x_array = numpy.asarray(testX[i:i+batch_size]).astype('float')
                if debug: print("test x_array:", str(x_array.shape))
                samples = torch.tensor(x_array, dtype=torch.float, device=tdevice)

            if X2 is not None:
                outputs = self(samples, x2_batch)
            else:
                outputs = self(samples)
            print("test outputs:", str(outputs.size()))
            num_items = outputs.size()[1]
            predicted = outputs.view(num_items).tolist()
            #_, predicted = torch.max(outputs.data, -1) # TODO: fix this
            print('predicted:', predicted)
            pred.append(predicted)
            if not keep_list:
                del sample_tensor
            i = i+batch_size
        return pred
