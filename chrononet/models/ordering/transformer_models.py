#!/usr/bin/python3

#import hurry.filesize
import math
import numpy
import random
import os
import time
import torch
import torch.nn as nn
#import torch.nn.functional as F
from allennlp.modules.elmo import Elmo, batch_to_ids
from GPUtil import showUtilization as gpu_usage
from torch import optim
from transformers import BertModel, BertTokenizer
#from sklearn.utils import shuffle

from .pytorch_models import Autoencoder, Embedder
from swarm_mod.swarmlayer import SwarmLayer

numpy.set_printoptions(threshold=numpy.inf)
debug = True
tdevice = 'cpu'
use_cuda = torch.cuda.is_available()
#use_cuda = False
if use_cuda:
    tdevice = torch.device('cuda')
options_file = "/h/sjeblee/research/data/elmo/weights/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "/h/sjeblee/research/data/elmo/weights/elmo_2x4096_512_2048cnn_2xhighway_weights_PubMed_only.hdf5"
#ae_file = '/h/sjeblee/research/data/va/chrono/ordergru_va_autoencoder_timeencoder/autoencoder.model'

import sys
print('__Python VERSION:', sys.version)
print('__pyTorch VERSION:', torch.__version__)
print('__CUDNN VERSION:', torch.backends.cudnn.version())
print('__Number CUDA Devices:', torch.cuda.device_count())
print('__Devices')
print('Active CUDA Device: GPU', torch.cuda.current_device())
print('Available devices ', torch.cuda.device_count())
print('Current cuda device ', torch.cuda.current_device())


# Set transformer models
class SetOrderBert(nn.Module):
    def __init__(self, input_size, encoding_size, time_encoding_size, hidden_size, output_size, encoder_file, dropout_p=0.1, use_autoencoder=False, autoencoder_file=None, checkpoint_dir=None, encoder_name='elmo'):
        super(SetOrderBert, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.encoding_size = encoding_size
        self.time_encoding_size = time_encoding_size
        self.output_size = output_size
        self.dropout = dropout_p
        self.checkpoint_dir = checkpoint_dir
        print('SetOrderBert checkpoint_dir:', self.checkpoint_dir)

        self.use_autoencoder = use_autoencoder
        #self.use_autoencoder = False

        if autoencoder_file is None:
            autoencoder_file = ''
        if self.use_autoencoder is True:
            if os.path.exists(autoencoder_file):
                self.autoencoder = torch.load(autoencoder_file)
            else:
                #self.elmo = Elmo(options_file, weight_file, 1, dropout=0).to(tdevice)
                self.autoencoder = Autoencoder(input_size, encoding_size, use_double=False, autoencoder_file=autoencoder_file, encoder_name=encoder_name)
        else:
            self.embedder = Embedder(encoder_name, encoding_size)

        self.set_layer = SwarmLayer(hidden_size, hidden_size, hidden_size, n_iter=10, n_dim=1, dropout=0.0, pooling='MEAN', channel_first=True, cache=False)

        print('time encoder file:', encoder_file)
        self.time_encoder = torch.load(encoder_file).model # TimeEncoder(self.input_size, self.time_encoding_size, self.elmo)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=2)

    ''' Input is a list of lists of numpy arrays
    '''
    def forward(self, input, X2=None, is_test=False):
        # input expected as (events, words, embedding_dim)
        encodings = []
        hn = None
        hn_e = None
        index = 0
        #extra_size = 0

        # Mini-batching for memory saving
        '''
        mini_batch = 16
        i = 0
        #if not (is_test and len(input) > mini_batch):
        if len(input) < mini_batch:
            mini_batch = len(input)
        '''

        output = None
        out1 = None
        '''
        while i < len(input):
            end = i + mini_batch
            if end > len(input):
                end = len(input)
            input_batch = input[i:i+mini_batch]
            print('mini_batch:', i, 'to', i+mini_batch, 'input_batch:', len(input_batch), 'should be', mini_batch)
            encodings = []
        '''
        for row in input:
            # ELMo embedding for each event (sequence of words)
            context = row[0]
            #print('context:', context)
            word_flags = row[1]
            time_words = row[2]
            tflags = row[3]
            time_val = row[4]
            #time_type = row[5]
            to_concat = []

            # Append the target flags
            #c_flags = torch.tensor(word_flags, dtype=torch.float, device=tdevice).view(1, -1, 1)
            #print('X:', uttX.size(), 'c_flags:', c_flags.size())
            #uttX = torch.cat((uttX, c_flags), dim=2)

            # Event encoding (BioBERT or ELMo)
            if self.use_autoencoder:
                enc = self.autoencoder.encode([row]).squeeze()
            else:
                enc = self.embedder(row)

            to_concat.append(enc)
            if debug: print('enc:', str(enc.size()))

            # Time phrase encoding
            if self.time_encoding_size > 0:
                if time_words is None:
                    time_emb = torch.zeros(self.time_encoding_size, dtype=torch.float, device=tdevice)
                else:
                    time_X = self.time_encoder.encode(time_words)
                    #time_char_ids = batch_to_ids([time_words]).to(tdevice)
                    #time_embeddings = self.elmo(time_char_ids)['elmo_representations']
                    #time_X = time_embeddings[0]
                    #time_X = time_X.view(1, -1, self.input_size) # should be (1, #words, input_dim)
                    print('time tensor:', time_X.size())
                    time_emb = time_X.view(self.time_encoding_size)

                    # Add the flags
                    #print('tflags:', str(tflags), 'twords:', time_words)
                    #t_flags = torch.tensor(tflags, dtype=torch.float, device=tdevice).view(1, -1, 1)
                    #print('time X:', time_X.size(), 't_flags:', t_flags.size())
                    #time_X = torch.cat((time_X, t_flags), dim=2)

                    #time_encoding, hn_t = self.gru_time(time_X, hn_t) # should be (1, #words, encoding_dim)
                    #time_emb = time_encoding[:, -1, :].view(self.time_encoding_size)

                '''
                if time_val is None:
                    time_enc = torch.zeros(2, dtype=torch.float64, device=tdevice)
                else:
                    time_enc = torch.tensor(time_val, dtype=torch.float64, device=tdevice)

                # Concatenate the time val and embedding
                time_enc = torch.cat((time_emb, time_enc), dim=0)
                '''
                to_concat.append(time_emb)

            # Structured features
            #flag_tensor = torch.tensor(flags, dtype=torch.float64, device=tdevice)

            # Concatenate the features
            event_vector = torch.cat(to_concat, dim=0)

            # Add other features
            if X2 is not None:
                x2_np = numpy.asarray([X2[index]]).astype('float')
                x2_vec = torch.tensor(x2_np, dtype=torch.float, device=tdevice)
                if debug: print('x2:', x2_vec.size())
                #extra_size = x2_vec.size()[0]
                enc = torch.cat((enc, x2_vec), dim=0)
            encodings.append(event_vector)
            index += 1

        conversation_orig = torch.stack(encodings)
        #print('conversation:', conversation.size())
        conversation = conversation_orig.view(-1, self.hidden_size, 1) # Treat whole conversation as a batch
        print('conversation resized:', conversation.size())

        # Set layer
        output_batch = self.set_layer(conversation)
        output_batch = output_batch.squeeze()
        print('output:', output_batch.size())
        #out1 = output_batch
        out1 = self.linear(output_batch)
        #output = output_batch
        '''
            if output is None:
                output = output_batch
            else:
                output = torch.cat((output, output_batch), dim=1)
            if out1 is None:
                out1 = out1_batch
            else:
                out1 = torch.cat((out1, out1_batch), dim=1)
            print('output size so far:', output.size())
        '''
        #out1 = self.softmax(self.linear(output))

        if is_test:
            encodings = []
            #del X
            del conversation
            #del embeddings
            torch.cuda.empty_cache()
            #i = i + mini_batch

        print('final out:', out1.size(), out1)
        if is_test:
            del encodings
        return out1, conversation_orig #conversation.detach()

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
                #input_dim = X[0][0][0].shape[0]
                #output_dim = Y[0][0].shape[0]
            #else:
                #input_dim = len(X[0][0])
                #output_dim = len(Y[0][0])
            print("X list:", str(len(X)), "Y list:", str(len(Y)))

        if X2 is not None:
            print("X2 list:", len(X2))

        #print("input_dim: ", str(input_dim))
        print("output_dim: ", str(output_dim))

        # Set up optimizer and loss function
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        #optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)
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

        # Train the autoencoder
        if self.use_autoencoder:
            self.autoencoder.fit(X)

        start_epoch = 0

        # Check for model checkpoint
        checkpoint_file = os.path.join(self.checkpoint_dir, 'checkpoint_setorder.pth')
        if os.path.exists(checkpoint_file):
            check = torch.load(checkpoint_file)
            self.load_state_dict(check['state_dict'])
            optimizer.load_state_dict(check['optimizer'])
            loss = check['loss']
            start_epoch = check['epoch'] + 1
            print('loading from checkpoint, restarting at epoch', start_epoch)

        # Train the model
        for epoch in range(start_epoch, num_epochs):
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
                    outputs, _ = self(batchX, batchX2)
                    #print('max_length:', max_length)
                    outputs.squeeze(0)
                    #outputs = outputs.view(max_length, -1)
                    print('outputs:', outputs.size())
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

            # Save checkpoint
            torch.save({'epoch': epoch, 'state_dict': self.state_dict(), 'optimizer': optimizer.state_dict(), 'loss': loss},
                       os.path.join(self.checkpoint_dir, 'checkpoint_setorder.pth'))
            print('Saved checkpoint for epoch', epoch)
        print("GRU_GRU training took", str(time.time()-start), "s")

    def predict(self, testX, X2=None, batch_size=1, keep_list=True, return_encodings=False):
        # Test the Model
        encodings = []
        print_every = 1
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
                #if debug: print("test x_array:", str(x_array.shape))
                samples = torch.tensor(x_array, dtype=torch.float, device=tdevice)

            if X2 is not None:
                outputs = self(samples, x2_batch)
            else:
                with torch.no_grad():
                    outputs, enc = self(samples, is_test=True)
                enc_cpu = enc.to('cpu')
                encodings.append(enc_cpu)
                del enc
                torch.cuda.empty_cache()
            outputs = outputs.squeeze()
            print("test outputs:", str(outputs.size()))
            #num_items = outputs.size()[1]
            #predicted = outputs.view(num_items).tolist()
            predicted = outputs.tolist()
            #_, predicted = torch.max(outputs.data, -1) # TODO: fix this
            print('predicted:', predicted)
            pred.append(predicted)
            del samples
            #print('mem allocated:', hurry.filesize.size(torch.cuda.memory_allocated()), 'mem cached:', hurry.filesize.size(torch.cuda.memory_cached()))
            #gpu_usage()

            i = i+batch_size
        if not return_encodings:
            del encodings
        if return_encodings:
            return pred, encodings
        else:
            return pred
