#!/usr/bin/python3

import math
import numpy
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.modules.elmo import Elmo, batch_to_ids
from torch import optim

#from models.loss_functions import Kendall_Tau_Loss

numpy.set_printoptions(threshold=numpy.inf)
debug = True
tdevice = 'cpu'
use_cuda = torch.cuda.is_available()
if use_cuda:
    tdevice = torch.device('cuda:2')

# GRU with GRU encoder, input: (conversations (1), utterances, words, embedding_dim)
class GRU_GRU(nn.Module):
    def __init__(self, input_size, encoding_size, time_encoding_size, hidden_size, output_size, dropout_p=0.1):
        super(GRU_GRU, self).__init__()
        options_file = "/u/sjeblee/research/data/elmo/weights/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "/u/sjeblee/research/data/elmo/weights/elmo_2x4096_512_2048cnn_2xhighway_weights_PubMed_only.hdf5"
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.encoding_size = encoding_size
        self.time_encoding_size = time_encoding_size
        self.output_size = output_size
        self.dropout = dropout_p
        self.elmo = Elmo(options_file, weight_file, 1, dropout=0).to(tdevice)
        self.gru0 = nn.GRU(input_size+1, int(encoding_size/2), bidirectional=True, dropout=dropout_p, batch_first=True)
        self.gru1 = nn.GRU(hidden_size, hidden_size, bidirectional=True, dropout=dropout_p, batch_first=True)
        self.gru_time = nn.GRU(self.input_size+1, int(self.time_encoding_size/2), bidirectional=True, dropout=dropout_p, batch_first=True)
        self.linear = nn.Linear(hidden_size*2, output_size)
        self.softmax = nn.Softmax(dim=2)

    ''' Input is a list of lists of numpy arrays
    '''
    def forward(self, input, X2=None):
        # input expected as (events, words, embedding_dim)
        encodings = []
        hn = None
        hn_t = None
        index = 0
        extra_size = 0
        for row in input:
            # ELMo embedding for each event (sequence of words)
            context = row[0]
            word_flags = row[1]
            time_words = row[2]
            tflags = row[3]
            time_val = row[4]
            to_concat = []
            character_ids = batch_to_ids([context]).to(tdevice)
            embeddings = self.elmo(character_ids)['elmo_representations']
            #print('elmo embeddings:', len(embeddings))
            X = embeddings[0].squeeze()
            #print('input_size:', self.input_size, 'X:', X.size())
            uttX = X.view(1, -1, self.input_size) # should be (1, #words, input_dim)

            # Append the target flags
            c_flags = torch.tensor(word_flags, dtype=torch.float, device=tdevice).view(1, -1, 1)
            print('X:', uttX.size(), 'c_flags:', c_flags.size())
            uttX = torch.cat((uttX, c_flags), dim=2)

            # Create a tensor
            '''
            uttXnp = numpy.asarray(row).astype('float')
            uttX = torch.tensor(uttXnp, dtype=torch.float, device=tdevice)
            '''

            #print('input tensor:', uttX)
            encoding, hn = self.gru0(uttX, hn) # should be (1, #words, encoding_dim)
            enc = encoding[:, -1, :].view(self.encoding_size) # Only keep the output of the last timestep
            to_concat.append(enc)
            #if debug: print('enc:', str(enc.size()), enc)

            # Time phrase encoding
            if self.time_encoding_size > 0:
                if time_words is None:
                    time_emb = torch.zeros(self.time_encoding_size, dtype=torch.float, device=tdevice)
                else:
                    time_char_ids = batch_to_ids([time_words]).to(tdevice)
                    time_embeddings = self.elmo(time_char_ids)['elmo_representations']
                    time_X = time_embeddings[0]
                    time_X = time_X.view(1, -1, self.input_size) # should be (1, #words, input_dim)
                    #print('time tensor:', time_X.size())

                    # Add the flags
                    print('tflags:', str(tflags), 'twords:', time_words)
                    t_flags = torch.tensor(tflags, dtype=torch.float, device=tdevice).view(1, -1, 1)
                    print('time X:', time_X.size(), 't_flags:', t_flags.size())
                    time_X = torch.cat((time_X, t_flags), dim=2)

                    time_encoding, hn_t = self.gru_time(time_X, hn_t) # should be (1, #words, encoding_dim)
                    time_emb = time_encoding[:, -1, :].view(self.time_encoding_size)

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
                extra_size = x2_vec.size()[0]
                enc = torch.cat((enc, x2_vec), dim=0)
            encodings.append(event_vector)
            index += 1

        conversation = torch.stack(encodings)
        conversation = conversation.view(1, -1, self.hidden_size) # Treat whole conversation as a batch
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

# Set to Sequence model for temporal ordering ##############################

class SetToSequence_encoder(nn.Module):
    def __init__(self, input_size, encoding_size, context_encoding_size, time_encoding_size, hidden_size, output_size, dropout_p=0.1, read_cycles=50):
        super(SetToSequence_encoder, self).__init__()

        options_file = "/u/sjeblee/research/data/elmo/weights/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "/u/sjeblee/research/data/elmo/weights/elmo_2x4096_512_2048cnn_2xhighway_weights_PubMed_only.hdf5"

        self.read_cycles = read_cycles
        self.input_size = input_size # word vector dim
        self.elmo = Elmo(options_file, weight_file, 1, dropout=0).to(tdevice)
        print('encoder input_size:', input_size)
        self.encoding_size = encoding_size
        self.context_encoding_size = context_encoding_size
        self.time_encoding_size = time_encoding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout_p
        self.gru0 = nn.GRU(self.input_size+1, int(self.encoding_size/2), bidirectional=True, dropout=dropout_p, batch_first=True).double()
        print('gr0 input_size:', self.gru0.input_size)
        self.gru_time = nn.GRU(self.input_size+1, int(self.time_encoding_size/2), bidirectional=True, dropout=dropout_p, batch_first=True).double()
        #self.gru_c = nn.GRU(self.input_size, int(self.encoding_size/2), bidirectional=True, dropout=dropout_p, batch_first=True).double()
        self.gru1 = nn.GRUCell(self.hidden_size, self.hidden_size).double() # encoder

        # Attention calculations
        self.bilinear = nn.Bilinear(self.hidden_size, self.hidden_size, 1).double()
        self.softmax = nn.Softmax(dim=1).double()

    ''' Input is a list of lists of numpy arrays
    '''
    def forward(self, input):
        encodings = []
        hn = None
        hn_t = None
        hn_c = None
        index = 0

        # Generate the event encodings
        print('events:', len(input))
        for row in input:
            # Create a tensor
            #Xnp = numpy.asarray(row).astype('float64')
            #print('row:', type(row), len(row))
            #X = torch.tensor(Xnp, dtype=torch.float64, device=tdevice)

            # ELMo embedding for each event (sequence of words)
            #words = row[0]
            context = row[0]
            word_flags = row[1]
            time_words = row[2]
            tflags = row[3]
            time_val = row[4]
            #flags = row[4]

            to_concat = []

            # EVENT encoding
            '''
            if self.encoding_size > 0:
                character_ids = batch_to_ids([words]).to(tdevice)
                embeddings = self.elmo(character_ids)['elmo_representations']
                #print('elmo embeddings:', len(embeddings))
                X = embeddings[0].double().squeeze()

                #print('X:', X.size())
                #flag_tensor = torch.tensor(word_flags, dtype=torch.double, device=tdevice).view(-1, 1)
                #X = torch.cat((X, flag_tensor), dim=1)
                X = X.view(1, -1, self.input_size) # should be (1, #words, input_dim)
                encoding, hn = self.gru0(X, hn) # should be (1, #words, encoding_dim)
                enc = encoding[:, -1, :].view(self.encoding_size) # Only keep the output of the last timestep
                #if debug: print('enc:', str(enc.size()), enc)
                to_concat.append(enc)
            '''

            # Context encoding
            if self.context_encoding_size > 0:
                character_ids = batch_to_ids([context]).to(tdevice)
                embeddings = self.elmo(character_ids)['elmo_representations']
                #print('elmo embeddings:', len(embeddings))
                X = embeddings[0].double()
                X = X.view(1, -1, self.input_size) # should be (1, #words, input_dim)

                # Append the target flags
                c_flags = torch.tensor(word_flags, dtype=torch.float64, device=tdevice).view(1, -1, 1)
                print('X:', X.size(), 'c_flags:', c_flags.size())
                X = torch.cat((X, c_flags), dim=2)

                #print('context tensor:', X.size())
                encoding_c, hn_c = self.gru0(X, hn_c) # should be (1, #words, encoding_dim)
                context_enc = encoding_c[:, -1, :].view(self.encoding_size) # Only keep the output of the last timestep
                to_concat.append(context_enc)

            # Time phrase encoding
            if self.time_encoding_size > 0:
                if time_words is None:
                    time_emb = torch.zeros(self.time_encoding_size, dtype=torch.float64, device=tdevice)
                else:
                    time_char_ids = batch_to_ids([time_words]).to(tdevice)
                    time_embeddings = self.elmo(time_char_ids)['elmo_representations']
                    time_X = time_embeddings[0].double()
                    time_X = time_X.view(1, -1, self.input_size) # should be (1, #words, input_dim)
                    #print('time tensor:', time_X.size())

                    # Add the flags
                    print('tflags:', str(tflags), 'twords:', time_words)
                    t_flags = torch.tensor(tflags, dtype=torch.float64, device=tdevice).view(1, -1, 1)
                    print('time X:', time_X.size(), 't_flags:', t_flags.size())
                    time_X = torch.cat((time_X, t_flags), dim=2)

                    time_encoding, hn_t = self.gru_time(time_X, hn_t) # should be (1, #words, encoding_dim)
                    time_emb = time_encoding[:, -1, :].view(self.time_encoding_size)

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
            #print('event_vector size:', event_vector.size())
            encodings.append(event_vector)
            index += 1

        mem_block = torch.stack(encodings)
        print('mem_block size:', mem_block.size())
        mem_block = mem_block.view(-1, self.hidden_size) # Treat whole conversation as a batch (n, enc_size*2)
        #print('mem_block:', mem_block.size())

        # Feed input into the encoder
        mem_size = mem_block.size(0)
        h_t = self.init_hidden()
        for t in range(self.read_cycles):
            # Calculate attention matrix
            e_list = []

            for i in range(mem_size): # For each event encoding
                e_ti = self.bilinear(mem_block[i], h_t.squeeze())
                e_list.append(e_ti)
            e_matrix = torch.stack(e_list).view(1, -1)
            #print('e_matrix:', e_matrix.size())
            attention_matrix = self.softmax(e_matrix).squeeze()
            #print('attention:', attention_matrix.size(), attention_matrix)

            if mem_size == 1:
                s_t = torch.mul(mem_block, attention_matrix.item())
            else:
                sum_list = []
                for i in range(mem_block.size(0)):
                    sum_list.append(torch.mul(mem_block[i], attention_matrix[i].item()))
                sum_matrix = torch.stack(sum_list)
                #print('sum_matrix:', sum_matrix.size())
                s_t = torch.sum(sum_matrix, dim=0)

            #print('s_t:', s_t.size())
            s_t = s_t.view(1, self.hidden_size) # Re-shape as a batch size of 1
            #print('s_t view:', s_t.size())

            # Feed the input into the encoder and update the hidden state
            h_t = self.gru1(s_t, h_t)

        return mem_block, h_t

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size, dtype=torch.float64, device=tdevice)


class SetToSequence_decoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, group_thresh=None):
        super(SetToSequence_decoder, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout_p
        self.gru = nn.GRUCell(hidden_size, hidden_size).double() # TODO: check this
        self.group_thresh = group_thresh

        # Attention calculations
        self.bilinear = nn.Bilinear(self.hidden_size, self.hidden_size, 1).double()
        self.softmax = nn.Softmax(dim=1).double()
        self.logsoftmax = nn.LogSoftmax(dim=1).double()

    ''' input: s_i of the previous predicted item
        hidden: the previous hidden state
    '''
    def forward(self, input, h_t, mem_block, mask, train=False):

        h_t = self.gru(input, h_t)

        # Calculate attention matrix
        e_list = []
        for i in range(mem_block.size(0)): # For each event encoding
            e_ti = self.bilinear(mem_block[i], h_t.squeeze())
            e_list.append(e_ti)
        e_matrix = torch.stack(e_list).squeeze()
        #print('dec attention:', attention_matrix.size(), attention_matrix)

        # Force the model to choose a new item that hasn't already been chosen
        #attention_matrix = self.softmax((e_matrix * mask).view(1, -1)).squeeze()
        attention_matrix = self.logsoftmax(e_matrix.view(1, -1)).squeeze() # log probs for NLL loss
        mask_matrix = attention_matrix * mask # Apply the mask

        #print('dec softmax attention w/ mask:', mask_matrix.size(), mask_matrix)

        # Select the item with the highest probability from attention
        #print('argmax:', torch.argmax(mask_matrix, dim=0))
        if self.group_thresh is not None:
            #attention_matrix = mask_matrix # Use the mask version for loss calcuation

            # Allow multiple events to be output at the same rank (prob threshold?)
            max_tensor, index_tensor = torch.max(mask_matrix, dim=0)
            target_index = int(index_tensor.item())
            max_prob = max_tensor.item()
            #print('max_prob:', max_prob)
            targets = []
            n = float(mask_matrix.size(0))
            for j in range(mask_matrix.size(0)):
                if mask[j] > 0.0 or max_prob == 0.0:
                    prob = mask_matrix[j]
                    if ((max_prob - prob) <= (self.group_thresh/n)):
                        targets.append(j)
                        #print('choosing item', j, 'with prob', mask_matrix[j].item())
                        mask[j] = 0.0
            #target = torch.stack(targets)
            target = targets
            attention_matrix = self.softmax(e_matrix.view(1, -1)).squeeze() # log probs for NLL loss

        else: # Linear prediction, use log domain
            #attention_matrix = self.logsoftmax(e_matrix.view(1, -1)).squeeze() # log probs for NLL loss

            if not train: # Fix nans and -inf in the mask
                for li in range(mask_matrix.size(0)):
                    if math.isnan(mask_matrix[li]) or mask[li] == float('-inf'):
                        mask_matrix[li] = float('-inf')
                print('fixed log_probs:', mask_matrix)

            target = torch.argmax(mask_matrix, dim=0)
            target_index = int(target.item())
            print('choosing item', target_index, 'with prob', mask_matrix[target_index].item())
            if not train:
                mask[target_index] = float('-inf') # Set the mask so the same event won't be output again
        #print('new mask:', mask)
        #x_i = mem_block[target_index]

        return target, h_t, mask, attention_matrix


class SetToSequence:

    def __init__(self, input_size, encoding_size, time_encoding_size, hidden_size, output_size, dropout_p=0.1, read_cycles=50, group_thresh=None, invert_ranks=False):
        self.encoder = SetToSequence_encoder(input_size, encoding_size, time_encoding_size, hidden_size, output_size, dropout_p, read_cycles)
        self.decoder = SetToSequence_decoder(hidden_size, output_size, dropout_p, group_thresh)
        self.group_thresh = group_thresh
        self.invert_ranks = invert_ranks

    ''' Creates and trains a recurrent neural network model. Supports SimpleRNN, LSTM, and GRU
        X: a list of training data
        Y: a list of training labels
        WARNING: Currently you can't use the encoding layer and use_prev_labels at the same time
    '''
    def fit(self, X, Y, activation='relu', num_epochs=10, batch_size=1, loss_function='mse'):
        start = time.time()

        # Parameters
        hidden_size = self.encoder.hidden_size
        encoding_size = self.encoder.encoding_size
        dropout = self.encoder.dropout
        output_dim = self.decoder.output_size
        learning_rate = 0.01
        print_every = 100
        teacher_forcing_ratio = 0.9

        print("hidden_size:", str(hidden_size), "dropout:", str(dropout), "epochs:", str(num_epochs))
        print("encoding size:", str(encoding_size))

        # Set up optimizer and loss function
        encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=learning_rate)

        # Loss function
        #criterion = nn.MSELoss()
        if self.group_thresh is not None:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.NLLLoss()

        if use_cuda:
            self.encoder = self.encoder.to(tdevice)
            self.decoder = self.decoder.to(tdevice)

        num_examples = len(X)
        print("X 000:", str(type(X[0][0][0])), "Y 00:", str(type(Y[0][0])))
        input_dim = self.encoder.input_size
        print("X list:", str(len(X)), "Y list:", str(len(Y)))

        print("input_dim: ", str(input_dim))
        print("output_dim: ", str(output_dim))

        # Train the model
        for epoch in range(num_epochs):
            print("epoch", str(epoch))
            i = 0
            while (i < num_examples):
                batchXnp = X[i]
                batchYnp = Y[i]
                #if debug: print("batchX len:", str(len(batchXnp)), "batchY len:", str(len(batchYnp)))
                if type(batchYnp) is list:
                    batchYnp = numpy.asarray(batchYnp)

                batchYnp = batchYnp.astype('float64')
                batchY = torch.tensor(batchYnp, dtype=torch.float64, device=tdevice, requires_grad=True)

                #print("batchX size:", str(batchX.size()), "batchY size:", str(batchY.size()))
                #if debug: print("batchX[0]:", str(batchXnp[0]))
                #if debug: print("batchY size:", str(batchY.size()))

                labels = batchY.view(1, -1, output_dim)
                seq_length = labels.size(1)
                #input_length = self.encoder.read_cycles
                if debug: print("seq_length:", str(seq_length))

                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                loss = 0

                # Run the encoder
                #encoder_hidden = self.encoder.init_hidden()
                mem_block, encoder_hidden = self.encoder(batchXnp)

                # Run the decoder
                decoder_hidden = encoder_hidden
                use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

                # Initialize the prediction
                x_i = torch.zeros(self.encoder.hidden_size, dtype=torch.float64, device=tdevice)
                output_indices = []
                output_probs = []
                correct_ranks = labels.squeeze()
                flatten = True
                if self.group_thresh is not None:
                    flatten = False
                correct_indices = ranks_to_indices(correct_ranks.tolist(), flatten)
                # INVERT the ranks?
                if self.invert_ranks:
                    correct_indices.reverse()
                num_timesteps = len(correct_indices)
                #if i==0: print('correct_indices:', correct_indices)
                done_mask = torch.ones(seq_length, dtype=torch.float64, device=tdevice)

                #di = 0
                #while torch.max(done_mask).item() > 0.0:
                for di in range(seq_length):
                    # Run until we've picked all the items
                    index, decoder_hidden, done_mask, log_probs = self.decoder(x_i.view(1, -1), decoder_hidden, mem_block, done_mask, True)
                    #print('di:', di, 'mask:', done_mask)

                    # Select multiple items at each timestep
                    if self.group_thresh is not None:
                        x_i_vecs = []
                        #if use_teacher_forcing:
                        #    selected = correct_indices[di]
                        #else:
                        selected = index
                        for ind in selected:
                            vec = mem_block[ind]
                            x_i_vecs.append(vec)
                        x_i = torch.mean(torch.stack(x_i_vecs), dim=0) # average the selected items

                    # Only select one item at each timestep
                    else:
                        index = int(index.item())
                        x_i = mem_block[index]
                        x_correct = mem_block[correct_indices[di]]
                        # Teacher forcing for training
                        if use_teacher_forcing:
                            x_i = x_correct

                    output_indices.append(index)
                    output_probs.append(log_probs)

                    '''
                    if self.group_thresh is not None:
                        target_tensor = torch.zeros(seq_length, dtype=torch.float64, device=tdevice, requires_grad=False)
                        if di < len(correct_indices):
                            for val in correct_indices[di]:
                                target_tensor[val] = 1.0
                        # If num of predicted timesteps is not the same as target tensor, pad one of them
                        print('output pred:', log_probs.size(), 'target:', target_tensor.size())
                        loss += criterion(log_probs.view(1, -1), target_tensor.view(1, -1))
                    '''
                    di += 1

                output_probs = torch.stack(output_probs).view(-1, seq_length)

                if self.group_thresh is not None:
                    target_tensor = torch.zeros(num_timesteps, seq_length, dtype=torch.float64, device=tdevice, requires_grad=False)
                    for timestep in range(len(correct_indices)):
                        indices = correct_indices[timestep]
                        for val in indices:
                            target_tensor[timestep][val] = 1.0
                    # output_probs: pred_timesteps * seq_length, target: num_timesteps * seq_length
                    print('pred:', output_probs.size(), 'target:', target_tensor.size())
                    # If num of predicted timesteps is not the same as target tensor, pad one of them
                    if output_probs.size(0) != target_tensor.size(0):
                        max_size = max(output_probs.size(0), target_tensor.size(0))
                        pad_size = max_size - output_probs.size(0)
                        if pad_size > 0:
                            pad_tensor = torch.zeros(pad_size, seq_length, dtype=torch.float64, device=tdevice)
                            output_probs = torch.cat((output_probs, pad_tensor))
                        pad_size = max_size - target_tensor.size(0)
                        if pad_size > 0:
                            pad_tensor = torch.zeros(pad_size, seq_length, dtype=torch.float64, device=tdevice)
                            target_tensor = torch.cat((target_tensor, pad_tensor))
                    print('padded pred:', output_probs.size(), 'target:', target_tensor.size())
                    loss = criterion(output_probs, target_tensor)
                else:
                    #if self.group_thresh is None:
                    loss = criterion(output_probs, torch.tensor(correct_indices, dtype=torch.long, device=tdevice, requires_grad=False))

                # Un-invert the ranks
                if self.invert_ranks:
                    output_indices.reverse()
                output_ranks = indices_to_ranks(output_indices)
                #if i==0: print('output_ranks:', output_ranks)
                #loss = criterion(torch.tensor(output_ranks, dtype=torch.float, device=tdevice, requires_grad=True), correct_ranks)
                #print('loss:', loss.item())

                loss.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()

                #return loss.item() / target_length
                if (i) % print_every == 0:
                    print('Epoch [%d/%d], Loss: %.4f' %(epoch, num_epochs, loss.item()))
                i += 1
        print('training took', time.time()-start, 's')

    def predict(self, testX, batch_size=1):
        print("X 000:", str(type(testX[0][0][0])))
        print("X list:", str(len(testX)))
        outputs = []

        # Run the model
        for i in range(len(testX)):
            batchXnp = testX[i]
            #if debug: print("batchX len:", len(batchXnp))

            #print("batchX size:", str(batchX.size()), "batchY size:", str(batchY.size()))
            #if debug: print("testX[0]:", str(batchXnp[0]))
            seq_length = len(batchXnp)
            #input_length = self.encoder.read_cycles
            #if debug: print("test seq_length:", str(seq_length))

            # Run the encoder
            mem_block, encoder_hidden = self.encoder(batchXnp)

            # Run the decoder
            decoder_hidden = encoder_hidden

            # Initialize the prediction
            x_i = torch.zeros(self.encoder.hidden_size, dtype=torch.float64, device=tdevice)
            output_indices = []
            done_mask = torch.ones(seq_length, dtype=torch.float64, device=tdevice)

            # Run until we've picked all the items
            di = 0
            while torch.max(done_mask).item() > 0.0:
                print('test di:', di, 'mask:', done_mask)

                # Make sure we don't accidentally get stuck in an infinite loop
                if di > done_mask.size(0):
                    print('ERROR: too many predict iterations')
                    exit(1)

                # Run the linear decoder
                index, decoder_hidden, done_mask, log_probs = self.decoder(x_i.view(1, -1), decoder_hidden, mem_block, done_mask)

                # Select multiple items at each timestep
                if self.group_thresh is not None:
                    x_i_vecs = []
                    selected = index
                    for ind in selected:
                        vec = mem_block[ind]
                        x_i_vecs.append(vec)
                    x_i = torch.mean(torch.stack(x_i_vecs), dim=0) # average the selected items

                # Only select one item at each timestep
                else:
                    index = int(index.item())
                    x_i = mem_block[index]

                output_indices.append(index)
                #output_probs.append(log_probs)
                di += 1

            # Un-invert the ranks
            if self.invert_ranks:
                output_indices.reverse()
            output_ranks = indices_to_ranks(output_indices)
            print('output_ranks:', output_ranks)
            outputs.append(output_ranks)
        return outputs


# Set2Seq GROUP #####################

# Set2seq linear w/ group transition prediction
class SetToSequenceGroup_decoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, group_thresh=None):
        super(SetToSequenceGroup_decoder, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout_p
        self.gru = nn.GRUCell(hidden_size, hidden_size).double() # TODO: check this
        self.group_thresh = group_thresh

        # Attention calculations
        self.bilinear = nn.Bilinear(self.hidden_size, self.hidden_size, 1).double()
        #self.softmax = nn.Softmax(dim=1).double()
        self.logsoftmax = nn.LogSoftmax(dim=1).double()

        #self.gru_out = nn.GRUCell(1, 2).double() # GRU to output 1 or 0
        #self.softmax_out = nn.Softmax(dim=1).double()

    ''' input: s_i of the previous predicted item
        hidden: the previous hidden state
    '''
    def forward(self, input, h_t, mem_block, mask, train=False):
        h_t = self.gru(input, h_t)

        # Calculate attention matrix
        e_list = []
        for i in range(mem_block.size(0)): # For each event encoding
            e_ti = self.bilinear(mem_block[i], h_t.squeeze())
            e_list.append(e_ti)
        e_matrix = torch.stack(e_list).squeeze()
        #print('dec attention:', attention_matrix.size(), attention_matrix)

        # Force the model to choose a new item that hasn't already been chosen
        attention_matrix = self.logsoftmax(e_matrix.view(1, -1)).squeeze()
        #if train:
        mask_matrix = attention_matrix
        #else:
        #mask_matrix = attention_matrix * mask # Apply the mask
        #print('dec softmax attention w/ mask:', mask_matrix.size(), mask_matrix)

        # Select the item with the highest probability from attention
        #attention_matrix = self.logsoftmax(e_matrix.view(1, -1)).squeeze() # log probs for NLL loss
        target = torch.argmax(mask_matrix, dim=0)
        #target_index = int(target.item())
        #if mask_matrix.dim() == 0:
        #    prob = mask_matrix.item()
        #else:
        #    prob = mask_matrix[target_index].item()
        #print('highest item', target_index, 'with prob', prob)

        #mask[target_index] = 0.0 # Set the mask so the same event won't be output again
        #print('new mask:', mask)
        '''
        out_hn = None
        output_vals = []
        attn = attention_matrix.view(-1, 1)
        print('attention matrix:', attn.size())
        for v in range(attn.size(0)):
            val = attn[v].view(1, 1)
            #print('val size:', val.size())
            out_hn = self.gru_out(val, out_hn)
            output_vals.append(self.softmax_out(out_hn))
        output_matrix = torch.stack(output_vals, dim=0).squeeze()
        print('output_matrix:', output_matrix.size())
        '''

        return target, h_t, mask, mask_matrix


class SetToSequenceGroup_transition(nn.Module):

    def __init__(self, hidden_size, dropout_p):
        super(SetToSequenceGroup_transition, self).__init__()
        self.hidden_size = hidden_size
        self.transition_gru = nn.GRU(self.hidden_size, self.hidden_size, bidirectional=False, dropout=dropout_p, batch_first=True).double()
        #self.fc1 = nn.Linear(self.hidden_size, self.hidden_size).double()
        #self.relu1 = nn.ReLU().double()
        self.dout = nn.Dropout(dropout_p).double()
        self.transition = nn.Linear(self.hidden_size, 1).double()
        self.sigmoid = nn.Sigmoid().double()
        #self.softmax = nn.Softmax(dim=2).double()

    def forward(self, input, h_t):
        _out, ht = self.transition_gru(input, h_t)
        #lin_out = self.relu1(self.fc1(ht))
        prob = self.sigmoid(self.transition(self.dout(ht)))
        #prob = self.transition(ht)
        #prob = self.softmax(self.transition(ht))
        return prob


class SetToSequenceGroup:

    def __init__(self, input_size, encoding_size, context_encoding_size, time_encoding_size, hidden_size, output_size, dropout_p=0.1, read_cycles=50, group_thresh=None, invert_ranks=False, sig=1):
        self.encoder = SetToSequence_encoder(input_size, encoding_size, context_encoding_size, time_encoding_size, hidden_size, output_size, dropout_p, read_cycles)
        self.decoder = SetToSequenceGroup_decoder(hidden_size, output_size, dropout_p, group_thresh)
        self.group_thresh = group_thresh
        self.hidden_size = hidden_size
        self.invert_ranks = invert_ranks
        self.sigma = sig

        # Group transition
        #self.group = SetToSequenceGroup_transition(hidden_size, dropout_p)

        #self.transition_prob = nn.Bilinear(self.hidden_size, self.hidden_size, 1).double()

    ''' Creates and trains a recurrent neural network model. Supports SimpleRNN, LSTM, and GRU
        X: a list of training data
        Y: a list of training labels
        WARNING: Currently you can't use the encoding layer and use_prev_labels at the same time
    '''
    def fit(self, X, Y, activation='relu', num_epochs=10, batch_size=1, loss_function='mse'):
        start = time.time()

        # Parameters
        hidden_size = self.encoder.hidden_size
        encoding_size = self.encoder.encoding_size
        dropout = self.encoder.dropout
        output_dim = self.decoder.output_size
        learning_rate = 0.01
        #group_learning_rate = 0.001
        print_every = 100
        teacher_forcing_ratio = 0.8

        print("hidden_size:", str(hidden_size), "dropout:", str(dropout), "epochs:", str(num_epochs))
        print("encoding size:", str(encoding_size))

        # Set up optimizer and loss function
        encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=learning_rate)
        #group_optimizer = optim.Adam(self.group.parameters(), lr=group_learning_rate)

        # Loss function
        #criterion = nn.KLDivLoss()
        criterion = nn.MSELoss()
        #criterion = nn.BCEWithLogitsLoss()
        #criterion = nn.MultiLabelSoftMarginLoss()
        #group_criterion = nn.BCELoss()
        #criterion = nn.NLLLoss()

        if use_cuda:
            self.encoder = self.encoder.to(tdevice)
            self.decoder = self.decoder.to(tdevice)
            #self.group = self.group.to(tdevice)

        num_examples = len(X)
        print("X 000:", str(type(X[0][0][0])), "Y 00:", str(type(Y[0][0])))
        input_dim = self.encoder.input_size
        print("X list:", str(len(X)), "Y list:", str(len(Y)))

        print("input_dim: ", str(input_dim))
        print("output_dim: ", str(output_dim))

        # Train the model
        for epoch in range(num_epochs):
            print("epoch", str(epoch))
            i = 0
            while (i < num_examples):
                batchXnp = X[i]
                batchYnp = Y[i]
                #if debug: print("batchX len:", str(len(batchXnp)), "batchY len:", str(len(batchYnp)))
                if type(batchYnp) is list:
                    batchYnp = numpy.asarray(batchYnp)

                batchYnp = batchYnp.astype('float64')
                batchY = torch.tensor(batchYnp, dtype=torch.float64, device=tdevice, requires_grad=True)

                #print("batchX size:", str(batchX.size()), "batchY size:", str(batchY.size()))
                #if debug: print("batchX[0]:", str(batchXnp[0]))
                #if debug: print("batchY size:", str(batchY.size()))

                labels = batchY.view(1, -1, output_dim)
                seq_length = labels.size(1)

                #input_length = self.encoder.read_cycles
                #if debug: print("seq_length:", str(seq_length))

                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                #group_optimizer.zero_grad()
                loss = 0

                # Run the encoder
                mem_block, encoder_hidden = self.encoder(batchXnp)

                # Run the decoder
                decoder_hidden = encoder_hidden
                #print('decoder hidden:', decoder_hidden.size())
                use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

                # Initialize the prediction
                x_i = torch.zeros(hidden_size, dtype=torch.float64, device=tdevice)
                output_indices = []
                output_probs = []
                correct_ranks = labels.squeeze()
                flatten = False
                #if self.group_thresh is not None:
                #    flatten = False
                correct_indices = ranks_to_indices(correct_ranks.tolist(), flatten)

                # INVERT the ranks?
                if self.invert_ranks:
                    correct_indices.reverse()

                num_timesteps = len(correct_indices)
                #if i==0: print('correct_indices:', correct_indices)
                done_mask = torch.ones(seq_length, dtype=torch.float64, device=tdevice)
                current_rank = []
                x_list = []

                # Run until we've picked all the items
                #di = 0
                #while torch.max(done_mask).item() > 0.0:
                output_probs = []
                target_probs = []
                for di in range(num_timesteps):
                    index, decoder_hidden, done_mask, log_probs = self.decoder(x_i.view(1, -1), decoder_hidden, mem_block, done_mask, train=True)
                    #print('di:', di, 'mask:', done_mask)
                    output_probs.append(log_probs)
                    #output_probs.append(output_matrix.view(1, -1, 2))

                    # Select multiple items at each timestep
                    index = int(index.item())
                    x_i = mem_block[index]
                    # Teacher forcing for training
                    if use_teacher_forcing:
                        x_correct = mem_block[correct_indices[di]]
                        x_i = x_correct
                        for ei in correct_indices[di]:
                            done_mask[ei] = float('-inf')
                    '''
                    else:
                        x_i_list = []
                        for ei in range(seq_length):
                            print('output:', ei, output_matrix[ei])
                            ei_guess = torch.argmax(output_matrix[ei]) # See if 0 (yes) or 1 (no) was predicted
                            if ei_guess == 1:
                                print('choosing', ei)
                                done_mask[ei] = float('-inf')
                                x_i_list.append(mem_block[ei])
                        if len(x_i_list) == 0:
                            x_i = torch.zeros(self.hidden_size, dtype=torch.float64, device=tdevice)
                        else:
                            x_i = torch.stack(x_i_list, dim=0)
                    '''

                    # Average the chosen items to use as input to the decoder
                    x_i = x_i.view(-1, hidden_size)
                    if x_i.size(0) > 1:
                        x_i = torch.mean(x_i, dim=0)
                    #print('x_i:', x_i.size())

                    x_list.append(x_i)

                    # Create the target prob distribution
                    '''
                    target_tensor_zero = torch.zeros((seq_length, 1), dtype=torch.float64, device=tdevice, requires_grad=False)
                    target_tensor_one = torch.ones((seq_length, 1), dtype=torch.float64, device=tdevice, requires_grad=False)
                    target_tensor = torch.cat((target_tensor_one, target_tensor_zero), dim=1)
                    '''
                    target_tensor = torch.zeros((seq_length), dtype=torch.float64, device=tdevice, requires_grad=False)
                    #print('target_tensor size:', target_tensor.size())
                    if di < len(correct_indices):
                        for val in correct_indices[di]:
                            target_tensor[val] = 1.0
                            #target_tensor[val][0] = 0.0
                        print('corr:', correct_indices[di], 'target_tensor:', target_tensor)
                    target_tensor = target_tensor.view(1, seq_length)
                    target_probs.append(target_tensor)
                    #print(di, 'target:', target_tensor)
                    #print(di, 'predic:', log_probs)

                    # Calculate loss per timestep
                    #loss += criterion(output_matrix.view(1, -1, 2), target_tensor)
                    #print('loss:', loss.item())
                    #loss.backward(retain_graph=True)
                    #encoder_optimizer.step()
                    #decoder_optimizer.step()

                    di += 1

                output_indices.append(current_rank)

                # Un-invert the ranks
                if self.invert_ranks:
                    output_indices.reverse()

                #output_ranks = indices_to_ranks(output_indices)
                #if i==0: print('output_indices:', output_indices)

                output_tensor = torch.stack(output_probs).view(-1, seq_length)
                if num_timesteps > 1:
                    target_tensor = torch.stack(target_probs).view(-1, seq_length)
                    target_tensor = smooth_distribution(target_tensor, self.sigma)
                else:
                    target_tensor = target_probs[0].view(1, -1)

                #print('target:', target_tensor)
                target_tensor = torch.log(target_tensor)
                print('output tensor:', output_tensor)
                print('target smoothed:', target_tensor)

                loss = criterion(output_tensor, target_tensor)

                #loss = criterion(torch.tensor(output_ranks, dtype=torch.float, device=tdevice, requires_grad=True), correct_ranks)
                print('loss:', loss.item())
                loss.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()

                #return loss.item() / target_length
                if (i) % print_every == 0:
                    print('Epoch [%d/%d], Loss: %.4f' %(epoch, num_epochs, loss.item()))
                i += 1
        print('training took', time.time()-start, 's')

    def predict(self, testX, batch_size=1):
        #print("X 000:", str(type(testX[0][0][0])))
        #print("X list:", str(len(testX)))
        outputs = []
        max_di = 1000

        # Run the model
        for i in range(len(testX)):
            batchXnp = testX[i]
            #if debug: print("batchX len:", len(batchXnp))

            #print("batchX size:", str(batchX.size()), "batchY size:", str(batchY.size()))
            #if debug: print("testX[0]:", str(batchXnp[0]))
            seq_length = len(batchXnp)
            #input_length = self.encoder.read_cycles
            #if debug: print("test seq_length:", str(seq_length))

            # Run the encoder
            mem_block, encoder_hidden = self.encoder(batchXnp)

            # Run the decoder
            decoder_hidden = encoder_hidden

            # Initialize the prediction
            x_i = torch.zeros(self.encoder.hidden_size, dtype=torch.float64, device=tdevice)
            output_indices = []
            done_mask = torch.ones(seq_length, dtype=torch.float64, device=tdevice)

            # Run until we've picked all the items
            #current_rank = []
            #x_list = []

            # Run until we've picked all the items
            chosen = []
            di = 0
            #avg_gap = 0.0
            while torch.max(done_mask).item() > 0.0 and (di < max_di):
                index, decoder_hidden, done_mask, log_probs = self.decoder(x_i.view(1, -1), decoder_hidden, mem_block, done_mask)

                # Select multiple items at each timestep
                index = int(index.item())

                # Allow multiple events to be output at the same rank (prob threshold?)
                log_probs = log_probs * done_mask
                # Fix any nans
                for li in range(log_probs.size(0)):
                    if math.isnan(log_probs[li]) or done_mask[li] == float('-inf'):
                        log_probs[li] = float('-inf')
                print('test di:', di, 'mask:', done_mask)
                print('log_probs:', log_probs)
                max_tensor, index_tensor = torch.max(log_probs, dim=0)
                #target_index = int(index_tensor.item())
                max_prob = max_tensor.item()
                #print('max_prob:', max_prob)
                targets = []
                n = float(log_probs.size(0))
                #print('n:', n)

                # Elastic probability threshold
                '''
                if di == 0:
                    if math.isinf(max_prob):
                        avg_gap = 0.0
                    else:
                        probs2 = log_probs.tolist()
                        sorted_probs = []
                        for prob in probs2:
                            if not math.isinf(prob): # Ignore -inf values
                                sorted_probs.append(prob)

                        # Remove outliers
                        elements = numpy.array(sorted_probs)
                        prob_mean = numpy.mean(elements, axis=0)
                        prob_sd = numpy.std(elements, axis=0)
                        sorted_probs = [x for x in sorted_probs if (x > prob_mean - 1 * prob_sd)]
                        sorted_probs = [x for x in sorted_probs if (x < prob_mean + 1 * prob_sd)]

                        if len(sorted_probs) < 2: # Make sure there are at least 2 probs left
                            avg_gap = 0.0
                        else:
                            sorted(sorted_probs, reverse=True)
                            gaps = []
                            for gindex in range(1, len(sorted_probs)):
                                gval = sorted_probs[gindex]
                                prev = sorted_probs[gindex-1]
                                diff = math.fabs(gval-prev)
                                gaps.append(diff)
                            avg_gap = torch.mean(torch.tensor(gaps, dtype=torch.float64, device=tdevice)).item()/2.0
                    #print('avg_gap:', avg_gap)
                '''

                for j in range(log_probs.size(0)):
                    if done_mask[j] > 0.0:# or max_prob == 0.0:
                        prob = log_probs[j]
                        if (math.fabs(max_prob - prob) <= (self.group_thresh)) or math.isinf(max_prob):
                            targets.append(j)
                            print('choosing item', j, 'with prob', log_probs[j].item())
                            done_mask[j] = float('-inf')


                # For binary function
                '''
                for j in range(seq_length):
                    ei_guess = torch.argmax(output_matrix[j]) # See if 0 (yes) or 1 (no) was predicted
                    print('ei:', j, 'output:', output_matrix[j])
                    if ei_guess == 1:
                        done_mask[j] = float('-inf')
                        if j not in chosen:
                            print('chooosing', j)
                            targets.append(j)
                            chosen.append(j)
                '''
                if len(targets) > 0:
                    output_indices.append(targets)

                xi_list = []
                x_i = torch.zeros(self.hidden_size, dtype=torch.float64, device=tdevice)
                print('targets:', len(targets))
                for ti in targets:
                    x_i = mem_block[ti]
                    xi_list.append(x_i)
                if len(xi_list) > 1:
                    x_i = torch.mean(torch.stack(xi_list), dim=0)
                #print('test x_i:', x_i.size())

                di += 1

            #output_indices.append(current_rank)
            # Un-invert the ranks
            if self.invert_ranks:
                output_indices.reverse()

            output_ranks = indices_to_ranks(output_indices)
            #print('output_ranks:', output_ranks)
            outputs.append(output_ranks)
        return outputs


''' Ranks should be a list of integers, NOT scaled
'''
def ranks_to_indices(ranks, flatten=True):
    # Given a list of n items, each with a corresponding rank
    # For now, rank them sequentially even if they have the same rank
    #print('ranks:', ranks)
    max_rank = int(numpy.max(numpy.asarray(ranks)))
    #print('ranks_to_indices: max rank:', max_rank)
    indices_multiple = [None] * (max_rank+1)
    print('ranks:', ranks)

    if type(ranks) == float:
        num_ranks = 1
        ranks = [ranks]
    else:
        num_ranks = len(ranks)

    for i in range(num_ranks):
        rank = int(ranks[i])
        #print('rank:', rank, 'index:', i)
        if indices_multiple[rank] is None:
            indices_multiple[rank] = []
        indices_multiple[rank].append(i)
    indices_multiple = [x for x in indices_multiple if x is not None] # Filter out none entries
    print('indices_multiple:', indices_multiple)

    if flatten:
        print('flatten')
        indices = []
        for index_list in indices_multiple:
            if index_list is not None:
                for item in index_list:
                    indices.append(item)
        return indices
    else:
        return indices_multiple


''' For now, indices is a single list of integers
'''
def indices_to_ranks(indices):
    #max_index = int(numpy.max(numpy.asarray(indices)))
    if type(indices) == float:
        num_indices = 1
    else:
        if type(indices[0]) is list:
            print('indices_to_rank: group')
            num_indices = 0
            for sub in indices:
                num_indices += len(sub)
        else:
            print('indices_to_rank: linear')
            num_indices = len(indices)

    print('indices_to_ranks:', num_indices, ':', indices)
    #print('num_indices:', num_indices)
    ranks = [None]*(num_indices)
    for i in range(len(indices)):
        if type(indices[i]) is list: # Handle multiple items at the same rank
            for item in indices[i]:
                ranks[int(item)] = i
        else:
            ranks[int(indices[i])] = i
    return ranks


''' Gaussian function for smoothing target distribution
'''
def gaussian(x, mu, sig):
    return numpy.exp(-numpy.power(x - mu, 2.) / (2 * numpy.power(sig, 2.)))


''' Smooth the target distribution
    target: a tensor of the target distribution (timesteps, n)
'''
def smooth_distribution(target, sig):
    n = target.size(1)
    #sig = float(n)/float(10)
    #sig = 0.5
    slices = []

    # For each slice of the tensor
    print('target', target.size())
    for y in range(n):
        slice = target[:, y].squeeze()
        #print('slice:', slice.size())
        one_index = torch.argmax(slice, dim=0).item()
        for k in range(slice.size(0)):
            slice[k] = max(slice[k].item(), gaussian(abs(one_index - k), 0, sig))
        slices.append(slice)

    # Concatenate the slices
    smoothed_target = torch.t(torch.stack(slices, dim=0))
    return smoothed_target
