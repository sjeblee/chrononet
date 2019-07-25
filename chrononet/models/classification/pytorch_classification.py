# @author sjeblee@cs.toronto.edu

import math
import numpy
import os
import random
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

numpy.set_printoptions(threshold=numpy.inf)
debug = True
tdevice = 'cpu'
use_cuda = torch.cuda.is_available()
if use_cuda:
    tdevice = torch.device('cuda:3')

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
class CNN_Text(nn.Module):

    def __init__(self, embed_dim, class_num, kernel_num=200, kernel_sizes=6, dropout=0.1, ensemble=False, hidden_size=100):
        super(CNN_Text, self).__init__()
        D = embed_dim
        C = class_num
        Ci = 1
        self.Co = kernel_num
        self.Ks = kernel_sizes
        self.ensemble = ensemble
        self.convs = []

        for n in range(1, self.kernel_sizes+1):
            self.convs.append(nn.Conv2d(Ci, self.Co, (n, D)).to(tdevice))

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.Co*self.Ks, C) # Use this layer when train with only CNN model, i.e. No ensemble

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = x.unsqueeze(1)  # (N, Ci, W, D)]
        x1 = self.conv_and_pool(x, self.conv11) # (N,Co)
        x2 = self.conv_and_pool(x, self.conv12) # (N,Co)
        x3 = self.conv_and_pool(x, self.conv13) # (N,Co)
        x4 = self.conv_and_pool(x, self.conv14) # (N,Co)
        x5 = self.conv_and_pool(x, self.conv15) # (N,Co)
        #x6 = self.conv_and_pool(x,self.conv16) #(N,Co)
        #x7 = self.conv_and_pool(x,self.conv17)
        #x8 = self.conv_and_pool(x,self.conv18)
        x = torch.cat((x1, x2, x3, x4, x5), 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)

        if not self.ensemble: # Train CNN with no ensemble
            logit = self.fc1(x)  # (N, C)
        else: # Train CNN with ensemble. Output of CNN will be input of another model
            logit = x
        return logit

    ''' Create and train a CNN model
        Hybrid features supported - pass structured feats as X2
        Does NOT support joint training yet
        returns: the CNN model
    '''
    def fit(X, Y, act=None, num_epochs=10, loss_func='crossentropy', dropout=0.1, pretrainX=[], pretrainY=[], query_vectors=None):
        # Train the CNN, return the model
        st = time.time()
        use_query = (query_vectors is not None)

        # Check for pretraining
        pretrain = False
        if len(pretrainX) > 0 and len(pretrainY) > 0:
            pretrain = True
            print("Using pretraining")

        # Params
        Xarray = numpy.asarray(X).astype('float')
        dim = Xarray.shape[-1]
        num_labels = Y.shape[-1]
        batch_size = 32
        learning_rate = 0.001
        #num_epochs = num_epochs
        #best_acc = 0
        #last_step = 0
        #log_interval = 1000

        if pretrain:
            print("pretraining...")
            for k in range(len(pretrainX)):
                trainX = pretrainX[k]
                trainY = pretrainY[k]
                pre_labels = trainY.shape[-1]

                if k == 0: # Init cnn model
                    cnn = CNN_Text(dim, pre_labels, dropout=dropout, kernel_sizes=kernel_sizes)
                else: # Replace the last layer
                    cnn.fc1 = nn.Linear(cnn.Co*cnn.Ks, pre_labels)

                # Pre-train the model
                if use_cuda:
                    cnn = cnn.cuda()
                cnn = train_cnn(cnn, trainX, trainY, batch_size, num_epochs, learning_rate)

            # Replace the last layer for the final model
            cnn.fc1 = nn.Linear(cnn.Co*cnn.Ks, num_labels)

        else: # No pre-training
            if use_query:
                dropout = 0.1
                cnn = CNN_Query(query_vectors, dim, num_labels, dropout=dropout, kernel_sizes=kernel_sizes)
            else:
                cnn = CNN_Text(dim, num_labels, dropout=dropout, kernel_sizes=kernel_sizes)

        if use_cuda:
            cnn = cnn.cuda()
            if use_query:
                cnn.query_vectors = cnn.query_vectors.cuda()

        # Train final model
        cnn = train_cnn(cnn, X, Y, batch_size, num_epochs, learning_rate, query=use_query)

        return cnn

    def train_cnn(cnn, X, Y, batch_size, num_epochs, learning_rate, query=False):
        Xarray = numpy.asarray(X).astype('float')
        Yarray = Y.astype('int')
        X_len = Xarray.shape[0]
        dim = Xarray.shape[-1]
        num_labels = Yarray.shape[-1]
        num_batches = math.ceil(X_len/batch_size)
        print("X numpy shape: ", str(Xarray.shape), "Y numpy shape:", str(Yarray.shape))

        optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

        steps = 0
        st = time.time()
        cnn.train()
        for epoch in range(num_epochs):
            print("epoch", str(epoch))
            i = 0
            numpy.random.seed(seed=1)
            permutation = torch.from_numpy(numpy.random.permutation(X_len)).long()
            Xiter = Xarray[permutation]
            Yiter = Yarray[permutation]

            while i+batch_size < X_len:
                batchX = Xiter[i:i+batch_size]
                batchY = Yiter[i:i+batch_size]
                Xtensor = torch.from_numpy(batchX).float()
                Ytensor = torch.from_numpy(batchY).long()
                if use_cuda:
                    Xtensor = Xtensor.cuda()
                    Ytensor = Ytensor.cuda()
                feature = Variable(Xtensor)
                target = Variable(Ytensor)
                i = i+batch_size

                optimizer.zero_grad()
                if query:
                    logit, attn_maps = cnn(feature)
                else:
                    logit = cnn(feature)

                loss = F.cross_entropy(logit, torch.max(target, 1)[1])
                print('loss: ', str(loss.data[0]))
                loss.backward()
                optimizer.step()
                steps += 1

            # Print epoch time
            ct = time.time() - st
            unit = "s"
            if ct > 60:
                ct = ct/60
                unit = "m"
            print("time so far: ", str(ct), unit)
        return cnn

    def test_cnn(model, testX, testids=None, probfile='/u/yoona/data/torch/probs_win200_epo10', labelencoder=None, collapse=False, threshold=0.1):
        y_pred = [] # Original prediction if threshold is not in used for ill-defined.
        y_pred_softmax = []
        y_pred_logsoftmax = []
        softmax = nn.Softmax(dim=1)
        logsoftmax = nn.LogSoftmax(dim=1)

        new_y_pred = [] # class prediction if threshold for ill-difined is used.
        if probfile is not None:
            probs = []

        for x in range(len(testX)):
            input_row = testX[x]

            icd = None
            if icd is None:
                input_tensor = torch.from_numpy(input_row.astype('float')).float().unsqueeze(0)
                if use_cuda:
                    #input_tensor = input_tensor.contiguous().cuda()
                    input_tensor = input_tensor.cuda()
                print('input_tensor:', input_tensor.size())
                print('input_tensor[0, 0]:', input_tensor[0, 0])
                icd_var, attn_maps = model(Variable(input_tensor))
                # Softmax and log softmax values
                icd_vec = logsoftmax(icd_var)
                icd_vec_softmax = softmax(icd_var)
                icd_code = torch.max(icd_vec, 1)[1].data[0]

                # Save the first example attn map
                if x == 0:
                    tempfile = '/u/sjeblee/research/va/data/cnn_query_cghr/attn_0.csv'
                    outf = open(tempfile, 'w')
                    for row in attn_maps[0]:
                        row = row.squeeze()
                        for i in range(row.size(0)):
                            outf.write(str(row.data[i]) + ',')
                        outf.write('\n')
                    outf.close()

    	    # Assign to ill-defined class if the maximum probabilities is less than a threshold.
    	    #icd_prob = torch.max(icd_vec_softmax,1)[0]
    	    #if icd_prob < threshold:
                #    new_y_pred.append(9)
    	    #else:
    	    #    new_y_pred.append(icd_code)

                # Save the probabilties
                #if probfile is not None:
                #    probs.append(icd_prob)

            y_pred.append(icd_code)

        #if probfile is not None:
        #    pickle.dump(probs, open(probfile, "wb"))

        #print "Probabilities: " + str(probs)

        return y_pred  # Uncomment this line if threshold is not in used.
        #return new_y_pred  # Comment this line out if threshold is not in used.
