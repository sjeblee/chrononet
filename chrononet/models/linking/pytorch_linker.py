# @author sjeblee@cs.toronto.edu

import numpy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.modules.elmo import Elmo, batch_to_ids
from torch import optim

numpy.set_printoptions(threshold=numpy.inf)
debug = True
tdevice = 'cpu'
use_cuda = torch.cuda.is_available()
if use_cuda:
    tdevice = torch.device('cuda')

options_file = "/h/sjeblee/research/data/elmo/weights/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "/h/sjeblee/research/data/elmo/weights/elmo_2x4096_512_2048cnn_2xhighway_weights_PubMed_only.hdf5"

class PairClassifier(nn.Module):

    def __init__(self, input_size, num_epochs=10, dropout_p=0.1, loss_func='crossentropy'):
        super(PairClassifier, self).__init__()
        print('elmo files:', options_file, weight_file)
        self.elmo = Elmo(options_file, weight_file, 1, dropout=0)
        if use_cuda:
            self.elmo = self.elmo.cuda()

        self.input_size = input_size
        self.epochs = num_epochs
        self.loss_func = loss_func

        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(self.input_size*2, 2) # Use this layer when train with only CNN model, i.e. No ensemble
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # print('x:', str(x))
        batch_size = len(x)
        character_ids = batch_to_ids(x)
        if use_cuda:
            character_ids = character_ids.cuda()
        embeddings = self.elmo(character_ids)['elmo_representations']
        #print('elmo embeddings:', embeddings[0].size())
        X = embeddings[0].view(batch_size, -1, 1024) # (N, W, D)

        # TODO: embed entity and time phrase

        x = self.dropout(x)  # (N, len(Ks)*Co)

        logit = self.fc1(x)  # (N, C)
        return logit

    ''' Create and train a CNN model
        Hybrid features supported - pass structured feats as X2
        Does NOT support joint training yet
        returns: the CNN model
    '''
    def fit(self, ann_maps, timex_maps, id_maps):
        # Train and return the model
        st = time.time()

        # TODO: create the pairs

        # Params
        batch_size = 16
        learning_rate = 0.001

        if use_cuda:
            self = self.cuda()

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

    def predict(self, test_anns, test_times, testids=None):
        y_pred = {}

        # TODO: create pairs

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

class DistanceClassifier():

    #def __init__(self):
    #    pass

    def fit(self, ann_maps, timex_maps, id_maps):
        pass

    def predict(self, test_anns, test_times, testids=None):
        y_pred = {}
        for annid in test_anns.keys():
            y_pred[annid] = None

        # For each time, assign it to the closest other entity
        for timeid in test_times:
            timex_feats = test_times[timeid]
            best_distance = 10000000
            best_id = None
            time_span = timex_feats[0]
            # Find the closest entity
            for annid in test_anns.keys():
                ann_feats = test_anns[annid]
                for span in ann_feats:
                    #if span.seqNo == time_span.seqNo:
                    if span.startIndex < time_span.startIndex:
                        distance = time_span.startIndex - span.startIndex
                    else:
                        distance = span.startIndex - time_span.startIndex
                    if distance < best_distance:
                        best_distance = distance
                        best_id = annid
            if best_id is not None:
                y_pred[best_id] = timeid
                print('TL found closest event for', timeid, ':', best_id)

        return y_pred
