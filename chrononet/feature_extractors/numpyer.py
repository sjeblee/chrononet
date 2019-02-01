#!/usr/bin/python3
# Convert features and labels to numpy arrays

import ast
import numpy

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from data_tools import data_util

debug = True


''' Convert the dataframe feature column to a numpy array for processing
'''
def to_feats(df, use_numpy=True):
    feats = []
    for i, row in df.iterrows():
        flist = row['feats']
        #flist = ast.literal_eval(row['feats'])
        if debug and i == 0:
            print('feats[0]:', flist)
        feats.append(flist)
    if debug:
        print('to_feats: ', len(feats))
    if use_numpy:
        return numpy.asarray(feats)
    else:
        return feats

def to_labels(df, labelname, labelencoder=None, encode=True):
    #data = ast.literal_eval(df[labelname])
    labels = []
    for i, row in df.iterrows():
        flist = row[labelname]
        flist = ast.literal_eval(flist)
        if debug and i == 0:
            print('labels[0]:', flist)
        labels.append(flist)
    if encode:
        if labelencoder is None:
            labelencoder = create_labelencoder(labels)
        labels = df.apply(encode_labels(labels))
    if debug:
        print('to_labels:', labelname, 'encode:', encode, 'labels:', len(labels))
    return labels, labelencoder

def create_labelencoder(data, num=0):
    global labelencoder, onehotencoder, num_labels
    print("create_labelencoder: data[0]: ", str(data[0]))
    labelencoder = LabelEncoder()
    labelencoder.fit(data)
    num_labels = len(labelencoder.classes_)
    #onehotencoder = OneHotEncoder()
    #onehotencoder.fit(data2)

    return labelencoder


''' Encodes labels as one-hot vectors (entire dataset: 2D array)
    data: a 1D array of labels
    num_labels: the number of label classes
'''
def encode_labels(data, labenc=None, max_len=50, pad=True):
    if labenc is None:
        labenc = labelencoder
    if labenc is None: # or onehotencoder == None:
        print("Error: labelencoder must be trained before it can be used!")
        return None
    #return onehotencoder.transform(labelencoder.transform(data))
    data2 = []

    num_labels = len(labenc.classes_)
    zero_vec = data_util.zero_vec(num_labels)
    if debug: print("data: ", str(len(data)))
    for item in data:
        #print "item len: " + str(len(item))
        new_item = []
        if len(item) > 0:
            item2 = labenc.transform(item)
            for lab in item2:
                onehot = []
                for x in range(num_labels):
                    onehot.append(0)
                onehot[lab] = 1
                new_item.append(onehot)
        # Pad vectors
        if pad:
            if len(new_item) > max_len:
                new_item = new_item[0:max_len]
            while len(new_item) < max_len:
                new_item.append(zero_vec)
        data2.append(new_item)
    return data2


''' Decodes one sequence of labels
'''
def decode_labels(data, labenc=None):
    #print "decode_labels"
    if labenc is None:
        labenc = labelencoder
    data2 = []
    for row in data:
        #print "- row: " + str(row)
        lab = numpy.argmax(numpy.asarray(row))
        #print "- lab: " + str(lab)
        data2.append(lab)
    #print "- data2: " + str(data2)
    return labenc.inverse_transform(data2)
    #return labelencoder.inverse_transform(onehotencoder.reverse_transform(data))


def decode_all_labels(data, labenc=None):
    decoded_labels = []
    for sequence in data:
        labs = decode_labels(sequence, labenc)
        decoded_labels.append(labs)
    return decoded_labels
