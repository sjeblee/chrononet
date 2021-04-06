#!/usr/bin/python3
# Convert features and labels to numpy arrays

import ast
import numpy

from lxml import etree
from sklearn.preprocessing import LabelEncoder

# Local imports
from data_tools import data_util

debug = True


''' Convert the dataframe feature column to a numpy array for processing
    use_numpy: True if we should convert to a numpy array
    doc_level: True if features should aggregate features at the document level (NOT IMPLEMENTED)
'''
def to_feats(df, use_numpy=True, doc_level=False, feat_names=None):
    print('to_feats: use_numpy:', use_numpy, 'doc_level:', doc_level, 'feats:', feat_names)
    feats = []
    feat_columns = []

    # Get the names of the feature columns for the df
    if feat_names is None:
        feat_columns.append('feats')
    else: # Handle multiple feature types
        for fname in feat_names:
            feat_columns.append('feats_' + fname)

    # Load a list of all the features
    for i, row in df.iterrows():
        if len(feat_columns) > 1:
            mini_feat_list = []
        for featname in feat_columns:
            flist = row[featname]
            if type(flist) is str:
                flist = ast.literal_eval(flist)

            if len(feat_columns) > 1:
                mini_feat_list.append(flist)
            else:
                feats.append(flist)
            #if debug:
            #    print('to_feats: ', row['docid'], 'feats:', flist)
            if debug and i == 0:
                print('feats:', flist)
                #if len(flist) > 0:
                #    print('feats[0]:', type(flist[0]), flist[0])
        if len(feat_columns) > 1:
            feats.append(mini_feat_list)

    if use_numpy:
        return numpy.asarray(feats).astype('float')
    else:
        return feats

def to_labels(df, labelname, labelencoder=None, encode=True):
    if debug:
        print('to_labels: ', labelname, ', encode: ', str(encode))
    #data = ast.literal_eval(df[labelname])
    labels = []

    # Extract the labels from the dataframe
    for i, row in df.iterrows():
        flist = row[labelname]
        if debug: print('flist:', type(flist), str(flist))
        #if type(flist) == str:
        #    flist = ast.literal_eval(flist)
        if debug and i == 0:
            print('labels[0]:', flist)
        labels.append(flist)

    # Normalize the rank values
    if labelname == 'event_ranks':
        enc_labels = []
        for rank_list in labels:
            norm_ranks = []
            if rank_list is not None and len(rank_list) > 0 and not rank_list == '[]':
                if type(rank_list) == str:
                    rank_list = ast.literal_eval(rank_list)
                min_rank = float(numpy.nanmin(numpy.array(rank_list, dtype=numpy.float), axis=None))
                # Scale min rank to 0
                if min_rank is not numpy.nan and min_rank > 0:
                    rank_list_scaled = []
                    for rank in rank_list:
                        if rank is None or rank is numpy.nan:
                            rank_list_scaled.append(-1)
                        else:
                            rank_list_scaled.append(rank - min_rank)
                    rank_list = rank_list_scaled
                if encode:
                    max_rank = float(numpy.nanmax(numpy.array(rank_list, dtype=numpy.float), axis=None))
                    if max_rank == numpy.nan or max_rank == 0:
                        print('WARNING: max rank is 0')
                        norm_ranks = rank_list # Don't normalize if they're all 0
                    else: # Normalize
                        norm_ranks = []
                        for rank in rank_list:
                            if rank is None or rank == -1:
                                norm_ranks.append(float(-1))
                            else:
                                norm_ranks.append(float(rank)/max_rank)
                        rank_list = norm_ranks
            print('normalized ranks', rank_list)
            enc_labels.append(numpy.asarray(rank_list))
        labels = enc_labels

    # Encode other labels
    elif encode:
        if labelencoder is None:
            labelencoder = create_labelencoder(labels)
        labels = encode_labels_plain(labels)
    if debug: print('to_labels:', labelname, 'encode:', encode, 'labels:', len(labels))
    return labels, labelencoder


def create_labelencoder(data, num=0):
    global labelencoder, onehotencoder, num_labels
    if debug: print("create_labelencoder: data[0]: ", str(data[0]))
    if type(data[0]) is list:
        data = [j for sub in data for j in sub]
    labelencoder = LabelEncoder()
    labelencoder.fit(data)
    num_labels = len(labelencoder.classes_)
    #onehotencoder = OneHotEncoder()
    #onehotencoder.fit(data2)

    return labelencoder


def encode_labels_plain(data, labenc=None):
    if labenc is None:
        labenc = labelencoder
    print('encode_labels_plain:', str(data))
    if type(data[0]) is list:
        new_lab = []
        for item in data:
            new_lab.append(labenc.transform(item))
    else:
        new_lab = labenc.transform(data)
    #print('encoded labels:', new_lab)
    return new_lab


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


''' Put 0 features corresponding to the labels if no features are required for the model
    (i.e. for random or mention order)
'''
def dummy_function(df, feat_name='feats', doc_level=False):
    print('dummy_function')
    df[feat_name] = '0'
    if not doc_level:
        for i, row in df.iterrows():
            fake_feats = []
            if row['events'] is not None:
                print('events:', row['events'], type(row['events']))
                if (type(row['events']) is list and len(row['events']) == 0) or (type(row['events']) is str and row['events'] == '[]'):
                    print('event list is empty')
                else:
                    if type(row['events']) is list:
                        event_list = row['events']
                    else:
                        event_list = etree.fromstring(str(row['events']))
                    if debug: print(row['docid'], 'dummy_function events:', type(event_list))#, etree.tostring(event_list))
                    if type(event_list) == str:
                        #event_list = eval(event_list)
                        event_list = ast.literal_eval(event_list)
                    if debug: print(row['docid'], 'dummy_function events len:', len(event_list))
                    for entry in event_list:
                        fake_feats.append(0)
            df.at[i, feat_name] = fake_feats
            print('dummy_function:', row['docid'], 'feats:', len(fake_feats))
    return df

def do_nothing(df, feat_name='feats', doc_level=False):
    if feat_name not in df.columns:
        df[feat_name] = '0'
    return df
