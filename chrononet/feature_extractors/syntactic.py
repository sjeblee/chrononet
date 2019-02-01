#!/usr/bin/python3

import ast

def check_feats(feats, seq, labels):
    if not len(feats) == len(labels):
        print('len mismatch:')
        print('feats:', len(feats))
        print('seq:', len(seq))
        print('labels:', len(labels))
    assert(len(feats) == len(labels))

def sent_features(df):
    df['feats'] = ''
    for i, row in df.iterrows():
        seq_list = ast.literal_eval(row['seq'])
        label_list = ast.literal_eval(row['seq_labels'])
        sent = list(zip(seq_list, label_list))
        feats = sent2features(sent)
        check_feats(feats, seq_list, label_list)
        df.at[i, 'feats'] = feats
        print('row', row['docid'], row['seqid'])
        #print('syntactic features:', str(feats))

    return df

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

def sent2tokens(sent):
    return [token for token, label in sent]

def word2features(sent, i):
    word = sent[i][0]
    #postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        #'postag': postag,
        #'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        #postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            #'-1:postag': postag1,
            #'-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        #print("sent[i+1]", str(sent[i+1]))
        word1 = sent[i+1][0]
        #postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            #'+1:postag': postag1,
            #'+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features
