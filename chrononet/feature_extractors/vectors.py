#!/usr/bin/python3
# Word or sentence vector features

import numpy
from gensim.models import KeyedVectors, Word2Vec, FastText
from lxml import etree

# Local imports
from data_tools import data_util

debug = True


''' Return a single vector per events (flattened context)
'''
def event_vectors_flat(df, vec_model):
    return event_vectors(df, vec_model, flatten=True)


''' Convert the dataframe feature column to a numpy array for processing
'''
def event_vectors(df, vec_model, flatten=False):
    df['feats'] = ''
    for i, row in df.iterrows():
        if debug: print('event_vectors', str(i))
        events = etree.fromstring(row['events'])
        text = row['text']
        event_vecs = []
        for event in events:
            # Get spans and attributes
            span = event.get('span').split(',')
            prev = text[0:int(span[0])]
            next = text[int(span[1]):]
            event_text = event.text
            pol = event.get('polarity')
            pol_flag = 0
            if pol is not None and pol.lower() == 'neg':
                pol_flag = 1
            flags = [1] # target flag
            flags.append(pol_flag) # polarity flag
            position = float(span[0])/float(len(text))
            flags.append(position) # position value
            #if debug: print('event_vector for:', event_text)
            vec = context_vector(vec_model, prev, event_text, next, flags=flags)
            if flatten:
                new_vec = vec[0]
                for x in range(1, len(vec)):
                    new_vec = numpy.append(new_vec, vec[x], axis=0)
                vec = new_vec
            event_vecs.append(vec)
        df.at[i, 'feats'] = event_vecs
    return df

def context_vector(vec_model, prev, target, next, max_len=10, lowercase=True, flags=[]):
    vecs = []
    target_words = data_util.split_words(target)
    prev_words = data_util.split_words(prev)
    next_words = data_util.split_words(next)
    prev_size = 0
    next_size = 0
    if len(target_words) > max_len:
        target_words = target_words[0:max_len]
    else:
        prev_size = int((max_len - len(target_words))/2)
        if prev_size < 1:
            prev_size = 0
        next_size = len(target_words) - prev_size
        if next_size < 1:
            next_size = 0

    if prev_size > 0:
        for word in prev_words[(-1*prev_size):]:
            vecs.append(vec_word(vec_model, word, lowercase=lowercase))
    for word in target_words:
        vecs.append(vec_word(vec_model, word, flags=flags, lowercase=lowercase))
    if next_size > 0:
        for word in next_words[0:next_size]:
            vecs.append(vec_word(vec_model, word, lowercase=lowercase))
    return vecs


def vec_word(vec_model, word, flags=[0, 0, -1], lowercase=True):
    if lowercase:
        word = word.lower()
    vec = get(word, vec_model)
    vec = numpy.append(vec, flags, axis=0)
    return vec


def word_vectors(df, vec_model):
    print("TODO")


def get(word, model):
    dim = model.vector_size
    if word in model:
        return list(model[word])
    else:
        return data_util.zero_vec(dim)


def load(filename):
    if '.bin' in filename:
        model = load_bin_vectors(filename, True)
    elif 'fasttext' in filename:
        model = FastText.load(filename)
    elif '.wtv' in filename:
        model = Word2Vec.load(filename)
    else:
        model = load_bin_vectors(filename, False)
    dim = model.vector_size
    return model, dim


def load_bin_vectors(filename, bin_vecs=True):
    word_vectors = KeyedVectors.load_word2vec_format(filename, binary=bin_vecs, unicode_errors='ignore')
    return word_vectors
