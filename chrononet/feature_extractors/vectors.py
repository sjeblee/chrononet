#!/usr/bin/python3
# Word or sentence vector features

import numpy
from gensim.models import KeyedVectors, Word2Vec, FastText
from lxml import etree
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer

# Local imports
from data_tools import data_util
from data_tools import temporal_util as tutil

debug = True
bert_file = "monologg/biobert_v1.1_pubmed"
bert_tokenizer = BertTokenizer.from_pretrained(bert_file)


''' Return a single vector per events (flattened context)
'''
def event_vectors_flat(df, vec_model):
    return event_vectors(df, vec_model, flatten=True)


''' Convert the dataframe feature column to a numpy array for processing
'''
def event_vectors(df, feat_name='feats', vec_model=None, flatten=False):
    df[feat_name] = ''
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
            event_text = text[int(span[0]): int(span[1])]
            #event_text = event.text

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


def context_words(prev, target, next, max_len=10, lowercase=True, use_bert=False):
    words = []
    c_words = []
    if use_bert:
        target_words = bert_tokenizer.tokenize(target)
        prev_words = bert_tokenizer.tokenize(prev)
        next_words = bert_tokenizer.tokenize(next)
    else:
        target_words = data_util.split_words(target)
        prev_words = data_util.split_words(prev)
        next_words = data_util.split_words(next)
    prev_size = 0
    next_size = 0
    word_flags = []
    #if len(target_words) > max_len:
    #    target_words = target_words[0:max_len]
    #else:
    #prev_size = int((max_len - len(target_words)))
    prev_size = min(len(prev_words), max_len)
    next_size = min(len(next_words), max_len)
    if prev_size < 1:
        prev_size = 0
        #next_size = len(target_words) - prev_size
    if next_size < 1:
        next_size = 0

    if prev_size > 0:
        for word in prev_words[(-1*prev_size):]:
            if lowercase:
                word = word.lower()
            c_words.append(word)
            word_flags.append(0)
    for word in target_words:
        if lowercase:
            word = word.lower()
        words.append(word)
        c_words.append(word)
        word_flags.append(1)
    if next_size > 0:
        for word in next_words[0:next_size]:
            if lowercase:
                word = word.lower()
            c_words.append(word)
            word_flags.append(0)
    print('words, flags:', len(c_words), len(word_flags))
    return words, c_words, word_flags
    #return words, c_words


def vec_word(vec_model, word, flags=[0, 0, -1], lowercase=True):
    if lowercase:
        word = word.lower()
    vec = get(word, vec_model)
    vec = numpy.append(vec, flags, axis=0)
    return vec


def word_vectors(df, vec_model):
    print("TODO")


def bert_event_vectors(df, feat_name='feats', flatten=False, use_iso_value=False, context_size=5):
    print('bert_event_vectors')
    return elmo_event_vectors(df, feat_name, flatten, use_iso_value, context_size, use_bert=True)


''' Convert the dataframe feature column to a numpy array for processing
'''
def elmo_event_vectors(df, feat_name='feats', flatten=False, use_iso_value=False, context_size=5, use_bert=False):
    verilogue = False
    filter_missing_ranks = True

    # Create the time type encoder
    time_types = ['UNK', 'DATE', 'TIME', 'DATETIME', 'DURATION', 'SET']
    timetypeencoder = LabelEncoder()
    timetypeencoder.fit(time_types)
    flags = []

    df[feat_name] = ''
    for i, row in df.iterrows():
        if debug: print('event_vectors', str(i))
        # Create timeid map
        timex_map = {}
        signal_map = {}
        utt_map = None
        text = row['text'].strip() # NOTE: we need to strip the extra whitespace to get the right event spans

        # Verilogue data processing
        if type(row['tags']) is list:
            verilogue = True
        if verilogue:
            tags = row['tags']
            events = row['events']
            for item in tags:
                if item.tag == 'TIMEX3':
                    timex_map[str(item.features['annotation_id'])] = item
            utt_map = {}
            utts = etree.fromstring(text)
            for utt in utts:
                print('verilogue utt:', str(utt))
                utt_map[int(utt.get('seqNo'))] = utt
            print('verilogue utts:', len(utts))
        else: # VA and THYME data
            print('narr text:', text)
            tags = data_util.load_xml_tags(row['tags'], decode=False)
            events = etree.fromstring(row['events'])

            # TIMEX map
            for timex in tags.findall('TIMEX3'):
                tid = timex.get('id')
                if tid is None:
                    tid = timex.get('tid')
                timex_map[tid] = timex
            # SIGNAL map
            for sig in tags.findall('SIGNAL'):
                sid = sig.get('id')
                if sid is None:
                    sid = sig.get('sid')
                signal_map[sid.lower()] = sig
                print('sid:', sid.lower())

        # Get the DCT (death date for VA)
        dct_string = row['dct']

        event_vecs = []
        for event in events:
            # Get spans and attributes
            if verilogue:
                event_text = ''
                for span in event.spans:
                    uid = int(span.seqNo)
                    word_text = utt_map[uid].text[span.startIndex:span.endIndex]
                    event_text += ' ' + word_text
                event_text = event_text.strip()
                print('verilogue event text:', event_text)
                prev_uid = int(event.spans[0].seqNo)
                next_uid = int(event.spans[-1].seqNo)
                prev = utt_map[prev_uid].text[0:event.spans[0].startIndex]
                next = utt_map[next_uid].text[event.spans[-1].endIndex:]
                pol_flag = 0
                if 'modality' in event.features and event.features['modality'] == 'negative':
                    pol_flag = 1
                flags = []
                flags.append(pol_flag) # polarity flag
                flags.append(0) # position flag
            else:
                span = event.get('span').split(',')
                prev = text[0:int(span[0])]
                next = text[int(span[1]):]
                event_text = text[int(span[0]): int(span[1])]
                #event_text = event.text
                print('event text:', event.text, 'span text:', event_text, span[0], span[1])
                #print('text:', text)
                print('prev:', prev)
                print('next:', next)
                pol = event.get('polarity')
                pol_flag = 0
                if pol is not None and pol.lower() == 'neg':
                    pol_flag = 1
                #flags = [1] # target flag
                flags = []
                flags.append(pol_flag) # polarity flag
                position = float(span[0])/float(len(text))
                flags.append(position) # position value
                #if debug: print('event_vector for:', event_text)

            # Get time phrase if there is one
            time_words = None
            time_val = None
            tflags = []
            if verilogue:
                if 'relatedToTime' in event.features:
                    time_id_string = event.features['relatedToTime']
                    time_id = time_id_string.split(',')[0] # Just get the first time phrase
                    if time_id not in timex_map:
                        print('WARNING: time_id not found:', time_id)
                    else:
                        timex = timex_map[time_id]
                        time_text = timex.features['rawText']
                        time_type = 'UNK'
                        print('ver time text:', time_text)
                        if 'timexType' in timex.features:
                            time_type = timex.features['timexType']
                            if time_type == '':
                                time_type = 'UNK'
                        if use_bert:
                            time_words = bert_tokenizer.tokenize(time_text)
                        else:
                            time_words = data_util.split_words(time_text)
                        for tw in time_words:
                            tflags.append(1)
                        # Add the time type
                        tflags = [0] + tflags
                        time_words = [time_type] + time_words # Append the time type to the beginning of the list
                        print('vectors.elmo_event_vectors: extracted time_words:', time_words)
                else:
                    time_id_string = None
            else:
                time_id_string = event.get('relatedToTime')
                if time_id_string is not None:
                    time_id = time_id_string.split(',')[0] # Just get the first time phrase
                    if time_id not in timex_map:
                        print('WARNING: time_id not found:', time_id)
                    else:
                        timex = timex_map[time_id]
                        time_text = timex.text
                        # Get the timex type and encode it
                        time_type = 'UNK'
                        if 'type' in timex.attrib:
                            time_type = timex.get('type')
                        #time_type_enc = timetypeencoder.transform([time_type])[0]

                        if use_iso_value:
                            tval = tutil.time_value(timex, dct_string, return_text=False)
                            print('time value of', time_text, ':', tval)
                            if tval is None:
                                date = 0
                                time = 0
                            else:
                                vals = tval.split('T')
                                date = int(vals[0].replace('-', ''))
                                time_string = vals[1]
                                if '.' in time_string:
                                    time_string = time_string.split('.')[0]
                                time = int(time_string.replace(':', ''))
                            time_val = [date, time]
                        #else:
                        if use_bert:
                            time_words = bert_tokenizer.tokenize(time_text)
                        else:
                            time_words = data_util.split_words(time_text)
                        for tw in time_words:
                            tflags.append(1)

                        # Get SIGNAL text
                        signal_id = event.get('signalID')
                        if signal_id is not None:
                            signal = signal_map[signal_id.lower()]
                            signal_text = signal.text
                            if use_bert:
                                signal_words = bert_tokenizer.tokenize(signal_text)
                            else:
                                signal_words = data_util.split_words(signal_text)
                            sflags = []
                            for sw in signal_words:
                                sflags.append(0)
                            time_words = signal_words + time_words # Add signal words to time phrase
                            tflags = sflags + tflags # Append time word flags

                        # Add the time type
                        tflags = [0] + tflags
                        time_words = [time_type] + time_words # Append the time type to the beginning of the list
                        print('vectors.elmo_event_vectors: extracted time_words:', time_words)

            #vec = data_util.split_words(event_text)
            vec, context, c_flags = context_words(prev, event_text, next, max_len=context_size, use_bert=use_bert)
            #words, word_flags = context_words(prev, event_text, next)
            time_type_enc = [0, 0, 0, 0, 0]
            event_vecs.append((context, c_flags, time_words, tflags, time_val, time_type_enc, flags))
        df.at[i, feat_name] = event_vecs
    return df


''' Convert the dataframe feature column to a numpy array for processing
'''
def elmo_word_vectors(df, feat_name='feats'):
    df[feat_name] = ''
    verilogue = False
    for i, row in df.iterrows():
        if debug: print('elmo words', str(i))
        text = row['text']
        if type(row['tags']) is list:
            verilogue = True
        if verilogue:
            xmlnode = etree.fromstring(text)
            new_text = ''
            for utt in xmlnode:
                new_text = new_text + ' ' + utt.text
            text = new_text
        print('narr text:', text)
        words = data_util.split_words(text)
        df.at[i, feat_name] = words
    return df

def elmo_vectors(elmo, prev, words, next, flags):
    sentences = [data_util.split_words(words)]
    character_ids = batch_to_ids(sentences)
    embeddings = elmo(character_ids)['elmo_representations']
    #print('num embeddings:', len(embeddings))
    #print('embeddings[0]', embeddings[0].size())
    emb_size = 1024
    emb_vecs = embeddings[0].view(-1, emb_size)
    # TODO: concat elmo vectors w/ flags
    return emb_vecs


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
