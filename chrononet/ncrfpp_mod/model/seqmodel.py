# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-10-17 16:47:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-03-30 16:20:07

import gc
import numpy as np
import random
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .wordsequence import WordSequence
from .crf import CRF
from ncrfpp_mod.utils.metric import get_ner_fmeasure

class SeqModel(nn.Module):
    def __init__(self, data):
        super(SeqModel, self).__init__()
        self.use_crf = data.use_crf
        print("build network...")
        print("use_char: ", data.use_char)
        if data.use_char:
            char_feature_extractor = data.char_feature_extractor
            print("char feature extractor: ", char_feature_extractor)
        word_feature_extractor = data.word_feature_extractor
        print("word feature extractor: ", word_feature_extractor)
        print("use crf: ", self.use_crf)

        self.device = data.device
        self.gpu = data.HP_gpu
        self.average_batch = data.average_batch_loss
        # add two more label for downlayer lstm, use original label size for CRF
        label_size = data.label_alphabet_size
        data.label_alphabet_size += 2
        print('data.label_alphabet_size:', data.label_alphabet_size)
        self.word_hidden = WordSequence(data)
        if self.use_crf:
            self.crf = CRF(label_size, self.gpu)

    def neg_log_likelihood_loss(self, word_inputs, batch_label, char_inputs=None, char_seq_lengths=None): #mask, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_label, mask):
        #outs = self.word_hidden(word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        outs = self.word_hidden(word_inputs, char_inputs=char_inputs, char_seq_lengths=char_seq_lengths)
        print('elmo embeddings:', outs.size())
        batch_size = outs.size(0)
        seq_len = outs.size(1)
        print('seq_len:', seq_len)
        #print('batch_label:', type(batch_label))
        mask = torch.ones((batch_size, seq_len)).byte()
        if batch_size > 1:
            for i in range(batch_size):
                #print('padding i=', i)
                orig_size = len(batch_label[i])
                pad_size = seq_len - orig_size
                batch_label[i] = np.append(batch_label[i], np.zeros(pad_size))
                for k in range(orig_size, seq_len):
                    mask[i][k] = 0
                #while len(batch_label[i]) < seq_len:
                #    index = len(batch_label[i])
                #    np.append(batch_label[i], [4])
                #    mask[i][index] = 0

        batch_label = torch.tensor(batch_label).long()
        #print('batch_label:', batch_label)
        print('mask:', mask.size(), mask)
        #mask = torch.ones((batch_size, seq_len)).byte()
        if self.gpu:
            mask = mask.to(self.device)
        if self.use_crf:
            total_loss = self.crf.neg_log_likelihood_loss(outs, mask, batch_label)
            scores, tag_seq = self.crf._viterbi_decode(outs, mask)
        else:
            loss_function = nn.NLLLoss(ignore_index=0, size_average=False)
            outs = outs.view(batch_size * seq_len, -1)
            score = F.log_softmax(outs, 1)
            total_loss = loss_function(score, batch_label.view(batch_size * seq_len))
            _, tag_seq = torch.max(score, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)
        if self.average_batch:
            total_loss = total_loss / batch_size
        #print('nll pred tag_seq:', tag_seq)
        return total_loss, tag_seq, mask

    def forward(self, word_inputs, char_inputs=None, char_seq_lengths=None): # feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask):
        outs = self.word_hidden(word_inputs, char_inputs=char_inputs, char_seq_lengths=char_seq_lengths) # feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        batch_size = outs.size(1)
        seq_len = outs.size(0)
        mask = torch.ones((batch_size, seq_len)).byte()
        if self.gpu:
            mask = mask.to(self.device)
        if self.use_crf:
            scores, tag_seq = self.crf._viterbi_decode(outs, mask)
        else:
            outs = outs.view(batch_size * seq_len, -1)
            _, tag_seq = torch.max(outs, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)
            # filter padded position with zero
            tag_seq = mask.long() * tag_seq
        return tag_seq

    # def get_lstm_features(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
    #     return self.word_hidden(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)

    def decode_nbest(self, word_inputs, nbest, char_inputs=None, char_seq_lengths=None): # feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask, nbest):
        if not self.use_crf:
            print("Nbest output is currently supported only for CRF! Exit...")
            exit(0)
        #outs = self.word_hidden(word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        outs = self.word_hidden(word_inputs, char_inputs=char_inputs, char_seq_lengths=char_seq_lengths)
        batch_size = outs.size(1)
        seq_len = outs.size(0)
        mask = torch.ones((batch_size, seq_len)).byte()
        if self.gpu:
            mask = mask.to(self.device)
        scores, tag_seq = self.crf._viterbi_decode_nbest(outs, mask, nbest)
        return scores, tag_seq

    def fit(self, X, Y, data, epochs=1):
        print("Training seqmodel...")
        #data.train_texts = X
        #data.train_Ids = ids
        data.show_data_summary()
        #save_data_name = data_model_dir +".dset"
        #data.save(save_data_name)
        #model = SeqModel(data)
        #loss_function = nn.NLLLoss()
        if data.optimizer.lower() == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=data.HP_lr, momentum=data.HP_momentum, weight_decay=data.HP_l2)
        elif data.optimizer.lower() == "adagrad":
            optimizer = optim.Adagrad(self.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
        elif data.optimizer.lower() == "adadelta":
            optimizer = optim.Adadelta(self.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
        elif data.optimizer.lower() == "rmsprop":
            optimizer = optim.RMSprop(self.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
        elif data.optimizer.lower() == "adam":
            optimizer = optim.Adam(self.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
        else:
            print('Optimizer illegal:', data.optimizer)
            exit(0)
        best_dev = -10
        #data_HP_iteration = 1
        # Start training
        for idx in range(epochs):
            epoch_start = time.time()
            temp_start = epoch_start
            print("Epoch: %s/%s" % (idx, epochs))
            if data.optimizer == "SGD":
                optimizer = lr_decay(optimizer, idx, data.HP_lr_decay, data.HP_lr)
            instance_count = 0
            #sample_id = 0
            sample_loss = 0
            total_loss = 0
            right_token = 0
            whole_token = 0
            #random.shuffle(data.train_Ids)
            #random.shuffle(data.train_texts)
            # Set model in train model
            self.train()
            self.zero_grad()
            # TODO: shuffle the training data
            batch_size = data.HP_batch_size
            batch_id = 0
            #train_num = len(data.train_Ids)
            train_num = len(X)
            total_batch = train_num//batch_size #+1
            for batch_id in range(total_batch):
                start = batch_id*batch_size
                end = (batch_id+1)*batch_size
                if end >train_num:
                    end = train_num
                #instance = data.train_Ids[start:end]
                #batch_word = []
                batch_word = X[start:end]
                batch_label = Y[start:end]

                # Map labels to index numbers (excludes 0)
                if idx == 0:
                    for row in batch_label:
                        for i in range(len(row)):
                            #print('map label:', row[i])
                            row[i] = data.label_alphabet.get_index(row[i])
                            print(row[i])
                print('epoch, batch:', idx, batch_id, 'batch_word:', batch_word, 'batch_label:', batch_label)
                #if not instance:
                #    continue
                #batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask = batchify_with_label(instance, data.HP_gpu)
                batch_char = None
                batch_charlen = None
                if data.use_char:
                    batch_char, batch_charlen = get_chars(batch_word, data)
                instance_count += 1
                #loss, tag_seq = self.neg_log_likelihood_loss(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_label, mask)
                loss, tag_seq, mask = self.neg_log_likelihood_loss(batch_word, batch_label, batch_char, batch_charlen)
                right, whole = predict_check(tag_seq, batch_label, mask)
                right_token += right
                whole_token += whole
                print('loss:', loss.item())
                sample_loss += loss.item()
                total_loss += loss.item()
                if end % 500 == 0:
                    temp_time = time.time()
                    temp_cost = temp_time - temp_start
                    temp_start = temp_time
                    print("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" % (end, temp_cost, sample_loss, right_token, whole_token, (right_token+0.)/whole_token))
                    if sample_loss > 1e8 or str(sample_loss) == "nan":
                        print("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
                        exit(0)
                    sys.stdout.flush()
                    sample_loss = 0
                loss.backward()
                optimizer.step()
                self.zero_grad()
            temp_time = time.time()
            temp_cost = temp_time - temp_start
            print("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" % (end, temp_cost, sample_loss, right_token, whole_token, (right_token+0.)/whole_token))

            epoch_finish = time.time()
            epoch_cost = epoch_finish - epoch_start
            print("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s" % (idx, epoch_cost, train_num/epoch_cost, total_loss))
            print("totalloss:", total_loss)
            if total_loss > 1e8 or str(total_loss) == "nan":
                print("WARNING: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
                #exit(0)
            # continue
            '''
            speed, acc, p, r, f, _, _ = self.evaluate(data, "dev")
            dev_finish = time.time()
            dev_cost = dev_finish - epoch_finish

            if data.seg:
                current_score = f
                print("Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (dev_cost, speed, acc, p, r, f))
            else:
                current_score = acc
                print("Dev: time: %.2fs speed: %.2fst/s; acc: %.4f" % (dev_cost, speed, acc))

            if current_score > best_dev:
                if data.seg:
                    print("Exceed previous best f score:", best_dev)
                else:
                    print("Exceed previous best acc score:", best_dev)
                model_name = data.model_dir +'.'+ str(idx) + ".model"
                print("Save current best model in file:", model_name)
                torch.save(self.state_dict(), model_name)
                best_dev = current_score
            '''
            # ## decode test
            '''
            speed, acc, p, r, f, _, _ = self.evaluate(data, "test")
            test_finish = time.time()
            test_cost = test_finish - dev_finish
            if data.seg:
                print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (test_cost, speed, acc, p, r, f))
            else:
                print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f" % (test_cost, speed, acc))
            '''
            gc.collect()

    def predict(self, X, data, nbest=1):
        #batch_size = data.HP_batch_size
        batch_size = 1
        total_batch = len(X)//batch_size+1
        pred_seq = []
        for batch_id in range(total_batch):
            start = batch_id*batch_size
            end = (batch_id+1)*batch_size
            if end > len(X):
                end = len(X)
            batch_word = X[start:end]
            if len(batch_word) == 0:
                tag_seq = []
            else:
                batch_char = None
                batch_charlen = None
                if data.use_char:
                    batch_char, batch_charlen = get_chars(batch_word, data)
                if nbest:
                    scores, nbest_tag_seq = self.decode_nbest(batch_word, nbest, batch_char, batch_charlen)
                    #nbest_pred_result = recover_nbest_label(nbest_tag_seq, mask, data.label_alphabet, batch_wordrecover)
                    #nbest_pred_results += nbest_pred_result
                    #pred_scores += scores[batch_wordrecover].cpu().data.numpy().tolist()
                    # Select the best sequence to evalurate
                    tag_seq = nbest_tag_seq[:, :, 0]
                else:
                    tag_seq = self(batch_word, batch_char, batch_charlen)
                tag_seq = tag_seq[0].tolist()
                for i in range(len(tag_seq)):
                    tag_seq[i] = data.label_alphabet.get_instance(tag_seq[i])
            print('pred tag seq:', tag_seq)
            pred_seq.append(tag_seq)
        return pred_seq

    ''' TODO - update this
    '''
    def evaluate(self, X, data, name, nbest=None):
        if name == "train":
            instances = data.train_Ids
        elif name == "dev":
            instances = data.dev_Ids
        elif name == 'test':
            instances = data.test_Ids
        elif name == 'raw':
            instances = data.raw_Ids
        else:
            print("Error: wrong evaluate name,", name)
        right_token = 0
        whole_token = 0
        nbest_pred_results = []
        pred_scores = []
        pred_results = []
        gold_results = []
        # Set model in eval model
        self.eval()
        batch_size = data.HP_batch_size
        start_time = time.time()
        train_num = len(instances)
        total_batch = train_num//batch_size+1
        for batch_id in range(total_batch):
            start = batch_id*batch_size
            end = (batch_id+1)*batch_size
            if end > train_num:
                end = train_num
            #instance = instances[start:end]
            #if not instance:
            #    continue
            batch_word = X[start:end]
            batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask  = batchify_with_label(instance, data.HP_gpu, True)
            if nbest:
                scores, nbest_tag_seq = self.decode_nbest(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask, nbest)
                nbest_pred_result = recover_nbest_label(nbest_tag_seq, mask, data.label_alphabet, batch_wordrecover)
                nbest_pred_results += nbest_pred_result
                pred_scores += scores[batch_wordrecover].cpu().data.numpy().tolist()
                # Select the best sequence to evalurate
                tag_seq = nbest_tag_seq[:, :, 0]
            else:
                tag_seq = self(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask)
            # print "tag:",tag_seq
            pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet, batch_wordrecover)
            pred_results += pred_label
            gold_results += gold_label
        decode_time = time.time() - start_time
        speed = len(instances)/decode_time
        acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagScheme)
        if nbest:
            return speed, acc, p, r, f, nbest_pred_results, pred_scores
        return speed, acc, p, r, f, pred_results, pred_scores


def get_chars(input_words, data):
    batch_size = len(input_words)
    # Pad_chars (batch_size, max_seq_len)
    chars = []
    max_seq_len = 0
    for row in input_words:
        chars_raw = []
        char_list = []
        print('row:', row)
        for word in row:
            print('word:', word)
            chs = list(word)
            chars_raw.append(chs)
            char_list_word = []
            for char in chs:
                char_list_word.append(data.char_alphabet.get_index(char))
            char_list.append(char_list_word)
        #chars_raw = list(' '.join(row)) # Get list of chars from the text string
        print('chars_raw:', chars_raw)
        if len(row) > max_seq_len:
            max_seq_len = len(row)
        # Map chars to vocab ids
        #char_list = []
        #for char in chars_raw:
        #    char_list.append(data.char_alphabet.get_index(char))
        chars.append(char_list)
    print('max_seq_len:', max_seq_len)

    # Original padding code
    pad_chars = []
    #for idx in range(len(chars)):
    #    pad_chars.append(chars[idx] + ([0] * (max_seq_len-len(chars[idx]))))
    pad_chars = [chars[idx] + [[0]] * (max_seq_len-len(chars[idx])) for idx in range(len(chars))]
    print('pad_chars:', len(pad_chars))

    length_list = []
    for pad_char in pad_chars:
        print('pad_char:', pad_char)
        mini_list = []
        for p in pad_char:
            mini_list.append(len(p))
        print('mini_list:', mini_list)
        length_list.append(mini_list)

    #length_list = list((len(p) for p in pad_chars))
    print('length list:', length_list)
    max_word_len = max(map(max, length_list))
    print('max_word_len:', max_word_len)
    char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_word_len)).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            # print len(word), wordlen
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

    char_seq_tensor = char_seq_tensor.view(batch_size*max_seq_len, -1)
    char_seq_lengths = char_seq_lengths.view(batch_size*max_seq_len,)
    print('char_seq_tensor:', char_seq_tensor.size())

    if data.HP_gpu:
        char_seq_tensor.to(data.device)
        char_seq_lengths.to(data.device)
    #char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    #char_seq_tensor = char_seq_tensor[char_perm_idx]

    return char_seq_tensor, char_seq_lengths


def batchify_with_label(input_batch_list, gpu, volatile_flag=False):
    """
        input: list of words, chars and labels, various length. [[words, chars, labels],[words,chars,labels],...]
            words: word ids for one sentence. (batch_size, sent_len)
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            word_seq_lengths: (batch_size,1) Tensor
            char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
            char_seq_lengths: (batch_size*max_sent_len,1) Tensor
            char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order
            label_seq_tensor: (batch_size, max_sent_len)
            mask: (batch_size, max_sent_len)
    """
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]

    features = [np.asarray(sent[1]) for sent in input_batch_list]
    feature_num = len(features[0][0])
    chars = [sent[2] for sent in input_batch_list]
    labels = [sent[3] for sent in input_batch_list]
    print('words[0]:', words[0])
    print('chars[0]:', chars[0])
    print('labels[0]:', labels[0])
    #word_seq_lengths = torch.LongTensor(map(len, words))
    word_seq_lengths = torch.LongTensor([len(w) for w in words])
    max_seq_len = word_seq_lengths.max().item()
    print('max_seq_len:', max_seq_len)
    word_seq_tensor = torch.zeros((batch_size, max_seq_len)).long()
    label_seq_tensor = torch.zeros((batch_size, max_seq_len)).long()
    feature_seq_tensors = []
    for idx in range(feature_num):
        feature_seq_tensors.append(torch.zeros((batch_size, max_seq_len)).long())
    mask = torch.zeros((batch_size, max_seq_len)).byte()
    for idx, (seq, label, seqlen) in enumerate(zip(words, labels, word_seq_lengths)):
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
        mask[idx, :seqlen] = torch.ones((seqlen))
        for idy in range(feature_num):
            feature_seq_tensors[idy][idx, :seqlen] = torch.LongTensor(features[idx][:, idy])
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    for idx in range(feature_num):
        feature_seq_tensors[idx] = feature_seq_tensors[idx][word_perm_idx]

    label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]
    # Deal with char
    # Pad_chars (batch_size, max_seq_len)
    pad_chars = []
    for idx in range(len(chars)):
        pad_chars.append(chars[idx] + ([0] * (max_seq_len-len(chars[idx]))))
    #pad_chars = [chars[idx] + [[0]] * (max_seq_len-len(chars[idx])) for idx in range(len(chars))]
    print('pad_chars:', pad_chars)
    length_list = []
    for pad_char in pad_chars:
        mini_list = []
        for p in pad_char:
            mini_list.append(len(p))
        length_list.append(mini_list)
    #length_list = [[(len(p) for p in pad_char)] for pad_char in pad_chars]
    print('length list:', length_list)
    max_word_len = max(map(max, length_list))
    print('max_word_len:', max_word_len)
    char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_word_len)).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            # print len(word), wordlen
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size*max_seq_len, -1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size*max_seq_len,)
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)
    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        for idx in range(feature_num):
            feature_seq_tensors[idx] = feature_seq_tensors[idx].cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_recover = char_seq_recover.cuda()
        mask = mask.cuda()
    return word_seq_tensor, feature_seq_tensors, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr/(1+decay_rate*epoch)
    print("Learning rate is set to:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def predict_check(pred_variable, gold_variable, mask_variable):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result, in numpy format
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    batch_size = pred_variable.size(0)
    pred = pred_variable.cpu().data.numpy()
    #gold = gold_variable.cpu().data.numpy()
    gold = np.array(gold_variable).astype('long')
    mask = mask_variable.cpu().data.numpy()
    print('predict_check mask:', mask)
    print('predict_check: pred:', pred, 'gold:', gold)
    overlaped = 0
    print('predict_check: batch_size:', batch_size)
    for x in range(batch_size):
        for k in range(gold[x].size):
            if gold[x][k] == pred[x][k]:
                overlaped += 1
    right_token = overlaped
    total_token = mask.sum()
    print("right: %s, total: %s" % (right_token, total_token))
    return right_token, total_token

def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """

    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = gold_variable.size(0)
    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        # print "g:", gold, gold_tag.tolist()
        assert(len(pred)==len(gold))
        pred_label.append(pred)
        gold_label.append(gold)
    return pred_label, gold_label


def recover_nbest_label(pred_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len, nbest): pred tag result
            mask_variable (batch_size, sent_len): mask variable
            word_recover (batch_size)
        output:
            nbest_pred_label list: [batch_size, nbest, each_seq_len]
    """
    # print "word recover:", word_recover.size()
    # exit(0)
    pred_variable = pred_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = pred_variable.size(0)
    seq_len = pred_variable.size(1)
    # print pred_variable.size()
    nbest = pred_variable.size(2)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    for idx in range(batch_size):
        pred = []
        for idz in range(nbest):
            each_pred = [label_alphabet.get_instance(pred_tag[idx][idy][idz]) for idy in range(seq_len) if mask[idx][idy] != 0]
            pred.append(each_pred)
        pred_label.append(pred)
    return pred_label
