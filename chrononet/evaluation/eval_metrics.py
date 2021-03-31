#!/usr/bin/python3
# Evaluation functions
# All functions should be of the form function(y_true, y_pred)

import torch
from allennlp.modules.elmo import Elmo, batch_to_ids
from ast import literal_eval
from sklearn import metrics

from data_tools import data_util

tdevice = 'cpu'
use_cuda = torch.cuda.is_available()
if use_cuda:
    tdevice = torch.device('cuda')

def class_report(y_true, y_pred, labels=None):
    sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
    return metrics.flat_classification_report(y_true, y_pred, labels=sorted_labels, digits=3)

def precision(y_true, y_pred):
    return metrics.precision_score(y_true, y_pred, average='weighted')

def recall(y_true, y_pred):
    return metrics.recall_score(y_true, y_pred, average='weighted')

def f1(y_true, y_pred, labels=None):
    return metrics.f1_score(y_true, y_pred, average='weighted', labels=labels)

def csmfa(y_true, y_pred):
    labels_correct = {}
    labels_pred = {}
    correct = []
    predicted = []
    for index in range(len(y_true)):
        pred = str(y_pred[index])
        cor = str(y_true[index])
        predicted.append(pred)
        correct.append(cor)
        #print('pred:', pred, 'cor:', cor)

        if cor in labels_correct:
            labels_correct[cor] = labels_correct[cor] + 1
        else:
            labels_correct[cor] = 1

        if pred in labels_pred:
            labels_pred[pred] = labels_pred[pred] + 1
        else:
            labels_pred[pred] = 1
    # Calculate CSMF accuracy
    n = len(correct)
    num_classes = len(labels_correct.keys())
    print('n:', str(n))
    print('num_classes:', str(num_classes))
    csmf_pred = {}
    csmf_corr = {}
    csmf_corr_min = 1
    csmf_sum = 0
    for key in labels_correct.keys():
        if key not in labels_pred:
            labels_pred[key] = 0
        num_corr = labels_correct[key]
        num_pred = labels_pred[key]
        csmf_c = num_corr/float(n)
        csmf_p = num_pred/float(n)
        csmf_corr[key] = csmf_c
        csmf_pred[key] = csmf_p
        print("csmf for " + key + " corr: " + str(csmf_c) + ", pred: " + str(csmf_p))
        if csmf_c < csmf_corr_min:
            csmf_corr_min = csmf_c
        csmf_sum = csmf_sum + abs(csmf_c - csmf_p)

    csmf_accuracy = 1 - (csmf_sum / (2 * (1 - csmf_corr_min)))
    return csmf_accuracy

def keyword_attr_similarity(df, elmo):
    print('keyword_attr_similarity')
    dim = 1024
    kw_sim_map = {} # for per-class sim
    kw_baseline_map = {}
    kw_sims = []
    kw_sims_baseline = []
    for i, row in df.iterrows():
        record_sims = []
        record_baselines = []
        print(row['docid'])
        text = row['feats_elmo_words']
        if type(text) is str:
            text = literal_eval(text)
        seq_len = len(text)
        # Per-class sim
        cod = row['diagnosis']
        if cod not in kw_sim_map.keys():
            kw_sim_map[cod] = []
            kw_baseline_map[cod] = []
        print('seq len:', seq_len, type(text), text)
        #print('original attrs:', row['attrs'].size())
        attrs = row['attrs'].view(-1, dim)
        # Normalize the attributions
        attrs = torch.exp(attrs)
        attrs -= attrs.min(1, keepdim=True)[0]
        attrs/= attrs.max(1, keepdim=True)[0]
        print('attrs scaled:', attrs)

        keywords = str(row['keywords'])
        print('keywords:', type(keywords), keywords)

        # Encode text
        text_indices = batch_to_ids([text]).to(tdevice)
        text_tensor = elmo(text_indices).view(-1, dim)
        print('text size:', text_tensor.size(), 'attrs size:', attrs.size())
        avg_narr = torch.mean(text_tensor, dim=0).view(dim)

        # Compute attr weight
        weighted_narr = attrs*text_tensor
        attr_avg_narr = torch.mean(weighted_narr, dim=0).view(dim)

        stop_words = ['and', 'or', 'of', 'in', 'the', 'a', 'on', 'at', 'had', 'has', 'due', 'to', 'was']

        if not keywords == 'nan':
            for kw in keywords.split(','):
                #print('kw:', kw)
                # Encode keywords
                kw_words = data_util.split_words(kw)
                kw_words = [w for w in kw_words if w not in stop_words] # Filter stopwords
                kw_indices = batch_to_ids([kw_words]).to(tdevice)
                kw_tensor = elmo(kw_indices).view(-1, dim)
                kw_avg = torch.mean(kw_tensor, dim=0).view(dim)

                # Compute cosine sim of kw_avg and avg_narr
                kw_sim_baseline = torch.nn.functional.cosine_similarity(kw_avg, avg_narr, dim=0)#.item()
                kw_sim = torch.nn.functional.cosine_similarity(kw_avg, attr_avg_narr, dim=0)#.item()
                print('kw sim:', kw_sim.item(), kw)
                print('kw sim baseline:', kw_sim_baseline.item(), kw)
                if not torch.isnan(kw_sim_baseline) and not torch.isnan(kw_sim):
                    record_sims.append(kw_sim.item())
                    record_baselines.append(kw_sim_baseline.item())

            # Average over individual records?
            if len(record_sims) > 0:
                rec_baselines = torch.tensor(record_baselines)
                rec_attr_sims = torch.tensor(record_sims)
                rec_avg_baseline = torch.mean(rec_baselines).item()
                rec_avg_attr = torch.mean(rec_attr_sims).item()
                print('kw sim baseline rec:', rec_avg_baseline)
                print('kw sim attr rec:', rec_avg_attr)
                kw_sims_baseline.append(rec_avg_baseline)
                kw_sims.append(rec_avg_attr)
                kw_baseline_map[cod].append(rec_avg_baseline)
                kw_sim_map[cod].append(rec_avg_attr)

    # Print avg sim for baseline and weighted
    print('kw sim baseline list:', kw_sims_baseline)
    print('kw sim attr list:', kw_sims)
    baselines = torch.tensor(kw_sims_baseline)
    attr_sims = torch.tensor(kw_sims)
    avg_baseline = torch.mean(baselines).item()
    avg_attr = torch.mean(attr_sims).item()
    std_baseline = torch.std(baselines).item()
    std_attr = torch.std(attr_sims).item()
    print('kw sim baseline avg:', avg_baseline, 'std:', std_baseline)
    print('kw sim attr avg:', avg_attr, 'std_avg:', std_attr)

    # Per class avg
    for cod_class in kw_sim_map.keys():
        print(cod_class, len(kw_sim_map[cod_class]))
        baselines = torch.tensor(kw_baseline_map[cod_class])
        attr_sims = torch.tensor(kw_sim_map[cod_class])
        avg_baseline = torch.mean(baselines).item()
        avg_attr = torch.mean(attr_sims).item()
        std_baseline = torch.std(baselines).item()
        std_attr = torch.std(attr_sims).item()
        print('class', cod_class, 'kw sim baseline avg:', avg_baseline, 'std:', std_baseline)
        print('class', cod_class, 'kw sim attr avg:', avg_attr, 'std_avg:', std_attr)
