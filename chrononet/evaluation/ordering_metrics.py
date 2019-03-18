#!/usr/bin/python3
# Temporal ordering evaluation functions
# All functions should be of the form function(y_true, y_pred)

import math
import numpy
import scipy

# Metric functions #########################

def kendalls_tau(true_ranks, pred_ranks):
    accuracies = []
    for n in range(len(true_ranks)):
        pr = pred_ranks[n]
        tr = true_ranks[n]
        print('tau: true:', tr)
        print('tau: pred:', pr)
        assert(len(pr) == len(tr))
        pval = None
        if len(tr) == 0:
            tau = 0
        elif len(tr) == 1:
            tau = 1
        else:
            tau, pval = scipy.stats.kendalltau(tr, pr)
            print('Kendalls tau: n=', n, 'tau:', tau, 'p-value:', pval)
        accuracies.append(tau)
    if len(accuracies) > 1:
        acc = numpy.average(numpy.asarray(accuracies))
    else:
        acc = accuracies[0]
    return acc


''' Calculate the mean squared error of the predicted ranks
'''
def rank_mse(true_ranks, pred_ranks):
    print('rank_mse: true:', len(true_ranks), 'pred:', len(pred_ranks))
    if len(true_ranks) != len(pred_ranks):
        print('ERROR: length mismatch of true and pred ranks')
    assert(len(true_ranks) == len(pred_ranks))
    mse_scores = []
    for n in range(len(true_ranks)):
        print('entry types: true:', type(true_ranks[n]), len(true_ranks[n]), 'pred:', type(pred_ranks[n]), len(pred_ranks[n]))
        num_samples = len(true_ranks[n])
        assert(num_samples == len(pred_ranks[n]))
        error_sum = 0
        for x in range(num_samples):
            error_sum += (true_ranks[n][x] - pred_ranks[n][x]) ** 2
        mse_scores.append(error_sum/float(num_samples))
    return numpy.average(numpy.asarray(mse_scores))


''' Calculate the pairwise accuracy of a listwise ranking
    Currently this is a macro average (every document has equal weight)
'''
def rank_pairwise_accuracy(true_ranks, pred_ranks, eps=0.001):
    accuracies = []
    for n in range(len(true_ranks)):
        pr = pred_ranks[n]
        se, so = get_ordered_pairs(true_ranks[n])
        num_pairs = len(so) + len(se)
        so_correct = 0
        se_correct = 0
        if num_pairs == 0:
            accuracy = 0
            print('WARNING: no ranks for evaluation')
        else:
            for pair in so:
                if pr[pair[0]] < pr[pair[1]]:
                    so_correct += 1
            for pair in se:
                if math.fabs(pr[pair[0]] - pr[pair[1]]) <= eps:
                    se_correct += 1
            accuracy = (so_correct + se_correct)/float(num_pairs)
        accuracies.append(accuracy)
    if len(accuracies) > 1:
        acc = numpy.average(numpy.asarray(accuracies))
    else:
        acc = accuracies[0]
    return acc

def epr(true_ranks, pred_ranks):
    return (events_per_rank(true_ranks), events_per_rank(pred_ranks))

# Utility functions ########################

def events_per_rank(labels):
    epr = []
    for ranks in labels:
        rank_to_num = {}
        num = 0
        for val in ranks:
            if val not in rank_to_num:
                rank_to_num[val] = 0
            rank_to_num[val] += 1
            num += 1
        for key in rank_to_num.keys():
            epr.append(rank_to_num[key])

    avg_epr = numpy.average(numpy.asarray(epr))
    return avg_epr


''' From the ranks, generate pairs of events with equal rank, and ordered ranks
'''
def get_ordered_pairs(ranks):
    num = len(ranks)
    equal_pairs = []
    ordered_pairs = []
    for x in range(num):
        first = ranks[x]
        for y in range(num):
            if x != y:
                second = ranks[y]
                if first == second:
                    equal_pairs.append((x, y))
                elif first < second:
                    ordered_pairs.append((x, y))
    return equal_pairs, ordered_pairs


''' Generate all pair relations for ranked events
'''
def pair_relations(events, ranks, eps=0.0):
    pairs = []
    relations = []
    for n in range(len(events)):
        event_list = events[n]
        rank_list = ranks[n]
        doc_pairs = []
        doc_labels = []
        for x in range(len(event_list)):
            for y in range(len(event_list)):
                if x != y:
                    event1 = event_list[x]
                    event2 = event_list[y]
                    rank1 = float(rank_list[x])
                    rank2 = float(rank_list[y])
                    rel_type = 'OVERLAP'
                    if math.fabs(rank1-rank2) <= eps:
                        rel_type = 'OVERLAP'
                    elif rank1 < rank2:
                        rel_type = 'BEFORE'
                    elif rank1 > rank2:
                        rel_type = 'AFTER'
                    #print("rank pair", str(rank1), str(rank2), rel_type)
                    doc_pairs.append((event1, event2))
                    doc_labels.append(rel_type)
        pairs.append(doc_pairs)
        relations.append(doc_labels)
    return pairs, relations

def str_pair(event_pair):
    return event_pair[0].attrib['eid'] + ' ' + event_pair[0].text + ' ' + event_pair[1].attrib['eid'] + ' ' + event_pair[1].text
