#!/usr/bin/python3
# Temporal ordering evaluation functions
# All functions should be of the form function(y_true, y_pred)

import ast
import math
import numpy
import scipy

from evaluation import eval_util
from data_tools import data_util
from data_tools import temporal_util as tutil

debug = True

# Metric functions #########################

def kendalls_tau(true_ranks, pred_ranks, avg=True):
    accuracies = []
    for n in range(len(true_ranks)):
        pr = numpy.asarray(pred_ranks[n])
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
            tau, pval = scipy.stats.kendalltau(tr, pr, nan_policy='raise')
            print('Kendalls tau: n=', n, 'tau:', tau, 'p-value:', pval)
        if not numpy.isnan(tau):
            accuracies.append(tau)
        else:
            print('WARNING: Tau score dropped because it was NaN')
    if avg:
        if len(accuracies) > 1:
            acc = numpy.average(numpy.asarray(accuracies))
        elif len(accuracies) == 0:
            acc = 0.0
        else:
            acc = accuracies[0]
        return acc
    else:
        return accuracies


''' Calculate the mean ABSOLUTE error of the predicted ranks
    Scale the ranks to 0-1
'''
def rank_mae(true_ranks, pred_ranks):
    print('rank_mse: true:', len(true_ranks), 'pred:', len(pred_ranks))
    if len(true_ranks) != len(pred_ranks):
        print('ERROR: length mismatch of true and pred ranks')
    assert(len(true_ranks) == len(pred_ranks))
    mae_scores = []

    # Decide whether or not to scale the ranks
    scale_pred = False
    scale_true = False
    pred_vals = [item for sublist in pred_ranks for item in sublist]
    true_vals = [item for sublist in true_ranks for item in sublist]
    if max(pred_vals) > 1:
        scale_pred = True
    if max(true_vals) > 1:
        scale_true = True

    print('mae scaling: true_ranks:', scale_true, 'pred_ranks:', scale_pred)

    for n in range(len(true_ranks)):
        print('entry types: true:', type(true_ranks[n]), len(true_ranks[n]), 'pred:', type(pred_ranks[n]), len(pred_ranks[n]))
        # Scale the ranks if needed
        if scale_true:
            true_n = scale_ranks(true_ranks[n])
        else:
            true_n = true_ranks[n]
        if scale_pred:
            pred_n = scale_ranks(pred_ranks[n])
        else:
            pred_n = pred_ranks[n]
        num_samples = len(true_ranks[n])
        assert(num_samples == len(pred_ranks[n]))
        error_sum = 0
        for x in range(num_samples):
            error_sum += abs(true_n[x] - pred_n[x])
        mae_scores.append(error_sum/float(num_samples))
    return numpy.average(numpy.asarray(mae_scores))


''' Calculate the mean squared error of the predicted ranks
    Scale the ranks to 0-1
'''
def rank_mse(true_ranks, pred_ranks):
    print('rank_mse: true:', len(true_ranks), 'pred:', len(pred_ranks))
    if len(true_ranks) != len(pred_ranks):
        print('ERROR: length mismatch of true and pred ranks')
    assert(len(true_ranks) == len(pred_ranks))
    mse_scores = []

    # Decide whether or not to scale the ranks
    scale_pred = False
    scale_true = False
    pred_vals = [item for sublist in pred_ranks for item in sublist]
    true_vals = [item for sublist in true_ranks for item in sublist]
    if max(pred_vals) > 1:
        scale_pred = True
    if max(true_vals) > 1:
        scale_true = True

    print('mse scaling: true_ranks:', scale_true, 'pred_ranks:', scale_pred)

    for n in range(len(true_ranks)):
        print('entry types: true:', type(true_ranks[n]), len(true_ranks[n]), 'pred:', type(pred_ranks[n]), len(pred_ranks[n]))
        # Scale the ranks if needed
        if scale_true:
            true_n = scale_ranks(true_ranks[n])
        else:
            true_n = true_ranks[n]
        if scale_pred:
            pred_n = scale_ranks(pred_ranks[n])
        else:
            pred_n = pred_ranks[n]
        num_samples = len(true_ranks[n])
        assert(num_samples == len(pred_ranks[n]))
        error_sum = 0
        for x in range(num_samples):
            error_sum += (true_n[x] - pred_n[x]) ** 2
        mse_scores.append(error_sum/float(num_samples))
    return numpy.average(numpy.asarray(mse_scores))


''' Calculate the pairwise accuracy of a listwise ranking
    Currently this is a macro average (every document has equal weight)
'''
def rank_pairwise_accuracy(true_ranks, pred_ranks, eps=0.00001, avg=True):
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
    if avg:
        if len(accuracies) > 1:
            acc = numpy.average(numpy.asarray(accuracies))
        else:
            acc = accuracies[0]
        return acc
    else:
        return accuracies

def epr(true_ranks, pred_ranks):
    return (events_per_rank(true_ranks), events_per_rank(pred_ranks))

def gpr(y_true, y_pred, ref_df):
    # Load gold pairs
    print("Extracting pair relations...")
    rec_ids, true_pairs, true_relations = eval_util.extract_relation_pairs(ref_df)
    events = []
    for i, row in ref_df.iterrows():
        event_elem = data_util.load_xml_tags(row['events'])
        event_list = []
        for child in event_elem:
            event_list.append(child)
        print('loaded events:', len(event_list))
        events.append(event_list)
    pred_pairs, pred_labels = eval_util.pair_relations(events, y_pred)

    gpr = score_relation_pairs(pred_pairs, pred_labels, true_pairs, true_relations)
    return gpr

# Utility functions ########################

def events_per_rank(labels, thresh=0.0):
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


def scale_ranks(rank_list):
    max_rank = max(rank_list)
    if max_rank == 0:
        return rank_list
    new_ranks = []
    for rank in rank_list:
        new_ranks.append(float(rank)/float(max_rank))
    return new_ranks


''' Score relations pairs against gold standard relation pairs
'''
def score_relation_pairs(pred_pairs, pred_labels, true_pairs, true_labels):
    doc_recalls = []
    doc_true_pairs = []
    doc_class_recalls = {}
    doc_class_recalls['BEFORE'] = []
    doc_class_recalls['AFTER'] = []
    doc_class_recalls['OVERLAP'] = []
    doc_class_totals = {}
    doc_class_totals['BEFORE'] = 0
    doc_class_totals['AFTER'] = 0
    doc_class_totals['OVERLAP'] = 0
    if debug: print('score_relation_pairs:', str(len(pred_pairs)), str(len(pred_labels)), str(len(true_pairs)), str(len(true_labels)))
    #assert(len(pred_labels) == len(true_labels))
    #assert(len(pred_pairs) == len(true_pairs))
    for x in range(len(pred_labels)):
        total = 0
        found = 0
        class_totals = {}
        class_totals['BEFORE'] = 0
        class_totals['AFTER'] = 0
        class_totals['OVERLAP'] = 0
        class_founds = {}
        class_founds['BEFORE'] = 0
        class_founds['AFTER'] = 0
        class_founds['OVERLAP'] = 0
        #print('tpairs:', str(len(true_pairs[x])))
        for y in range(len(true_pairs[x])):
            tpair = true_pairs[x][y]
            tlabel = tutil.map_rel_type(true_labels[x][y], 'simple')
            if debug: print('- tpair:', eval_util.str_pair(tpair), 'tlabel:', str(tlabel))
            total += 1
            class_totals[tlabel] += 1
            #print('pred_pair[0]:', str(pred_pairs[x][0][0]), str(pred_pairs[x][0][1]))
            for z in range(len(pred_pairs[x])):
                ppair = pred_pairs[x][z]
                if tutil.are_pairs_equal(tpair, ppair):
                    plabel = pred_labels[x][z]
                    if debug: print("-- checking pair:", eval_util.str_pair(ppair), str(plabel))
                    if tlabel == plabel:
                        found += 1
                        class_founds[tlabel] += 1
                        if debug: print('--- correct')
                    # Count before and before/overlap as the same since we're ranking on start time
                    elif tlabel == 'BEFORE/OVERLAP' and plabel == 'BEFORE':
                        if debug: print('--- correct (before/overlap)')
                        found += 1
                        class_founds[plabel] += 1
        if total == 0:
            print('WARNING: no reference relations found!')
            doc_recall = 0
        else:
            doc_recall = found/total
            for key in class_totals.keys():
                if class_totals[key] == 0:
                    val = 0.0
                else:
                    val = float(class_founds[key]) / class_totals[key]
                doc_class_recalls[key].append(val)
                doc_class_totals[key] += class_totals[key]
            doc_recalls.append(doc_recall)
            doc_true_pairs.append(total)

    # Calculate the weighted average recall
    avg_recall = numpy.average(doc_recalls, weights=doc_true_pairs)
    for key in doc_class_recalls.keys():
        avg_class_recall = numpy.average(numpy.asarray(doc_class_recalls[key]))
        print('GPR Recall', key, str(avg_class_recall), 'num=', str(doc_class_totals[key]))

    return avg_recall
