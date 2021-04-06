#!/usr/bin/python3
# Evaluation functions
# All functions should be of the form function(y_true, y_pred)

from sklearn import metrics

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
