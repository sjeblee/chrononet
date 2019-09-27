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
