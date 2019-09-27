#!/usr/bin/python3

ORIG = ['docid', 'text', 'tags', 'events', 'event_ranks' 'diagnosis', 'dct']
SEQ = ['docid', 'seqid', 'text', 'text_orig', 'seq', 'seq_labels']
ORDER = ['docid', 'text', 'text_orig', 'tags', 'seq', 'seq_labels', 'events', 'event_ranks', 'diagnosis', 'dct']
RELATION = ['docid', 'relid' 'entity1', 'entity2', 'rel_type']

def orig():
    return ORIG

def seq():
    return SEQ

def order():
    return ORDER

def relation():
    return RELATION
