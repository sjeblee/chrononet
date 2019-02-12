#!/usr/bin/python3

ORIG = ['docid', 'text', 'tags', 'event_ranks' 'diagnosis']
SEQ = ['docid', 'seqid', 'text', 'seq', 'seq_labels']
ORDER = ['docid', 'text', 'events', 'event_ranks', 'diagnosis']
RELATION = ['docid', 'relid' 'entity1', 'entity2', 'rel_type']

def orig():
    return ORIG

def seq():
    return SEQ

def order():
    return ORDER

def relation():
    return RELATION
