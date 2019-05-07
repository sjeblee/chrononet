#!/usr/bin/python3
# Evaluation functions
# All functions should be of the form function(y_true, y_pred)

import math
import numpy
import time

from lxml import etree

from data_tools import data_util
from data_tools import temporal_util as tutil

debug = True
none_label = 'NONE'


''' Extract relation pairs
'''
def extract_relation_pairs(df, relation_set='exact', pairtype='ee', train=False, nodename='narr_timeml_simple'):
    if debug: print("extracting relation pairs: ", relation_set, pairtype, "train: ", str(train))
    all_labels = []
    all_pairs = []
    ids = []

    for i, row in df.iterrows():
        rec_id = row['docid']
        ids.append(rec_id)
        tags = data_util.load_xml_tags(row['tags'])
        pairs, pair_labels = extract_pairs(tags, relation_set, pairtype, train)
        all_pairs.append(pairs)
        all_labels.append(pair_labels)

    return ids, all_pairs, all_labels


''' Extract time-event or event-event pairs from xml data
'''
def extract_pairs(xml_node, relation_set='exact', pairtype='ee', under=1):
    pairs = []
    labels = []
    events = xml_node.xpath("EVENT")
    times = xml_node.xpath("TIMEX3")
    tlinks = xml_node.xpath("TLINK")
    if debug: print("events: ", str(len(events)), " times: ", str(len(times)), " tlinks: ", str(len(tlinks)))
    for event in events:
        if 'eid' not in event.attrib:
            print("no eid: ", etree.tostring(event))
        event_id = event.attrib['eid']

        # Get the position of the event
        event_position = xml_node.index(event)
        event.attrib['position'] = str(event_position)

        # Make a pair out of this event and all time phrases
        if pairtype == 'et' or pairtype == 'te':
            for timex in times:
                time_id = timex.attrib['tid']
                # Get the position of the time
                if 'position' not in timex.attrib.keys():
                    time_position = xml_node.index(timex)
                    timex.attrib['position'] = str(time_position)
                pairs.append((event, time))
                labels.append(tutil.map_rel_type(none_label, relation_set))
        # Make pairs of events
        elif pairtype == 'ee':
            for event2 in events:
                if event2.attrib['eid'] != event.attrib['eid']:
                    pairs.append((event, event2))
                    labels.append(tutil.map_rel_type(none_label, relation_set))

    #print("pairs:", str(len(pairs)))
    # Find actual relations
    for tlink in tlinks:
        if pairtype == 'ee':
            if 'relatedToEventID' in tlink.attrib and 'eventID' in tlink.attrib:
                event_id = tlink.attrib['eventID']
                event2_id = tlink.attrib['relatedToEventID']
                rel_type = tlink.attrib['relType']
                mapped_type = tutil.map_rel_type(rel_type, relation_set)
                for x in range(len(labels)):
                    pair = pairs[x]
                    if pair[0].attrib['eid'] == event_id and pair[1].attrib['eid'] == event2_id:
                        labels[x] = mapped_type
        else:
            if 'relatedToTime' in tlink.attrib and 'eventID' in tlink.attrib:
                event_id = tlink.attrib['eventID']
                time_id = tlink.attrib['relatedToTime']
                rel_type = tlink.attrib['relType']
                mapped_type = tutil.map_rel_type(rel_type, relation_set)
                for x in range(len(labels)):
                    pair = pairs[x]
                    if pair[0].attrib['eid'] == event_id and pair[1].attrib['tid'] == time_id:
                        labels[x] = mapped_type
    # Undersample the NONE examples for training or all
    do_undersample = True
    if do_undersample:
        #if debug: print("undersampling NONE class: ", str(under))
        index = 0
        while index < len(labels):
            if labels[index] == none_label:
                #r = 0 # Temp for no NONE class
                #r = numpy.random.random() # undersampling probability
                #if r < under:
                del labels[index]
                del pairs[index]
                index = index-1
            index = index+1
        if debug: print("pairs after undersample:", str(len(pairs)))

    return pairs, labels


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
