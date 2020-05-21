#!/usr/bin/python3

import ast
import pandas

from lxml import etree

from data_tools import data_util

def extract_entities(df, target_df):
    target_df = df.copy()
    target_df['entities'] = ''
    target_df['relations'] = ''
    for i, row in df.iterrows():
        tag_string = row['tags']
        node = etree.fromstring(tag_string)
        rel_node = etree.Element('relations')
        # Save the relations in their own column
        for child in node:
            if child.tag == 'TLINK':
                rel_node.append(child)
        # TODO: extract entities from xml string
    return target_df

def extract_relations(df, target_df):
    if 'entities' not in df.keys():
        df = extract_entities(df, df.copy())

    target_df = pandas.DataFrame(columns=['docid', 'relid', 'relation', 'text1', 'text2'])
    # TODO: get relation objects from the tags column, create them as rows

    return target_df

def extract_time_pairs(df, feat_name='feats', verilogue=False):
    #verilogue = True
    print('extract_time_pairs')
    df[feat_name] = ''
    df['time_order'] = ''
    for i, row in df.iterrows():
        #if debug: print('event_vectors', str(i))
        ranks = row['event_ranks']
        if type(ranks) is str:
            ranks = ast.literal_eval(ranks)
        time_phrases = []
        time_ranks = []
        # Create timeid map
        timex_map = {}
        signal_map = {}
        #text = row['text']
        #print('narr text:', text)

        #if type(row['events']) is list:
        if verilogue:
            print('found verilogue data')
            verilogue = True
            events = row['events']
            tags = row['tags']
            print('tags:', tags)
            # TIMEX map
            for tag in tags:
                if tag.tag.upper() == 'TIMEX3':
                    tid = str(tag.features['annotation_id'])
                    timex_map[tid] = tag
                    print('timex_map added:', tid, tag.features['rawText'])
        else:
            events = etree.fromstring(row['events'])
            tags = data_util.load_xml_tags(row['tags'], decode=False)

            # TIMEX map
            for timex in tags.findall('TIMEX3'):
                print(etree.tostring(timex).decode('utf8'), timex.attrib)
                tid = str(timex.get('id'))
                if tid is None:
                    tid = str(timex.attrib['tid'])
                timex_map[tid] = timex
                print('timex_map', tid)

            # SIGNAL map
            for sig in tags.findall('SIGNAL'):
                sid = sig.get('id')
                if sid is None:
                    sid = sig.get('sid')
                signal_map[sid.lower()] = sig
                print('sid:', sid.lower())

        for event_num in range(len(events)):
            event = events[event_num]
            rank = ranks[event_num]
            # Get spans and attributes
            #span = event.get('span').split(',')

            # Get time phrase if there is one
            if verilogue:
                time_id_string = None
                if 'relatedToTime' in event.features:
                    time_id_string = event.features['relatedToTime']
            else:
                time_id_string = event.get('relatedToTime')
            time_words = None
            if time_id_string is not None:
                time_id = str(time_id_string.split(',')[0]) # Just get the first time phrase
                if time_id not in timex_map:
                    print('WARNING: time_id not found:', time_id)
                else:
                    timex = timex_map[time_id]
                    if verilogue:
                        time_text = timex.features['rawText']
                    else:
                        time_text = timex.text
                    time_type = 'UNK'
                    if 'type' in timex.attrib:
                        time_type = timex.attrib['type']
                    time_words = data_util.split_words(time_text)

                    # Get SIGNAL text
                    signal_id = event.get('signalID')
                    if signal_id is not None:
                        signal = signal_map[signal_id.lower()]
                        signal_text = signal.text
                        signal_words = data_util.split_words(signal_text)
                        #sflags = []
                        #for sw in signal_words:
                        #    sflags.append(0)
                        time_words = signal_words + time_words # Add signal words to time phrase

                    # Add the time type
                    time_words = [time_type] + time_words # Append the time type to the beginning of the list
                    time_phrases.append(time_words)
                    time_ranks.append(rank)
                    print('found time phrase:', time_words, rank)
        time_pairs = []
        time_pair_orders = []
        for tp in range(len(time_phrases)):
            for tp2 in range(len(time_phrases)):
                if not tp == tp2:
                    time1 = time_phrases[tp]
                    rank1 = int(time_ranks[tp])
                    time2 = time_phrases[tp2]
                    rank2 = int(time_ranks[tp2])
                    print('time pair:', time1, rank1, time2, rank2)
                    if rank1 is not None and rank2 is not None:
                        time_pairs.append((time1, time2))
                        label = 'SIMULTANEOUS'
                        if rank1 < rank2:
                            label = 'BEFORE'
                        elif rank1 > rank2:
                            label = 'AFTER'
                        time_pair_orders.append(label)

        df.at[i, feat_name] = time_pairs
        df.at[i, 'time_order'] = time_pair_orders
    return df

def extract_time_pairs_from_file(filename, feat_name='feats'):
    df[feat_name] = ''
    df['time_order'] = ''
    for i, row in df.iterrows():
        #if debug: print('event_vectors', str(i))
        events = etree.fromstring(row['events'])
        text = row['text']
        print('narr text:', text)
        tags = data_util.load_xml_tags(row['tags'], decode=False)
        ranks = row['event_ranks']
        time_phrases = []
        time_ranks = []
        # Create timeid map
        timex_map = {}
        # TIMEX map
        for timex in tags.findall('TIMEX3'):
            tid = timex.get('id')
            if tid is None:
                tid = timex.get('tid')
            timex_map[tid] = timex

        for event_num in range(len(events)):
            event = events[event_num]
            rank = ranks[event_num]
            # Get spans and attributes
            #span = event.get('span').split(',')

            # Get time phrase if there is one
            time_id_string = event.get('relatedToTime')
            time_words = None
            if time_id_string is not None:
                time_id = time_id_string.split(',')[0] # Just get the first time phrase
                if time_id not in timex_map:
                    print('WARNING: time_id not found:', time_id)
                else:
                    timex = timex_map[time_id]
                    time_text = timex.text
                    time_words = data_util.split_words(time_text)
                    time_phrases.append(time_words)
                    time_ranks.append(rank)
        time_pairs = []
        time_pair_orders = []
        for tp in range(len(time_phrases)):
            for tp2 in range(len(time_phrases)):
                if not tp == tp2:
                    time1 = time_phrases[tp]
                    rank1 = time_ranks[tp]
                    time2 = time_phrases[tp2]
                    rank2 = time_ranks[tp2]
                    time_pairs.append((time1, time2))
                    label = 'SIMULTANEOUS'
                    if rank1 < rank2:
                        label = 'BEFORE'
                    elif rank1 > rank2:
                        label = 'AFTER'
                    time_pair_orders.append(label)

        df.at[i, feat_name] = time_pairs
        df.at[i, 'time_order'] = time_pair_orders
    return df
