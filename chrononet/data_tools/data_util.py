#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Util functions
# @author sjeblee@cs.toronto.edu

from lxml import etree
from lxml.etree import tostring
from itertools import chain
from nltk.tokenize import wordpunct_tokenize
from random import shuffle
from sklearn import metrics
import numpy
import operator
import pandas
import re
import subprocess
import torch

debug = False

#def clean_file(filename):
    # remove blank lines | remove extra spaces| remove leading and trailing spaces  | fix utf-8 chars
    #command = r"sed '/^\s*$/d' $file | sed -e 's/  */ /g' | sed -e 's/^ //g' | sed -e 's/ $//g' | sed -e 's/&amp;/and/g' | sed -e 's/&#13;/ /g' | sed -e 's/&#8217;/\'/g' | sed -e 's/&#8221;/\"/g' | sed -e 's/&#8220;/\"/g' | sed -e 's/&#65533;//g' | sed -e 's/&#175\7;//g'| sed -e 's/&#1770;/\'/g'"
    # TODO

def add_labels(df, labels, labelname):
    print('add_labels:', len(labels))
    df[labelname] = ''
    for i, row in df.iterrows():
        #print('add_labels i=', i)
        if i < len(labels):
            #print('add_labels labelname:', labelname, labels[i])
            df.at[i, labelname] = labels[i]
        else:
            print('WARNING, add_labels i out of range:', i)

    return df

def add_time_ids(event_elem, tag_elem):
    time_map = {}
    for tlink in tag_elem.findall('TLINK'):
        eventid = None
        timeid = None
        if 'eventID' in tlink.attrib and 'relatedToTime' in tlink.attrib:
            eventid = tlink.get('eventID')
            timeid = tlink.get('relatedToTime')
        elif 'timeID' in tlink.attrib and 'relatedToEventID' in tlink.attrib:
            eventid = tlink.get('relatedToEventID')
            timeid = tlink.get('timeID')
        if timeid is not None and eventid is not None:
            if eventid not in time_map:
                time_map[eventid] = []
            time_map[eventid].append(timeid)

    for event in event_elem:
        eid = event.get('eid')
        time_ids = time_map[eid]
        if time_ids is not None:
            tid_string = ','.join(time_ids)
            event.set('relatedToTime', tid_string)
    return event_elem


''' (In progress)
'''
def add_thyme_labels(filename, outfile):
    brain = []
    colon = []
    xmltree = etree.parse(filename)
    root = xmltree.getroot()
    for child in root:
        idname = child.get('record_id').text.split('_')[0]
        id = idname[2:]
        if int(id) in brain:
            label = 'brain_cancer'
        elif int(id) in colon:
            label = 'colon_cancer'
        else:
            print('WARNING: id not found:', id)
            label = 'none'
        labelnode = etree.SubElement(child, 'diagnosis')
        labelnode.text = label
    etree.write(outfile)


def create_df(df):
    return pandas.DataFrame(columns=['ID'])


def collapse_labels(labels):
    flat_labels = []
    for lab in labels:
        for item in lab:
            flat_labels.append(item)
    return flat_labels

def extract_ranks(events, event_list=None, allow_empty=False):
    elem = load_xml_tags(events, decode=False)
    ranks = []
    event_map = {}
    if debug: print('extract_ranks: events:', type(events))# 'elem:', etree.tostring(elem))
    if debug: print('extract_ranks: event_list:', type(event_list))

    if event_list is not None:
        for event in event_list:
            if event.tag == 'EVENT':
                #print(etree.tostring(event))
                id = event.get('eid')
                rank = event.get('rank')
                if rank is None:
                    print('ERROR: no rank attribute found:', etree.tostring(event))
                    rank = 0
                    if not allow_empty:
                        exit(1)
                event_map[id] = rank

    event_count = 0
    for event in elem:
        if debug: print('child tag:', event.tag)
        if event.tag == 'EVENT':
            event_count += 1
            #print('elem event:', etree.tostring(event))
            if event_list is None:
                rank = event.get('rank')
            else:
                eventid = event.get('eid')
                #print('looking up eid', eventid)
                rank = event_map[eventid]
            if rank is None:
                print('ERROR: no rank attribute found:', etree.tostring(event))
                rank = 0
                if not allow_empty:
                    exit(1)
                #ranks.append(0)

            #if int(rank) == 0:
            #    print('WARNING: rank is 0:', etree.tostring(event))
            ranks.append(int(rank))
            #if int(rank) == 0:
            #    print('WARNING: rank is 0:', etree.tostring(event))
    if debug: print('events:', event_count, 'ranks:', len(ranks))
    assert(len(ranks) == event_count)
    return ranks


''' Convert arrows in text to non-arrows (for xml processing)
    filename: the file to fix (file will be overwritten)
'''
def fix_arrows(filename):
    sed_command = r"sed -e 's/-->/to/g' " + filename + r" | sed -e 's/->/to/g' | sed -e 's/ < / lt /g' | sed -e 's/ > / gt /g'"
    print("sed_command: ", sed_command)
    #f = open("temp", 'wb')
    ps = subprocess.Popen(sed_command, shell=True, stdout=subprocess.PIPE)
    output = ps.communicate()[0]
    out = open(filename, 'w')
    out.write(output)
    out.close()


def fix_escaped_chars(filename):
    subprocess.call(["sed", "-i", "-e", 's/&lt;/ </g', filename])
    subprocess.call(["sed", "-i", "-e", 's/&gt;/> /g', filename])
    subprocess.call(["sed", "-i", "-e", 's/  / /g', filename])
    subprocess.call(["sed", "-i", "-e", "s/‘/'/g", filename])
    subprocess.call(["sed", "-i", "-e", "s/’/'/g", filename])
    subprocess.call(["sed", "-i", "-e", "s/&#8216;/'/g", filename])
    subprocess.call(["sed", "-i", "-e", "s/&#8217;/'/g", filename])
    subprocess.call(["sed", "-i", "-e", "s/&#8211;/,/g", filename])


''' Remove blank lines, convert \n to space, remove double spaces, insert a line break before each record
    filename: the file to fix (file will be overwritten)
    rec_type: the type of record: adult, child, or neonate
'''
def fix_line_breaks(filename, rec_type):
    tag = "<Adult_Anonymous>"
    if rec_type == "child":
        tag = "<Child_Anonymous>"
    elif rec_type == "neonate":
        tag = "<Neonate_Anonymous>"
    sed_command = "s/" + tag + r"/\n" + tag + "/g"
    sed_command2 = r"sed -e 's/<\/root>/\n<\/root>/g'"
    #print "sed_command: " + sed_command
    tr_command = "tr " + r"'\n' " + "' '"
    #print "tr_command: " + tr_command
    #f = open("temp", 'wb')
    command = "sed -e '/^\s$/d' " + filename + " | " + tr_command + " | sed -e 's/  / /g' | sed -e '" + sed_command + "'" + " | " + sed_command2
    ps = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    output = ps.communicate()[0]
    out = open(filename, 'w')
    out.write(output)
    out.close()


def fix_xml_tags(text):
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;EVENT&gt;', '<EVENT>').replace('&lt;/EVENT&gt;', '</EVENT>')
    text = text.replace('&lt;EVENT', '<EVENT')
    text = text.replace('&amp;lt;EVENT&amp;gt;;', '<EVENT>').replace('&amp;lt;/EVENT&amp;gt;', '</EVENT>')
    text = text.replace('&lt;TIMEX3&gt;', '<TIMEX3>').replace('&lt;/TIMEX3&gt;', '</TIMEX3>')
    text = text.replace('&lt;TIMEX3', '<TIMEX3')
    text = text.replace('&amp;lt;TIMEX3&amp;gt;', '<TIMEX3>').replace('&amp;lt;/TIMEX3&amp;gt;', '</TIMEX3>')
    text = text.replace('&lt;SIGNAL&gt;', '<SIGNAL>').replace('&lt;/SIGNAL&gt;', '</SIGNAL>')
    text = text.replace('&lt;SIGNAL', '<SIGNAL')
    text = text.replace('&lt;TLINK', '<TLINK').replace('/&gt;', '/>')
    text = text.replace('" &gt;', '">')
    text = text.replace('"&gt;', '">').replace(' >', '>')
    text = text.replace('&', '&amp;') # escape any leftover and signs
    return text


''' Shuffle events within the same rank value, produce one shuffled example for every in-order example
'''
def generate_permutations(ids, x, y):
    new_ids = ids
    new_x = x
    new_y = y
    for n in range(len(x)):
        #rank_map = {}
        doc_id = ids[n]
        x_list = x[n]
        y_list = y[n]
        new_x_list = []
        new_y_list = []
        temp_list = list(zip(x_list, y_list))
        shuffle(temp_list)
        new_lists = [list(t) for t in zip(*temp_list)]
        new_x_list = new_lists[0]
        new_y_list = new_lists[1]
        new_ids.append(doc_id)
        new_x.append(new_x_list)
        new_y.append(new_y_list)
        #print('Shuffle entry:', str(new_y_list))

    # Shuffle the final training list
    temp_pairs = list(zip(new_ids, new_x, new_y))
    shuffle(temp_pairs)
    #print('shuffle temp pairs[0]:', str(temp_pairs[0]))
    new_lists = [list(t) for t in zip(*temp_pairs)]
    new_ids = new_lists[0]
    new_x = new_lists[1]
    new_y = new_lists[2]
    #print('shuffle new_y[0]', str(new_y[0]))
    return new_ids, new_x, new_y


def load_time_pairs(filename):
    print('load time pairs:', filename)
    time_df = pandas.read_csv(filename, header=None, index_col=False)
    time_df.columns = ['time1', 'time2', 'order']
    pairs = []
    labels = []
    for i, row in time_df.iterrows():
        pairs.append((split_words(row['time1']), split_words(row['time2'])))
        labels.append(row['order'])
        #print('loaded time pair:', pairs[-1], labels[-1])
    return pairs, labels


def load_xml_tags(ann, unwrap=True, decode=False):
    if debug: print('load_xml_tags:', ann)
    if decode or type(ann) is not str:
        ann = ann.decode('utf8')
    if unwrap:
        ann_xml = etree.fromstring(ann)
        ann_text = stringify_children(ann_xml)
    else:
        ann_text = ann
    ann_text = fix_xml_tags(ann_text) # Escape & signs that might have been unescaped
    #if len(ann_text) > 830:
    #    print(ann_text[820:])
    ann_element = etree.fromstring("<root>" + ann_text + "</root>")
    return ann_element


def reorder_encodings(encodings, orderings):
    print('reorder encodings:', len(encodings), orderings)
    assert(len(encodings) == len(orderings))
    dim = encodings[0].size(-1)
    new_encodings = []
    for x in range(len(encodings)):
        enc = encodings[x].view(-1, dim)
        order = orderings[x]

        indices = []
        for y in range(len(order)):
            indices.append((y, order[y]))
        indices.sort(key=lambda k: k[1])
        #shuffle(indices)
        enc_list = []
        for pair in indices:
            rank = pair[1]
            index = pair[0]
            print('picking rank:', rank, 'at index:', index)
            enc_list.append(enc[index])
        new_enc = torch.stack(enc_list)
        print('encodings size:', new_enc.size())
        new_encodings.append(new_enc)

    return new_encodings


def score_majority_class(true_labs):
    pred_labs = []
    majority_lab = None
    count_map = {}
    for lab in true_labs:
        if lab not in count_map.keys():
            count_map[lab] = 0
        count_map[lab] = count_map[lab]+1
    majority_lab = max(count_map.iteritems(), key=operator.itemgetter(1))[0]
    for lab in true_labs:
        pred_labs = majority_lab
    # Score
    precision = metrics.precision_score(true_labs, pred_labs, average="weighted")
    recall = metrics.recall_score(true_labs, pred_labs, average="weighted")
    f1 = metrics.f1_score(true_labs, pred_labs, average="weighted")
    return precision, recall, f1


''' Scores vector labels with binary values
    returns: avg precision, recall, f1 of 1 labels (not 0s)
'''
def score_vec_labels(true_labs, pred_labs):
    p_scores = []
    r_scores = []
    f1_scores = []
    micro_pos = 0
    micro_tp = 0
    micro_fp = 0
    assert(len(true_labs) == len(pred_labs))
    for x in range(len(true_labs)):
        true_lab = true_labs[x]
        pred_lab = pred_labs[x]
        pos = 0
        tp = 0
        fp = 0
        for y in range(len(true_lab)):
            true_val = true_lab[y]
            pred_val = pred_lab[y]
            if true_val == 1:
                pos = pos+1
                micro_pos = micro_pos+1
                if pred_val == 1:
                    tp = tp+1
                    micro_tp=micro_tp+1
            else:
                if pred_val == 1:
                    fp = fp+1
                    micro_fp = micro_fp+1

        p = 0.0
        r = 0.0
        if (tp+fp) > 0:
            p = float(tp) / float(tp+fp)
        if pos > 0:
            r = float(tp) / float(pos)
        if p == 0.0 and r == 0.0:
            f1 = float(0)
        else:
            f1 = 2*(p*r)/(p+r)
        p_scores.append(p)
        r_scores.append(r)
        f1_scores.append(f1)
    precision = numpy.average(p_scores)
    recall = numpy.average(r_scores)
    f1 = numpy.average(f1_scores)
    micro_p = 0.0
    micro_r = 0.0
    if (micro_tp+micro_fp) > 0:
        micro_p = float(micro_tp) / float(micro_tp+micro_fp)
    if micro_pos > 0:
        micro_r = float(micro_tp) / float(micro_pos)
    if micro_p == 0.0 and micro_r == 0.0:
        micro_f1 = float(0)
    else:
        micro_f1 = 2*(micro_p*micro_r)/(micro_p+micro_r)
    return precision, recall, f1, micro_p, micro_r, micro_f1


''' A function for separating words and punctuation
    Not using NLTK because it would split apart contractions and we don't want that
'''
def split_words(text):
    return wordpunct_tokenize(text)
    #return re.findall(r"[\w']+|[.,!?;$=/\-\[\]]", text.strip())


''' Get content of a tree node as a string
    node: etree.Element
'''
def stringify_children(node):
    parts = ([str(node.text)] + list(chain(*([tostring(c)] for c in node.getchildren()))))
    # filter removes possible Nones in texts and tails
    for x in range(len(parts)):
        if type(parts[x]) != str:
            parts[x] = str(parts[x])
    return ''.join(filter(None, parts))


''' Get contents of tags as a list of strings
    text: the xml-tagged text to process
    tags: a list of the tags to extract
    atts: a list of attributes to extract as well
'''
def phrases_from_tags(text, tags, atts=[]):
    for x in range(len(tags)):
        tags[x] = tags[x].lower()
    text = "<root>" + text + "</root>"
    phrases = []
    root = etree.fromstring(text)
    #print "phrases_from tags text: " + text
    for child in root:
        if child.tag.lower() in tags:
            print("found tag: ", child.tag)
            phrase = {}
            if child.text is not None:
                phrase['text'] = child.text
            for att in atts:
                if att in child.keys():
                    phrase[att] = child.get(att)
            phrases.append(phrase)
    return phrases


''' Get contents of tags as a list of strings
    text: the xml-tagged text to process
    tags: a list of the tags to extract
'''
def text_from_tags(text, tags):
    for x in range(len(tags)):
        tags[x] = tags[x].lower()
    text = "<root>" + text + "</root>"
    newtext = ""
    root = etree.fromstring(text)
    print("text: ", text)
    for child in root:
        print("--child")
        if child.tag.lower() in tags:
            print("found tag: ", child.tag)
            if child.text is not None:
                newtext = newtext + ' ' + child.text
    return newtext


''' matrix: a list of dictionaries
    dict_keys: a list of the dictionary keys
    outfile: the file to write to
'''
def write_to_file(matrix, dict_keys, outfile):
    # Write the features to file
    print("writing ", str(len(matrix)), " feature vectors to file...")
    output = open(outfile, 'w')
    for feat in matrix:
        #print "ICD_cat: " + feat["ICD_cat"]
        feat_string = str(feat).replace('\n', '')
        output.write(feat_string + "\n")
    output.close()

    key_output = open(outfile + ".keys", "w")
    key_output.write(str(dict_keys))
    key_output.close()
    return dict_keys


def xml_to_txt(filename):
    name = filename.split(".")[0]
    sed_command = r"sed '$d' < " + filename + r" | sed '1d' > " + name + ".txt"
    ps = subprocess.Popen(sed_command, shell=True, stdout=subprocess.PIPE)
    ps.communicate()


def zero_vec(dim):
    vec = []
    for x in range(dim):
        vec.append(0)
    return vec
