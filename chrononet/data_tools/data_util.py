#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Util functions
# @author sjeblee@cs.toronto.edu

from lxml import etree
from lxml.etree import tostring
from itertools import chain
from sklearn import metrics
import numpy
import operator
import pandas
import subprocess

#def clean_file(filename):
    # remove blank lines | remove extra spaces| remove leading and trailing spaces  | fix utf-8 chars
    #command = r"sed '/^\s*$/d' $file | sed -e 's/  */ /g' | sed -e 's/^ //g' | sed -e 's/ $//g' | sed -e 's/&amp;/and/g' | sed -e 's/&#13;/ /g' | sed -e 's/&#8217;/\'/g' | sed -e 's/&#8221;/\"/g' | sed -e 's/&#8220;/\"/g' | sed -e 's/&#65533;//g' | sed -e 's/&#175\7;//g'| sed -e 's/&#1770;/\'/g'"
    # TODO

def add_labels(df, labels, labelname):
    df[labelname] = ''
    for i, row in df.iterrows():
        df.at[i, labelname] = labels[i]
    return df

def create_df(df):
    return pandas.DataFrame(columns=['ID'])


def collapse_labels(labels):
    flat_labels = []
    for lab in labels:
        for item in lab:
            flat_labels.append(item)
    return flat_labels


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
