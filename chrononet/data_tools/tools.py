#!/usr/bin/python3
# Chrononet data tools

import random
import re
import subprocess
from itertools import chain
from lxml.etree import tostring
from lxml import etree
from xml.sax.saxutils import unescape

def escape(text):
    text = re.sub('<', '&lt;', text)
    text = re.sub('>', '&gt;', text)
    return text


def escape_and(text):
    return re.sub('&', '&amp;', text)


''' Unescape arrows in an xml file
'''
def fix_arrows(filename):
    subprocess.call(["sed", "-i", "-e", 's/&lt;/</g', filename])
    subprocess.call(["sed", "-i", "-e", 's/&gt;/>/g', filename])
    subprocess.call(["sed", "-i", "-e", 's/  / /g', filename])


def fix_arrows_only(filename):
    subprocess.call(["sed", "-i", "-e", 's/&lt;/</g', filename])
    subprocess.call(["sed", "-i", "-e", 's/&gt;/>/g', filename])


def stringify_node(node):
    return unescape(etree.tostring(node, encoding='utf-8').decode('utf-8'))


''' Get content of a tree node as a string
    node: etree.Element
'''
def stringify_children(node):
    # filter removes possible Nones in texts and tails
    parts = []
    node_text = node.text if node.text is not None else ""
    node_text = node_text.strip()
    parts.append(node_text)
    for c in node.getchildren():
        parts = parts + list(chain(*([tostring(c, encoding='utf-8').decode('utf-8')])))

    for x in range(len(parts)):
        if type(parts[x]) != str and parts[x] is not None:
            parts[x] = str(parts[x])
        #print("parts[x]:", parts[x])
    return ''.join(filter(None, parts))


def unsplit_punc(filename):
    subprocess.call(["sed", "-i", "-e", 's/ ,/,/g', filename])
    subprocess.call(["sed", "-i", "-e", 's/ \././g', filename])
    subprocess.call(["sed", "-i", "-e", "s/ '/'/g", filename])
    subprocess.call(["sed", "-i", "-e", "s/ :/:/g", filename])
    subprocess.call(["sed", "-i", "-e", "s/ - /-/g", filename])
    subprocess.call(["sed", "-i", "-e", "s/\[ /[/g", filename])
    subprocess.call(["sed", "-i", "-e", "s/ \]/]/g", filename])
    subprocess.call(["sed", "-i", "-e", "s/ = /=/g", filename])
    subprocess.call(["sed", "-i", "-e", "s/ \/ /\//g", filename])


def split_file(filename, prefix, num_chunks=10):
    xml_tree = etree.parse(filename)
    root = xml_tree.getroot()
    documents = []
    for child in root:
        documents.append(child)

    random.shuffle(documents) # shuffle the order of the entries
    num = len(documents)
    num_per_file = int(num / 10)
    print('num documents per file:', num_per_file)
    sets = []
    index = 0
    for i in range(num_chunks):
        sets.append([])
        for k in range(0, num_per_file):
            doc = documents[index]
            index += 1
            sets[i].append(doc)

    for j in range(1, num_chunks+1):
        test_outfile = prefix + '_test_' + str(j) + '.xml'
        train_outfile = prefix + '_train_' + str(j) + '.xml'
        print(train_outfile, test_outfile)
        train_root = etree.Element("root")
        test_root = etree.Element("root")
        for i in range(len(sets)):
            if i == j-1:
                for item in sets[i]:
                    test_root.append(item)
            else:
                for item in sets[i]:
                    train_root.append(item)
        traintree = etree.ElementTree(train_root)
        traintree.write(train_outfile)
        testtree = etree.ElementTree(test_root)
        testtree.write(test_outfile)
