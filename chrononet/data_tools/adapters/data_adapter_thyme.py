#!/usr/bin/python3
# Data adapter for the THYME dataset

import ast
import os
import pandas

from lxml import etree

from .data_adapter import DataAdapter
#from chrononet.data_tools import data_util

class DataAdapterThyme(DataAdapter):

    def load_data(self, filename):
        print('DataAdapterThyme.load_data', filename)
        df = pandas.DataFrame(columns=self.column_names)

        # Loop through data entries and create a row for each record
        tree = etree.parse(filename)
        root = tree.getroot()
        for child in root:
            row = {}
            row['docid'] = child.find('record_id').text
            row['text'] = child.find('narrative').text
            row['tags'] = etree.tostring(child.find('narr_timeml_simple'), encoding='utf8')
            # No diagnosis yet for THYME
            df = df.append(row, ignore_index=True)
        return df

    # TODO: implement data loading from the original thyme files

    '''
       seqs: dict[id] -> [(word, label),...]
       filename: optional, the xml file to add the sequences to. If blank, will create a new tree
       tag: the tag to use for new elements if creating a new tree
    '''
    def seq_to_xml(self, df, filename=None, tag="narr_timeml", elementname="Record", id_name="record_id"):
        if self.debug: print("seq_to_xml")

        tree = None
        root = etree.Element("root")
        for i, row in df.iterrows():
            docid = row['docid']
            seq = row['seq']
            if type(seq) == 'str':
                seq = ast.literal_eval(seq)
            labels = row['seq_labels']
            if type(labels) == 'str':
                labels = ast.literal_eval(labels)
            seq_labels = list(zip(seq, labels))
            child = etree.SubElement(root, elementname)
            id_node = etree.SubElement(child, id_name)
            id_node.text = docid
            text_node = etree.SubElement(child, 'narrative')
            text_node.text = row['text']
            narr_node = etree.SubElement(child, tag)
            narr_node.text = self.to_xml(seq_labels)
            if self.debug: print("added seq: ", narr_node.text)
        tree = etree.ElementTree(root)
        return tree

    def write_output(self, df, outdir):
        outfile = os.path.join(outdir, 'out.xml')
        self.seq_to_xml(df).write(outfile)
