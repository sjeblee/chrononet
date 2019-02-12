#!/usr/bin/python3
# Data adapter for the VA dataset

import ast
import os
import pandas

from lxml import etree

from .data_adapter import DataAdapter
from data_tools import data_util

class DataAdapterVA(DataAdapter):

    def load_data(self, filename):
        print('DataAdapterVA.load_data', filename)
        df = pandas.DataFrame(columns=self.column_names)

        # Loop through data entries and create a row for each record
        tree = etree.parse(filename)
        root = tree.getroot()
        for child in root:
            row = {}
            row['docid'] = child.find('MG_ID').text
            row['text'] = child.find('narrative').text
            tag_narr = etree.tostring(child.find('narr_timeml_simple'), encoding='utf8')
            #tag_narr = self.fix_xml_tags(tag_narr)
            row['tags'] = tag_narr
            row['event_ranks'] = data_util.extract_ranks(row['tags'])
            row['diagnosis'] = child.find('cghr_cat')
            df = df.append(row, ignore_index=True)
        return df

    def write_output(self, df, outdir):
        outfile = os.path.join(outdir, 'out.xml')
        self.seq_to_xml(df, tag="narr_crf", elementname="AdultAnonymous", id_name="MG_ID").write(outfile)
