#!/usr/bin/python3
# Data adapter for the THYME dataset

import ast
import os
import pandas

from lxml import etree

from .data_adapter import DataAdapter
from data_tools import data_util

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
            event_list = child.find('event_list')
            print('event_list:', type(event_list))
            row['event_ranks'] = data_util.extract_ranks(row['tags'], event_list) # TODO
            # No diagnosis yet for THYME
            df = df.append(row, ignore_index=True)
        return df

    # TODO: implement data loading from the original thyme files

    def write_output(self, df, outdir):
        outfile = os.path.join(outdir, 'out.xml')
        self.seq_to_xml(df).write(outfile)
