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
        #df['diagnosis'] = ''

        # Loop through data entries and create a row for each record
        tree = etree.parse(filename)
        root = tree.getroot()
        for child in root:
            row = {}
            row['docid'] = child.find('record_id').text
            docid = row['docid']
            row['text'] = child.find('narrative').text
            row['tags'] = etree.tostring(child.find('narr_timeml_simple'), encoding='utf8')
            event_list = child.find('event_list')
            elem = data_util.load_xml_tags(row['tags'])
            event_elem = etree.Element('events')
            for child in elem:
                if child.tag == 'EVENT':
                    event_elem.append(child)
            row['events'] = etree.tostring(event_elem).decode('utf8')
            print(docid, 'event_list:', type(event_list))
            if event_list is None:
                print('WARNING: event_list not found:', row['docid'])
            row['event_ranks'] = data_util.extract_ranks(row['tags'], event_list)
            print(docid, 'event_ranks:', row['event_ranks'])
            # No diagnosis yet for THYME
            row['diagnosis'] = ''
            df = df.append(row, ignore_index=True)
        return df

    # TODO: implement data loading from the original thyme files

    def write_output(self, df, outdir, doc_level=False):
        outfile = os.path.join(outdir, 'out.xml')
        xmltree = self.seq_to_xml(df, doc_level=doc_level)

        # Add event list if ordering was predicted
        if 'event_ranks' in df:
            xmltree = self.add_ranks(xmltree, df)

        xmltree.write(outfile)
