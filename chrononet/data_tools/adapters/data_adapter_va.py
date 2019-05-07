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
            row['text'] = child.find('narrative').text.strip()
            #print('narrative:', row['text'])
            #row['tags'] = etree.tostring(child.find('narr_timeml_simple'), encoding='utf8').decode('utf8')
            tag_node = child.find('narr_timeml_simple')
            tag_narr = etree.tostring(tag_node, encoding='utf8').decode('utf8')
            #print('tag_narr orig:', tag_narr)
            tag_narr = data_util.fix_xml_tags(tag_narr)
            print('tag_narr fixed:', tag_narr)
            row['tags'] = tag_narr
            elem = data_util.load_xml_tags(row['tags'], decode=False, unwrap=True)
            event_elem = etree.Element('events')
            num_events = 0
            for elem_child in elem:
                if elem_child.tag == 'EVENT':
                    event_elem.append(elem_child)
                    num_events += 1

            # Drop records with no annotated events
            if num_events == 0:
                print('Dropping record with no events:', row['docid'])
                continue
            else:
                print('Loaded events:', num_events)

            row['events'] = etree.tostring(event_elem).decode('utf8')
            row['event_ranks'] = data_util.extract_ranks(row['tags'])
            print(row['docid'], 'event_ranks:', row['event_ranks'])
            row['diagnosis'] = child.find('cghr_cat').text
            date_node = child.find('DeathDate')
            if date_node is None:
                date_node = child.find('InterviewDate')
            row['dct'] = date_node.text
            df = df.append(row, ignore_index=True)
        return df

    def write_output(self, df, outdir, doc_level=True):
        outfile = os.path.join(outdir, 'out.xml')
        xmltree = self.seq_to_xml(df, tag="narr_crf", elementname="AdultAnonymous", id_name="MG_ID", doc_level=doc_level)

        # Add event list if ordering was predicted
        if 'event_ranks' in df:
            xmltree = self.add_ranks(xmltree, df, record_name='MG_ID')

        xmltree.write(outfile)
