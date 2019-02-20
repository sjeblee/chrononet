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
            for child in xmltree.getroot():
                docid = child.find('record_id').text
                print(docid, 'write_output', df.loc[df['docid'] == docid].to_string())
                row = df.loc[df['docid'] == docid].iloc[0] # There should only be one row if this is a doc-level df
                # Get the events and ranks
                events = row['events']
                event_ranks = row['event_ranks']
                if type(events) == str:
                    event_list = []
                    events = etree.fromstring(events)
                    for event_child in events:
                        event_list.append(event_child)
                        print(docid, 'event before:', etree.tostring(event_child))
                    events = event_list
                    #events = ast.literal_eval(events)
                if type(event_ranks) == str:
                    event_ranks = ast.literal_eval(event_ranks)
                print('events:', len(events), 'ranks:', len(event_ranks))
                assert(len(events) == len(event_ranks))
                rank_event_pairs = list(zip(event_ranks, events))

                # Sort the events by rank
                sorted_event_pairs = sorted(rank_event_pairs, key=lambda x: x[0])
                eventlist_elem = etree.SubElement(child, 'event_list')
                for event_pair in sorted_event_pairs:
                    #event = etree.fromstring(event_pair[1])
                    event = event_pair[1]
                    rank_val = event_pair[0]
                    print('rank:', str(rank_val), 'event:', etree.tostring(event))
                    event.set('rank', str(rank_val))
                    print(docid, 'event after', etree.tostring(event))
                    eventlist_elem.append(event)
                #child.append(eventlist_elem)

        xmltree.write(outfile)
