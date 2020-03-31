#!/usr/bin/python3
# Data adapter for the VA dataset

import ast
import os
import pandas

from lxml import etree

from .data_adapter import DataAdapter
from data_tools import data_util

class DataAdapterVA(DataAdapter):

    def load_data(self, filename, drop_unlabeled=True):
        if self.debug: print('DataAdapterVA.load_data', filename)
        df = pandas.DataFrame(columns=self.column_names)

        # Loop through data entries and create a row for each record
        tree = etree.parse(filename)
        root = tree.getroot()
        for child in root:
            row = {}
            row['docid'] = child.find('MG_ID').text
            narr_node = child.find('narrative')

            # Ignore records with no narrative
            if narr_node is not None:
                row['text'] = child.find('narrative').text
                #row['text_orig'] = child.find('narrative').text
                #print('narrative:', row['text'])
                #row['tags'] = etree.tostring(child.find('narr_timeml_simple'), encoding='utf8').decode('utf8')
                tag_node = child.find('narr_timeml_simple')
                if tag_node is None:
                    tag_node = child.find('narr_crf')
                tag_narr = ''
                event_elem = None
                num_events = 0
                if tag_node is not None:
                    tag_narr = etree.tostring(tag_node, encoding='utf8').decode('utf8')
                    #print('tag_narr orig:', tag_narr)
                    tag_narr = data_util.fix_xml_tags(tag_narr)
                    if self.debug: print('tag_narr fixed:', tag_narr)
                    row['tags'] = tag_narr
                    elem = data_util.load_xml_tags(row['tags'], decode=False, unwrap=True)
                    event_elem = etree.Element('events')
                    for elem_child in elem:
                        if elem_child.tag == 'EVENT':
                            event_elem.append(elem_child)
                            num_events += 1
                else:
                    row['tags'] = ''

                # Drop records with no annotated events
                if drop_unlabeled and num_events == 0:
                    if self.debug: print('Dropping record with no events:', row['docid'])
                    continue
                else:
                    if self.debug: print('Loaded events:', num_events)

                if event_elem is not None:
                    row['events'] = etree.tostring(event_elem).decode('utf8')
                    row['event_ranks'] = data_util.extract_ranks(row['tags'])
                    if self.debug: print(row['docid'], 'event_ranks:', row['event_ranks'])
                else:
                    row['events'] = ''
                    row['event_ranks'] = ''
                row['diagnosis'] = str(child.find('cghr_cat').text)
                if self.debug: print('cghr_cat:', row['diagnosis'])
                date_node = child.find('DeathDate')
                if date_node is None:
                    date_node = child.find('Death_YR_Month')
                if date_node is None:
                    date_node = child.find('InterviewDate')
                row['dct'] = date_node.text
                df = df.append(row, ignore_index=True)
        return df

    def write_output(self, df, outdir, doc_level=True):
        outfile = os.path.join(outdir, 'out.xml')
        xmltree = None
        if 'sequence' in self.stages:
            xmltree = self.seq_to_xml(df, tag="narr_crf", elementname="AdultAnonymous", id_name="MG_ID", doc_level=doc_level)

        # Add event list if ordering was predicted
        if 'ordering' in self.stages:
            xmltree = self.add_ranks(xmltree, df, record_name='MG_ID')

        # Add CoD classification to the xml file
        if 'classification' in self.stages:
            xmltree = self.class_to_xml(df, xmltree, tag='cghr_cat', elementname='AdultAnonymous', id_name='MG_ID')

        xmltree.write(outfile)
