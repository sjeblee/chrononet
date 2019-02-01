#!/usr/bin/python3
# Data adapter for the THYME dataset

import pandas

from lxml import etree

from data_tools import data_util

def extract_entities(df, target_df):
    target_df = df.copy()
    target_df['entities'] = ''
    target_df['relations'] = ''
    for i, row in df.iterrows():
        tag_string = row['tags']
        node = etree.fromstring(tag_string)
        rel_node = etree.Element('relations')
        # Save the relations in their own column
        for child in node:
            if child.tag == 'TLINK':
                rel_node.append(child)
        # TODO: extract entities from xml string
    return target_df

def extract_relations(df, target_df):
    if 'entities' not in df.keys():
        df = extract_entities(df, df.copy())

    target_df = pandas.DataFrame(columns=['docid', 'relid', 'relation', 'text1', 'text2'])
    # TODO: get relation objects from the tags column, create them as rows

    return target_df
