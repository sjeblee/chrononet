#!/usr/bin/python3
# Data adapter for the THYME dataset

import ast
import os
import pandas

from lxml import etree

from .data_adapter import DataAdapter
from data_tools import data_util

# Structs for json annotations
def Struct(*fields):

    class _Struct:
        def __init__(self, *args):

            if len(args) != len(fields):
                raise IncompleteStruct(fields)

            self.__dict__.update(zip(fields, args))

    return _Struct

class IncompleteStruct(Error):
    "Expected {} fields: {}"

    def create(self, doc, fields):
        return doc.format(len(fields), ", ".join(fields))


Annotation = Struct("tag", "features", "spans")
Span = Struct("seqNo", "startIndex", "endIndex")

class AnnotationSaveObject(Jsonable):

    def __init__(self):
        self.annotations = []

    def fromObj(self, obj):
        for item in obj:
            #for tag, features, rawSpans in obj:
            if type(item) is list or type(item) is tuple:
                tag = item[0]
                features = item[1]
                rawSpans = item[2]
                #print('fromObj:', tag, features, rawSpans)
                spans = [Span(*rspan) for rspan in rawSpans]
                #if len(spans) == 0:
                #    print('WARNING: no spans found when loading:', tag, features)
                self.annotations.append(Annotation(tag, features, spans))
            else:
                print('WARNING: annotation item is a not a tuple or list:', item)
                self.annotations.append(item)

    def json(self):
        object = [[ann.tag, ann.features, [self.span_json(span) for span in ann.spans]] for ann in self.annotations]
        return json.dumps(object)

    def span_json(self, span):
        return [span.seqNo, span.startIndex, span.endIndex]

    def __str__(self):
        return str(self.json())

class DataAdapterVerilogue(DataAdapter):

    def load_data(self, filename, drop_unlabeled=True):
        print('DataAdapterVerilogue.load_data', filename)
        df = pandas.DataFrame(columns=self.column_names)
        #df['diagnosis'] = ''

        # Loop through data entries and create a row for each record
        tree = etree.parse(filename)
        root = tree.getroot()
        transcripts = root.find("transcripts")
        ann_node = transcripts.find("annotations")
        annObj = AnnotationSaveObject()

        for child in root:
            row = {}
            row['docid'] = child.find('record_id').text
            docid = row['docid']
            row['text'] = child.find('narrative').text
            row['tags'] = etree.tostring(child.find('narr_timeml_simple'), encoding='utf8').decode('utf8')
            event_list = child.find('event_list')
            elem = data_util.load_xml_tags(row['tags'])
            event_elem = etree.Element('events')
            dct = ''
            for child in elem:
                if child.tag == 'EVENT':
                    event_elem.append(child)
                elif child.tag == 'TIMEX3':
                    if 'functionInDocument' in child.attrib and child.attrib['functionInDocument'] == 'CREATION_TIME':
                        dct = child.text
            event_elem = data_util.add_time_ids(event_elem, elem)
            row['events'] = etree.tostring(event_elem).decode('utf8')
            print(docid, 'event_list:', type(event_list))
            if event_list is None:
                print('WARNING: event_list not found:', row['docid'])
            row['event_ranks'] = data_util.extract_ranks(row['tags'], event_list)
            print(docid, 'event_ranks:', row['event_ranks'])
            # No diagnosis yet for THYME
            row['diagnosis'] = ''
            row['dct'] = dct
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
