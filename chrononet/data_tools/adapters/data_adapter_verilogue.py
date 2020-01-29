#!/usr/bin/python3
# Data adapter for the THYME dataset

import ast
import json
import os
import pandas

from lxml import etree

from .data_adapter import DataAdapter, Element
from data_tools import data_scheme
#from data_tools import data_util

class Jsonable:

    def dump(self, fname):
        with open(fname, "w") as f:
            json.dump(self.json(), f)

    def load(self, fname):
        with open(fname, "r") as f:
            obj = json.load(f)
            self.fromObj(obj)

    # === ABSTRACT PROTECTED ===
    def fromObj(self, obj):
        "Loads features of obj into itself."
        raise NotImplementedError

    def json(self):
        "Returns the JSON form of itself."
        raise NotImplementedError

class Error(Exception):
    ""

    def __init__(self, *args, **kwargs):
        doc = self.__doc__.strip()
        msg = self.create(doc, *args, **kwargs)
        Exception.__init__(self, msg)

    def create(self, doc, *args, **kwargs):
        return doc.format(*args, **kwargs)

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


BE = 'B-E'
IE = 'I-E'
BT = 'B-T'
IT = 'I-T'
OO = 'O'


class DataAdapterVerilogue(DataAdapter):

    def load_data(self, filename, drop_unlabeled=True):
        print('DataAdapterVerilogue.load_data', filename)
        df = pandas.DataFrame(columns=self.column_names)
        #df['diagnosis'] = ''

        # Loop through data entries and create a row for each record
        tree = etree.parse(filename)
        root = tree.getroot()

        for child in root:
            #child_root = child.find('interaction')
            transcripts = child.find('transcripts')
            ann_node = transcripts.find('annotations')
            annObj = AnnotationSaveObject()
            annObj.fromObj(json.loads(ann_node.text))
            row = {}
            row['docid'] = child.find('id').text.strip()
            docid = row['docid']
            row['tags'] = annObj.annotations # Save anns as a list

            # Get annotations and ranks
            ann_ranks = []
            events = []
            for ann in annObj.annotations:
                if ann.tag != 'TIMEX3':
                    events.append(ann)
                    if 'rank' in ann.features:
                        ann_ranks.append(ann.features['rank'])
                    else:
                        ann_ranks.append(None)
            row['events'] = events
            row['event_ranks'] = ann_ranks
            print(docid, 'event_ranks:', row['event_ranks'])

            # Get utterances
            turns = transcripts.find("turns")
            #utts = turns.xpath("//utterance")
            row['text'] = etree.tostring(turns, encoding='utf8')

            # No diagnosis yet for THYME
            row['diagnosis'] = ''
            dct = ''
            row['dct'] = dct
            df = df.append(row, ignore_index=True)

        return df

    def write_output(self, df, outdir, doc_level=False):
        outfile = os.path.join(outdir, 'out.xml')
        xmltree = self.seq_to_xml(df, doc_level=doc_level)

        # Add event list if ordering was predicted
        if 'event_ranks' in df:
            xmltree = self.add_ranks(xmltree, df)

        xmltree.write(outfile)

    '''
    Get sequence tags from separate xml tags and text
    narr: the text-only narrative
    ann: just the xml tags
    split_sents: NOT IMPLEMENTED YET
    '''
    def ann_to_seq(self, narr, ann, split_sents, ncrf=False):
        print('ann:', type(ann))
        if ann is not None:
            #annObj = AnnotationSaveObject()
            #annObj.fromObj(json.loads(ann))
            print('loaded annotations:', len(ann))
        narr_ref = etree.fromstring(narr.decode('utf8')) # Load utterances
        utt_map = {}
        for entry in narr_ref:
            #utt = etree.fromstring(entry)
            uid = entry.attrib['seqNo']
            print('ann_to_seq: uid:', uid)
            utt_map[int(uid)] = entry

        '''
        if ncrf:
            narr_ref = narr.replace('\n', '$')
            if split_sents:
                narr_ref = re.sub("\.  ", ". $", narr_ref)
        else:
            narr_ref = narr.replace('\n', ' ').strip() # Replace newlines with spaces so words get separated
        print('narr_ref:', narr_ref)
        '''
        #tags = []
        tag_map = {}
        seqs = [] # a list of tuples (word, label)

        # Process unlabeled narratives
        if ann is None:
            for entry in narr_ref:
                utt_seqs = []
                for word in self.split_words(entry.text):
                    utt_seqs.append((word, OO))
                seqs.append(utt_seqs)
        else:
            for item in ann:
                print('element:', item.tag)
                #if item.tag != 'TIMEX3':
                utt_num = -1
                for span in item.spans:
                    num = int(span.seqNo)
                    if num != utt_num:
                        if num not in tag_map:
                            tag_map[num] = []
                        tag_map[num].append(Element.from_ann(item, num))
                        utt_num = num
                        #print('element:', item.tag)
                    # Convert the annotation into an xml element
                    '''
                    elem = etree.Element(item.tag)
                    for key in item.features:
                        elem.attrib[key] = item.features[key]
                    tags.append(Element(item))
                    '''

            if self.debug:
                print("tag_map: ", len(tag_map.keys()))

            for uttid in tag_map.keys():
                utt_tags = tag_map[uttid]
                # Sort tags by span start
                utt_tags.sort(key=lambda x: x.start)
                utt_text = utt_map[uttid].text
                print('utt_text:', utt_text)
                index = 0
                utt_seqs = []

                for tag in utt_tags:
                    print('- tag span:', tag.start, tag.end, 'index:', index)

                    if tag.start < index:
                        print("WARNING: dropping overlapping reference tag")
                        continue
                    if tag.start > index:
                        text = utt_text[index:tag.start]
                        print('getting OO seq:', index, tag.start, text)
                        index = tag.start
                        self.get_seqs(text, OO, utt_seqs)
                    if tag.element.tag != 'TIMEX3':
                        label = BE
                    elif tag.element.tag == 'TIMEX3':
                        label = BT
                    print('getting label seq:', tag.start, tag.end, utt_text[tag.start:tag.end])
                    for word in self.split_words(utt_text[tag.start:tag.end]):
                        utt_seqs.append((word, label))
                        if label == BE:
                            label = IE
                        elif label == BT:
                            label = IT
                    index = tag.end

                # Add the tail of the narrative
                if index < len(utt_text):
                    text = utt_text[index:]
                    self.get_seqs(text, OO, utt_seqs)
                seqs.append(utt_seqs)

        # Split sentences
        '''
        if split_sents and not ncrf:
            print("split_sents")
            narr = re.sub("\.  ", ". \n", narr) # Add line breaks after sentence breaks
            narr_splits = narr.splitlines()
            #print "split_sents: " + str(len(narr_splits))
            split_seqs = []
            sent_seqs = []
            index = 0
            for chunk in narr_splits:
                chunk = chunk.strip()
                if len(chunk) > 0:
                    #print "chunk: " + chunk
                    num_words = len(self.split_words(chunk))
                    #print "num_words: " + str(num_words)
                    index2 = index+num_words
                    sent_seqs = seqs[index:index2]
                    #if self.debug:
                    #    print("seq: ", sent_seqs)
                    split_seqs.append(sent_seqs)
                    index += num_words
            seqs = split_seqs
        else:
            seqs = [seqs] # Embed the sequences in a second list for proper post-processing
        '''

        return seqs

    ''' Convert a doc-level df in SEQ format to a dataframe for ordering (use to_doc)
    doc_df: a doc-level df in SEQ format
    df_orig: a df in ORIG format (with original event tags)
    use_gold_tags: True to use gold tags, False to use predicted event tags (NOT IMPLEMENTED)
    '''
    def to_order(self, doc_df, df_orig, use_gold_tags=True):
        print('to_order')
        order_df = pandas.DataFrame(columns=data_scheme.order())
        for i, row in doc_df.iterrows():
            docid = row['docid']
            seq = row['seq']
            if type(seq) == 'str':
                seq = ast.literal_eval(seq)
            labels = row['seq_labels']
            if type(labels) == 'str':
                labels = ast.literal_eval(labels)
            seq_labels = list(zip(seq, labels))

            # Create new row for the order df
            new_row = {}
            new_row['docid'] = docid
            new_row['seq'] = row['seq']
            new_row['seq_labels'] = row['seq_labels']

            orig_row = df_orig.loc[df_orig['docid'] == docid].iloc[0]
            new_row['diagnosis'] = orig_row['diagnosis']
            new_row['tags'] = orig_row['tags']
            new_row['text'] = orig_row['text']
            new_row['text_orig'] = row['text_orig']
            new_row['dct'] = orig_row['dct']

            if use_gold_tags:
                #if self.debug: print(docid, 'orig_row[events]', str(orig_row['events']))
                #events_xml = etree.fromstring(str(orig_row['events']))
                #if self.debug: print(docid, 'events_xml: ', etree.tostring(events_xml))
                new_row['events'] = orig_row['events'] # Keep as xml node
                event_ranks = orig_row['event_ranks']
                if type(event_ranks) == str:
                    event_ranks = ast.literal_eval(event_ranks)
                if self.debug:
                    #for e in events_xml:
                    #    print(docid, 'to_order: event:', etree.tostring(e))
                    print(docid, 'to_order: event_ranks:', type(event_ranks), str(event_ranks))
                #if type(event_ranks) == str:
                #    event_ranks = list(ast.literal_eval(event_ranks))
                new_row['event_ranks'] = event_ranks # These will be in text order
            '''
            else:
                # Reconstruct events from seq
                print('ERROR: using pred sequences as events not implemented yet!')
                xml_text = self.to_xml(seq_labels)
                elem = data_util.load_xml_tags(xml_text, unwrap=False)
                #if self.debug: print('elem:', etree.tostring(elem))
                events = etree.fromstring()
                for child in elem:
                    if child.tag == 'EVENT':
                        events.append(etree.tostring(child).decode('utf8'))
                        if self.debug: print('added event:', etree.tostring(child).decode('utf8'))

                # TODO: propagate gold ranks to pred events based on partial overlap
            '''
            order_df = order_df.append(new_row, ignore_index=True)

        return order_df
