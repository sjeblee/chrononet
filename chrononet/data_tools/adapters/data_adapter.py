#!/usr/bin/python3
# General-purpose data conversion tools for Chrononet

import ast
import re
import pandas

from lxml import etree
#from xml.sax.saxutils import unescape

# Local imports
from data_tools import data_scheme, data_util

BE = 'BE'
IE = 'IE'
BT = 'BT'
IT = 'IT'
OO = 'O'

class Element:
    element = None
    start = 0
    end = 0

    def __init__(self, element):
        self.element = element
        span_text = element.attrib['span']
        span = span_text.split(',')
        self.start = int(span[0])
        self.end = int(span[1])

class DataAdapter:
    debug = False
    column_names = data_scheme.orig()

    def __init__(self, debug):
        self.debug = debug

    def get_labelname(self, stage_name):
        if stage_name == 'sequence':
            return 'seq_labels'
        elif stage_name == 'ordering':
            return 'event_ranks'
        elif stage_name == 'classification':
            return 'diagnosis'
        else:
            return None

    def load_data(self, filename):
        df = pandas.DataFrame(columns=self.column_names)
        return df

    ''' Convert seq level df to doc level df
    '''
    def to_doc(self, df):
        doc_df = pandas.DataFrame(columns=data_scheme.seq())
        doc_map = {}
        for i, row in df.iterrows():
            docid = row['docid']
            text = row['text']
            if docid not in doc_map:
                doc_map[docid] = {'docid': docid, 'seqid': 0, 'text': text, 'seq': [], 'seq_labels': []}
            else:
                doc_map[docid]['text'] += ' ' + text
            row_seq = row['seq']
            seq_labels = row['seq_labels']
            if type(row_seq) == str:
                row_seq = ast.literal_eval(row_seq)
            if type(seq_labels) == str:
                seq_labels = ast.literal_eval(seq_labels)
            for item in row_seq:
                doc_map[docid]['seq'].append(item)
            for item in seq_labels:
                doc_map[docid]['seq_labels'].append(item)

        # Put everything into a new dataframe
        for key in doc_map.keys():
            doc_df = doc_df.append(doc_map[key], ignore_index=True)

        return doc_df

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
            new_row['dct'] = orig_row['dct']

            if use_gold_tags:
                #if self.debug: print(docid, 'orig_row[events]', str(orig_row['events']))
                events_xml = etree.fromstring(str(orig_row['events']))
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
            order_df = order_df.append(new_row, ignore_index=True)

        return order_df

    ''' Convert dataframe into one row per sequence (sentence) and sequence labels
        df: the dataframe in ORIG format
        split_sents: True if each row should be a sentence, False if each row should be a record
    '''
    def to_seq(self, df, split_sents=False):
        seq_columns = data_scheme.seq()
        target_df = pandas.DataFrame(columns=seq_columns)
        for i, row in df.iterrows():
            #print('to_seq row:', str(row))
            labeled_seqs = self.ann_to_seq(row['text'], row['tags'], split_sents)
            if self.debug:
                print('labeled_seqs:', len(labeled_seqs))
            sid = 0
            for labeled_seq in labeled_seqs:
                seq = []
                seq_labels = []
                #print('items:', len(labeled_seq))
                for item in labeled_seq:
                    seq.append(item[0])
                    seq_labels.append(item[1])
                target_row = {}
                target_row['docid'] = row['docid']
                target_row['seqid'] = sid
                target_row['seq'] = seq
                target_row['seq_labels'] = seq_labels
                target_row['text'] = ' '.join(seq)
                if self.debug:
                    print('adding feat row:', target_row['docid'], target_row['seqid'], target_row['text'], target_row['seq_labels'])
                target_df = target_df.append(target_row, ignore_index=True)
                sid += 1

        return target_df

    '''
    Get sequence tags from separate xml tags and text
    narr: the text-only narrative
    ann: just the xml tags
    split_sents: NOT IMPLEMENTED YET
    '''
    def ann_to_seq(self, narr, ann, split_sents, ncrf=False):
        #print(type(ann))
        ann_element = data_util.load_xml_tags(ann, decode=False, unwrap=True)
        if ncrf:
            narr_ref = narr.replace('\n', '$')
            if split_sents:
                narr_ref = re.sub("\.  ", ". $", narr_ref)
        else:
            narr_ref = narr.replace('\n', ' ') # Replace newlines with spaces so words get separated
        tags = []
        seqs = [] # a list of tuples (word, label)
        for child in ann_element:
            if child.tag in ['EVENT', 'TIMEX3']:
                tags.append(Element(child))

                #print "element: " + etree.tostring(child).decode('utf8')
        if self.debug:
            print("tags: ", len(tags))
        # Sort tags by span start
        tags.sort(key=lambda x: x.start)
        index = 0
        for tag in tags:
            if tag.start < index:
                print("WARNING: dropping overlapping reference tag")
                continue
            if tag.start > index:
                text = narr_ref[index:tag.start]
                index = tag.start
                self.get_seqs(text, OO, seqs)
            if tag.element.tag == 'EVENT':
                label = BE
            elif tag.element.tag == 'TIMEX3':
                label = BT
            for word in self.split_words(tag.element.text):
                seqs.append((word, label))
                if label == BE:
                    label = IE
                elif label == BT:
                    label = IT
            index = tag.end

        # Add the tail of the narrative
        if index < len(narr):
            text = narr_ref[index:]
            self.get_seqs(text, OO, seqs)

        # Split sentences
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

        return seqs

    def get_seqs(self, text, label, seqs):
        for word in self.split_words(text):
            seqs.append((word, OO))

    ''' A function for separating words and punctuation
        Not using NLTK because it would split apart contractions and we don't want that
    '''
    def split_words(self, text):
        return re.findall(r"[\w']+|[.,!?;$=/\-\[\]]", text.strip())

    def write_output(self, data, outdir):
        pass

    ############################

    def add_ranks(self, xmltree, df, record_name='record_id'):
        for child in xmltree.getroot():
            docid = child.find(record_name).text
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
                    #print(docid, 'event before:', etree.tostring(event_child))
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
                print(docid, 'true:', event.get('rank'), 'pred:', rank_val, 'event:', event.text)
                event.set('rank', str(rank_val))
                eventlist_elem.append(event)
        return xmltree

    def closelabel(self, prevlabel, elem_text):
        t_labels = ['BT', 'IT']
        e_labels = ['BE', 'IE']
        text = ""
        if prevlabel in t_labels:
            text = elem_text.strip() + '</TIMEX3>'
        elif prevlabel in e_labels:
            text = elem_text.strip() + '</EVENT>'
        return text

    '''
       seqs: dict[id] -> [(word, label),...]
       filename: optional, the xml file to add the sequences to. If blank, will create a new tree
       tag: the tag to use for new elements if creating a new tree
    '''
    def seq_to_xml(self, df, filename=None, tag="narr_timeml", elementname="Record", id_name="record_id", doc_level=False):
        if self.debug: print("seq_to_xml")

        tree = None
        root = etree.Element("root")
        for i, row in df.iterrows():
            docid = row['docid']
            seq = row['seq']
            if type(seq) == str:
                seq = ast.literal_eval(seq)
            labels = row['seq_labels']
            if type(labels) == str:
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

    def to_xml(self, seq):
        text = ""
        elem_text = ""
        tid = 0
        eid = 0
        prevlabel = OO
        in_elem = False
        for word, label in seq:
            #print(word, label)
            # Add xml tags if necessary
            if label == OO:
                in_elem = False
                if prevlabel != OO:
                    text = text + self.closelabel(prevlabel, elem_text)
                    elem_text = ""
            elif label == BT or (label == IT and (prevlabel != BT and prevlabel != IT)):
                text = text + self.closelabel(prevlabel, elem_text) + ' <TIMEX3 tid="t' + str(tid) + '" type="TIME" value="">'
                tid = tid+1
                in_elem = True
                elem_text = ""
            elif label == BE or (label == IE and (prevlabel != BE and prevlabel != IE)):
                text = text + self.closelabel(prevlabel, elem_text) + ' <EVENT eid="e' + str(eid) + '" class="OCCURRENCE">'
                eid = eid+1
                in_elem = True
                elem_text = ""

            # Add word
            if in_elem:
                elem_text = elem_text + word + ' '
            else:
                text = text + ' ' + word
            prevlabel = label

        # Close the final tag
        text = text + self.closelabel(prevlabel, elem_text)
        in_elem = False
        elem_text = ""
        text = re.sub('LINEBREAK', '\n', text.strip())

        # Fix lines
        lines = []
        for line in text.splitlines():
            lines.append(line.strip())
        return '\n'.join(lines) + '\n'

    ''' Converts inline xml tags to a sequence of word, label pairs
        text: the xml narrative as text
        returns: a list of (word, label) pairs
    '''
    def xml_to_seq(self, text):
        # List of tuples
        seq = []

        if self.debug: print("text: ", text)
        event_start = "<EVENT"
        time_start = ["<TIMEX3", "<SECTIME"]
        event_end = "</EVENT>"
        time_end = ["</TIMEX3>", "</SECTIME>"]
        ignore_tags = ["<TLINK", "<SLINK", "<ALINK", "<MAKEINSTANCE"]
        in_event = False
        b_event = False
        b_time = False
        in_time = False
        keep_linebreaks = True
        text = re.sub('\n', ' LINEBREAK ', text)
        chunks = text.split()
        x = 0

        while x < len(chunks):
            chunk = chunks[x].strip()
            if len(chunk) > 0:
                if self.debug: print("chunk: ", chunk)

                # Escape brackets for the NCRF model
                if chunk == "[":
                    chunk = "LB"
                elif chunk == "]":
                    chunk = "RB"

                # Handle EVENTs
                if in_event:
                    if chunk == event_end:
                        in_event = False
                    else:
                        word = chunk
                        label = IE
                        if event_end in chunk:
                            ind = chunk.index(event_end)
                            word = chunk[0:ind]
                            in_event = False
                        elif ">" in chunk:
                            ind = chunk.index('>')
                            word = chunk[ind:]
                        if b_event:
                            label = BE
                            b_event = False
                        pair = (word, label)
                        seq.append(pair)
                # Handle TIMEX3
                elif in_time:
                    if chunk in time_end:
                        in_time = False
                    else:
                        word = chunk
                        label = IT
                        for te in time_end:
                            if te in chunk:
                                ind = chunk.index(te)
                                word = chunk[0:ind]
                                in_time = False
                        if in_time and ">" in chunk:
                            ind = chunk.index('>')
                            word = chunk[ind:]
                        if b_time:
                            label = BT
                            b_time = False
                        pair = (word, label)
                        seq.append(pair)
                elif chunk == event_start:
                    in_event = True
                    b_event = True
                    # Process rest of start tag
                    while x < len(chunks) and '>' not in chunks[x]:
                        x = x+1
                elif chunk in time_start:
                    in_time = True
                    b_time = True
                    while x < len(chunks) and '>' not in chunks[x]:
                        x = x+1
                elif chunk in ignore_tags:
                    # Ignore the whole tag
                    while x < len(chunks) and '>' not in chunks[x]:
                        x = x+1
                    if x < len(chunks) and chunks[x] == 'LINEBREAK': # Ignore line breaks after ignored tags
                        x = x+1
                    keep_linebreaks = False # Discard blank lines after main text
                elif chunk not in ['<narr_timeml_simple>', '</narr_timeml_simple>']:
                    pair = (chunk, OO)
                    if chunk != 'LINEBREAK' or keep_linebreaks:
                        seq.append(pair)
            x = x+1
        if self.debug: print("seq: ", str(seq))
        return seq
