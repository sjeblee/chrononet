#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Create a temporally ordered list from a graph of events and times

#from xml.sax.saxutils import unescape
import argparse
import networkx as nx
import time
from lxml import etree

# Local imports
from data_tools.thyme import graphify
from data_tools import temporal_util as tutil
from data_tools import data_util

# Global variables
debug = True
unk = "UNK"
none_label = "NONE"
list_name = "event_list"

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', '--in', action="store", dest="infile")
    argparser.add_argument('-r', '--relset', action="store", dest="relset")
    argparser.add_argument('-o', '--out', action="store", dest="outfile")
    args = argparser.parse_args()

    if not (args.infile):
        print("usage: ./listify.py --in [file_timeml.xml] --out [file.xml] --relset [binary/simple/exact]")
        exit()

    relset = 'exact'
    if args.relset:
        relset = args.relset
    get_lists(args.infile, args.outfile, relset)

def get_lists(filename, outfile, relation_set='exact'):
    #print("creating graph: ", relation_set)
    starttime = time.time()
    # Get the xml from file
    tree = etree.parse(filename)
    root = tree.getroot()
    timelines = []
    ids = []
    records = 0
    dropped = 0

    for child in root:
        id_node = child.find("record_id")
        rec_id = id_node.text
        #print("rec_id:", rec_id)
        records += 1
        node = child.find("narr_timeml_simple")
        list_node = child.find(list_name)
        #print("node is None:", str(node is None))
        try:
            node = etree.fromstring('<narr_timeml_simple>' + data_util.stringify_children(node).encode('utf8').decode('utf8') + '</narr_timeml_simple>')
        except etree.XMLSyntaxError as e:
            dropped += 1
            position = e.position[1]
            print("XMLSyntaxError at ", e.position, str(e), data_util.stringify_children(node)[position-5:position+5])
        if node is not None: # and list_node is None: # Skip records that already have a list
            timeline = listify(node, relation_set)
            ids.append(rec_id)
            timelines.append(timeline)
            if list_node is not None:
                child.remove(list_node)
            timeline_node = etree.SubElement(child, list_name)
            timeline_to_xml(timeline, timeline_node)
            # Write the file after every record in case one of them fails
            tree.write(outfile)

    print("records:", str(records))
    print("dropped:", str(dropped))
    tutil.print_time(time.time()-starttime)
    return timelines


''' Create a graph from xml data
'''
def listify(xml_node, relation_set='exact'):
    events = xml_node.xpath("EVENT")
    times = xml_node.xpath("TIMEX3")
    tlinks = xml_node.xpath("TLINK")
    #if debug: print("events: ", str(len(events)), " times: ", str(len(times)), " tlinks: ", str(len(tlinks)))

    # Create timex map
    timex_map = {}
    for tm in times:
        timex_map[tm.attrib['tid']] = tm

    # Create the graph
    graph, node_to_events = graphify.create_digraph(xml_node, relation_set, return_elements=True)

    #if debug: print("graph nodes:", str(len(graph.nodes())))
    # Create a binned list based on graph topological order
    timeline = []
    for node in nx.algorithms.dag.topological_sort(graph):
        timeline.append(node_to_events[node])

    # Print the timeline and create maps for verification
    event_map = {}
    id_to_bin = {}
    for x in range(len(timeline)):
        if debug: print("---", str(x), "---")
        for item in timeline[x]:
            event_map[item.eid] = item
            id_to_bin[item.eid] = x
            if debug: print_event(item)# etree.tostring(item.element))

    # Verify the timeline
    #verify_list(timeline, tlinks, event_map, timex_map, id_to_bin)

    return timeline

def print_event(item):
    print(str(item))

def timeline_to_string(timeline):
    string = ""
    for x in range(len(timeline)):
        print("---", str(x), "---")
        for item in timeline[x]:
            string += str(item) + "\n"

def timeline_to_xml(timeline, xml_parent):
    for x in range(len(timeline)):
        for item in timeline[x]:
            event_node = item.element
            event_node.attrib['rank'] = str(x)
            event_node.attrib['start_time'] = item.start
            event_node.attrib['start_type'] = str(item.start_type)
            event_node.attrib['end_time'] = item.end
            event_node.attrib['end_type'] = str(item.end_type)
            event_node.attrib['overlap_time'] = item.overlap
            xml_parent.append(event_node)


''' Make sure all the original relations have been preserved in the final list
'''
def verify_list(timeline, tlinks, event_map, timex_map, id_to_bin):
    print('verifying list...')

    verify_links(tlinks, id_to_bin)
    verify_time_info(tlinks, event_map, timex_map)


def verify_links(tlinks, id_to_bin):
    # Verify that all pairwise relations are still valid
    for tlink in tlinks:
        if 'eventID' in tlink.attrib and 'relatedToEventID' in tlink.attrib:
            eid1 = tlink.attrib['eventID']
            eid2 = tlink.attrib['relatedToEventID']
            rel_type = tlink.attrib['relType']
            if rel_type == 'BEFORE':
                if not (id_to_bin[eid1] < id_to_bin[eid2]):
                    print('VERIFY FAILED:', eid1, 'should be before', eid2)
            elif rel_type == 'AFTER':
                if not (id_to_bin[eid1] > id_to_bin[eid2]):
                    print('VERIFY FAILED:', eid1, 'should be after', eid2)
            elif rel_type == 'OVERLAP':
                if not (id_to_bin[eid1] == id_to_bin[eid2]):
                    print('VERIFY FAILED:', eid1, 'should overlap', eid2)


def verify_time_info(tlinks, event_map, timex_map):
    before_unk = ['BEFORE', 'UNK']
    for tlink in tlinks:
        reverse = False
        if 'eventID' in tlink.attrib and 'relatedToTimeID' in tlink.attrib:
            eid = tlink.attrib['eventID']
            tid = tlink.attrib['relatedToTimeID']
        elif 'timeID' in tlink.attrib and 'relatedToEventID' in tlink.attrib:
            eid = tlink.attrib['relatedToEventID']
            tid = tlink.attrib['timeID']
            reverse = True
        else:
            continue # Not an event-time relation
        rel_type = tlink.attrib['relType']
        if eid not in event_map:
            print("VERIFY FAILED: could not find event", eid)
        else:
            event = event_map[eid]
            tval = tutil.time_value(timex_map[tid])
            # TODO: check time info
            if rel_type == 'OVERLAP':
                # TODO: check if this is a link we dropped due to a cycle
                if event.overlap != tval:
                    print("VERIFY FAILED: time overlap doesn't match:", tval, str(event))
                elif rel_type == 'BEFORE' or (reverse and rel_type == 'AFTER'):
                    # Check all times
                    if not (tutil.compare_event_times(event, tval) in before_unk):
                        print("VERIFY FAILED: event before time doesn't match:", tval, str(event))
                #elif rel_type == 'BEFORE/OVERLAP':
                # TODO


if __name__ == "__main__": main()
