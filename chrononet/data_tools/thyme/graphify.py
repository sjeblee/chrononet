#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Create a graph from the temporal relations in the data

from lxml import etree
import argparse
import pydot
import networkx as nx
import time

# Local imports
from data_tools import data_util
from data_tools import temporal_util as tutil
from data_tools.temporal_util import Event, TimeRel


# Global variables
debug = False
unk = "UNK"
none_label = "NONE"
node_to_ids = {}
id_to_node = {}
before_types = ['BEFORE']
overlap_types = ['OVERLAP', 'CONTAINS']

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', '--in', action="store", dest="infile")
    argparser.add_argument('-r', '--relset', action="store", dest="relset")
    args = argparser.parse_args()

    if not (args.infile):
        print("usage: ./graphify.py --in [file_timeml.xml] --relset [binary/simple/exact]")
        exit()

    relset = 'exact'
    if args.relset:
        relset = args.relset
    get_graphs(args.infile, relset)

def get_graphs(filename, relation_set='exact'):
    if debug: print("creating graph: ", relation_set)
    starttime = time.time()
    # Get the xml from file
    tree = etree.parse(filename)
    root = tree.getroot()
    graphs = []
    ids = []
    records = 0
    dropped = 0

    for child in root:
        id_node = child.find("record_id")
        rec_id = id_node.text
        print("rec_id:", rec_id)
        records += 1
        node = child.find("narr_timeml_simple")
        #print("node is None:", str(node is None))
        try:
            node = etree.fromstring(etree.tostring(node, encoding='utf8').decode('utf8'))
            #node = etree.fromstring('<narr_timeml_simple>' + data_util.stringify_children(node).encode('utf8').decode('utf8') + '</narr_timeml_simple>')
        except etree.XMLSyntaxError as e:
            dropped += 1
            position = e.position[1]
            print("XMLSyntaxError at ", e.position, str(e), data_util.stringify_children(node)[position-5:position+5])
        if node is not None:
            graph = create_graph(node, relation_set, find_more_relations=False)
            digraph = create_digraph(node, relation_set)
            ids.append(rec_id)
            graphs.append(digraph)
            records += 1

            # Save an image of the graph
            pdot_graph = nx.drawing.nx_pydot.to_pydot(graph)
            pdot_graph.write_png(filename + "." + rec_id + ".graph.png")
            pdot_digraph = nx.drawing.nx_pydot.to_pydot(digraph)
            pdot_digraph.write_png(filename + "." + rec_id + ".digraph.png")

    # Print the first few feature vectors as a sanity check
    print("records:", str(records))
    print("dropped:", str(dropped))
    print("time:", str(time.time()-starttime), "s")
    return graphs


''' Create a graph from xml data
'''
def create_graph(xml_node, relation_set='exact', return_event_map=False, find_more_relations=True):
    events = xml_node.xpath("EVENT")
    times = xml_node.xpath("TIMEX3")
    tlinks = xml_node.xpath("TLINK")
    if debug: print("events: ", str(len(events)), " times: ", str(len(times)), " tlinks: ", str(len(tlinks)))

    # Create the graph
    graph = nx.DiGraph()
    event_map = {}
    timex_map = {}
    dct = None
    for event in events:
        if 'eid' not in event.attrib:
            print("no eid: ", etree.tostring(event))
        event_id = event.attrib['eid']
        graph.add_node(event_id, label=event.text)
        #print("adding event: ", event_id, event.text)
        event_map[event_id] = Event(event_id, event)

    for timex in times:
        time_id = timex.attrib['tid']
        #print("adding time: ", time_id, timex.text)
        graph.add_node(time_id, label=timex.text)
        timex_map[time_id] = timex
        if 'functionInDocument' in timex.attrib and timex.attrib['functionInDocument'] == 'CREATION_TIME':
            dct = timex
        elif dct is not None:
            timex.attrib['dct'] = tutil.time_value(dct)

    if debug:
        print("nodes:", str(len(graph.nodes())))
        print("events:", str(len(event_map.keys())))
        print("times:", str(len(timex_map.keys())))

    # Add event relations to the graph
    for tlink in tlinks:
        # event-event relations
        if 'relatedToEventID' in tlink.attrib and 'eventID' in tlink.attrib:
            event_id = tlink.attrib['eventID']
            event2_id = tlink.attrib['relatedToEventID']
            rel_type = tlink.attrib['relType']
            rel_type = tutil.map_rel_type(rel_type, relation_set)
            if event_id not in event_map or event2_id not in event_map:
                print("WARNING: missing event:", event_id)
                #event_map[event_id] = Event(event_id, None)
            #if event2_id not in event_map:
            #    print("adding event from tlink:", event2_id)
            #    event_map[event2_id] = Event(event2_id, None)
            # Flip after relations when adding to the graph
            id1, id2, rel_type = tutil.flip_relations(event_id, event2_id, rel_type)
            graph.add_edge(id1, id2, label=rel_type, priority=0)

    # Add time information to events, and add transitive time-event relations
    for tlink in tlinks:
        rel_type = tlink.attrib['relType']
        # event-time
        if 'relatedToTime' in tlink.attrib and 'eventID' in tlink.attrib:
            event_id = tlink.attrib['eventID']
            time_id = tlink.attrib['relatedToTime']
            # Flip relations if necessary
            id1, id2, rel_type = tutil.flip_relations(event_id, time_id, rel_type)
        # time-event
        elif 'timeID' in tlink.attrib and 'relatedToEventID' in tlink.attrib:
            event_id = tlink.attrib['relatedToEventID']
            time_id = tlink.attrib['timeID']
            id1, id2, rel_type = tutil.flip_relations(time_id, event_id, rel_type)
        # event-event relations handled in the next section
        else:
            continue
        #elif 'timeID' in tlink.attrib and 'relatedToTimeID' in tlink.attrib:
        #    time_id = tlink.attrib['timeID']
        #    time2_id = tlink.attrib['relatedToTimeID']
        #    id1, id2, rel_type = tutil.flip_relations(time_id, event_id, rel_type)
        graph.add_edge(id1, id2, label=rel_type, priority=0)

        # Add time value to the event
        e = event_map[event_id]
        if id2 == event_id:
            rel_type = tutil.reverse_relation(rel_type)
        add_time_to_event(e, timex_map[time_id], rel_type)

    # Resolve cycles in original graph
    graph, num_cycles = resolve_cycles(graph)
    print("num_cycles (original):", str(num_cycles))

    # Add additional relations that weren't in the original graph
    if find_more_relations:
        # Add transitive event relations (BEFORE only)
        for tlink in tlinks:
            # Transitive time-event before relations
            if 'relatedToTime' in tlink.attrib and 'eventID' in tlink.attrib:
                event_id = tlink.attrib['eventID']
                time_id = tlink.attrib['relatedToTime']
                rel_type = tlink.attrib['relType']
                if rel_type in before_types:
                    for other_event in graph.successors(time_id):
                        second_rel = graph.get_edge_data(time_id, other_event)['label']
                        if not graph.has_edge(event_id, other_event) and second_rel in before_types:
                            add_edge(graph, event_id, other_event, 'BEFORE', 1)

            # Propagate time information across contains relations
            elif 'timeID' in tlink.attrib and 'relatedToEventID' in tlink.attrib:
                event_id = tlink.attrib['relatedToEventID']
                time_id = tlink.attrib['timeID']
                rel_type = tlink.attrib['relType']
                if rel_type == 'CONTAINS':
                    for other_event in graph.successors(event_id):
                        second_rel = graph.get_edge_data(time_id, other_event)
                        if (not (graph.has_edge(time_id, other_event) or graph.has_edge(other_event, time_id))) and second_rel == 'CONTAINS':
                            add_time_to_event(other_event, timex_map[time_id], rel_type)

            # Transitive event-event relations
            elif 'eventID' in tlink.attrib and 'relatedToEventID' in tlink.attrib:
                # Handle transitive ends-on etc. relations
                event_id = tlink.attrib['eventID']
                event2_id = tlink.attrib['relatedToEventID']
                rel_type = tlink.attrib['relType']
                if rel_type == 'BEGINS-ON':
                    for nex in graph.successors(event2_id):
                        if graph.get_edge_data(event2_id, nex) == 'OVERLAP' and nex in timex_map:
                            if event_map[event_id].start == 'UNK':
                                event_map[event_id].start = tutil.time_value(timex_map[nex])
                    for prev in graph.predecessors(event2_id):
                        if graph.get_edge_data(prev, event2_id) == 'CONTAINS' and prev in timex_map:
                            if event_map[event_id].start == 'UNK':
                                event_map[event_id].start = tutil.time_value(timex_map[prev])
                elif rel_type == 'ENDS-ON':
                    for nex in graph.successors(event2_id):
                        if graph.get_edge_data(event2_id, nex) == 'OVERLAP' and nex in timex_map:
                            if event_map[event_id].end == 'UNK':
                                event_map[event_id].end = tutil.time_value(timex_map[nex])
                    for prev in graph.predecessors(event2_id):
                        if graph.get_edge_data(prev, event2_id) == 'CONTAINS' and prev in timex_map:
                            if event_map[event_id].end == 'UNK':
                                event_map[event_id].end = tutil.time_value(timex_map[prev])

        # Add event relations based on start and end times
        for eid in event_map.keys():
            for eid2 in event_map.keys():
                if eid != eid2 and (not graph.has_edge(eid, eid2)) and (not graph.has_edge(eid2, eid)):
                    event1 = event_map[eid]
                    event2 = event_map[eid2]
                    rel = tutil.compare_events(event1, event2)
                    #if debug: print("event-event rel:", rel)
                    if rel == 'BEFORE':
                        add_edge(graph, eid, eid2, 'BEFORE')
                        id1 = eid
                        id2 = eid2
                        if debug: print("found time BEFORE relation: ", str(event1), "|", str(event2))
                    elif rel == 'AFTER':
                        add_edge(graph, eid2, eid, 'BEFORE')
                        id1 = eid2
                        id2 = eid
                        if debug: print("found time BEFORE relation: ", str(event2), "|", str(event1))
                    #elif rel == 'OVERLAP':
                    #    graph.add_edge(eid, eid2, label='OVERLAP')
                    # Make sure we haven't just created a cycle
                    #    try:
                    #        nx.algorithms.cycles.find_cycle(graph, eid)
                        #if debug: print('not adding edge because it would create a cycle')
                    #        graph.remove_edge(eid, eid2)
                    #    except:
                    #        if debug: print("found time OVERLAP relation:", str(event1), "|", str(event2))
                    #        pass

        # Resolve cycles:
        graph, num_cycles = resolve_cycles(graph)
        print("num_cycles (after):", str(num_cycles))

    if return_event_map:
        print("event_map events:", str(len(event_map.keys())))
        return graph, event_map, timex_map
    else:
        return graph


''' Create the directed graph with groups of events
'''
def create_digraph(xml_node, relation_set='exact', return_elements=False):
    if debug: print('create_digraph')
    graph, event_map, timex_map = create_graph(xml_node, relation_set, True)

    # Reset node maps
    global node_to_ids, id_to_node
    node_to_ids = {}
    id_to_node = {}

    # Remove time nodes from the graph
    #print("Removing time nodes...")
    #for tid in timex_map:
    #    graph.remove_node(tid)
    if debug: print("Event nodes in original graph:", str(len(graph.nodes())))

    # Create a directed graph based on simple relation set
    # Nodes in this graph will be lists of events
    node_num = 0
    digraph = nx.DiGraph()
    for (node,neigh,data) in graph.edges(data=True):
        node_found = node in id_to_node
        neigh_found = neigh in id_to_node
        node_id = node_num
        if node_found:
            node_id = id_to_node[node]
        #elif neigh_found:
        #    node_id = id_to_node[neigh]
        else:
            node_id = node_num
            node_num += 1
        # Put overlapping events in the same node only if there's no other incoming links
        data_label = tutil.map_rel_type(data['label'], 'simple')
        if (data_label in overlap_types) and ((graph.in_degree(neigh) <= 1 and graph.out_degree(neigh) <= 1) or ((data['priority'] == 0) and not do_edges_conflict(graph, node, neigh))):
            update_node(digraph, node, node_id)
            update_node(digraph, neigh, node_id)
        else: #data['label'] == 'BEFORE':
            if not node_found:
                update_node(digraph, node, node_num)
                node_num += 1
            if not neigh_found:
                update_node(digraph, neigh, node_num)
                node_num += 1
            if node not in id_to_node:
                print('id_to_node:', str(id_to_node))
            startnode = id_to_node[node]
            endnode = id_to_node[neigh]
            if startnode != endnode:
                add_edge(digraph, startnode, endnode, data_label)

    # Remove timex ids from the digraph
    for item in id_to_node.keys():
        if item[0] == 't':
            index = id_to_node[item]
            if index in node_to_ids:
                node_to_ids[index].remove(item)
            id_to_node[item] = None

    # Delete empty nodes
    for node in list(digraph.nodes()):
        if len(node_to_ids[node]) == 0:
            digraph.remove_node(node)

    # TODO: what about nodes with no edges?
    for node in list(graph.nodes):
        if node[0] == 'e' and node not in id_to_node:
            print("WARNING: event with no edges dropped:", str(event_map[node]))

    # Combine nodes via overlap links (as long as the other links don't conflict
    for (node1, node2, label) in digraph.edges(data=True):
        if label == 'OVERLAP':
            if do_edges_conflict(digraph, node1, node2):
                continue
            else:
                # If the checks have passed, merge the nodes
                combine_nodes(digraph, node1, node2)
                print("combined overlapping nodes", str(node1), str(node2))

    # Combine nodes with the same in and out relations
    nid = 0
    node_list = list(digraph.nodes(data=True))
    while nid < len(node_list):
        if node_list[nid] is not None:
            node, data = node_list[nid]
            for nid2 in range(len(node_list)):
                node2 = None
                if node_list[nid2] is not None:
                    node2, data2 = node_list[nid2]
                if (node2 is not None) and (node2 != node):
                    # If the two nodes have the same relations, combine them
                    same = True # Do node and node2 have the same relations?
                    for nextnode in digraph.successors(node):
                        edge_data = digraph[node][nextnode]
                        if 'label' not in edge_data:
                            print('no label:', str(node), str(nextnode), str(edge_data))
                        rel = edge_data['label']
                        if (not digraph.has_edge(node2, nextnode)) or (digraph[node2][nextnode]['label'] != rel):
                            same = False
                    for prevnode in digraph.predecessors(node):
                        edge_data = digraph[prevnode][node]
                        if 'label' not in edge_data:
                            print('no label:', str(node), str(nextnode), str(edge_data))
                        rel = edge_data['label']
                        if (not digraph.has_edge(prevnode, node2)) or (digraph[prevnode][node2]['label'] != rel):
                            same = False
                    if same:
                        combine_nodes(digraph, node, node2)
                        # Remove the combined node, but keep nid the same
                        node_list[nid2] = None
        # Move to the next node
        nid += 1

    # Check for cycles
    for cycle in nx.simple_cycles(digraph):
        print("digraph cycle found!", str(cycle))
        break

    if return_elements:
        num_elements = 0
        for nid in node_to_ids:
            num_elements += len(node_to_ids[nid])
        if debug: print("num_elements:", str(num_elements))
        node_to_event = {}
        for nid in node_to_ids:
            #print("nid:", nid)
            node_to_event[nid] = []
            for eid in node_to_ids[nid]:
                #print("- eid:", eid)
                node_to_event[nid].append(event_map[eid])
        return digraph, node_to_event
    else:
        return digraph

#------------ Helper Functions ------------


''' Add an edge to the graph as long as it doesn't create a cycle
'''
def add_edge(graph, node1, node2, rel, priority=2):
    # Do nothing if an edge already exists
    if graph.has_edge(node1, node2):
        return

    graph.add_edge(node1, node2, label=rel, priority=priority)
    try:
        nx.algorithms.cycles.find_cycle(graph, node1)
        if debug: print('not adding edge because it would create a cycle')
        graph.remove_edge(node1, node2)
    except nx.exception.NetworkXNoCycle:
        if debug: print("added relation to graph:", rel, str(node1), "|", str(node2))
        pass


''' Add time information to an event object
    event: Event object
    time: xml element representing a TIMEX
    rel_type: the relation type (presumably from event to time)
'''
def add_time_to_event(event, time, rel_type):
    tag = ""
    tval = tutil.time_value(time)
    if rel_type == 'BEGINS-ON':
        if event.start == unk:
            event.start = tval
            event.start_type = TimeRel.ON
            tag = "start"
    elif rel_type in 'ENDS-ON':
        if event.end == unk:
            event.end = tval
            event.end_type = TimeRel.ON
            tag = "end"
    elif rel_type == 'BEFORE':
        if event.start == unk:
            event.start = tval
            event.start_type = TimeRel.BEFORE
            tag = "start"
        if event.end == unk:
            event.end = tval
            event.end_type = TimeRel.BEFORE
            tag += "/end"
    elif rel_type == 'AFTER':
        if event.start == unk:
            event.start = tval
            event.start_type = TimeRel.AFTER
            tag = "start"
        if event.end == unk:
            event.end = tval
            event.end_type = TimeRel.AFTER
            tag += "/end"
    elif rel_type == 'BEFORE/OVERLAP':
        if event.start == unk:
            event.start = tval
            event.start_type = TimeRel.BEFORE
        if event.overlap == unk:
            event.overlap = tval
            tag = "start/overlap"
    elif rel_type in overlap_types:
        #if event.start == unk and event.end == unk:
        if event.overlap == unk:
            event.overlap = tval
            tag = "overlap"
        # Update start and end information based on overlap
        if event.start != unk:
            rel = tutil.compare_times(event.start, event.overlap)
            if rel == 'AFTER' and event.start_type == TimeRel.BEFORE:
                event.start = tval
                event.start_type = TimeRel.BEFORE
        # TODO: update end information?
    else:
        if debug: print("unhandled rel type:", rel_type)
    if debug: print("added time to event:", tag, tval, "-->", str(event))


''' Check to see if two nodes have the same relation edges
    graph: the directed graph
'''
def check_edges(node1, node2, graph):
    # Check that they have the same relations
    for pre in graph.predecessors(node1):
        if not graph.has_edge(pre, node2):
            return False
        rel = graph.get_edge_data(pre, node1)
        if not graph.get_edge_data(pre, node2) == rel:
            return False
    for nex in graph.successors(node1):
        if not graph.has_edge(node2, nex):
            return False
        rel = graph.get_edge_data(node1, nex)
        if not graph.get_edge_data(node2, nex) == rel:
            return False
    return True


''' Combine two nodes
'''
def combine_nodes(digraph, node1, node2):
    #print("combine_nodes", str(node1), str(node2))
    for eid in node_to_ids[node2]:
        update_node(digraph, eid, node1)
    digraph.remove_node(node2)


''' Check if the edges of two nodes conflict
'''
def do_edges_conflict(graph, node1, node2):
    for nextnode in graph.successors(node1):
        if nextnode != node2:
            if nextnode in graph.predecessors(node2):
                return True
            elif nextnode in graph.successors(node2):
                if graph.edges[node1, nextnode]['label'] != graph.edges[node2, nextnode]['label']:
                    return True
    for prevnode in graph.predecessors(node1):
        if prevnode != node2:
            if prevnode in graph.successors(node2):
                return True
            elif prevnode in graph.predecessors(node2):
                if graph.edges[prevnode, node1]['label'] != graph.edges[prevnode, node2]['label']:
                    return True
    return False


''' Resolve cycles by dropping overlap links
'''
def resolve_cycles(graph):
    count = 0
    for cycle in nx.simple_cycles(graph):
        count += 1
        fixed = False
        print("cycle found!", str(cycle))
        try:
            edges = nx.algorithms.cycles.find_cycle(graph, source=cycle)
        except:
            fixed = True
            print("cycle already fixed")
            continue
        # First try reversing links
        for from_node, to_node in edges:
            edge_data = graph.get_edge_data(from_node, to_node)
            if edge_data['label'] == 'OVERLAP':
                graph.remove_edge(from_node, to_node)
                # Try just reversing the link
                graph.add_edge(to_node, from_node, label=tutil.reverse_relation(edge_data['label']), priority=edge_data['priority'])
                # Check if the cycle is still there
                try:
                    nx.algorithms.cycles.find_cycle(graph, source=from_node)
                    #graph.remove_edge(to_node, from_node)
                    #print("removed OVERLAP link")
                    #nx.algorithms.cycles.find_cycle(graph, source=from_node)
                    fixed = False
                except:
                    fixed = True
                    print("cycle fixed by reversing OVERLAP links")
                    pass
        if not fixed: # Try reversing other links
            for from_node, to_node in edges:
                if fixed:
                    break
                edge_data = graph.get_edge_data(from_node, to_node)
                if edge_data is not None and (edge_data['label'] == 'BEGINS-ON' or edge_data['label'] == 'ENDS-ON'):
                    graph.remove_edge(from_node, to_node)
                    # Try just reversing the link
                    graph.add_edge(to_node, from_node, label=tutil.reverse_relation(edge_data['label']), priority=edge_data['priority'])
                    # Check if the cycle is still there
                    try:
                        nx.algorithms.cycles.find_cycle(graph, source=from_node)
                        fixed = False
                    except:
                        fixed = True
                        print("cycle fixed by reversing links")
                        pass
        if not fixed: # Try removing overlap links
            for from_node, to_node in edges:
                if fixed:
                    break
                edge_data = graph.get_edge_data(from_node, to_node)
                if edge_data is not None and (edge_data['label'] in overlap_types):
                    graph.remove_edge(from_node, to_node)
                    # Check if the cycle is still there
                    try:
                        nx.algorithms.cycles.find_cycle(graph, source=from_node)
                        fixed = False
                    except:
                        fixed = True
                        print("cycle fixed by removing OVERLAP links")
                        pass
        if not fixed:
            print("WARNING: cycle could not be resolved!")
    return graph, count


def update_node(digraph, eid, node_num):
    #print("update node:", eid, str(node_num))
    global node_to_ids, id_to_node
    # Make sure the node is in the graph
    if node_num not in node_to_ids:
        digraph.add_node(node_num)
        node_to_ids[node_num] = []
    # Collapse two nodes if necessary
    if eid in id_to_node:
        # If the event is already in the right node, do nothing
        if id_to_node[eid] == node_num:
            return
        # Move other members of the old node to the new node
        old_node = id_to_node[eid]
        for eidn in node_to_ids[old_node]:
            node_to_ids[node_num].append(eidn)
            id_to_node[eidn] = node_num
        node_to_ids[old_node] = []
        # Move edges from old node to new node
        for eidn in list(digraph.predecessors(old_node)):
            edge_data = digraph[eidn][old_node]
            edge_label = edge_data['label']
            digraph.remove_edge(eidn, old_node)
            if eidn != node_num:
                add_edge(digraph, eidn, node_num, edge_label, edge_data['priority'])
        for eidn in list(digraph.successors(old_node)):
            edge_data = digraph[old_node][eidn]
            edge_label = edge_data['label']
            digraph.remove_edge(old_node, eidn)
            if eidn != node_num:
                add_edge(digraph, node_num, eidn, edge_label, edge_data['priority'])
    # Add the eid to the correct node
    id_to_node[eid] = node_num
    if eid not in node_to_ids[node_num]:
        node_to_ids[node_num].append(eid)


if __name__ == "__main__":main()
