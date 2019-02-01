#!/usr/bin/python3
# Temporal utility functions
import sys
sys.path.append('/u/sjeblee/research/git/dateparser')

from dateparser import parse
from enum import Enum

none_label = "NONE"
unk = "UNK"

class TimeRel(Enum):
    BEFORE = 'B'
    AFTER = 'A'
    ON = 'ON'

class Event:
    start = unk
    start_type = unk
    overlap = unk
    overlap_type = TimeRel.ON
    end = unk
    end_type = unk

    def __init__(self, eid, element):
        self.eid = eid
        self.element = element

    def __str__(self):
        return self.eid + ": " + self.element.text + "\n\t" + str(self.start_type) + " " + self.start + " - " + self.overlap + " - " + str(self.end_type) + " " + self.end

    def equals(self, other):
        if self.eid == other.eid:
            return True
        else:
            return False
        #if not ((self.element.text == other.element.text) and (self.start == other.start) and (self.end == other.end)):
        #    return False
        # Check attributes
        #for key in self.element.attrib:
        #    if (key not in other.element.attrib) or (other.element.attrib[key] != self.element.attrib[key]):
        #        return False
        #return True

    def times_unknown(self):
        if self.start == unk and self.end == unk and self.overlap == unk:
            return True
        else:
            return False

def flip_relations(id1, id2, rel_type):
    if rel_type == 'AFTER':
        return id2, id1, 'BEFORE'
    elif rel_type == 'TERMINATES':
        return id2, id1, 'ENDS-ON'
    elif rel_type == 'INITIATES':
        return id2, id1, 'BEGINS-ON'
    else:
        return id1, id2, rel_type

def reverse_relation(rel):
    if rel == 'BEFORE':
        return 'AFTER'
    if rel == 'AFTER':
        return 'BEFORE'
    if rel == 'BEFORE/OVERLAP':
        return 'OVERLAP/AFTER'
    if rel == 'TERMINATES':
        return 'ENDS-ON'
    if rel == 'ENDS-ON':
        return 'TERMINATES'
    if rel == 'INITIATES':
        return 'BEGINS-ON'
    if rel == 'BEGINS-ON':
        return 'INITIATES'
    else:
        return rel

def map_rel_type(rel_type, relation_set):
    rel_type = rel_type.upper()
    if relation_set == 'exact':
        return rel_type
    if relation_set == 'binary':
        if rel_type == none_label:
            return 0
        else:
            return 1
    elif relation_set == 'simple':
        if rel_type == none_label:
            return rel_type
        elif rel_type in ['BEFORE', 'IBEFORE', 'BEFORE/OVERLAP', 'ENDS-ON']:
            return 'BEFORE'
        elif rel_type in ['AFTER', 'IAFTER']:
            return 'AFTER'
        else: # BEGINS-ON, INITIALIZES, CONTAINS
            return 'OVERLAP'
    elif relation_set == 'contains':
        if rel_type == 'CONTAINS':
            return rel_type
        else:
            return none_label

def print_time(t):
    unit = "s"
    if t>60:
        t = t/60
        unit = "mins"
    if t>60:
        t = t/60
        unit = "hours"
    print("time:", str(t), unit)
    return str(t) + " " + unit

# ------------- Datetime functions ----------------

''' Check if two time types are comparable, i.e. if the times have the 'BEFORE' relation, is this valid?
'''
def are_comparable(timerel1, timerel2):
    if timerel1 == timerel2:
        return True
    elif (timerel1 == TimeRel.BEFORE and timerel2 == TimeRel.ON) or (timerel1 == TimeRel.ON and timerel2 == TimeRel.AFTER):
        return True
    return False


''' Compare time types, assuming that the associated times are ordered
'''
def compare_types(timerel1, timerel2):
    if ((timerel1 == TimeRel.BEFORE or timerel1 == TimeRel.ON) and timerel2 == TimeRel.AFTER) or ((timerel1 == TimeRel.BEFORE) and (timerel2 == TimeRel.ON or timerel2 == TimeRel.AFTER)):
        return TimeRel.BEFORE
    else:
        return unk

''' Compare two events and see if one is before the other
'''
def compare_events(event1, event2):
    # If one of the events has no time info, return UNK
    if event1.times_unknown() or event2.times_unknown():
        return unk

    # Compare endpoints
    endpoint_rel = compare_times(event1.end, event2.start)
    if endpoint_rel == 'OVERLAP' and compare_types(event1.end_type, event2.start_type) == TimeRel.BEFORE:
        return 'BEFORE'
    if endpoint_rel == 'BEFORE' and are_comparable(event1.end_type, event2.start_type):
            #print("end before start")
            return 'BEFORE'
    endpoint_rel = compare_times(event2.end, event1.start)
    if endpoint_rel == 'OVERLAP' and compare_types(event2.end_type, event1.start_type) == TimeRel.BEFORE:
        return 'AFTER'
    if endpoint_rel == 'BEFORE' and are_comparable(event2.end_type, event1.start_type):
            #print("end before start inverse")
            return 'AFTER'

    # If all start and end are unk, or only one of the starts and ends are unk
    if (event1.start == unk or event2.start == unk or (event1.start == event2.start and event1.start_type == event2.start_type)) and (event1.end == unk or event2.end == unk or (event1.end == event2.end and event1.end_type == event2.end_type)):
        if event1.overlap != unk and event2.overlap != unk:
            #print("comparing overlap...")
            if event1.overlap == event2.overlap:
                return 'OVERLAP'
            else:
                return compare_times(event1.overlap, event2.overlap)
    # If start and end times are equal (including types)
    #if (event1.start == event2.start) and (event1.end == event2.end) and (event1.start != unk) and (event1.end != unk) and (event2.start != unk) and (event2.end != unk) and (event1.start_type == event2.start_type) and (event1.end_type == event2.end_type):
     #   if event1.overlap != unk and event2.overlap != unk:
     #       #print("comparing overlap...")
     #       if event1.overlap == event2.overlap:
     #           return 'OVERLAP'
     #       else:
     #           return compare_times(event1.overlap, event2.overlap)

     # Handle before/overlap
    if event1.start == unk and event1.overlap != unk and event2.start != unk and compare_times(event1.overlap, event2.start) == 'BEFORE' and are_comparable(event1.overlap_type, event2.start_type):
        #print("overlap before start")
        return 'BEFORE'
    if event2.start == unk and event2.overlap != unk and event1.start != unk and compare_times(event2.overlap, event1.start) == 'BEFORE' and are_comparable(event2.overlap_type, event1.start_type):
        #print("overlap before start inverse")
        return 'AFTER'

    # Handle events with ends before, and another event with overlap but no start info
    if event1.end_type == TimeRel.BEFORE and event2.start == unk and event2.overlap != unk:
        if event1.end == event2.overlap or (compare_times(event1.end, event2.overlap) == 'BEFORE'):
            return 'BEFORE'
    if event1.end == unk and event1.overlap != unk and event2.start != unk and event2.start_type == TimeRel.AFTER:
        if event1.overlap == event2.start or (compare_times(event1.overlap, event2.start) == 'BEFORE'):
            return 'BEFORE'
    return unk


''' Compare event times
    event: the event to check the times for
    time: the time to check against
'''
def compare_event_times(event, time):
    if compare_times(event.overlap, time) == 'OVERLAP':
        return 'OVERLAP'
    elif (event.start != unk and compare_times(event.start, time) == 'BEFORE') and (event.end != unk and compare_times(event.end, time) == 'BEFORE') and (event.start_type != TimeRel.AFTER) and (event.end_type != TimeRel.AFTER):
        return 'BEFORE'
    elif (event.end_type == TimeRel.ON and compare_times(event.end, time) == 'BEFORE') or (event.end == unk and event.start_type == TimeRel.ON and compare_times(event.start, time) == 'BEFORE'):
        if event.overlap == unk or (compare_times(event.overlap, time) == 'BEFORE'):
            return 'BEFORE'
    # TODO: check for BEFORE/OVERLAP
    return unk


''' Compare times
'''
def compare_times(time1, time2):
    if time1 == unk or time2 == unk or (len(time1) != 19) or (len(time2) != 19):
        return unk
    if time1 == time2:
        return 'OVERLAP'
    if is_datetime(time1) and is_datetime(time2):
        if time1 < time2:
            return 'BEFORE'
        elif time2 < time1:
            return 'AFTER'
        elif time1 == time2:
            return 'OVERLAP'
    return unk

def is_datetime(text):
    if parse(text) is None:
        return False
    else:
        return True


''' Get the actual time value of a TIMEX tag
    Return the ISO string if parseable, otherwise just the string
    timex_element: the xml element representing a TIMEX
    dct: a datetime object representing the DCT
'''
def time_value(timex_element):
    time_string = timex_element.text
    dct = None
    if 'dct' in timex_element.attrib:
        dct_string = timex_element.attrib['dct']
        if is_datetime(dct_string):
            dct = to_date(dct_string)
    if is_datetime(time_string):
        iso_datetime = to_date(time_string, dct)
        return iso_datetime.isoformat()
    else:
        return timex_element.text

'''
Convert a text representing a date to a date object
text: the string to convert
dct: the DCT as a datetime object
'''
def to_date(text, dct=None):
    parser_settings = {}
    if dct is not None:
        parser_settings['RELATIVE_BASE'] = dct
    return parse(text, settings=parser_settings)

# ---------- Event functions ------------------

def are_pairs_equal(pair1, pair2):
    eid1a = pair1[0].attrib['eid']
    eid1b = pair1[1].attrib['eid']
    eid2a = pair2[0].attrib['eid']
    eid2b = pair2[1].attrib['eid']
    return (eid1a == eid2a and eid1b == eid2b)
