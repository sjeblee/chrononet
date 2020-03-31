import numpy
import pandas
from lxml import etree

# Local imports
import data_util
from adapters import data_adapter_va, data_adapter_thyme, data_adapter_verilogue

def main():
    va_train_file = '/h/sjeblee/research/data/va/test_all_cat_A-HJK_rank.xml'
    va_test_file = '/h/sjeblee/research/data/va/test_all_cat_I_rankSJ.xml'
    thyme_train_file = '/h/sjeblee/research/data/thyme/train_dctrel_list_combined.xml'
    thyme_test_file = '/h/sjeblee/research/data/thyme/devtest_dctrel_list.xml'
    ver_train_file = '/h/sjeblee/research/data/verilogue/train_verilogue_165.xml'
    ver_test_file = '/h/sjeblee/research/data/verilogue/test_verilogue_165.xml'

    print('STATISTICS: Verbal autopsy')
    statistics(va_train_file, va_test_file, data_adapter_va.DataAdapterVA(debug=False))
    print('STATISTICS: THYME')
    statistics(thyme_train_file, thyme_test_file, data_adapter_thyme.DataAdapterThyme(debug=False))
    print('STATISTICS: Verilogue')
    statistics(ver_train_file, ver_test_file, data_adapter_verilogue.DataAdapterVerilogue(debug=False), verilogue=True)

def statistics(train_file, test_file, adapter, verilogue=False):
    train_df = adapter.load_data(train_file)
    test_df = adapter.load_data(test_file)
    combined_df = pandas.concat((train_df, test_df), ignore_index=True)
    narr_len_train = avg_narr_len(train_df, verilogue)
    narr_len_test = avg_narr_len(test_df, verilogue)
    narr_len_total = avg_narr_len(combined_df, verilogue)
    print('Num docs: train:', len(train_df), 'test:', len(test_df), 'total:', len(combined_df))
    print('Avg narr len: train:', narr_len_train, 'test:', narr_len_test, 'total:', narr_len_total)

    # num of events, percentage w/ associated time phrases
    #if not verilogue:
    events, event_avg, percent_events_time, timexes, timexes_avg = event_stats(combined_df, verilogue)
    print('Total events:', events, 'events per doc:', event_avg, 'fraction of events w/ timex:', percent_events_time)
    print('Total timexes:', timexes, 'timexes per doc:', timexes_avg)

def avg_narr_len(df, verilogue=False):
    lengths = []
    for i, row in df.iterrows():
        narr = row['text']
        if verilogue:
            turns = etree.fromstring(narr.decode('utf8'))
            narr = ''
            for utt in turns:
                narr += utt.text + '\n'
        if narr is None:
            num_words = 0
        else:
            #print('narr:', narr)
            num_words = len(data_util.split_words(narr))
        lengths.append(num_words)

    avg_len = numpy.average(numpy.array(lengths))
    return avg_len

def event_stats(df, verilogue=False):
    event_nums = []
    events_with_time = []
    timex_nums = []
    for i, row, in df.iterrows():
        num_events = 0
        num_timexes = 0
        num_events_with_time = 0

        if verilogue:
            elem = row['events']
            time_elem = row['tags']
            for child in elem:
                num_events += 1
                if 'relatedToTime' in child.features:
                    num_events_with_time += 1
            for time_child in time_elem:
                if time_child.tag == 'TIMEX3':
                    num_timexes += 1
        else:
            elem = data_util.load_xml_tags(row['events'], decode=False, unwrap=True)
            time_elem = data_util.load_xml_tags(row['tags'], decode=False, unwrap=True)
            #time_elem = etree.fromstring(row['tags'])
            #print('time_elem:', etree.tostring(time_elem))
            for elem_child in elem:
                if elem_child.tag == 'EVENT':
                    num_events += 1
                    if 'relatedToTime' in elem_child.attrib:
                        num_events_with_time += 1
            for elemt in time_elem:
                #print('time child:', etree.tostring(elemt).decode('utf8'))
                print('time child tag:', elemt.tag)
                if elemt.tag == 'TIMEX3':
                    #print('-- found timex!')
                    num_timexes += 1

        event_nums.append(num_events)
        events_with_time.append(num_events_with_time)
        timex_nums.append(num_timexes)

    event_nums = numpy.array(event_nums)
    timex_nums = numpy.array(timex_nums)
    events_with_time = numpy.array(events_with_time)
    total_events = numpy.sum(event_nums)
    avg_events = numpy.average(event_nums)
    total_events_time = numpy.sum(events_with_time)
    #avg_events_time = numpy.average(events_with_time)
    percent_events_time = float(total_events_time)/float(total_events)
    total_timexes = numpy.sum(timex_nums)
    avg_timexes = numpy.average(timex_nums)

    return total_events, avg_events, percent_events_time, total_timexes, avg_timexes


if __name__ == "__main__": main()
