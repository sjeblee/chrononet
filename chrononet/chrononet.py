#!/usr/bin/python3
# Main Chrononet script

# Imports
import ast
import argparse
import configparser
import os
import pandas

# Local imports
from data_tools.adapters import data_adapter_thyme, data_adapter_va
from data_tools import data_util
from evaluation import eval_metrics, ordering_metrics
from feature_extractors import numpyer, relations, syntactic
from models.sequence.crf import CRFfactory
from models.ordering.random_order import RandomOrderFactory, MentionOrderFactory

# SETUP
fe_map = {'relations': relations.extract_relations, 'syntactic': syntactic.sent_features, 'none': numpyer.dummy_function}
model_map = {'crf': CRFfactory, 'random': RandomOrderFactory, 'mention': MentionOrderFactory}
metric_map = {'p': eval_metrics.precision, 'r': eval_metrics.recall, 'f1': eval_metrics.f1,
            'mse': ordering_metrics.rank_mse, 'poa': ordering_metrics.rank_pairwise_accuracy}
debug = True

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-c', '--config', action="store", dest="configfile")
    args = argparser.parse_args()

    if not (args.configfile):
        print('usage: ./chrononet.py --config [config.ini]')
        exit()

    config = configparser.ConfigParser()
    config.read(args.configfile)

    # INITIALIZATION
    # Parse config file
    data_config = config['data']
    train_dataset = data_config['train_dataset']
    test_dataset = data_config['test_dataset']
    trainfile = data_config['trainfile']
    testfile = data_config['testfile']
    outdir = data_config['output_dir']
    save_intermediate = ast.literal_eval(data_config['save_intermediate_files'])
    score_report = []

    # Create output directory
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    inter_prefix = '_chrono_'

    # PREPROCESSING
    train_data_adapter = get_data_adapter(train_dataset)
    test_data_adapter = get_data_adapter(test_dataset)
    print('train data_adapter:', type(train_data_adapter), 'test_data_adapter:', type(test_data_adapter))
    train_filename = os.path.join(outdir, inter_prefix + 'train_df.csv')
    test_filename = os.path.join(outdir, inter_prefix + 'test_df.csv')
    train_df = None
    test_df = None
    if os.path.exists(train_filename):
        train_df = pandas.read_csv(train_filename)
    if os.path.exists(test_filename):
        test_df = pandas.read_csv(test_filename)
    if train_df is None:
        train_df = train_data_adapter.load_data(trainfile)
    if test_df is None:
        test_df = test_data_adapter.load_data(testfile)
    if save_intermediate:
        if debug:
            print('Saving preprocessed df...')
        train_df.to_csv(train_filename)
        test_df.to_csv(test_filename)
    orig_train_df = train_df.copy()
    orig_test_df = test_df.copy()

    # SEQUENCE TAGGER STAGE
    train_filename = os.path.join(outdir, inter_prefix + 'train_df_seq.csv')
    test_filename = os.path.join(outdir, inter_prefix + 'test_df_seq.csv')
    if os.path.exists(train_filename) and os.path.exists(test_filename):
        seq_train_df = pandas.read_csv(train_filename)
        seq_test_df = pandas.read_csv(test_filename)
    else:
        seq_train_df = train_data_adapter.to_seq(train_df, split_sents=True)
        seq_test_df = test_data_adapter.to_seq(test_df, split_sents=True)
    if save_intermediate:
        if debug:
            print('Saving feature df...')
        seq_train_df.to_csv(train_filename)
        seq_test_df.to_csv(test_filename)
    scores, train_df, test_df = run_stage('sequence', config, train_data_adapter, test_data_adapter, seq_train_df, seq_test_df)
    score_report.append(scores)

    train_df = train_data_adapter.to_doc(train_df)
    test_df = test_data_adapter.to_doc(test_df)

    # THYME dataset eval script
    #if dataset == 'thyme':
    #    data_adapter.output()

    # TEMPORAL ORDERING STAGE
    if 'ordering' in config:
        order_train_df = train_data_adapter.to_order(train_df, orig_train_df)
        order_test_df = test_data_adapter.to_order(test_df, orig_test_df)
        scores, train_df, test_df = run_stage('ordering', config, train_data_adapter, test_data_adapter, order_train_df, order_test_df, doc_level=True)
        score_report.append(scores)

    # OUTPUT
    if debug:
        print('Writing output files...')
    test_data_adapter.write_output(test_df, outdir)

    # PRINT the scores
    for line in score_report:
        print(str(line))


''' Check the format of the config file and make sure at least one of the sections is there.
'''
def check_config(config):
    if 'sequence' not in config:
        return False
    return True


''' Load the appropriate data adapter
'''
def get_data_adapter(dataname):
    if dataname.lower() == 'thyme':
        return data_adapter_thyme.DataAdapterThyme(debug)
    elif dataname.lower() == 'va':
        return data_adapter_va.DataAdapterVA(debug)


''' Run one stage of chrononet
    stage_name: sequence|ordering
    config: the config object
    train_df: the training dataframe, with results of previous stages
    test_df: the test dataframe, with the results of previous stages
'''
def run_stage(stage_name, config, train_data_adapter, test_data_adapter, train_df, test_df, doc_level=False):

    # Load config info
    stage_config = config[stage_name]
    features = stage_config['features'].split(',')
    models = stage_config['models'].split(',')
    metrics = stage_config['metrics'].split(',')
    labelname = test_data_adapter.get_labelname(stage_name)
    print('Running stage: ', stage_name, 'with models:', str(models), 'and feats:', str(features))

    # FEATURE EXTRACTION
    train_feat_df = train_df.copy()# data_util.create_df(train_df)
    test_feat_df = test_df.copy()
    for fe in features:
        extractor = fe_map[fe]
        train_feat_df = extractor(train_feat_df)
        test_feat_df = extractor(test_feat_df)

    # MODELS
    score_string = ''
    for modelname in models:
        if modelname == 'ground_truth':
            model = None
        else:
            model = model_map[modelname].get_model()
        if modelname == 'crf':
            should_encode = False
            use_numpy = False
        else:
            should_encode = True
            use_numpy = True

        # Get the labels
        should_encode = False
        train_Y, labelencoder = numpyer.to_labels(train_feat_df, labelname, encode=should_encode)
        test_Y, labelencoder = numpyer.to_labels(test_feat_df, labelname, labelencoder, encode=should_encode)

        # Ground truth model
        if modelname == 'ground_truth':
            y_pred = test_Y
        # Real models
        else:
            # Encode features and labels
            train_X = numpyer.to_feats(train_feat_df, use_numpy, doc_level=False)
            test_X = numpyer.to_feats(test_feat_df, use_numpy, doc_level=False)
            print('train X:', len(train_X), 'Y:', len(train_Y))
            print('test X:', len(test_X), 'Y:', len(test_Y))

            if debug:
                model.debug = True
                print('Running model', modelname)

            # RUN MODEL
            model.fit(train_X, train_Y)
            y_pred = model.predict(test_X)

        # Save results to dataframe
        test_feat_df = data_util.add_labels(test_feat_df, y_pred, labelname)

        if stage_name == 'sequence':
            y_pred = data_util.collapse_labels(y_pred)
            y_true = data_util.collapse_labels(test_Y)
            print('collapsed labels: true:', len(y_true), 'pred:', len(y_pred))
        else:
            y_true = test_Y

        # EVALUATION
        score_string += modelname
        for metric in metrics:
            metric_func = metric_map[metric]
            # TODO: do we need to convert these back to text labels?
            score = metric_func(y_true, y_pred)
            print(metric, score)
            score_string += '\t' + metric + ': ' + str(score)
        score_string += '\n'

    return score_string, train_feat_df, test_feat_df





if __name__ == "__main__": main()
