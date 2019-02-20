#!/usr/bin/python3
# Main Chrononet script

# Imports
import ast
import argparse
import configparser
import joblib
import os
import pandas
import time
import torch

# Local imports
from data_tools.adapters import data_adapter_thyme, data_adapter_va
from data_tools import data_util
from evaluation import eval_metrics, ordering_metrics
from feature_extractors import numpyer, relations, syntactic, vectors
from models.sequence.crf import CRFfactory
from models.ordering.neural_order import NeuralOrderFactory
from models.ordering.random_order import RandomOrderFactory, MentionOrderFactory

# SETUP
fe_map = {'relations': relations.extract_relations, 'syntactic': syntactic.sent_features,
          'event_vectors': vectors.event_vectors, 'none': numpyer.dummy_function}
vector_feats = ['event_vectors']
model_map = {'crf': CRFfactory, 'random': RandomOrderFactory, 'mention': MentionOrderFactory, 'neural': NeuralOrderFactory}
metric_map = {'p': eval_metrics.precision, 'r': eval_metrics.recall, 'f1': eval_metrics.f1,
              'mse': ordering_metrics.rank_mse, 'poa': ordering_metrics.rank_pairwise_accuracy}
debug = True

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-c', '--config', action="store", dest="configfile")
    args = argparser.parse_args()

    start_time = time.time()

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
    doc_level_df = False

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
        if debug:
            print('loading seq df')
        seq_train_df = pandas.read_csv(train_filename)
        seq_test_df = pandas.read_csv(test_filename)
    else:
        seq_train_df = train_data_adapter.to_seq(train_df, split_sents=True)
        seq_test_df = test_data_adapter.to_seq(test_df, split_sents=True)
    if save_intermediate:
        if debug:
            print('Saving seq df...')
        seq_train_df.to_csv(train_filename)
        seq_test_df.to_csv(test_filename)
    scores, train_df, test_df = run_stage('sequence', config, train_data_adapter, test_data_adapter, seq_train_df, seq_test_df, outdir)
    score_report.append(scores)

    del seq_train_df
    del seq_test_df

    train_df = train_data_adapter.to_doc(train_df)
    test_df = test_data_adapter.to_doc(test_df)
    doc_level_df = True

    # TEMPORAL ORDERING STAGE
    if 'ordering' in config:
        train_filename = os.path.join(outdir, inter_prefix + 'train_df_order.csv')
        test_filename = os.path.join(outdir, inter_prefix + 'test_df_order.csv')
        if os.path.exists(train_filename) and os.path.exists(test_filename):
            if debug:
                print('Loading ordering df...')
            order_train_df = pandas.read_csv(train_filename)
            order_test_df = pandas.read_csv(test_filename)
        else:
            order_train_df = train_data_adapter.to_order(train_df, orig_train_df)
            order_test_df = test_data_adapter.to_order(test_df, orig_test_df)
        if save_intermediate:
            if debug:
                print('Saving ordering df...')
            order_train_df.to_csv(train_filename)
            order_test_df.to_csv(test_filename)

        # Run temporal ordering
        scores, train_df, test_df = run_stage('ordering', config, train_data_adapter, test_data_adapter, order_train_df, order_test_df, outdir, doc_level=True)
        score_report.append(scores)

    # OUTPUT
    if debug:
        print('Writing output files...')
    test_data_adapter.write_output(test_df, outdir, doc_level=doc_level_df)

    # PRINT the scores
    for line in score_report:
        print(str(line))

    end_time = time.time()
    print('total time:', time_string(end_time - start_time))


''' Print a time in seconds in a human-readable format
'''
def time_string(seconds):
    if seconds < 60:
        return str(seconds) + ' s'
    else:
        mins = float(seconds)/60
        if mins < 60:
            return str(mins) + ' mins'
        else:
            hours = mins/60
            return str(hours) + ' hours'


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
def run_stage(stage_name, config, train_data_adapter, test_data_adapter, train_df, test_df, outdir, doc_level=False):

    # Load config info
    stage_config = config[stage_name]
    features = stage_config['features'].split(',')
    models = stage_config['models'].split(',')
    metrics = stage_config['metrics'].split(',')
    vecfile = None
    if 'vecfile' in stage_config:
        vecfile = stage_config['vecfile']
    labelname = test_data_adapter.get_labelname(stage_name)
    vec_model = None
    dim = 0
    print('Running stage: ', stage_name, 'with models:', str(models), 'and feats:', str(features))

    if vecfile is not None:
        print('loading vectors:', vecfile)
        vec_model, dim = vectors.load(vecfile)
        print('dim:', str(dim))

    # FEATURE EXTRACTION
    train_feat_df = train_df.copy()# data_util.create_df(train_df)
    test_feat_df = test_df.copy()
    f_time = time.time()
    for fe in features:
        extractor = fe_map[fe]
        if fe in vector_feats:
            train_feat_df = extractor(train_feat_df, vec_model)
            test_feat_df = extractor(test_feat_df, vec_model)
        else:
            train_feat_df = extractor(train_feat_df)
            test_feat_df = extractor(test_feat_df)
    print('feature extraction time:', time_string(time.time()-f_time))

    # MODELS
    score_string = ''
    for modelname in models:
        modelfile = os.path.join(outdir, modelname + '.model')
        if modelname == 'ground_truth':
            model = None
        else:
            model_factory = model_map[modelname]
            if model_factory.requires_dim:
                model = model_factory.get_model(dim)
            else:
                model = model_factory.get_model()
        if modelname == 'crf' or modelname == 'ground_truth':
            should_encode = False
            use_numpy = False
        elif stage_name == 'ordering':
            should_encode = True
            use_numpy = False
        else:
            should_encode = True # Should we encode labels
            use_numpy = True # Should we use numpy to encode the features

        # Get the labels
        #should_encode = False
        # Encode features and labels
        m_time = time.time()
        train_X = numpyer.to_feats(train_feat_df, use_numpy, doc_level=False)
        test_X = numpyer.to_feats(test_feat_df, use_numpy, doc_level=False)
        train_Y, labelencoder = numpyer.to_labels(train_feat_df, labelname, encode=should_encode)
        test_Y, labelencoder = numpyer.to_labels(test_feat_df, labelname, labelencoder, encode=should_encode)
        train_ids = train_feat_df['docid'].tolist()
        test_ids = test_feat_df['docid'].tolist()
        print('train X:', len(train_X), 'Y:', len(train_Y))
        print('test X:', len(test_X), 'Y:', len(test_Y))
        if debug:
            print('train X[0]:', train_X[0])
            print('train Y[0]:', train_Y[0])
            print('test X[0]:', test_X[0])
            print('test Y[0]:', test_Y[0])
        if stage_name == 'ordering':
            check_alignment(train_ids, train_X, train_Y)
            check_alignment(test_ids, test_X, test_Y)

        # Ground truth model
        if modelname == 'ground_truth':
            y_pred = test_Y
        # Real models
        else:
            if debug:
                model.debug = True
                print('Running model', modelname)

            # LOAD previously trained model
            if os.path.exists(modelfile):
                print('Loading pretrained model:', modelfile)
                model = load(modelfile, 'torch')
            else:
                model.fit(train_X, train_Y)
                if modelname in ['neural']:
                    print('Saving model...')
                    save(model, modelfile, 'torch')

            # RUN MODEL
            y_pred = model.predict(test_X)
        print('time for model', modelname, ':', time_string(time.time()-m_time))

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


def check_alignment(ids, X, Y):
    for index in range(0, len(ids)):
        recid = ids[index]
        x_row = X[index]
        y_row = Y[index]
        if not len(x_row) == len(y_row):
            print('ERROR: feature/label mismatch:', recid, 'has', len(x_row), 'features and', len(y_row), 'labels')
        assert(len(x_row) == len(y_row))


def save(model, modelfile, model_type):
    #if self.model is None:
    #    raise NoModelError(self)
    #if not os.path.exists(self.directory):
    #    os.mkdir(self.directory)
    if model_type == 'torch':
        torch.save(model, modelfile)
    elif model_type == 'sklearn':
        joblib.dump(model, modelfile)
    else:
        print('Error in save: unrecognized model_type:', model_type)


def load(modelfile, model_type):
    #if not os.path.exists(self.modelfile):
    #    raise NoModelError(self)
    if model_type == 'torch':
        return torch.load(modelfile)
    elif model_type == 'sklearn':
        return joblib.load(modelfile)
    else:
        print('Error in save: unrecognized model_type:', model_type)


if __name__ == "__main__": main()
