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
from scipy.stats import ttest_rel
from sklearn.metrics import classification_report

# Local imports
from data_tools.adapters import data_adapter_thyme, data_adapter_va, data_adapter_verilogue
from data_tools import data_util
from evaluation import eval_metrics, ordering_metrics
from feature_extractors import numpyer, relations, syntactic, vectors
from models.sequence.crf import CRFfactory
from models.sequence.ncrfpp import NCRFppFactory
from models.encoding.time_encoding import TimeEncodingFactory
from models.ordering.neural_order import NeuralOrderFactory, HyperoptNeuralOrderFactory, NeuralLinearFactory
from models.ordering.random_order import RandomOrderFactory, MentionOrderFactory
from models.classification.cnn import CNNFactory, MatrixCNNFactory, RNNFactory, MatrixRNNFactory
from models.classification.random import RandomFactory

# SETUP
fe_map = {'relations': relations.extract_relations, 'syntactic': syntactic.sent_features,
          'event_vectors': vectors.event_vectors, 'elmo_vectors': vectors.elmo_event_vectors, 'elmo_words': vectors.elmo_word_vectors,
          'none': numpyer.dummy_function, 'timeline': numpyer.do_nothing, 'time_pairs': relations.extract_time_pairs}
vector_feats = ['event_vectors']
model_map = {'crf': CRFfactory, 'random': RandomOrderFactory, 'mention': MentionOrderFactory, 'neural': NeuralOrderFactory, 'neurallinear': NeuralLinearFactory, 'hyperopt': HyperoptNeuralOrderFactory,
             'cnn': CNNFactory, 'rnn': RNNFactory, 'matrixcnn': MatrixCNNFactory, 'matrixrnn': MatrixRNNFactory, 'ncrfpp': NCRFppFactory,
             'randclass': RandomFactory, 'time_encoding': TimeEncodingFactory}
metric_map = {'p': eval_metrics.precision, 'r': eval_metrics.recall, 'f1': eval_metrics.f1, 'mae': ordering_metrics.rank_mae,
              'mse': ordering_metrics.rank_mse, 'poa': ordering_metrics.rank_pairwise_accuracy, 'tau': ordering_metrics.kendalls_tau,
              'epr': ordering_metrics.epr, 'gpr': ordering_metrics.gpr, 'csmfa': eval_metrics.csmfa}
debug = True

time_modelfile = None
tdevice = 'cpu'

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
    extra_trainfile = None
    extra_train_df = None
    if 'extra_train' in data_config:
        extra_trainfile = data_config['extra_train']
    outdir = data_config['output_dir']
    save_intermediate = ast.literal_eval(data_config['save_intermediate_files'])
    score_report = []
    doc_level_df = False
    should_eval = True
    if 'eval' in data_config:
        should_eval = ast.literal_eval(data_config['eval'])

    # Create output directory
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    global inter_prefix
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
        test_df = test_data_adapter.load_data(testfile, drop_unlabeled=should_eval)
    if save_intermediate:
        if debug:
            print('Saving preprocessed df...')
        train_df.to_csv(train_filename)
        test_df.to_csv(test_filename)
    orig_train_df = train_df.copy()
    orig_test_df = test_df.copy()

    if extra_trainfile is not None:
        extra_train_filename = os.path.join(outdir, inter_prefix + 'extra_train_df.csv')
        extra_train_df = train_data_adapter.load_data(extra_trainfile)
        orig_extra_train_df = extra_train_df.copy()
        if save_intermediate:
            extra_train_df.to_csv(extra_train_filename)

    # SEQUENCE TAGGER STAGE
    if 'sequence' in config:
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

        seq_extra_train_df = None
        if extra_train_df is not None:
            seq_extra_train_df = train_data_adapter.to_seq(extra_train_df, split_sents=True)

        scores, train_df, test_df, extra_train_df = run_stage('sequence', config, train_data_adapter, test_data_adapter, seq_train_df, seq_test_df, seq_extra_train_df, outdir)
        score_report.append(scores)

        del seq_train_df
        del seq_test_df

        train_df = train_data_adapter.to_doc(train_df)
        test_df = test_data_adapter.to_doc(test_df)
        doc_level_df = True

    # TEMPORAL ORDERING STAGE
    if 'ordering' in config or 'encoding' in config:
        train_filename = os.path.join(outdir, inter_prefix + 'train_df_order.csv')
        test_filename = os.path.join(outdir, inter_prefix + 'test_df_order.csv')
        if os.path.exists(train_filename):
            if debug:
                print('Loading ordering df...')
            order_train_df = pandas.read_csv(train_filename)
        else:
            order_train_df = train_data_adapter.to_order(train_df, orig_train_df)
        if os.path.exists(test_filename):
            order_test_df = pandas.read_csv(test_filename)
        else:
            order_test_df = test_data_adapter.to_order(test_df, orig_test_df)
        if save_intermediate:
            if debug:
                print('Saving ordering df...')
            order_train_df.to_csv(train_filename)
            order_test_df.to_csv(test_filename)

        order_extra_train_df = None
        if extra_train_df is not None:
            order_extra_train_df = train_data_adapter.to_order(extra_train_df, orig_extra_train_df)

        # ENCODING stage
        if 'encoding' in config:
            scores, train_df, test_df, _ = run_stage('encoding', config, train_data_adapter, test_data_adapter, order_train_df, order_test_df, None, outdir, doc_level=True)
            score_report.append(scores)

        # Run temporal ordering stage
        if 'ordering' in config:
            scores, train_df, test_df, extra_train_df = run_stage('ordering', config, train_data_adapter, test_data_adapter, order_train_df, order_test_df, order_extra_train_df, outdir, doc_level=True)
            score_report.append(scores)

    #else:
    #    train_df = train_data_adapter.to_order(train_df, orig_train_df)
    #    test_df = test_data_adapter.to_order(test_df, orig_test_df)

    # CLASSIFICATION
    if 'classification' in config:
        train_filename = os.path.join(outdir, inter_prefix + 'train_df_class.csv')
        test_filename = os.path.join(outdir, inter_prefix + 'test_df_class.csv')
        if os.path.exists(train_filename):
            if debug:
                print('Loading classification df...')
            train_df = pandas.read_csv(train_filename)
        elif save_intermediate:
            if debug:
                print('Saving classification df...')
            train_df.to_csv(train_filename)
        if os.path.exists(test_filename):
            test_df = pandas.read_csv(test_filename)
        elif save_intermediate:
            test_df.to_csv(test_filename)

        # Run classification
        scores, train_df, test_df, _ = run_stage('classification', config, train_data_adapter, test_data_adapter, train_df, test_df, extra_train_df, outdir, doc_level=True)
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
    elif dataname.lower() == 'verilogue':
        return data_adapter_verilogue.DataAdapterVerilogue(debug)


''' Run one stage of chrononet
    stage_name: sequence|ordering
    config: the config object
    train_df: the training dataframe, with results of previous stages
    test_df: the test dataframe, with the results of previous stages
'''
def run_stage(stage_name, config, train_data_adapter, test_data_adapter, train_df, test_df, extra_train_df=None, outdir='', doc_level=False):

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
    dim = 1024

    order_classify = False
    if stage_name == 'classification' and 'rnn' in models:
        order_classify = True

    # Load model params from config
    stage_params = config._sections[stage_name + '_params']

    print('Running STAGE: ', stage_name, 'with models:', str(models), 'and feats:', str(features))
    for entry in stage_params.keys():
        print('param:', entry, stage_params[entry])

    if vecfile is not None:
        print('loading vectors:', vecfile)
        vec_model, dim = vectors.load(vecfile)
    print('dim:', str(dim))

    # FEATURE EXTRACTION
    f_time = time.time()
    train_filename = os.path.join(outdir, inter_prefix + 'train_df_' + stage_name + '_feats.csv')
    test_filename = os.path.join(outdir, inter_prefix + 'test_df_' + stage_name + '_feats.csv')
    if os.path.exists(train_filename):
        if debug: print('Loading train feat df...')
        train_feat_df = pandas.read_csv(train_filename)
    else:
        if debug: print('Extracting train features...')
        train_feat_df = train_df.copy()# data_util.create_df(train_df)

        for fe in features:
            extractor = fe_map[fe]
            if fe in vector_feats:
                train_feat_df = extractor(train_feat_df, vec_model)
            else:
                train_feat_df = extractor(train_feat_df)
        if debug:
            print('Saving train feat df...')
        train_feat_df.to_csv(train_filename)

    # extra train features
    if extra_train_df is not None:
        train_feat_df = train_df.copy()
        for fe in features:
            extractor = fe_map[fe]
            if fe in vector_feats:
                extra_train_feat_df = extractor(train_feat_df, vec_model)
            else:
                extra_train_feat_df = extractor(train_feat_df)

    # Load test features
    if os.path.exists(test_filename):
        if debug: print('Loading test feat df...')
        test_feat_df = pandas.read_csv(test_filename)
    else:
        if debug: print('Extracting test features...')
        test_feat_df = test_df.copy()
        for fe in features:
            extractor = fe_map[fe]
            if fe in vector_feats:
                test_feat_df = extractor(test_feat_df, vec_model)
            else:
                test_feat_df = extractor(test_feat_df)
        if debug:
            print('Saving test feat df...')
        test_feat_df.to_csv(test_filename)
    print('feature extraction time:', time_string(time.time()-f_time))

    # MODELS
    score_string = ''
    model_results = {}
    for modelname in models:

        if modelname == 'crf' or modelname == 'ground_truth':
            should_encode = False
            use_numpy = False
        elif stage_name == 'ordering':
            should_encode = False
            use_numpy = False
            if modelname == 'neurallinear':
                should_encode = True
        elif stage_name == 'sequence' and not modelname == 'ncrfpp':
            should_encode = True # Should we encode labels
            use_numpy = True # Should we use numpy to encode the features
        else:
            should_encode = True # Should we encode labels
            use_numpy = False # Should we use numpy to encode the features
        if modelname == 'ncrfpp':
            should_encode = False

        print('encode labels:', should_encode)
        print('use numpy:', use_numpy)

        # Encode features and labels
        m_time = time.time()
        train_X = numpyer.to_feats(train_feat_df, use_numpy, doc_level=False)
        test_X = numpyer.to_feats(test_feat_df, use_numpy, doc_level=False)
        train_Y, labelencoder = numpyer.to_labels(train_feat_df, labelname, encode=should_encode)
        test_Y, labelencoder = numpyer.to_labels(test_feat_df, labelname, labelencoder, encode=should_encode)
        if should_encode and not stage_name == 'ordering':
            print('labels:', labelencoder.classes_)

        if stage_name == 'encoding':
            # Linearize training data
            events = []
            labels = []
            for row in train_X:
                for ev in row:
                    events.append(ev)
            for row in train_Y:
                for ev in row:
                    labels.append(ev)
            train_X = events
            train_Y = labels
            assert(len(train_X) == len(train_Y))
            print('time encoding trainX:', len(train_X))

            # Limit test data for speed
            #if len(test_X) > 1000:
            #    test_X = test_X[0:1000]
            #    test_Y = test_Y[0:1000]

            # Save the time pairs to a file
            '''
            if config['data']['train_dataset'] == 'thyme':
                outfile = open('/u/sjeblee/research/data/thyme/train_time_pairs.csv', 'w+')
                for i in range(len(train_X)):
                    pair0 = ' '.join(train_X[i][0])
                    pair1 = ' '.join(train_X[i][1])
                    print(train_Y[i])
                    label = labelencoder.inverse_transform([train_Y[i]])[0]
                    print('writing:', pair0, pair1, label)
                    outfile.write(pair0 + ',' + pair1 + ',' + label + '\n')
                outfile.close()
                print('Wrote train time pairs to file')
            '''
            # Load extra time pairs for training
            '''
            train_extra, labels_extra = data_util.load_time_pairs(stage_config['train_time_pairs'])
            train_X = train_extra #+ train_X
            train_Y = labelencoder.transform(labels_extra).tolist() #+ train_Y
            '''

        # Get rank labels for joint ordering/classification model
        if order_classify:
            labelname2 = test_data_adapter.get_labelname('ordering')
            train_Y2, rankencoder = numpyer.to_labels(train_feat_df, labelname2, encode=should_encode)
            test_Y2, rankencoder = numpyer.to_labels(test_feat_df, labelname2, rankencoder, encode=should_encode)

        if modelname in ['cnn', 'matrixcnn', 'rnn', 'matrixrnn']:
            num_classes = len(labelencoder.classes_)
            stage_params['num_classes'] = num_classes
            print('added num_classes to params')

        train_ids = train_feat_df['docid'].tolist()
        test_ids = test_feat_df['docid'].tolist()
        print('train X:', len(train_X), 'Y:', len(train_Y))
        print('test X:', len(test_X), 'Y:', len(test_Y))

        if extra_train_df is not None:
            extra_train_X = numpyer.to_feats(extra_train_feat_df, use_numpy, doc_level=False)

        if modelname == 'matrixcnn' or modelname == 'matrixrnn':
            print('train X[0]:', train_X[0])
            dim = train_X[0].size(-1)
            #dim = 40
        print('dim:', str(dim))
        if debug:
            print('train X[0]:', train_X[0])
            print('train Y[0]:', train_Y[0])
            print('test X[0]:', test_X[0])
            print('test Y[0]:', test_Y[0])
        if stage_name == 'ordering':
            check_alignment(train_ids, train_X, train_Y)
            check_alignment(test_ids, test_X, test_Y)
            #train_ids, train_X, train_Y = data_util.generate_permutations(train_ids, train_X, train_Y)

        print('Running', modelname)
        modelfile = os.path.join(outdir, modelname + '.model')

        # Save the time encoding modelfile for the ordering model to use
        if stage_name == 'encoding':
            global time_modelfile
            time_modelfile = modelfile
            print('saved time_modelfile')

        if modelname in ['neural', 'neurallinear', 'rnn']:
            stage_params['encoder_file'] = time_modelfile

        if modelname == 'ground_truth':
            model = None
        else:
            model_factory = model_map[modelname]
            if model_factory.requires_dim:
                model = model_factory.get_model(dim, stage_params)
            else:
                model = model_factory.get_model(stage_params)

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
                if order_classify:
                    model.fit(train_X, train_Y, train_Y2)
                else:
                    model.fit(train_X, train_Y)
                # Save the model
                if modelname in ['neural', 'neurallinear', 'cnn', 'ncrfpp', 'rnn', 'matrixcnn', 'matrixrnn', 'time_encoding']:
                    print('Saving model...')
                    save(model, modelfile, 'torch')

            # RUN MODE
            '''
            # TEMP: load second model
            if stage_name == 'ordering':
                modelfile2 = '/u/sjeblee/research/data/thyme/chrono/order_test_gru_context_time/neurallinear.model'
                print('Loading pretrained model 2:', modelfile2)
                model2 = load(modelfile2, 'torch')
                y_pred2 = model2.predict(test_X)
            '''

            if modelname == 'hyperopt':
                y_pred = model.predict(test_X, test_Y)
            else:
                if modelname in ['neurallinear', 'neural']:
                    print('Predict and retrieve encodings...')
                    y_pred = model.predict(test_X)
                    '''
                    y_pred, encodings = model.predict(test_X)
                    print('test_ids:', len(test_ids), 'encodings:', len(encodings))
                    # Save encodings to the dataframe
                    encodings = data_util.reorder_encodings(encodings, test_Y) # GOLD order
                    test_feat_df = data_util.add_labels(test_feat_df, encodings, 'feats')
                    trainy_pred, train_encodings = model.predict(train_X)
                    train_encodings = data_util.reorder_encodings(train_encodings, train_Y)
                    train_feat_df = data_util.add_labels(train_feat_df, train_encodings, 'feats')
                    '''
                else:
                    # TEMP for THYME dataset
                    if stage_name == 'encoding':
                        y_pred = test_Y
                    else:
                        y_pred = model.predict(test_X)
            if extra_train_df is not None:
                extra_train_labels = model.predict(extra_train_X)
        print('time for model', modelname, ':', time_string(time.time()-m_time))

        # Save results to dataframe
        if should_encode and not modelname == 'neurallinear':
            print('decoding labels...', labelencoder.classes_)
            if stage_name == 'sequence' or stage_name == 'encoding':
                pred_labels = []
                for row in y_pred:
                    print('decoding row:', row)
                    pred_labels.append(labelencoder.inverse_transform(row))
                    print(pred_labels[-1])
            else:
                pred_labels = labelencoder.inverse_transform(y_pred)
            # extra train
            if extra_train_df is not None:
                extra_train_labels = labelencoder.inverse_transform(extra_train_labels)
        else:
            pred_labels = y_pred
        test_feat_df = data_util.add_labels(test_feat_df, pred_labels, labelname)
        test_data_adapter.stages.append(stage_name)

        # TEMP: Save the dataframe with predicted labels
        outdir = config['data']['output_dir']
        out_filename = os.path.join(outdir, inter_prefix + 'test_out.csv')
        test_feat_df.to_csv(out_filename)

        # Check that the correct and predicted labels are the same length
        check_alignment(test_ids, test_Y, y_pred)

        if extra_train_df is not None:
            extra_train_df = data_util.add_labels(extra_train_df, extra_train_labels, labelname)

        # Collapse labels for sequence task
        if stage_name == 'sequence' or stage_name == 'encoding':
            y_pred = data_util.collapse_labels(y_pred)
            y_true = data_util.collapse_labels(test_Y)
            print('collapsed labels: true:', len(y_true), 'pred:', len(y_pred))
        else:
            y_true = test_Y

        '''
        if stage_name == 'encoding':
            test_synth, labels_synth = data_util.load_time_pairs(stage_config['test_time_pairs'])
            test_synth = [[item] for item in test_synth] # Wrap each item in a list to de-linearize
            labels_synth = labelencoder.transform(labels_synth).tolist()
            print('test_synth:', test_synth[0:10])
            print('labels_synth:', labels_synth[0:10])
            print('synth test size:', len(test_synth))
            synth_pred = data_util.collapse_labels(model.predict(test_synth))
            for metric in metrics:
                metric_func = metric_map[metric]
                score = metric_func(labels_synth, synth_pred)
                print('synth test:', metric, score)
        '''

        # EVALUATION
        score_string += modelname
        for metric in metrics:
            metric_func = metric_map[metric]
            if metric == 'gpr':
                score = metric_func(y_true, y_pred, test_df)
            else:
                # TODO: do we need to convert these back to text labels?
                score = metric_func(y_true, y_pred)
            print(metric, score)
            score_string += '\t' + metric + ': ' + str(score)

            if metric in ['poa', 'tau']:
                ind_scores = metric_func(y_true, y_pred, avg=False)
                if metric not in model_results:
                    model_results[metric] = []
                model_results[metric].append(ind_scores)

                # TEMP: stat sig
                '''
                if stage_name == 'ordering':
                    ind_scores2 = metric_func(y_true, y_pred2, avg=False)
                    model_results[metric].append(ind_scores2)
                '''

                # Score grouped POA
                #poa_score = metric_func(y_true, y_pred, eps=0.01)
                #print(metric, poa_score)
                #score_string += '\t' + metric + ' (0.01): ' + str(score)
        score_string += '\n'

        if stage_name == 'classification':
            print('classification_report scores:')
            print(classification_report(y_true, y_pred))

    # Calculate statistical significance
    stat_sig = False
    #if stage_name == 'ordering':
    #    stat_sig = True
    #if len(models) > 1:
    if stat_sig:
        for metric_name in model_results.keys():
            print(str(model_results))
            print('Stat sig for metric:', metric_name)
            set_1 = model_results[metric_name][0]
            set_2 = model_results[metric_name][1]
            tval, pval = ttest_rel(set_1, set_2)
            print('pval:', pval)

    return score_string, train_feat_df, test_feat_df, extra_train_df


def check_alignment(ids, X, Y):
    print('ids:', len(ids), 'x:', len(X), 'y:', len(Y))
    assert(len(X) == len(Y))
    for index in range(0, len(ids)):
        recid = ids[index]
        x_row = X[index]
        y_row = Y[index]
        if not len(x_row) == len(y_row):
            print('ERROR: feature/label mismatch:', recid, 'has', len(x_row), 'features and', len(y_row), 'labels')
            print('feats:', str(x_row))
            print('labels:', str(y_row))
        assert(len(x_row) == len(y_row))


''' Save the model to a file
    model: the model object
    modefile: the file path to save the model to
    model_type: torch or sklearn
'''
def save(model, modelfile, model_type):
    if model_type == 'torch':
        torch.save(model, modelfile)
    elif model_type == 'sklearn':
        joblib.dump(model, modelfile)
    else:
        print('Error in save: unrecognized model_type:', model_type)


def load(modelfile, model_type):
    if model_type == 'torch':
        return torch.load(modelfile, map_location=tdevice) # TEMP
    elif model_type == 'sklearn':
        return joblib.load(modelfile)
    else:
        print('Error in save: unrecognized model_type:', model_type)


if __name__ == "__main__": main()
