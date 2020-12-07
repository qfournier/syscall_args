import argparse
from os import path


def get_arguments():
    # create parser
    parser = argparse.ArgumentParser()

    # misc
    parser.add_argument('--it',
                        type=int,
                        default=0,
                        help='iteration number (log files and figures')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='device(s) (e.g., "cpu", "cuda:0",  "cuda:0,cuda:1")')
    parser.add_argument('--log',
                        type=str,
                        default='INFO',
                        help='logging level (DEBUG, INFO, WARNING, '
                        'ERROR, CRITICAL)')

    # data
    parser.add_argument('--data',
                        type=str,
                        default='data/requests',
                        help='path of the trace folder')
    parser.add_argument('--save_corpus',
                        action='store_true',
                        help='save the corpus in the data folder')
    parser.add_argument('--load_corpus',
                        action='store_true',
                        help='load the corpus from the data folder')
    parser.add_argument('--requests',
                        action='store_true',
                        help='consider individual requests')
    parser.add_argument('--limit',
                        type=int,
                        default=None,
                        help='maximum number of sequence to load')
    parser.add_argument('--max_length',
                        type=int,
                        default=None,
                        help='maximum sequence lengths')
    parser.add_argument('--plot_hist',
                        action='store_true',
                        help='Plot the system calls and processes histograms')

    # model
    parser.add_argument('--load_model',
                        type=int,
                        default=None,
                        help='model number to load')
    parser.add_argument('--model',
                        type=str,
                        default='transformer',
                        choices=['ngram', 'lstm', 'transformer'],
                        help='model to use')
    parser.add_argument('--order',
                        type=int,
                        default=2,
                        help='N-gram order (value of N)')

    # model hyperparameters
    parser.add_argument('--emb_sys',
                        type=int,
                        default=32,
                        help='embedding dimension of system '
                        'call names and entry/exit')
    parser.add_argument('--emb_proc',
                        type=int,
                        default=16,
                        help='embedding dimension of process names')
    parser.add_argument('--emb_pid',
                        type=int,
                        default=4,
                        help='embedding dimension of the process if')
    parser.add_argument('--emb_tid',
                        type=int,
                        default=4,
                        help='embedding dimension of the thread id')
    parser.add_argument('--emb_time',
                        type=int,
                        default=8,
                        help='embedding dimension of the timestamp')
    parser.add_argument('--emb_order',
                        type=int,
                        default=8,
                        help='embedding dimension of the ordering')
    parser.add_argument('--heads',
                        type=int,
                        default=8,
                        help='number of attention heads')
    parser.add_argument('--hiddens',
                        type=int,
                        default=128,
                        help='number of hidden units of each encoder MLP')
    parser.add_argument('--layers',
                        type=int,
                        default=2,
                        help='number of layers')
    parser.add_argument('--dropout',
                        type=float,
                        default=0.5,
                        help='model dropout rate (embedding & encoder)')
    # training
    parser.add_argument('--batch', type=int, default=64, help='batch size')
    parser.add_argument('--valid',
                        type=float,
                        default=0.25,
                        help='percentage of the test set used for valdiation')
    parser.add_argument('--p_mask',
                        type=float,
                        default=0.25,
                        help='percentage of the input masked')
    parser.add_argument('--mlm_epochs',
                        type=int,
                        default=0,
                        help='number of epochs using MLM (pre-training)')
    parser.add_argument('--lm_epochs',
                        type=int,
                        default=0,
                        help='number of epochs using LM (fine-tuning)')
    parser.add_argument('--eval',
                        type=int,
                        default=1000,
                        help='number of update before evaluating the model '
                        '(impact early stopping)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--early_stopping',
                        type=int,
                        default=5,
                        help='number of iteration before early stopping')
    parser.add_argument('--checkpoint',
                        action='store_true',
                        help='trade computation for memory')

    # ablation study
    parser.add_argument('--disable_entry',
                        action='store_true',
                        help='Do not use entry/exit for the embedding')
    parser.add_argument('--disable_ret',
                        action='store_true',
                        help='Do not use the return value for the embedding')
    parser.add_argument('--disable_time',
                        action='store_true',
                        help='Do not use the timestamp')
    parser.add_argument('--disable_proc',
                        action='store_true',
                        help='Do not use the process name for the embedding')
    parser.add_argument('--disable_pid',
                        action='store_true',
                        help='Do not use the process id for the embedding')
    parser.add_argument('--disable_tid',
                        action='store_true',
                        help='Do not use the thread id for the embedding')
    parser.add_argument('--disable_order',
                        action='store_true',
                        help='Do not use the event order')

    args = parser.parse_args()

    # check arguments
    assert path.exists(args.data), 'data folder not found'
    assert path.exists('{}/train'.format(
        args.data)), 'train data folder not found'
    assert path.exists('{}/test'.format(
        args.data)), 'test data folder not found'
    assert not (args.save_corpus
                and args.load_corpus), 'cannot save and load the Corpus'
    assert args.max_length is None or args.max_length > 0, \
        'max_length must be greater than 0'
    assert args.load_model is None or path.exists('models/{}'.format(
        args.load_model)), 'model not found'
    assert args.order > 1, 'order must be greater than 1'
    assert args.batch > 0, 'batch must be greater than 0'
    assert args.p_mask > 0, 'p_mask must be greater than 0'
    assert args.emb_sys > 0, 'emb_sys must be greater than 0'
    assert args.disable_time or args.emb_time > 0, \
        'emb_time must be greater than 0'
    assert args.disable_proc or args.emb_proc > 0, \
        'emb_sys must be greater than 0'
    assert args.disable_pid or args.emb_pid > 0, \
        'emb_pid must be greater than 0'
    assert args.disable_tid or args.emb_tid > 0, \
        'emb_tid must be greater than 0'
    assert args.heads > 0, 'heads must be greater than 0'
    assert args.hiddens > 0, \
        'hiddens must be greater than 0'
    assert args.layers > 0, 'layers must be greater than 0'
    assert args.dropout >= 0 and args.dropout < 1, \
        'dropout must be between in [0, 1)'
    if args.model.lower() == 'lstm':
        assert args.mlm_epochs == 0, 'MLM not compatible with LSTM'
    assert args.early_stopping > 0, 'early_stopping must be greater than 0'

    return args
