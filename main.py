'''Main file.
'''
import sys
import random
import logging
import numpy as np
from time import time
from datetime import timedelta

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from source.Dictionary import Dictionary
from source.arguments import get_arguments
from source.functions import nltk_ngram
from source.functions import ngram_acc
from source.functions import train
from source.functions import collate_fn
from source.functions import plot_loss
from source.functions import plot_accuracy
from source.functions import datatset_stats
from source.functions import evaluate
from source.Corpus import Corpus
from source.CorpusReq import CorpusReq
from source.LMDataset import LMDataset
from source.MLMDataset import MLMDataset
from source.Transformer import Transformer
from source.LSTM import LSTM

###############################################################################
# Miscellaneous
###############################################################################

# get hyperparameters
args = get_arguments()

# set seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# convert args.log to numerical logging level
numeric_level = getattr(logging, args.log.upper(), None)
if not isinstance(numeric_level, int):
    raise ValueError('Invalid log level: {}'.format(args.log))

# create logger
logger = logging.getLogger('logger')
logger.setLevel(numeric_level)
# create console handler for info messages
ch = logging.StreamHandler()
ch.setLevel(numeric_level)
# create formatter and add it to the handlers
ch.setFormatter(logging.Formatter('%(message)s'))
# add the handlers to the logger
logger.addHandler(ch)
# create file handler to store log in a file
fh = logging.FileHandler('logs/{}_log.txt'.format(args.it))
fh.setLevel(numeric_level)
# create formatter and add it to the handlers
fh.setFormatter(logging.Formatter('%(levelname)8s - %(message)s'))
# add the handlers to the logger
logger.addHandler(fh)

# log the arguments
logger.info('=' * 89)
logger.info('{:^89s}'.format('Arguments'))
logger.info('=' * 89)
for arg in vars(args):
    logger.info('{:25s}: {:10}'.format(arg, str(getattr(args, arg))))

###############################################################################
# Load data
###############################################################################

# a single dictionary for the two traces
# they were collected separately to avoid information leaks
if args.load_corpus:
    dict_sys = Dictionary(path='{}/dict_sys'.format(args.data))
    dict_proc = Dictionary(path='{}/dict_proc'.format(args.data))
else:
    dict_sys = Dictionary()
    dict_proc = Dictionary()

if args.requests:
    corpus_train = CorpusReq('{}/train'.format(args.data), dict_sys, dict_proc,
                             args.max_length, args.limit, args.save_corpus,
                             args.load_corpus)

    corpus_test = CorpusReq('{}/test'.format(args.data), dict_sys, dict_proc,
                            args.max_length, args.limit, args.save_corpus,
                            args.load_corpus)
else:
    corpus_train = Corpus('{}/train'.format(args.data), dict_sys, dict_proc,
                          args.max_length, args.limit, args.save_corpus,
                          args.load_corpus)

    corpus_test = Corpus('{}/test'.format(args.data), dict_sys, dict_proc,
                         args.max_length, args.limit, args.save_corpus,
                         args.load_corpus)

if args.save_corpus:
    dict_sys.save(path='{}/dict_sys'.format(args.data))
    dict_proc.save(path='{}/dict_proc'.format(args.data))

# create a training set, a validation set and a test set
dataset_size = len(corpus_test)
indices = list(range(dataset_size))
np.random.shuffle(indices)
split = int(np.floor(args.valid * dataset_size))

valid_idx, test_idx = indices[split:], indices[:split]
valid_sampler = SubsetRandomSampler(valid_idx)
test_sampler = SubsetRandomSampler(test_idx)

mlm_train_loader = DataLoader(MLMDataset(corpus_train, args.p_mask),
                              batch_size=args.batch,
                              shuffle=True,
                              collate_fn=collate_fn,
                              pin_memory=True,
                              num_workers=0)
mlm_valid_loader = DataLoader(MLMDataset(corpus_test, args.p_mask),
                              batch_size=args.batch,
                              sampler=valid_sampler,
                              collate_fn=collate_fn,
                              pin_memory=True,
                              num_workers=0)
mlm_test_loader = DataLoader(MLMDataset(corpus_test, args.p_mask),
                             batch_size=args.batch,
                             sampler=test_sampler,
                             collate_fn=collate_fn,
                             pin_memory=True,
                             num_workers=0)

lm_train_loader = DataLoader(LMDataset(corpus_train),
                             batch_size=args.batch,
                             shuffle=True,
                             collate_fn=collate_fn,
                             pin_memory=True,
                             num_workers=0)
lm_valid_loader = DataLoader(LMDataset(corpus_test),
                             batch_size=args.batch,
                             sampler=valid_sampler,
                             collate_fn=collate_fn,
                             pin_memory=True,
                             num_workers=0)
lm_test_loader = DataLoader(LMDataset(corpus_test),
                            batch_size=args.batch,
                            sampler=test_sampler,
                            collate_fn=collate_fn,
                            pin_memory=True,
                            num_workers=0)

###############################################################################
# Data analysis
###############################################################################

n_syscall = len(dict_sys)
n_process = len(dict_proc)

logger.info('=' * 89)
logger.info('{:^89s}'.format('Vocabulary'))
logger.info('=' * 89)
logger.info('{:25}: {:10d}'.format('Vocabulary size', n_syscall))
logger.info('{:25}: {:10d}'.format('Number of process', n_process))

datatset_stats(
    corpus_train,
    dict_sys,
    dict_proc,
    args.plot_hist,
    name='{}_train'.format('request' if 'request' in args.data else 'startup'))
datatset_stats(
    corpus_test,
    dict_sys,
    dict_proc,
    args.plot_hist,
    name='{}_test'.format('request' if 'request' in args.data else 'startup'))

###############################################################################
# Build and train the model
###############################################################################

if args.device == 'auto':
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

if args.load_model is None:
    if args.model.lower() == 'ngram':
        logger.info('=' * 89)
        logger.info('{:^89s}'.format('{}-gram model'.format(args.order)))
        logger.info('=' * 89)
        start = time()
        # no validation set for ngrams
        pred = nltk_ngram([s for i, s in enumerate(corpus_train.call)],
                          dict_sys.idx2word, args.order)
        logger.info('Training done in {}'.format(
            timedelta(seconds=round(time() - start))))
        train_acc = ngram_acc(pred,
                              [s for i, s in enumerate(corpus_train.call)],
                              dict_sys.idx2word, args.order)
        logger.info('{:25}: {:6.1%}'.format('Train set accuracy', train_acc))
        val_acc = ngram_acc(pred, [s for i, s in enumerate(corpus_test.call)],
                            dict_sys.idx2word, args.order)
        logger.info('{:25}: {:6.1%}'.format('Validation set accuracy',
                                            val_acc))
        sys.exit()
    elif args.model.lower() == 'lstm':
        model = LSTM(n_syscall, n_process, args)
    elif args.model.lower() == 'transformer':
        model = Transformer(n_syscall, n_process, args)

    if len(args.device.split(',')) > 1:
        ids = [int(x.split(":")[1]) for x in args.device.split(',')]
        model = nn.DataParallel(model, device_ids=ids)
        args.device = 'cuda'

    model.to(args.device)

    model_params = filter(lambda p: p.requires_grad, model.parameters())
    train_params = sum([np.prod(p.size()) for p in model_params])
    logger.info('{:25}: {:10d}'.format('Trainable parameters', train_params))

    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    mlm_done, lm_done = 0, 0

    if args.mlm_epochs > 0:
        logger.info('=' * 89)
        logger.info('{:^89s}'.format('Pre-training using MLM on {}'.format(
            args.device)))
        logger.info('=' * 89)

        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        _train_loss, _val_loss, _train_acc, _val_acc = train(
            model,
            mlm_train_loader,
            mlm_valid_loader,
            args.mlm_epochs,
            args.early_stopping,
            optimizer,
            criterion,
            n_syscall,
            args.eval,
            args.device,
            mlm=True,
            chk=args.checkpoint,
            it=args.it)

        mlm_done = len(_train_loss)
        train_loss += _train_loss
        val_loss += _val_loss
        train_acc += _train_acc
        val_acc += _val_acc

        # load the best saved model
        with open('models/{}'.format(args.it), 'rb') as f:
            model = torch.load(f)
            logger.info('Best model loaded')

        # evaluate the model
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        test_loss, test_acc = evaluate(model,
                                       mlm_test_loader,
                                       criterion,
                                       n_syscall,
                                       args.device,
                                       mlm=True)
        logger.info('=' * 89)
        logger.info('Test loss {:5.3f} acc {:5.1%}'.format(
            test_loss, test_acc))

    if args.lm_epochs > 0:
        logger.info('=' * 89)
        logger.info('{:^89s}'.format('Fine-tuning using LM on {}'.format(
            args.device)))
        logger.info('=' * 89)

        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        _train_loss, _val_loss, _train_acc, _val_acc = train(
            model,
            lm_train_loader,
            lm_valid_loader,
            args.lm_epochs,
            args.early_stopping,
            optimizer,
            criterion,
            n_syscall,
            args.eval,
            args.device,
            mlm=False,
            chk=args.checkpoint,
            it=args.it)

        lm_done = len(_train_loss)
        train_loss += _train_loss
        val_loss += _val_loss
        train_acc += _train_acc
        val_acc += _val_acc

        # load the best saved model
        with open('models/{}'.format(args.it), 'rb') as f:
            model = torch.load(f)
            logger.info('Best model loaded')

        # evaluate the model
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        test_loss, test_acc = evaluate(model,
                                       lm_test_loader,
                                       criterion,
                                       n_syscall,
                                       args.device,
                                       mlm=False)
        logger.info('=' * 89)
        logger.info('Test loss {:5.3f} acc {:5.1%}'.format(
            test_loss, test_acc))

    plot_loss(train_loss, val_loss, mlm_done, lm_done, args.it)
    plot_accuracy(train_acc, val_acc, mlm_done, lm_done, args.it)

else:
    with open('models/{}'.format(args.load_model), 'rb') as f:
        model = torch.load(f)
        logger.info('Model {} loaded'.format(args.load_model))

###############################################################################
# Model analysis
###############################################################################

# Not implemented