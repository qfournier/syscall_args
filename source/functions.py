import torch
import operator
import numpy as np
import statistics
from time import time
from datetime import timedelta
import itertools as it
import babeltrace as bt
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

from nltk import ngrams
from nltk.lm import NgramCounter

import torch.nn.functional as F

import logging
logger = logging.getLogger('logger')

###############################################################################
# Trace processing
###############################################################################


def load_trace(path):
    """Load the trace located in path.

    Args:
        path (string): Path to the LTTng trace folder.

    Returns:
        babeltrace.TraceCollection: a collection of one trace.
    """
    trace_collection = bt.TraceCollection()
    trace_collection.add_trace(path, 'ctf')
    return trace_collection


def get_events(trace_collection, keys=None, syscall=True):
    """Return a generator of events. An event is a dict with the key the
    arguement's name.

    Args:
        trace_collection (babeltrace.TraceCollection): Trace from which
            to read the events.
        keys (dict, optional): dict of the multiple ways of the arguments
            to consider in addition to name and timestamp.
        syscall (bool, optional): only syscall should be considered

    Returns:
        generator: a generator of events.
    """
    return (
        {
            **{
                'name': event.name,
                'timestamp': event.timestamp
            },
            **{
                keys[k]: event[k]
                # scope 3 = Stream event context (procname, pid, tid)
                for k in event.field_list_with_scope(3) if keys and k in keys
            },
            **{
                keys[k]: event[k]
                # scope 5 = Event fields (return value)
                for k in event.field_list_with_scope(5) if keys and k in keys
            }
        } for event in trace_collection.events
        if not syscall or "syscall" in event.name)


def get_individual_requests(events):
    """Split individual requests delimited by accept4 and close/shutdown systam
    calls.

    Args:
        events (generator): Generator of event.
    """
    # dictionary of threads
    threads = {}

    for event in events:
        tid = event['tid']
        # start the request for a specific thread
        if event['name'] == "syscall_entry_accept4" and event['procname'] == 'apache2':
            threads[tid] = []

        # add event in all currently recording thread
        for req in threads.values():
            req.append(event)

        # end the request for a specific thread
        if event['name'] == "syscall_exit_close" and event[
                'procname'] == 'apache2' and tid in threads.keys():
            yield threads[tid]
            del threads[tid]


###############################################################################
# Data
###############################################################################


def collate_fn(data):
    """Construct a bacth by padding the sequence to the size of the longest.

    Args:
        data (tuple): tensors

    Returns:
        tuple: padded tensors
    """
    # Construct a bacth by padding the sequence to the size of the longest
    size = [len(_x) for _x in list(zip(*data))[0]]
    pad_data = [torch.zeros(len(size), max(size)) for _ in zip(*data)]
    pad_mask = torch.ones(len(size), max(size))

    for i, _data in enumerate(data):
        end = size[i]
        pad_mask[i, :end] = 0
        for j, d in enumerate(_data):
            pad_data[j][i, :end] = d

    return [d.type(torch.int64)
            for d in pad_data] + [pad_mask.type(torch.bool)]

    # not yet supported by yapf and black formatter (allowed in Python 3.8)
    # return *[d.type(torch.int64) for d in pad_data],
    #  pad_mask.type(torch.bool)


def datatset_stats(corpus, dict_sys, dict_proc, plot, name=''):

    logger.info('=' * 89)
    logger.info('{:^89s}'.format('{} Data'.format(name)))
    logger.info('=' * 89)
    lengths = [len(x) for x in corpus.call]
    logger.info('{:25}: {:10d}'.format('Number of sequence', len(lengths)))

    if plot:
        plot_hist(corpus.call, dict_sys.idx2word, "syscall_{}".format(name))
        plot_hist(corpus.proc, dict_proc.idx2word, "process_{}".format(name))


###############################################################################
# N-gram
###############################################################################


def nltk_ngram(call, vocab, n):
    """Compute n-grams using the nltk library.

    Args:
        call (list): list of system call name (as integer) sequences
        vocab (list): mapping from integer to system call name
        n (int): the n-gram order

    Returns:
        tuple: list of n-grams, list of n-grams count, list of n-grams
         probability, dictionary {context: prediction}
    """
    # convert sequences of integer into sequences of string and call NLTK
    counter = NgramCounter([ngrams([vocab[w] for w in s], n) for s in call])
    # store predictions in a dictionary {context: prediction}
    return {
        context: max(counter[context].items(), key=operator.itemgetter(1))[0]
        for context in it.product(vocab, repeat=n - 1) if counter[context]
    }


def ngram_acc(pred, call, vocab, order):
    """Compute the n-grams accuracy.

    Args:
        pred (dict): dictionary {context: (prediction, probability)}
        call (list): list of system call name sequences as integer
        vocab (list): mapping from integer to system call name
        order (int): the n-gram order

    Returns:
        float: accuracy
    """
    acc = (1 if tuple(s[i:i + order - 1]) in pred.keys()
           and pred[tuple(s[i:i + order - 1])] == s[i + order] else 0
           for s in map(lambda x: [vocab[w] for w in x], call)
           for i in range(len(s) - order - 1))
    return statistics.mean(acc)


###############################################################################
# Train & evaluate the model
###############################################################################


# https://github.com/bearpaw/pytorch-classification/blob/cc9106d598ff1fe375cc030873ceacfea0499d77/utils/eval.py
def correct(output, target, tokens):
    """Computes the number of correct predictions.

    Args:
        output (torch.tensor): output of the model
        target (torch.tensor): masked labels
        tokens (int): vocabulary size

    Returns:
        int: number of correct predictions
    """
    with torch.no_grad():
        mask = target.type(torch.bool)
        labels = torch.masked_select(target, mask)
        mask = mask.unsqueeze(-1).expand_as(output)
        output = torch.masked_select(output, mask).reshape(-1, tokens)
        _, predicted = torch.max(output, dim=-1)
    return (predicted == labels).sum().item()


def train(model, train_loader, valid_loader, epochs, early_stopping, optimizer,
          criterion, tokens, eval, device, mlm, chk, it):

    model.train()
    steps = 1
    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    total_train_loss, total_train_pred, total_train_correct = 0, 0, 0
    n_batch_train = len(train_loader)
    best_val_acc = 0

    start, mlm_time = time(), time()

    # pretrain with MLM
    for epoch in range(epochs):
        for i, data in enumerate(train_loader, 1):
            # send tensors to device
            data = [d.to(device) for d in data]

            # get the pad_mask and the output from the data
            data, y, pad_mask = data[:-2], data[-2], data[-1]

            # get prediction
            out = model(*data, pad_mask, mlm, chk)

            # compute loss
            loss = criterion(out.reshape(-1, tokens), y.reshape(-1))

            # update the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # collect metric
            total_train_loss += float(loss.item())
            total_train_pred += float(torch.nonzero(y).size(0))
            total_train_correct += correct(out, y, tokens)

            # every 1000 updates, evaluate and collect metrics
            if (epoch * n_batch_train + i) % eval == 0:
                # get average duration per batch in ms
                avg_d = (time() - start)

                # evaluate model
                _val_loss, _val_acc = evaluate(model, valid_loader, criterion,
                                               tokens, device, mlm)

                # append metric
                train_loss.append(total_train_loss / eval)
                train_acc.append(total_train_correct / total_train_pred)
                val_loss.append(_val_loss)
                val_acc.append(_val_acc)

                # display summary of the epochs
                summary = [
                    'Updates {:6d}'.format(epoch * n_batch_train + i),
                    '(epoch {:3d} '.format(epoch + 1),
                    '@ {:3.0f}ms/batch)'.format(avg_d),
                    'loss {:5.3f} '.format(train_loss[-1]),
                    'val_loss {:5.3f}'.format(val_loss[-1]),
                    'acc {:5.1%} '.format(train_acc[-1]),
                    'val_acc {:5.1%}'.format(val_acc[-1])
                ]
                logger.info(' '.join(summary))

                # save the model if the validation loss is the best so far
                if len(val_acc) == 1 or val_acc[-1] > best_val_acc + 0.001:
                    with open('models/{}'.format(it), 'wb') as f:
                        torch.save(model, f)
                        logger.debug('Done: save model')
                        best_val_acc = val_acc[-1]
                        steps = 1
                else:
                    steps += 1

                # early stopping
                if early_stopping and steps > early_stopping:
                    logger.info('Early stopping')
                    logger.info('Training done in {}'.format(
                        timedelta(seconds=round(time() - mlm_time))))
                    return train_loss, val_loss, train_acc, val_acc

                # prepare to resume training
                model.train()
                total_train_loss = 0
                total_train_pred = 0
                total_train_correct = 0
                start = time()

    logger.info('Training done in {}'.format(
        timedelta(seconds=round(time() - mlm_time))))
    return train_loss, val_loss, train_acc, val_acc


def evaluate(model, test_loader, criterion, tokens, device, mlm):

    # evaluate model
    model.eval()
    total_val_loss, total_val_pred, total_val_correct = 0, 0, 0
    with torch.no_grad():
        for data in test_loader:
            # send tensors to device
            data = [d.to(device) for d in data]

            # get the pad_mask and the output from the data
            data, y, pad_mask = data[:-2], data[-2], data[-1]

            # get prediction
            out = model(*data, pad_mask, mlm=False, chk=False)

            # compute loss
            loss = criterion(out.reshape(-1, tokens), y.reshape(-1))

            # collect metric
            total_val_loss += float(loss.item())
            total_val_pred += float(torch.nonzero(y).size(0))
            total_val_correct += correct(out, y, tokens)

    return total_val_loss / len(
        test_loader), total_val_correct / total_val_pred


###############################################################################
# Visualization
###############################################################################


def plot_hist(x, mapping, name):
    # Pre-count to save memory
    count = [0 for _ in mapping]
    _, count = np.unique([int(w) for _x in x for w in _x], return_counts=True)
    # Convert to probability and add 0 for mask
    count = [0] + [c / sum(count) for c in count]
    # Sort and keep the 20 most probable
    count, mapping = map(list, zip(*sorted(zip(count, mapping))))
    count = count[-9:]
    mapping = mapping[-9:]
    # Add 'other'
    count.insert(0, 1 - sum(count))
    mapping.insert(0, 'other')
    # Create figure
    fig = plt.figure(figsize=(10, 6), tight_layout=True)
    ax = fig.add_subplot(111)
    # Change font to Helvetica
    plt.rc('font', family='Helvetica')
    # Set colors
    dark_gray = '#808080'
    light_gray = '#D3D3D3'
    # Plot
    bins = [x - 0.5 for x in range(len(mapping) + 1)]
    n, bins, patches = plt.hist(mapping,
                                bins=bins,
                                weights=count,
                                rwidth=0.8,
                                orientation='horizontal')
    # Hide the bottom, right and top spines and ticks
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tick_params(axis='y',
                    which='both',
                    left=False,
                    right=False,
                    labelleft=True)
    # Change color of other
    patches[0].set_fc(light_gray)
    # For each bar: Place a label
    for i, (c, p) in enumerate(zip(count, patches)):
        x_value = p.get_width()
        y_value = p.get_y() + p.get_height() / 2
        if x_value > 0.01:
            plt.annotate("{:.0%}".format(c), (x_value, y_value),
                         color='w' if i != 0 else 'k',
                         xytext=(-2, 0),
                         textcoords="offset points",
                         va='center',
                         ha='right')
    # Change colors and labels of Y axis
    ax.spines["left"].set_color(dark_gray)
    # Add the name to the y-axis
    ax.tick_params(axis='y', colors=dark_gray)
    ax.set_yticks(range(len(mapping)))
    ax.set_yticklabels(mapping)
    ax.tick_params(axis='x', colors='w')
    # Make title the length of the graph
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="11%", pad=0)
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    for x in cax.spines:
        cax.spines[x].set_visible(False)
    cax.spines['top'].set_visible(False)
    cax.set_facecolor(dark_gray)
    at = AnchoredText("Histogram of {} names".format(name),
                      loc=6,
                      pad=0,
                      prop=dict(backgroundcolor=dark_gray, size=20, color='w'))
    at.patch.set_edgecolor('none')
    cax.add_artist(at)
    # Save figure
    plt.savefig('figures/dataset/hist_{}.png'.format(name))
    plt.close()


def plot_loss(train, val, mlm_epochs, lm_epochs, it):
    """Plot the loss.

    Args:
        train (list): list of loss per epoch on the training set
        val (list): list of loss per epoch on the validation set
        mlm_epochs (int): number of mlm epochs completed
        lm_epochs (int): number of lm epochs completed
        it (int): iteration number (to name the figure)
    """
    mlm_epochs, lm_epochs = mlm_epochs - 1, lm_epochs - 1
    # Create figure
    fig = plt.figure(figsize=(10, 6), tight_layout=True)
    ax = fig.add_subplot(111)
    # Change font to Helvetica
    plt.rc('font', family='Helvetica')
    # Set colors
    dark_gray = '#808080'
    light_gray = '#D3D3D3'
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Change axes and tick color
    ax.spines["bottom"].set_color(dark_gray)
    ax.tick_params(axis='x', colors=dark_gray)
    ax.spines["left"].set_color(dark_gray)
    ax.tick_params(axis='y', colors=dark_gray)
    # Plot
    ax.plot(range(1, len(train) + 1), train, color='C0')
    ax.annotate('Train {:6.3f}'.format(train[-1]),
                xy=(len(train), train[-1]),
                xytext=(5, -5 if train[-1] < val[-1] else 5),
                size=12,
                textcoords='offset points',
                va='center',
                color='C0')
    ax.plot(range(1, len(val) + 1), val, color='C1')
    ax.annotate('Valid {:6.3f}'.format(val[-1]),
                xy=(len(val), val[-1]),
                xytext=(5, 5 if train[-1] < val[-1] else -5),
                size=12,
                textcoords='offset points',
                va='center',
                color='C1')
    mx, mn = max(*train, *val), min(*train, *val)
    # Vertical line delimiting MLM and LM
    if mlm_epochs > 0 and lm_epochs > 0:
        plt.vlines(mlm_epochs, mn, mx, colors=light_gray)
    # Increase left margin
    lim = ax.get_xlim()
    right = lim[1] + (lim[1] - lim[0]) * 0.1
    ax.set_xlim(lim[0], right)
    # Labels
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy')
    # Make title the length of the graph
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="11%", pad=0)
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    for x in cax.spines:
        cax.spines[x].set_visible(False)
    cax.spines['top'].set_visible(False)
    cax.set_facecolor(dark_gray)
    at = AnchoredText("Cross-entropy During Training",
                      loc=6,
                      pad=0,
                      prop=dict(backgroundcolor=dark_gray, size=20, color='w'))
    at.patch.set_edgecolor('none')
    cax.add_artist(at)
    # Save figure
    plt.savefig('figures/model/{}_loss.png'.format(it))
    plt.close()


def plot_accuracy(train, val, mlm_epochs, lm_epochs, it):
    """Plot the accuracy.

    Args:
        train (list): list of accuracy per epoch on the training set
        val (list): list of accuracy per epoch on the validation set
        mlm_epochs (int): number of mlm epochs completed
        lm_epochs (int): number of lm epochs completed
        it (int): iteration number (to name the figure)
    """
    mlm_epochs, lm_epochs = mlm_epochs - 1, lm_epochs - 1
    # Create figure
    fig = plt.figure(figsize=(10, 6), tight_layout=True)
    ax = fig.add_subplot(111)
    # Change font to Helvetica
    plt.rc('font', family='Helvetica')
    # Set colors
    dark_gray = '#808080'
    light_gray = '#D3D3D3'
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Change axes and tick color
    ax.spines["bottom"].set_color(dark_gray)
    ax.tick_params(axis='x', colors=dark_gray)
    ax.spines["left"].set_color(dark_gray)
    ax.tick_params(axis='y', colors=dark_gray)
    # Plot
    mn, mx = min(*train, *val), max(*train, *val)
    ax.plot(range(1, len(train) + 1), train, color='C0')
    ax.annotate('Train {:6.1%}'.format(train[-1]),
                xy=(len(train), train[-1]),
                xytext=(5, -5 if train[-1] < val[-1] else 5),
                size=12,
                textcoords='offset points',
                va='center',
                color='C0')
    ax.plot(range(1, len(val) + 1), val, color='C1')
    ax.annotate('Valid {:6.1%}'.format(val[-1]),
                xy=(len(val), val[-1]),
                xytext=(5, 5 if train[-1] < val[-1] else -5),
                size=12,
                textcoords='offset points',
                va='center',
                color='C1')
    # Vertical line delimiting MLM and LM
    if mlm_epochs > 0 and lm_epochs > 0:
        plt.vlines(mlm_epochs, mn, mx, colors=light_gray)
    # Increase left margin
    lim = ax.get_xlim()
    right = lim[1] + (lim[1] - lim[0]) * 0.1
    ax.set_xlim(lim[0], right)
    # Labels
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    # Make title the length of the graph
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="11%", pad=0)
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    for x in cax.spines:
        cax.spines[x].set_visible(False)
    cax.spines['top'].set_visible(False)
    cax.set_facecolor(dark_gray)
    at = AnchoredText("Accuracy During Training",
                      loc=6,
                      pad=0,
                      prop=dict(backgroundcolor=dark_gray, size=20, color='w'))
    at.patch.set_edgecolor('none')
    cax.add_artist(at)
    # Save figure
    plt.savefig('figures/model/{}_accuracy.png'.format(it))
    plt.close()