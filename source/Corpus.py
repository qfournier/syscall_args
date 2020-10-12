"""Corpus is a collection of sequences of events.
"""
import torch
import numpy as np

from source.functions import load_trace
from source.functions import get_events

import logging
logger = logging.getLogger('logger')


def to_int64_tensor(array):
    return torch.tensor(array).type(torch.int64)


class Corpus(object):
    def __init__(self, path, dict_sys, dict_proc, max_length, limit, save,
                 load):
        # dictionary of system call names
        self.dict_sys = dict_sys
        # dictionary of process names
        self.dict_proc = dict_proc

        if load:
            data = np.load('{}/corpus.npz'.format(path), allow_pickle=True)
            self.call = data['call']
            self.entry = data['entry']
            self.time = data['time']
            self.proc = data['proc']
            self.pid = data['pid']
            self.tid = data['tid']
            self.ret = data['ret']
        else:
            # item: list of system call names as integers
            self.call = []
            # item: list of entry or exit (0:padding, 1:entry, 2:exit)
            self.entry = []
            # item: list of integers (time in ns)
            self.time = []
            # item: list of process names as integer
            self.proc = []
            # item: list of pids
            self.pid = []
            # item: list of tids
            self.tid = []
            # item: list of return values (0:padding, 1:success, 2:failure)
            self.ret = []

            # Load data
            trace = load_trace('{}/kernel/'.format(path))
            # mapping to consider the multiple way of denoting each argument
            # e.g., the tid may be stored as 'tid' or 'vtid'
            keys = {
                'vtid': 'tid',
                'tid': 'tid',
                'vpid': 'pid',
                'pid': 'pid',
                'procname': 'procname',
                'ret': 'ret'
            }
            events = get_events(trace, keys)
            n_seq = 0
            _call, _entry, _time, _proc = [], [], [], []
            _pid, _tid, _ret = [], [], []
            for i, event in enumerate(events, 1):

                if n_seq + 1 % 100 == 0:
                    print('\rProcessing sequences {}'.format(n_seq), end='')

                if limit and n_seq == limit:
                    break

                # get system call name
                name = event['name'].replace('entry_', '').replace(
                    'exit_', '').replace('syscall_', '')
                # add system call name to dictionary
                self.dict_sys.add_word(name)
                # append system call name
                _call.append(self.dict_sys.word2idx[name])
                # append entry (1), exit (2), or none (0)
                if 'entry' in event['name']:
                    _entry.append(1)
                elif 'exit' in event['name']:
                    _entry.append(2)
                else:
                    _entry.append(0)
                # append timestamp (with an offset to avoid numerical instability)
                _time.append(event['timestamp'] if i %
                             max_length == 1 else event['timestamp'] -
                             _time[0])
                # add process name to dicitonary
                self.dict_proc.add_word(event['procname'])
                # append process name
                _proc.append(self.dict_proc.word2idx[event['procname']])
                # append pid
                _pid.append(event['pid'])
                # append tid
                _tid.append(event['tid'])
                # append return value
                if 'entry' in event['name']:
                    _ret.append(0)
                elif event['ret'] >= 0:
                    _ret.append(1)
                else:
                    _ret.append(2)

                # add sequence of size max_length to dataset
                if i % max_length == 0:
                    self.call.append(_call)
                    self.entry.append(_entry)
                    self.proc.append(_proc)
                    self.pid.append(_pid)
                    self.tid.append(_tid)
                    self.ret.append(_ret)
                    self.time.append(_time)
                    # reset sequence
                    _call, _entry, _time, _proc = [], [], [], []
                    _pid, _tid, _ret = [], [], []
                    # increment number of sequences
                    n_seq += 1
            if save:
                self.save(path)
        print('\r')

        # converting list into tensors
        self.call = list(map(to_int64_tensor, self.call))
        self.entry = list(map(to_int64_tensor, self.entry))
        self.time = list(map(to_int64_tensor, self.time))
        self.proc = list(map(to_int64_tensor, self.proc))
        self.pid = list(map(to_int64_tensor, self.pid))
        self.tid = list(map(to_int64_tensor, self.tid))
        self.ret = list(map(to_int64_tensor, self.ret))

    def __len__(self):
        return len(self.call)

    def save(self, path):
        logger.debug("Saving corpus")
        np.savez('{}/corpus'.format(path),
                 call=self.call,
                 entry=self.entry,
                 time=self.time,
                 proc=self.proc,
                 pid=self.pid,
                 tid=self.tid,
                 ret=self.ret)
        logger.debug("Corpus saved")