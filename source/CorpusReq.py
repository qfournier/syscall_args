"""Corpus is a collection of requests.
"""
import torch
import numpy as np

from source.functions import load_trace
from source.functions import get_events
from source.functions import get_individual_requests

import logging
logger = logging.getLogger('logger')


def to_int32_tensor(array):
    return torch.tensor(array).type(torch.int32)


class CorpusReq(object):
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

            for n_request, request in enumerate(
                    get_individual_requests(events)):

                if n_request + 1 % 100 == 0:
                    print('\rProcessing request {}'.format(n_request,end=''))

                if limit and n_request > limit:
                    break

                _call, _entry, _time, _proc = [], [], [], []
                _pid, _tid, _ret = [], [], []

                for event in request:
                    # get system call name
                    name = event['name'].replace('entry_', '').replace(
                        'exit_', '').replace('syscall_', '')
                    # add system call name to dictionary
                    self.dict_sys.add_word(name)
                    # append system call name
                    _call.append(self.dict_sys.word2idx[name])
                    # append 0 if entry, else 1
                    _entry.append(1 if 'entry' in event['name'] else 2)
                    # append time
                    _time.append(event['timestamp'])
                    # add process name to dicitonary
                    self.dict_proc.add_word(event['procname'])
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

                # add request to dataset, if requests is longer than max_length
                # split at every multiple of max_length
                if max_length is None:
                    self.call.append(_call)
                    self.entry.append(_entry)
                    self.proc.append(_proc)
                    self.pid.append(_pid)
                    self.tid.append(_tid)
                    self.ret.append(_ret)
                    # offset timestamps to avoid numerical instability
                    self.time.append([t - _time[0] for t in _time])
                else:
                    for j in range(0, len(_call) - 1, max_length):
                        self.call.append(_call[j:j + max_length])
                        self.entry.append(_entry[j:j + max_length])
                        self.proc.append(_proc[j:j + max_length])
                        self.pid.append(_pid[j:j + max_length])
                        self.tid.append(_tid[j:j + max_length])
                        self.ret.append(_ret[j:j + max_length])
                        # offset timestamps to avoid numerical instability
                        self.time.append(
                            [t - _time[j] for t in _time[j:j + max_length]])
            if save:
                self.save(path)
        print('\r')

        # converting list into tensors
        self.call = list(map(to_int32_tensor, self.call))
        self.entry = list(map(to_int32_tensor, self.entry))
        self.time = list(map(to_int32_tensor, self.time))
        self.proc = list(map(to_int32_tensor, self.proc))
        self.pid = list(map(to_int32_tensor, self.pid))
        self.tid = list(map(to_int32_tensor, self.tid))
        self.ret = list(map(to_int32_tensor, self.ret))

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