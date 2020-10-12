from math import ceil

import torch
from torch.utils.data import Dataset


class MLMDataset(Dataset):
    """Language modeling dataset."""
    def __init__(self, corpus, p_mask):
        self.corpus = corpus
        self.p_mask = p_mask

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        size = len(self.corpus.call[idx])

        # generate a random mask
        mask = torch.zeros(size).scatter_(
            0, torch.randint(0, size - 1, (ceil(self.p_mask * size), )),
            1.).type(torch.bool)

        # generate a vector of random values in [0, 1]
        rnd = torch.rand(size) * mask

        # if rnd_i < 0.1, keep same word (10%)
        # if 0.1 < rnd_i < 0.2, random word (10%)
        # if 0.2 < rnd_i, mask word (80%)
        mask_call = torch.where(
            rnd < 0.1, self.corpus.call[idx],
            torch.where(rnd < 0.2,
                        torch.randint(1, len(self.corpus.dict_sys), (size, )),
                        torch.zeros(size, dtype=torch.int64)))

        # if rnd_i < 0.1, keep same argument (10%)
        # if 0.1 < rnd_i < 0.2, keep same argument (10%)
        # if 0.2 < rnd_i, mask argument (80%)
        mask_entry = torch.where(rnd < 0.2, self.corpus.entry[idx],
                                 torch.zeros(size, dtype=torch.int64))
        mask_ret = torch.where(rnd < 0.2, self.corpus.ret[idx],
                               torch.zeros(size, dtype=torch.int64))
        mask_time = torch.where(rnd < 0.2, self.corpus.time[idx],
                                torch.zeros(size, dtype=torch.int64))
        mask_proc = torch.where(rnd < 0.2, self.corpus.proc[idx],
                                torch.zeros(size, dtype=torch.int64))
        mask_pid = torch.where(rnd < 0.2, self.corpus.pid[idx],
                               torch.zeros(size, dtype=torch.int64))
        mask_tid = torch.where(rnd < 0.2, self.corpus.tid[idx],
                               torch.zeros(size, dtype=torch.int64))
        mask_y = self.corpus.call[idx] * mask

        return mask_call, mask_entry, mask_ret, mask_time, mask_proc, mask_pid, mask_tid, mask_y